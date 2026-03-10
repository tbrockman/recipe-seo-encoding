#!/usr/bin/env python3
"""
Arithmetic Steganography via Language Model
============================================

Hides secret messages in natural-looking text by exploiting the probability
distributions of a language model. Each generated token encodes bits of the
secret message by constraining which token is selected from the model's
predicted distribution, using arithmetic coding for optimal efficiency.

Requirements:
    pip install torch transformers bitsandbytes accelerate numpy

Usage:
    # Encode a secret message
    python stego.py encode \
        --message "attack at dawn" \
        --prompt "The best thing about making scrambled eggs is: "

    # Decode from stego text (prompt must match!)
    python stego.py decode \
        --stego-file stego_output.txt \
        --prompt "The best thing about making scrambled eggs is: "

    # Use a specific model
    python stego.py encode \
        --model "mistralai/Mistral-7B-v0.3" \
        --message "secret message" \
        --prompt "My grandmother always"

CRITICAL: Both encoder and decoder must use the exact same model, quantization,
          top-k value, and prompt. Any difference causes decoding failure.
"""

import argparse
import struct
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Arithmetic coding constants
# We use 64-bit integer precision. The interval [lo, hi] lives in [0, WHOLE).
# Renormalization keeps the interval width >= QUARTER, so we never run out
# of precision regardless of message length.
# ---------------------------------------------------------------------------
PRECISION = 48  # bits of precision in the interval
WHOLE = 1 << PRECISION
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER
MASK = WHOLE - 1


# ---------------------------------------------------------------------------
# Bit I/O helpers
# ---------------------------------------------------------------------------
class BitReader:
    """Reads individual bits from a byte sequence, MSB first."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self._total_read = 0

    def read(self) -> int:
        """Read one bit. Returns 0 after data is exhausted (implicit padding)."""
        if self.byte_pos >= len(self.data):
            self._total_read += 1
            return 0  # zero-padding beyond the message
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        self._total_read += 1
        return bit

    @property
    def total_bits_read(self) -> int:
        return self._total_read


class BitWriter:
    """Collects individual bits and converts to bytes."""

    def __init__(self):
        self.bits: list[int] = []

    def write(self, bit: int):
        self.bits.append(bit & 1)

    def write_with_pending(self, bit: int, pending: int):
        """Write a bit followed by `pending` copies of its complement."""
        self.write(bit)
        for _ in range(pending):
            self.write(bit ^ 1)

    def to_bytes(self) -> bytes:
        # Pad to a full byte
        padded = self.bits + [0] * ((8 - len(self.bits) % 8) % 8)
        out = bytearray()
        for i in range(0, len(padded), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | padded[i + j]
            out.append(byte)
        return bytes(out)

    def __len__(self):
        return len(self.bits)


# ---------------------------------------------------------------------------
# CDF construction from logits
# ---------------------------------------------------------------------------
def build_cdf(
    logits: torch.Tensor,
    top_k: int = 256,
    cdf_total: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build an integer cumulative frequency table from model logits.

    1. Take top-k tokens by logit value.
    2. Softmax over those to get probabilities.
    3. Quantize to integer frequencies summing to `cdf_total`.
    4. Build cumulative frequency array.

    Returns:
        token_ids:  int array of shape (K,)    — the token IDs in this CDF
        cum_freq:   int array of shape (K+1,)  — cumulative frequencies
                    cum_freq[0] = 0, cum_freq[-1] = cdf_total
        cdf_total:  the total frequency mass
    """
    # Top-k filtering
    values, indices = torch.topk(logits.float(), min(top_k, logits.shape[-1]))

    # Softmax to probabilities
    probs = torch.softmax(values, dim=-1)

    # Quantize to integer frequencies, minimum 1 per token
    freqs = torch.clamp((probs * cdf_total).floor().long(), min=1)

    # Correct rounding error so frequencies sum to exactly cdf_total
    diff = cdf_total - freqs.sum().item()
    if diff > 0:
        # Distribute surplus to highest-probability tokens
        freqs[0] += diff
    elif diff < 0:
        # Remove from highest-prob tokens (preserving min=1)
        for i in range(len(freqs)):
            take = min(-diff, freqs[i].item() - 1)
            freqs[i] -= take
            diff += take
            if diff == 0:
                break

    # Build cumulative frequencies
    cum_freq = np.zeros(len(freqs) + 1, dtype=np.int64)
    freq_np = freqs.cpu().numpy()
    np.cumsum(freq_np, out=cum_freq[1:])
    cum_freq[-1] = cdf_total  # ensure exact

    token_ids = indices.cpu().numpy()
    return token_ids, cum_freq, cdf_total


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class StegoModel:
    """Wraps a HuggingFace causal LM for token-by-token logit access."""

    def __init__(self, model_name: str, device: str = "cuda", quantize_4bit: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {model_name} ({'4-bit' if quantize_4bit else 'fp16'})")
        load_kwargs = dict(torch_dtype=torch.float16, device_map=device)

        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("Warning: bitsandbytes not found, loading in fp16")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.device = device
        print("Model loaded.\n")

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def tokenize_no_special(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def logits_at(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for the next token given input_ids context."""
        with torch.no_grad():
            return self.model(input_ids).logits[0, -1, :]

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Encoder: secret bits → stego text
#
# This is conceptually an arithmetic DECODER: it reads bits from the secret
# message and emits symbols (tokens). The secret bitstream defines a point
# in [0, 1), and we "decode" that point into a token sequence using the
# model's probability distributions as the codebook.
# ---------------------------------------------------------------------------
def encode(
    model: StegoModel,
    secret: bytes,
    prompt: str,
    top_k: int = 256,
    max_tokens: int = 512,
) -> str:
    """
    Encode `secret` bytes into natural text, guided by `prompt`.

    The output text is statistically close to normal model output for the
    given prompt — each token was a plausible next token in context.
    """
    # Prepend a 4-byte big-endian length header so the decoder knows
    # how many bytes to extract.
    payload = struct.pack(">I", len(secret)) + secret
    reader = BitReader(payload)
    total_payload_bits = len(payload) * 8

    # --- Arithmetic coder state ---
    lo = 0
    hi = WHOLE - 1

    # Fill the value register with the first PRECISION bits of the message.
    # This register represents "where in the current interval our target
    # point lies."
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | reader.read()

    # --- Generation loop ---
    input_ids = model.tokenize(prompt)
    generated: list[int] = []

    for step in range(max_tokens):
        logits = model.logits_at(input_ids)
        token_ids, cum_freq, freq_total = build_cdf(logits, top_k)

        # Which token's sub-interval contains our value?
        #
        # The current interval [lo, hi] is subdivided proportionally to
        # token frequencies. We scale `value` into [0, freq_total) and
        # find which bin it lands in.
        rng = hi - lo + 1
        scaled = ((value - lo + 1) * freq_total - 1) // rng

        # Binary search via numpy
        idx = int(np.searchsorted(cum_freq, scaled, side="right")) - 1
        idx = max(0, min(idx, len(token_ids) - 1))

        # Narrow the interval to this token's sub-interval
        sym_lo = int(cum_freq[idx])
        sym_hi = int(cum_freq[idx + 1])
        hi = lo + (rng * sym_hi) // freq_total - 1
        lo = lo + (rng * sym_lo) // freq_total

        # Emit the token
        token_id = int(token_ids[idx])
        generated.append(token_id)

        # Extend context for next step
        next_tok = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # --- Renormalization ---
        # Shift out the MSBs that lo and hi agree on, and read new
        # message bits into the value register to maintain precision.
        while True:
            if hi < HALF:
                # Both start with 0 — shift out
                lo = (lo << 1) & MASK
                hi = ((hi << 1) | 1) & MASK
                value = ((value << 1) | reader.read()) & MASK
            elif lo >= HALF:
                # Both start with 1 — subtract HALF, shift, restore
                lo = ((lo - HALF) << 1) & MASK
                hi = (((hi - HALF) << 1) | 1) & MASK
                value = (((value - HALF) << 1) | reader.read()) & MASK
            elif lo >= QUARTER and hi < THREE_QUARTER:
                # Underflow / near-convergence — shift out second MSB
                lo = ((lo - QUARTER) << 1) & MASK
                hi = (((hi - QUARTER) << 1) | 1) & MASK
                value = (((value - QUARTER) << 1) | reader.read()) & MASK
            else:
                break

        # Are we done? We need to consume all payload bits plus a full
        # register's worth of padding bits to flush the state.
        if reader.total_bits_read >= total_payload_bits + PRECISION:
            break

        # Stop on EOS
        if token_id == model.eos_id:
            break

        # Progress indicator
        bits_encoded = min(reader.total_bits_read, total_payload_bits)
        if step % 20 == 0:
            pct = bits_encoded / total_payload_bits * 100
            bpt = bits_encoded / (step + 1) if step > 0 else 0
            print(
                f"  step {step:4d} | {bits_encoded}/{total_payload_bits} bits "
                f"({pct:.0f}%) | {bpt:.1f} bits/token"
            )

    stego_text = model.detokenize(generated)

    # Final stats
    bits_encoded = min(reader.total_bits_read, total_payload_bits)
    n_tokens = len(generated)
    print(f"\n  Done: {n_tokens} tokens, ~{bits_encoded / n_tokens:.2f} bits/token")
    print(f"  Payload: {len(secret)} bytes in {n_tokens} tokens ({len(stego_text)} chars)")

    return stego_text


# ---------------------------------------------------------------------------
# Decoder: stego text → secret bits
#
# This is conceptually an arithmetic ENCODER: it reads symbols (tokens from
# the stego text) and outputs bits (the secret message). It narrows the
# interval exactly as the encoder did, and the MSBs that become decided
# during renormalization ARE the recovered message bits.
# ---------------------------------------------------------------------------
def decode(
    model: StegoModel,
    stego_text: str,
    prompt: str,
    top_k: int = 256,
) -> bytes:
    """
    Decode a secret message from stego text.

    The model and prompt must be identical to those used for encoding.
    """
    stego_ids = model.tokenize_no_special(stego_text)
    input_ids = model.tokenize(prompt)

    # --- Arithmetic coder state ---
    lo = 0
    hi = WHOLE - 1
    pending = 0  # pending bits for underflow resolution

    writer = BitWriter()

    for i, token_id in enumerate(stego_ids):
        logits = model.logits_at(input_ids)
        token_ids, cum_freq, freq_total = build_cdf(logits, top_k)

        # Find this token in our top-k vocabulary
        matches = np.where(token_ids == token_id)[0]
        if len(matches) == 0:
            # Token fell outside top-k. This means the encoder wouldn't have
            # chosen it, so something is wrong — but we continue gracefully.
            print(f"  Warning: token {token_id!r} not in top-{top_k} at position {i}")
            next_tok = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            continue

        idx = int(matches[0])

        # Narrow interval identically to the encoder
        rng = hi - lo + 1
        sym_lo = int(cum_freq[idx])
        sym_hi = int(cum_freq[idx + 1])
        hi = lo + (rng * sym_hi) // freq_total - 1
        lo = lo + (rng * sym_lo) // freq_total

        # Update context
        next_tok = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # --- Renormalization ---
        # When the MSBs of lo and hi agree, those bits are decided —
        # they are recovered message bits. Pending bits handle the
        # underflow case where lo and hi straddle a boundary.
        while True:
            if hi < HALF:
                # MSB is 0 for both
                writer.write_with_pending(0, pending)
                pending = 0
                lo = (lo << 1) & MASK
                hi = ((hi << 1) | 1) & MASK
            elif lo >= HALF:
                # MSB is 1 for both
                writer.write_with_pending(1, pending)
                pending = 0
                lo = ((lo - HALF) << 1) & MASK
                hi = (((hi - HALF) << 1) | 1) & MASK
            elif lo >= QUARTER and hi < THREE_QUARTER:
                # Underflow: lo = 01..., hi = 10... — MSBs don't agree yet
                # but will on the next decided bit. Track as pending.
                pending += 1
                lo = ((lo - QUARTER) << 1) & MASK
                hi = (((hi - QUARTER) << 1) | 1) & MASK
            else:
                break

    # Flush: emit one final bit to resolve any remaining pending bits
    if lo >= QUARTER:
        writer.write_with_pending(1, pending)
    else:
        writer.write_with_pending(0, pending)

    # Convert recovered bits to bytes and parse the length-prefixed message
    raw = writer.to_bytes()

    if len(raw) < 4:
        raise ValueError(f"Decoded only {len(raw)} bytes — not enough for length header")

    msg_len = struct.unpack(">I", raw[:4])[0]
    print(f"  Decoded length header: {msg_len} bytes")
    print(f"  Total recovered: {len(raw)} bytes ({len(writer)} bits)")

    if msg_len > len(raw) - 4:
        print(f"  Warning: expected {msg_len} bytes but only have {len(raw) - 4}")
        msg_len = len(raw) - 4

    return raw[4 : 4 + msg_len]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Arithmetic Steganography via Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stego.py encode --message "meet me at noon" \\
      --prompt "The best scrambled eggs I ever had"

  python stego.py decode --stego-file stego_output.txt \\
      --prompt "The best scrambled eggs I ever had"

  python stego.py encode --message "hello" \\
      --model mistralai/Mistral-7B-v0.3 \\
      --prompt "Every Sunday morning my grandmother would"
        """,
    )

    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B",
        help="HuggingFace model name/path (default: Llama-3.2-3B)",
    )
    parser.add_argument(
        "--prompt",
        default="You are writing a typical online recipe preamble, aiming to gamify SEO. Write your recipe about scrambled eggs: ",
        help="Context prompt (MUST match between encode/decode)",
    )
    parser.add_argument("--message", help="Secret message to encode (encode mode)")
    parser.add_argument("--stego-text", help="Stego text string to decode")
    parser.add_argument("--stego-file", help="File containing stego text to decode")
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Top-k vocabulary size for CDF (default: 256)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses fp16; needs more VRAM)",
    )
    parser.add_argument(
        "--output",
        default="stego_output.txt",
        help="Output file for stego text (encode mode)",
    )

    args = parser.parse_args()

    # Load model
    stego_model = StegoModel(
        args.model,
        device=args.device,
        quantize_4bit=not args.no_4bit,
    )

    if args.mode == "encode":
        secret_msg = args.message or input("Enter secret message: ")
        secret_bytes = secret_msg.encode("utf-8")

        print(f"Secret: {len(secret_bytes)} bytes ({len(secret_bytes) * 8} bits)")
        print(f"Prompt: {args.prompt!r}")
        print(f"Top-k:  {args.top_k}")
        print()

        stego_text = encode(
            stego_model,
            secret_bytes,
            args.prompt,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )

        print("\n" + "=" * 70)
        print("STEGO TEXT (paste this wherever recipe preambles go):")
        print("=" * 70)
        print(stego_text)
        print("=" * 70)

        with open(args.output, "w") as f:
            f.write(stego_text)
        print(f"\nSaved to {args.output}")

    elif args.mode == "decode":
        if args.stego_file:
            with open(args.stego_file) as f:
                stego_text = f.read()
        elif args.stego_text:
            stego_text = args.stego_text
        else:
            print("Paste stego text (Ctrl-D when done):")
            stego_text = sys.stdin.read()

        print(f"Stego text: {len(stego_text)} chars")
        print(f"Prompt: {args.prompt!r}")
        print(f"Top-k:  {args.top_k}")
        print()

        try:
            recovered = decode(
                stego_model,
                stego_text,
                args.prompt,
                top_k=args.top_k,
            )
            print("\n" + "=" * 70)
            print("RECOVERED SECRET MESSAGE:")
            print("=" * 70)
            print(recovered.decode("utf-8", errors="replace"))
            print("=" * 70)
        except Exception as e:
            print(f"\nDecode failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()