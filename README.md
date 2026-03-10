# recipe-seo-encoding

> [!WARNING]
> This project is **entirely vibe-coded** <sup>(likely partially plagiarized from [this repository](https://github.com/artkpv/arithmetic-coding-steganography))</sup> and the author is a _dumb-dumb_ who just thought the idea was funny

Use SEO-gaming recipe preambles as a vehicle for encoding secret messages (the prompt can really be anything though, the shared prompt is simply a key that both encoder and decoder agree on).

Hides secret messages in natural-looking text using [neural linguistic steganography](https://aclanthology.org/D19-1115/) over a language model's token distributions.

## How it works

1. The encoder takes your secret message and a shared prompt.
2. At each generation step, it builds a probability distribution (CDF) over the model's top-k tokens.
3. [Arithmetic coding](https://www.artkpv.net/Tool-Arithmetic-Coding-for-LLM-Steganography/) maps secret message bits onto token selections — each chosen token is both a plausible continuation *and* an encoding of hidden data.
4. The decoder, given the same model and prompt, reconstructs the identical CDF at each step and recovers the secret bits from the token choices.

Both sides must use the **exact same model, quantization, top-k value, and prompt**. Any difference causes decoding failure.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An NVIDIA GPU with CUDA support (for 4-bit quantized inference)
  - CPU mode is available via `--device cpu --no-4bit` but will be significantly slower
- Sufficient VRAM for your chosen model (~3 GB for Llama-3.2-3B in 4-bit)

## Installation

```bash
# Create and activate a virtual environment, and install required packages
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# or, using `uv`
uv sync
uv pip install -r requirements.txt
```

## Usage

> [!NOTE]
> Set the `HF_TOKEN` environment variable to authenticate to HuggingFace

### Encode a secret message

```bash
python main.py encode --message "attack at dawn"
```

This uses the default recipe preamble prompt and Llama-3.2-3B. The stego text is printed to stdout and saved to `stego_output.txt`.

### Decode from stego text

```bash
python main.py decode --stego-file stego_output.txt
```

### Custom prompt and model

```bash
# Encode with a custom prompt
python main.py encode \
    --message "meet me at noon" \
    --prompt "The best thing about making scrambled eggs is: "

# Decode (prompt must match exactly!)
python main.py decode \
    --stego-file stego_output.txt \
    --prompt "The best thing about making scrambled eggs is: "

# Use a different model
python main.py encode \
    --model "mistralai/Mistral-7B-v0.3" \
    --message "secret message" \
    --prompt "Every Sunday morning my grandmother would"
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3.2-3B` | HuggingFace model name/path |
| `--prompt` | *(recipe SEO preamble)* | Context prompt (must match between encode/decode) |
| `--message` | *(interactive)* | Secret message to encode |
| `--stego-text` | | Stego text string to decode |
| `--stego-file` | | File containing stego text to decode |
| `--top-k` | `256` | Top-k vocabulary size for CDF |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--no-4bit` | off | Disable 4-bit quantization (uses fp16; needs more VRAM) |
| `--output` | `stego_output.txt` | Output file for stego text |
