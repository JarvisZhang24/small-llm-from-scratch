"""Generate text from a JarvisLM training checkpoint.

Example:
    PYTHONPATH=src python scripts/generate_samples.py \\
        --checkpoint checkpoints/jarvislm-350m/last.pt --ema \\
        --prompt "The meaning of life is"
"""

import argparse

import torch

from jarvislm.inference import generate_text, load_inference_model
from jarvislm.tokenizer import GPT2Tokenizer

DEFAULT_PROMPTS = (
    "The meaning of life is",
    "In a distant galaxy,",
    "def fibonacci(n):",
    "The president announced that",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    parser.add_argument(
        "--prompt",
        action="append",
        help="prompt to generate from; repeat for multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, default: CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use EMA weights when present (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_inference_model(args.checkpoint, args.device, use_ema=args.ema)
    tokenizer = GPT2Tokenizer()
    tokenizer.validate_model_vocab_size(model.config.vocab_size)

    for prompt in args.prompt or DEFAULT_PROMPTS:
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("=" * 70)
        print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()
