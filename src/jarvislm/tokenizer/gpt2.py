"""The GPT-2 tokenizer used by JarvisLM pretraining.

JarvisLM follows the reference project: tokenization uses tiktoken's ``gpt2``
encoding (50,257 usable token IDs) while :class:`ModelConfig` pads the model
vocabulary to 50,304 for hardware-friendly matrix dimensions.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tiktoken


class GPT2Tokenizer:
    """Small, explicit wrapper around tiktoken's GPT-2 encoding.

    Text that looks like a tiktoken special token is deliberately encoded as
    ordinary text, matching the reference data-preparation code.  A document
    boundary is represented by appending GPT-2's EOT token (ID 50,256).
    """

    ENCODING_NAME = "gpt2"
    vocab_size = 50_257
    eos_id = 50_256
    padded_vocab_size = 50_304

    def __init__(self, encoding: Any | None = None) -> None:
        self._encoding = encoding if encoding is not None else self._load_encoding()
        if self._encoding.n_vocab != self.vocab_size:
            raise ValueError(
                f"expected GPT-2 vocabulary size {self.vocab_size}, got "
                f"{self._encoding.n_vocab}"
            )
        if self._encoding.eot_token != self.eos_id:
            raise ValueError(
                f"expected GPT-2 EOT ID {self.eos_id}, got "
                f"{self._encoding.eot_token}"
            )

    @staticmethod
    def _load_encoding() -> "tiktoken.Encoding":
        try:
            import tiktoken
        except ImportError as error:
            raise ImportError(
                "GPT2Tokenizer requires tiktoken. Install the project "
                "dependencies with `python -m pip install -e '.[dev]'`."
            ) from error
        return tiktoken.get_encoding(GPT2Tokenizer.ENCODING_NAME)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text, optionally appending the GPT-2 EOT token."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = self._encoding.encode(text, disallowed_special=())
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode GPT-2 token IDs into text."""
        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise TypeError("token IDs must be integers")
        return self._encoding.decode(token_ids)

    def validate_model_vocab_size(self, model_vocab_size: int) -> None:
        """Ensure the embedding table can represent every GPT-2 token ID."""
        if model_vocab_size < self.vocab_size:
            raise ValueError(
                f"model vocab_size={model_vocab_size} is smaller than GPT-2 "
                f"vocab_size={self.vocab_size}"
            )


__all__ = ["GPT2Tokenizer"]
