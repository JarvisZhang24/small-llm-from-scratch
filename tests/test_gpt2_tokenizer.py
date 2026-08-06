import pytest

from jarvislm import ModelConfig
from jarvislm.tokenizer import GPT2Tokenizer


class FakeGPT2Encoding:
    n_vocab = 50_257
    eot_token = 50_256

    def __init__(self) -> None:
        self.last_disallowed_special: tuple[object, ...] | None = None

    def encode(self, text: str, *, disallowed_special: tuple[object, ...]) -> list[int]:
        self.last_disallowed_special = disallowed_special
        return [len(text)]

    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


def test_gpt2_tokenizer_matches_reference_document_boundary_behavior() -> None:
    encoding = FakeGPT2Encoding()
    tokenizer = GPT2Tokenizer(encoding=encoding)

    assert tokenizer.encode("hello", add_eos=True) == [5, 50_256]
    assert encoding.last_disallowed_special == ()
    assert tokenizer.decode([1, 50_256]) == "1,50256"


def test_gpt2_tokenizer_and_default_model_config_are_compatible() -> None:
    tokenizer = GPT2Tokenizer(encoding=FakeGPT2Encoding())
    config = ModelConfig()

    assert config.vocab_size == tokenizer.padded_vocab_size
    tokenizer.validate_model_vocab_size(config.vocab_size)
    with pytest.raises(ValueError, match="smaller"):
        tokenizer.validate_model_vocab_size(tokenizer.vocab_size - 1)


def test_gpt2_tokenizer_rejects_invalid_inputs_and_encodings() -> None:
    with pytest.raises(TypeError, match="text"):
        GPT2Tokenizer(encoding=FakeGPT2Encoding()).encode(b"text")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="token IDs"):
        GPT2Tokenizer(encoding=FakeGPT2Encoding()).decode(["bad"])  # type: ignore[list-item]

    bad_encoding = FakeGPT2Encoding()
    bad_encoding.n_vocab = 10
    with pytest.raises(ValueError, match="vocabulary"):
        GPT2Tokenizer(encoding=bad_encoding)
