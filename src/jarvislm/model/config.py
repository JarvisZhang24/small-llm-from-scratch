from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the JarvisLM decoder-only Transformer."""

    # Tokenizer and sequence
    vocab_size: int = 50_304
    max_seq_len: int = 1_024

    # Model architecture
    d_model: int = 1_024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4

    # Feedforward network
    # None reproduces the original project's int(8 / 3 * d_model).
    ffn_hidden_dim: int | None = None

    # Numerical / architectural options
    dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    bias: bool = False
    tie_weights: bool = True
    use_flash: bool = True

    def __post_init__(self) -> None:
        """Validate relationships between architecture hyperparameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")

        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even when using RoPE")

        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = int(8 / 3 * self.d_model)

        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        """Hidden dimension handled by one attention head."""
        return self.d_model // self.n_heads

    @property
    def uses_gqa(self) -> bool:
        """Whether Key/Value heads are shared by groups of Query heads."""
        return self.n_kv_heads < self.n_heads