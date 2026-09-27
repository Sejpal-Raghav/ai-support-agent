from support_agent.intent.transformer import TransformerTrainingConfig


def test_transformer_defaults_match_voice_plan() -> None:
    config = TransformerTrainingConfig()
    assert config.model_id == "vinai/bertweet-base"
    assert (
        config.max_length,
        config.batch_size,
        config.learning_rate,
        config.epochs,
        config.seed,
    ) == (128, 16, 2e-5, 3, 1729)
