from src.training.trainer import TrainConfig


def test_train_config_accepts_train_ratio():
    cfg = TrainConfig(train_ratio=0.9)
    assert cfg.train_ratio == 0.9


def test_train_config_default_train_ratio_is_0_8():
    cfg = TrainConfig()
    assert cfg.train_ratio == 0.8
