"""Tests for configuration system."""
from ucef.core.config import (
    UCEFConfig, HyperbolicConfig, QuantumConfig, MemorySystemConfig,
    CompressionConfig, QualityConfig,
)


class TestUCEFConfig:
    def test_default_config_creation(self):
        config = UCEFConfig()
        assert config.hyperbolic is not None
        assert config.quantum is not None
        assert config.memory is not None
        assert config.compression is not None
        assert config.quality is not None

    def test_nested_config_access(self):
        config = UCEFConfig()
        assert config.hyperbolic.embedding_dim == 128
        assert config.hyperbolic.curvature == -1.0
        assert config.quality.quality_threshold > 0

    def test_custom_hyperbolic(self):
        config = UCEFConfig(
            hyperbolic=HyperbolicConfig(embedding_dim=64, curvature=-2.0),
        )
        assert config.hyperbolic.embedding_dim == 64
        assert config.hyperbolic.curvature == -2.0


class TestHyperbolicConfig:
    def test_negative_curvature(self):
        cfg = HyperbolicConfig(curvature=-1.0)
        assert cfg.curvature == -1.0

    def test_embedding_dim(self):
        cfg = HyperbolicConfig(embedding_dim=256)
        assert cfg.embedding_dim == 256


class TestMemorySystemConfig:
    def test_default_budget_ratios(self):
        cfg = MemorySystemConfig()
        total = cfg.hot_budget_pct + cfg.warm_budget_pct + cfg.cold_budget_pct
        assert abs(total - 1.0) < 0.05


class TestQualityConfig:
    def test_default_values(self):
        cfg = QualityConfig()
        assert 0 < cfg.quality_threshold <= 1.0
        assert cfg.monitor_window_size >= 10
        assert cfg.max_refinement_iterations >= 1
