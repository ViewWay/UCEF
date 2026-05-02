"""
UCEF Configuration Management

Dual-backend configuration: uses Pydantic v2 when available for full validation,
falls back to pure dataclasses when Pydantic is not installed.

All config classes maintain identical field names and defaults regardless of backend.
When Pydantic is missing, runtime validation is skipped but type hints remain correct.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel as _PydanticBase, Field as _PydanticField
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


# ──────────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    from pydantic import BaseModel, Field, field_validator, model_validator
else:
    # Pure-stdlib fallback: BaseModel → dataclass, Field → plain default
    BaseModel = object  # type: ignore[assignment, misc]

    def Field(default=..., *, description: str = "", **kwargs) -> Any:  # type: ignore[no-redef]
        """Pydantic Field stub — returns the plain default value."""
        if default is ...:
            return None
        return default


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperbolic Geometry Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class HyperbolicConfig(BaseModel):
        """Configuration for hyperbolic geometry operations."""
        embedding_dim: int = Field(default=128, ge=16, le=1024)
        curvature: float = Field(default=-1.0)
        max_norm: float = Field(default=0.9, gt=0.0, lt=1.0)
        learning_rate: float = Field(default=0.01, gt=0.0)
        n_epochs: int = Field(default=100, ge=1)
        burn_in_epochs: int = Field(default=20, ge=0)
        burn_in_lr: float = Field(default=0.001, gt=0.0)
        n_neighbors: int = Field(default=50, ge=1, le=1000)

        @field_validator("curvature")
        @classmethod
        def validate_curvature(cls, v: float) -> float:
            if v >= 0:
                raise ValueError("Curvature must be negative for hyperbolic space")
            return v
else:
    @dataclass
    class HyperbolicConfig:
        """Configuration for hyperbolic geometry operations."""
        embedding_dim: int = 128
        curvature: float = -1.0
        max_norm: float = 0.9
        learning_rate: float = 0.01
        n_epochs: int = 100
        burn_in_epochs: int = 20
        burn_in_lr: float = 0.001
        n_neighbors: int = 50

        def __post_init__(self) -> None:
            if self.curvature >= 0:
                raise ValueError("Curvature must be negative for hyperbolic space")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Quantum-Inspired Selection Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class QuantumConfig(BaseModel):
        """Configuration for quantum-inspired context selection."""
        enabled: bool = True
        initial_amplitude: str = "equal"
        entanglement_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
        measurement_method: str = "top_k"
        top_k_measurements: int = Field(default=10, ge=1, le=100)
        use_interference: bool = True
        interference_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
else:
    @dataclass
    class QuantumConfig:
        """Configuration for quantum-inspired context selection."""
        enabled: bool = True
        initial_amplitude: str = "equal"
        entanglement_threshold: float = 0.3
        measurement_method: str = "top_k"
        top_k_measurements: int = 10
        use_interference: bool = True
        interference_threshold: float = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 3. Memory System Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class HotMemoryConfig(BaseModel):
        enabled: bool = True
        redis_url: str = "redis://localhost:6379/0"
        max_tokens: int = Field(default=20_000, ge=1000)
        ttl_seconds: int = Field(default=3600, ge=60)

    class WarmMemoryConfig(BaseModel):
        enabled: bool = True
        persist_directory: str = "./data/chroma"
        collection_name: str = "ucef_documents"
        embedding_model: str = "all-MiniLM-L6-v2"
        distance_metric: str = "cosine"

    class ColdMemoryConfig(BaseModel):
        enabled: bool = True
        storage_path: str = "./data/cold"
        format: str = "hdf5"
        compression: str = "gzip"

    class MemorySystemConfig(BaseModel):
        hot: HotMemoryConfig = Field(default_factory=HotMemoryConfig)
        warm: WarmMemoryConfig = Field(default_factory=WarmMemoryConfig)
        cold: ColdMemoryConfig = Field(default_factory=ColdMemoryConfig)
        hot_budget_pct: float = Field(default=0.10, ge=0.0, le=1.0)
        warm_budget_pct: float = Field(default=0.60, ge=0.0, le=1.0)
        cold_budget_pct: float = Field(default=0.30, ge=0.0, le=1.0)

        @model_validator(mode="after")
        def validate_total_budget(self) -> MemorySystemConfig:
            total = self.hot_budget_pct + self.warm_budget_pct + self.cold_budget_pct
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Memory budget percentages must sum to ~1.0, got {total:.2f}"
                )
            return self
else:
    @dataclass
    class HotMemoryConfig:
        enabled: bool = True
        redis_url: str = "redis://localhost:6379/0"
        max_tokens: int = 20_000
        ttl_seconds: int = 3600

    @dataclass
    class WarmMemoryConfig:
        enabled: bool = True
        persist_directory: str = "./data/chroma"
        collection_name: str = "ucef_documents"
        embedding_model: str = "all-MiniLM-L6-v2"
        distance_metric: str = "cosine"

    @dataclass
    class ColdMemoryConfig:
        enabled: bool = True
        storage_path: str = "./data/cold"
        format: str = "hdf5"
        compression: str = "gzip"

    @dataclass
    class MemorySystemConfig:
        hot: HotMemoryConfig = field(default_factory=HotMemoryConfig)
        warm: WarmMemoryConfig = field(default_factory=WarmMemoryConfig)
        cold: ColdMemoryConfig = field(default_factory=ColdMemoryConfig)
        hot_budget_pct: float = 0.10
        warm_budget_pct: float = 0.60
        cold_budget_pct: float = 0.30

        def __post_init__(self) -> None:
            total = self.hot_budget_pct + self.warm_budget_pct + self.cold_budget_pct
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Memory budget percentages must sum to ~1.0, got {total:.2f}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Quality Assurance Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class QualityConfig(BaseModel):
        quality_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
        relevance_weight: float = Field(default=0.30, ge=0.0, le=1.0)
        completeness_weight: float = Field(default=0.30, ge=0.0, le=1.0)
        coherence_weight: float = Field(default=0.20, ge=0.0, le=1.0)
        accuracy_weight: float = Field(default=0.20, ge=0.0, le=1.0)
        max_regeneration_attempts: int = Field(default=3, ge=1, le=10)
        use_self_consistency: bool = True
        consistency_samples: int = Field(default=5, ge=1, le=20)
        calibration_method: str = "temperature_scaling"
else:
    @dataclass
    class QualityConfig:
        quality_threshold: float = 0.75
        relevance_weight: float = 0.30
        completeness_weight: float = 0.30
        coherence_weight: float = 0.20
        accuracy_weight: float = 0.20
        max_regeneration_attempts: int = 3
        use_self_consistency: bool = True
        consistency_samples: int = 5
        calibration_method: str = "temperature_scaling"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Compression Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class CompressionConfig(BaseModel):
        default_strategy: str = "adaptive"
        aggressive_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
        moderate_ratio: float = Field(default=0.30, gt=0.0, le=1.0)
        light_ratio: float = Field(default=0.50, gt=0.0, le=1.0)
        use_mdl: bool = True
        description_length_weight: float = Field(default=0.5, ge=0.0, le=1.0)
        use_entropy: bool = True
        entropy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
else:
    @dataclass
    class CompressionConfig:
        default_strategy: str = "adaptive"
        aggressive_ratio: float = 0.10
        moderate_ratio: float = 0.30
        light_ratio: float = 0.50
        use_mdl: bool = True
        description_length_weight: float = 0.5
        use_entropy: bool = True
        entropy_threshold: float = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# 6. Top-Level UCEF Configuration
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class UCEFConfig(BaseModel):
        project_name: str = "ucef"
        version: str = "0.1.0"
        hyperbolic: HyperbolicConfig = Field(default_factory=HyperbolicConfig)
        quantum: QuantumConfig = Field(default_factory=QuantumConfig)
        memory: MemorySystemConfig = Field(default_factory=MemorySystemConfig)
        quality: QualityConfig = Field(default_factory=QualityConfig)
        compression: CompressionConfig = Field(default_factory=CompressionConfig)
        target_extended_context: int = Field(default=1_000_000, ge=10_000)
        max_retrieval_time_ms: float = Field(default=500.0, ge=10.0)
        log_level: str = "INFO"
        data_dir: Path = Path("./data")
        cache_dir: Path = Path("./data/cache")

        model_config = {"env_prefix": "UCEF_"}

        @classmethod
        def from_env(cls) -> UCEFConfig:
            kwargs: Dict[str, Any] = {}
            if val := os.getenv("UCEF_TARGET_CONTEXT"):
                kwargs["target_extended_context"] = int(val)
            if val := os.getenv("UCEF_LOG_LEVEL"):
                kwargs["log_level"] = val
            if val := os.getenv("UCEF_DATA_DIR"):
                kwargs["data_dir"] = Path(val)
            return cls(**kwargs)

        @classmethod
        def from_file(cls, path: Path) -> UCEFConfig:
            content = path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(content)
            elif path.suffix == ".json":
                import json
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            return cls(**data)

        def ensure_directories(self) -> None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            Path(self.memory.warm.persist_directory).mkdir(parents=True, exist_ok=True)
            Path(self.memory.cold.storage_path).mkdir(parents=True, exist_ok=True)
else:
    @dataclass
    class UCEFConfig:
        project_name: str = "ucef"
        version: str = "0.1.0"
        hyperbolic: HyperbolicConfig = field(default_factory=HyperbolicConfig)
        quantum: QuantumConfig = field(default_factory=QuantumConfig)
        memory: MemorySystemConfig = field(default_factory=MemorySystemConfig)
        quality: QualityConfig = field(default_factory=QualityConfig)
        compression: CompressionConfig = field(default_factory=CompressionConfig)
        target_extended_context: int = 1_000_000
        max_retrieval_time_ms: float = 500.0
        log_level: str = "INFO"
        data_dir: Path = Path("./data")
        cache_dir: Path = Path("./data/cache")

        @classmethod
        def from_env(cls) -> UCEFConfig:
            kwargs: Dict[str, Any] = {}
            if val := os.getenv("UCEF_TARGET_CONTEXT"):
                kwargs["target_extended_context"] = int(val)
            if val := os.getenv("UCEF_LOG_LEVEL"):
                kwargs["log_level"] = val
            if val := os.getenv("UCEF_DATA_DIR"):
                kwargs["data_dir"] = Path(val)
            return cls(**kwargs)

        @classmethod
        def from_file(cls, path: Path) -> UCEFConfig:
            content = path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(content)
            elif path.suffix == ".json":
                import json
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            return cls(**data)

        def ensure_directories(self) -> None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            Path(self.memory.warm.persist_directory).mkdir(parents=True, exist_ok=True)
            Path(self.memory.cold.storage_path).mkdir(parents=True, exist_ok=True)
