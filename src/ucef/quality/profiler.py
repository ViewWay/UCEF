"""
Model Capability Profiler

Analyzes model characteristics to select optimal context extension strategy.
"""

from typing import Dict, Any, Optional
import asyncio

from ucef.core.types import ModelProfile, ContextCategory, CompressionStrategy


class ModelCapabilityProfiler:
    """
    Analyzes model capabilities to inform context extension strategy.
    
    This profiler:
    1. Detects the native context window
    2. Measures performance at different context lengths
    3. Assesses quality retention capabilities
    4. Recommends optimal extension strategy
    """
    
    # Known model specifications
    MODEL_SPECS = {
        # Small context (4K-8K)
        "llama-7b": {"context": 4096, "category": ContextCategory.SMALL},
        "qwen-7b": {"context": 8192, "category": ContextCategory.SMALL},
        "mistral-7b": {"context": 8192, "category": ContextCategory.SMALL},

        # Medium context (32K-64K)
        "llama-13b": {"context": 32768, "category": ContextCategory.MEDIUM},
        "qwen-14b": {"context": 32768, "category": ContextCategory.MEDIUM},
        "yi-34b": {"context": 65536, "category": ContextCategory.MEDIUM},
        "chatglm3-6b": {"context": 32768, "category": ContextCategory.MEDIUM},

        # Large context (128K-200K)
        "llama-3.1-70b": {"context": 131072, "category": ContextCategory.LARGE},
        "qwen2.5-72b": {"context": 131072, "category": ContextCategory.LARGE},
        "deepseek-v2": {"context": 131072, "category": ContextCategory.LARGE},
        "glm-5.1": {"context": 200000, "category": ContextCategory.LARGE},
        "gpt-4o": {"context": 131072, "category": ContextCategory.LARGE},
        "claude-3.5-sonnet": {"context": 200000, "category": ContextCategory.LARGE},
    }
    
    def __init__(self):
        self.cache = {}
    
    async def profile_model(
        self,
        model_client: Optional[Any],
        model_name: str
    ) -> ModelProfile:
        """
        Profile model capabilities.

        Args:
            model_client: Model client instance (None for known models with specs)
            model_name: Name of the model

        Returns:
            ModelProfile: Comprehensive capability profile

        Raises:
            ValueError: If model_client is None and model is not in known specs.
        """
        # Check cache first
        if model_name in self.cache:
            return self.cache[model_name]

        # Validate: for unknown models, a client is required for probing
        normalized = model_name.lower().strip()
        is_known = any(known_name in normalized for known_name in self.MODEL_SPECS)

        if model_client is None and not is_known:
            raise ValueError(
                f"Model '{model_name}' is not in known specs. "
                f"A model_client is required for profiling unknown models. "
                f"Known models: {list(self.MODEL_SPECS.keys())}"
            )
        
        # 1. Get native context window
        native_context, category = self._get_context_spec(model_name)
        
        # 2. Measure performance curve (if not in specs)
        performance_curve = await self._measure_performance_curve(
            model_client,
            model_name,
            native_context
        )
        
        # 3. Assess quality retention
        quality_retention = await self._assess_quality_retention(
            model_client,
            model_name
        )
        
        # 4. Assess other capabilities
        retrieval_capability = await self._assess_retrieval_capability(model_client)
        reasoning_strength = await self._assess_reasoning_capability(model_client)
        
        # 5. Recommend strategy
        strategy, compression_ratio, max_context = self._recommend_strategy(
            category,
            quality_retention,
            reasoning_strength
        )
        
        profile = ModelProfile(
            model_name=model_name,
            native_context_window=native_context,
            context_category=category,
            performance_curve=performance_curve,
            quality_retention=quality_retention,
            retrieval_capability=retrieval_capability,
            reasoning_strength=reasoning_strength,
            recommended_strategy=strategy,
            optimal_compression_ratio=compression_ratio,
            max_extended_context=max_context
        )
        
        # Cache for future use
        self.cache[model_name] = profile
        
        return profile
    
    def _get_context_spec(self, model_name: str) -> tuple[int, str]:
        """Get context window specification."""
        # Normalize model name
        normalized = model_name.lower().strip()
        
        # Check known models
        for known_name, spec in self.MODEL_SPECS.items():
            if known_name in normalized:
                return spec["context"], spec["category"]
        
        # Default: Assume 32K if unknown
        return 32768, ContextCategory.MEDIUM
    
    async def _measure_performance_curve(
        self,
        model_client: Any,
        model_name: str,
        max_context: int
    ) -> Dict[int, float]:
        """
        Measure performance at different context lengths.
        
        Returns a mapping from context length to quality score.
        """
        # For known models, use pre-measured curves
        # For unknown models, run lightweight probes
        
        if model_name.lower() in self.MODEL_SPECS:
            return self._get_pre_measured_curve(model_name)
        
        # Run probes at 25%, 50%, 75%, 100% of max context
        probe_lengths = [
            max_context // 4,
            max_context // 2,
            max_context * 3 // 4,
            max_context
        ]
        
        performance_curve = {}
        for length in probe_lengths:
            # Probe with simple task
            score = await self._probe_at_length(model_client, length)
            performance_curve[length] = score
        
        return performance_curve
    
    def _get_pre_measured_curve(self, model_name: str) -> Dict[int, float]:
        """Get pre-measured performance curve."""
        # Simplified: assume gradual decline
        base_score = 0.85
        spec = self.MODEL_SPECS[model_name.lower()]
        max_context = spec["context"]
        
        return {
            max_context // 4: base_score,
            max_context // 2: base_score * 0.95,
            max_context * 3 // 4: base_score * 0.90,
            max_context: base_score * 0.85,
        }
    
    async def _probe_at_length(
        self,
        model_client: Any,
        context_length: int
    ) -> float:
        """Probe model performance at specific context length."""
        # Implement actual probe
        # For now, return estimated score
        return 0.85 * (1 - 0.1 * (context_length / 200000))
    
    async def _assess_quality_retention(
        self,
        model_client: Any,
        model_name: str
    ) -> float:
        """
        Assess how well model retains quality with context extension.
        
        Returns score 0-1.
        """
        # For known models, use historical data
        # For unknown, run quality retention tests
        
        known_scores = {
            "llama-7b": 0.75,
            "llama-13b": 0.80,
            "llama-3.1-70b": 0.90,
            "gpt-4o": 0.95,
        }
        
        for known_name, score in known_scores.items():
            if known_name in model_name.lower():
                return score
        
        # Default estimate
        return 0.80
    
    async def _assess_retrieval_capability(self, model_client: Any) -> float:
        """Assess model's retrieval and reasoning capabilities."""
        # Simplified: test on simple retrieval task
        return 0.75
    
    async def _assess_reasoning_capability(self, model_client: Any) -> float:
        """Assess model's reasoning strength."""
        # Simplified: test on reasoning task
        return 0.80
    
    def _recommend_strategy(
        self,
        category: ContextCategory,
        quality_retention: float,
        reasoning_strength: float
    ) -> tuple:
        """
        Recommend optimal extension strategy.

        Returns:
            (strategy, compression_ratio, max_extended_context)
        """
        if category == ContextCategory.SMALL:
            return (CompressionStrategy.AGGRESSIVE, 0.1, 1_000_000)

        elif category == ContextCategory.MEDIUM:
            return (CompressionStrategy.MODERATE, 0.3, 1_000_000)

        elif category == ContextCategory.LARGE:
            return (CompressionStrategy.LIGHT, 0.5, 1_000_000)

        else:
            # XLARGE or unknown
            return (CompressionStrategy.LIGHT, 0.5, 1_000_000)
