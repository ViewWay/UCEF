"""
Adaptive Context Extension Strategy

Automatically selects optimal extension strategy based on model capabilities.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio


class ContextExtensionStrategy(ABC):
    """Base class for context extension strategies."""
    
    @abstractmethod
    async def extend(
        self,
        model_profile: Any,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Extend context using specific strategy.
        
        Args:
            model_profile: Model capability profile
            documents: Available documents
            query: User query
            
        Returns:
            Selected context documents
        """
        pass


class SmallContextStrategy(ContextExtensionStrategy):
    """
    Strategy for small context models (4K-32K).
    
    Characteristics:
    - Aggressive compression (10%)
    - Precision retrieval
    - Multi-round fusion
    - Fine-grained budget allocation
    """
    
    def __init__(self):
        self.compression_ratio = 0.1
    
    async def extend(
        self,
        model_profile: Any,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Extend context for small models.
        """
        # 1. Aggressive compression
        compressed = await self._aggressive_compression(documents)
        
        # 2. High-precision retrieval
        layer1 = await self._retrieve_high_relevance(query, compressed, top_k=50)
        layer2 = await self._retrieve_semantic_neighbors(query, compressed, top_k=50)
        
        # 3. Multi-round fusion
        fused = self._multi_round_fusion(layer1, layer2)
        
        # 4. Fine-grained budget allocation
        budget = model_profile.native_context_window * 0.9  # Use 90% of budget
        allocated = self._fine_grained_allocation(fused, budget)
        
        return allocated
    
    async def _aggressive_compression(self, documents: List[Dict]) -> List[Dict]:
        """Compress documents to 10% of original size."""
        # Simplified: return top 10%
        n_keep = max(1, int(len(documents) * self.compression_ratio))
        return documents[:n_keep]
    
    async def _retrieve_high_relevance(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Retrieve high-relevance documents."""
        # Simplified: use keyword matching
        query_words = set(query.lower().split())
        
        scored = []
        for doc in documents:
            text = doc.get('text', '').lower()
            score = len(query_words & set(text.split()))
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]
    
    async def _retrieve_semantic_neighbors(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Retrieve semantically similar documents."""
        # Placeholder: would use embedding similarity
        return documents[:top_k]
    
    def _multi_round_fusion(
        self,
        layer1: List[Dict],
        layer2: List[Dict]
    ) -> List[Dict]:
        """Fuse results from multiple retrieval rounds."""
        # Remove duplicates
        seen = set()
        fused = []
        
        for doc in layer1 + layer2:
            doc_id = doc.get('id')
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                fused.append(doc)
        
        return fused
    
    def _fine_grained_allocation(
        self,
        documents: List[Dict],
        budget: int
    ) -> List[Dict]:
        """Allocate tokens with fine-grained control."""
        selected = []
        total_tokens = 0
        
        for doc in documents:
            tokens = doc.get('token_count', 100)
            if total_tokens + tokens <= budget:
                selected.append(doc)
                total_tokens += tokens
            else:
                break
        
        return selected


class MediumContextStrategy(ContextExtensionStrategy):
    """
    Strategy for medium context models (32K-128K).
    
    Characteristics:
    - Moderate compression (30%)
    - Hyperbolic space retrieval
    - Quantum selection
    """
    
    def __init__(self):
        self.compression_ratio = 0.3
    
    async def extend(
        self,
        model_profile: Any,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Extend context for medium models.
        """
        # 1. Moderate compression
        compressed = await self._moderate_compression(documents)
        
        # 2. Hyperbolic space retrieval
        hyperbolic_results = await self._hyperbolic_retrieve(query, compressed)
        
        # 3. Quantum selection
        selected = await self._quantum_selection(hyperbolic_results, query)
        
        return selected
    
    async def _moderate_compression(self, documents: List[Dict]) -> List[Dict]:
        """Compress documents to 30% of original size."""
        n_keep = max(1, int(len(documents) * self.compression_ratio))
        return documents[:n_keep]
    
    async def _hyperbolic_retrieve(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """Retrieve using hyperbolic space."""
        # Placeholder: would use hyperbolic geometry
        # For now, return top results
        return documents[:100]
    
    async def _quantum_selection(
        self,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """Select using quantum-inspired mechanism."""
        # Placeholder: would use quantum superposition
        # For now, return top 50
        return documents[:50]


class LargeContextStrategy(ContextExtensionStrategy):
    """
    Strategy for large context models (128K-200K).
    
    Characteristics:
    - Light compression (50%)
    - Structure preservation
    - Attention optimization
    """
    
    def __init__(self):
        self.compression_ratio = 0.5
    
    async def extend(
        self,
        model_profile: Any,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Extend context for large models.
        """
        # 1. Light compression
        compressed = await self._light_compression(documents)
        
        # 2. Preserve structure
        structured = await self._preserve_structure(compressed)
        
        # 3. Optimize attention order
        optimized = await self._optimize_attention_order(structured, query)
        
        return optimized
    
    async def _light_compression(self, documents: List[Dict]) -> List[Dict]:
        """Compress documents to 50% of original size."""
        n_keep = max(1, int(len(documents) * self.compression_ratio))
        return documents[:n_keep]
    
    async def _preserve_structure(self, documents: List[Dict]) -> List[Dict]:
        """Preserve document structure (sections, hierarchy)."""
        # Simplified: maintain original order
        return documents
    
    async def _optimize_attention_order(
        self,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """Optimize document order for better attention."""
        # Reorder by relevance
        query_words = set(query.lower().split())
        
        scored = []
        for doc in documents:
            text = doc.get('text', '').lower()
            score = len(query_words & set(text.split()))
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]


class AdaptiveContextExtender:
    """
    Automatically selects optimal extension strategy based on model profile.
    
    This is the main entry point for adaptive context extension.
    """
    
    def __init__(self):
        self.strategies = {
            'small': SmallContextStrategy(),
            'medium': MediumContextStrategy(),
            'large': LargeContextStrategy(),
            'xlarge': LargeContextStrategy(),  # Use large strategy
        }
    
    async def extend_context(
        self,
        model_profile: Any,
        documents: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Extend context using optimal strategy for the model.
        
        Args:
            model_profile: Model capability profile
            documents: Available documents
            query: User query
            
        Returns:
            Selected context documents
        """
        # 1. Determine strategy type from profile
        strategy_type = model_profile.context_category
        
        # 2. Get strategy
        strategy = self.strategies.get(strategy_type)
        
        if not strategy:
            # Default to medium strategy
            strategy = self.strategies['medium']
        
        # 3. Apply strategy
        extended_context = await strategy.extend(
            model_profile,
            documents,
            query
        )
        
        return extended_context
