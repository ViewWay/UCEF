# Three-Layer Memory

A tiered storage architecture with automatic data migration based on access patterns.

## Architecture

```
┌────────────────────────────────────────────┐
│  Hot Layer    (Redis / OrderedDict)        │
│  Latency: <10ms   Capacity: ~10K docs      │
│  LRU + TTL eviction                        │
├────────────────────────────────────────────┤
│  Warm Layer   (ChromaDB / numpy)           │
│  Latency: <100ms  Capacity: ~100K docs     │
│  Semantic retrieval with hyperbolic emb.   │
├────────────────────────────────────────────┤
│  Cold Layer   (HDF5 / JSON / Parquet)      │
│  Latency: <500ms  Capacity: Unlimited      │
│  Compressed archival storage               │
└────────────────────────────────────────────┘
```

## Data Flow

1. New documents enter **Hot** layer
2. When Hot is full, LRU items demoted to **Warm**
3. When Warm is full, oldest items archived to **Cold**
4. Accessed Cold items promoted back to **Warm**

## API

```python
from ucef.memory import ThreeLayerMemory

memory = ThreeLayerMemory()

# Store (auto-distributes across layers)
await memory.store(document)

# Retrieve (checks Hot → Warm → Cold)
result = await memory.retrieve(query_vector, top_k=10)

# Batch operations
await memory.store_batch(documents)
results = await memory.retrieve_batch(query_vectors, top_k=10)
```

## Graceful Degradation

| Dependency | Status | Fallback |
|-----------|--------|----------|
| Redis | Not installed | `OrderedDict` with LRU |
| ChromaDB | Not installed | numpy arrays with cosine similarity |
| h5py | Not installed | JSON files |
| All missing | — | Fully functional with in-memory storage |

## Configuration

```python
from ucef.core.config import MemoryConfig

config = MemoryConfig(
    hot_capacity=10000,
    warm_capacity=100000,
    hot_ttl_seconds=3600,
    migration_threshold=0.8,
)
```
