# Memory API

## ThreeLayerMemory

```python
from ucef.memory.three_layer import ThreeLayerMemory

memory = ThreeLayerMemory()

# Store
await memory.store(document)
await memory.store_batch(documents)

# Retrieve
result = await memory.retrieve(query_vector, top_k=10)
results = await memory.retrieve_batch(query_vectors, top_k=10)

# Stats
stats = memory.get_stats()
```

## Hot Layer (Redis / OrderedDict)

```python
from ucef.memory.hot import RedisHotMemory

hot = RedisHotMemory(
    capacity=10000,
    ttl_seconds=3600,
    redis_url="redis://localhost:6379",
)

await hot.store(key, document)
result = await hot.retrieve(key)
await hot.evict()
```

Falls back to `OrderedDict` with LRU when Redis is unavailable.

## Warm Layer (ChromaDB / numpy)

```python
from ucef.memory.warm import ChromaWarmMemory

warm = ChromaWarmMemory(
    capacity=100000,
    collection_name="ucef_warm",
)

await warm.store(document)
results = await warm.search(query_vector, top_k=10)
```

Falls back to numpy arrays with cosine/hyperbolic similarity when ChromaDB is unavailable.

## Cold Layer (HDF5 / JSON)

```python
from ucef.memory.cold import FileSystemColdMemory

cold = FileSystemColdMemory(
    storage_dir="./data/cold",
    format="json",  # or "hdf5", "parquet"
)

await cold.store(document)
results = await cold.search(query_vector, top_k=10)
```

Falls back to JSON when h5py/parquet is unavailable.
