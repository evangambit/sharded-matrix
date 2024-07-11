# Sharded Matrix

A relatively modest repository to create and read sharded matrices, as well as perform some operations on them.

Simple Example:

```Python
from sharded_matrix import ShardedWriter, ShardedLoader
import numpy as np

# Write a 2970x128x128 tensor, 99 at a time.
with ShardedWriter('test', dtype=np.int8, shape=(128, 128)) as writer:
  for _ in range(30):
    x = np.zeros((99, 128, 128), dtype=np.int8)
    writer.write_many(x)


# There are now 2 files: "test-00000.sm" and "test-00001.sm"


# Read the matrix, shard-by-shard.
loader = ShardedLoader('test')
for a in loader.iterator():
    print(a.shape)

# Prints:
# (1525, 128, 128)
# (1445, 128, 128)

```

Notworthy features:

1. Multi-process inner-product computation that splits the work across the sharded files. Notably this is enough to support...

2. Parallelized linear regression. Even if your dataset doesn't fit into memory, if you can invert a DxD matrix, then you can perform the linear regression.

2. Boolean matrices are packed (i.e. 8x smaller on disk), however each row must be a multiple of 8.
