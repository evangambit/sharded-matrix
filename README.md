# Sharded Matrix

Python code to read/write/compute with sharded matrices.

Simple Example:

```Python
from sharded_matrix import ShardedWriter, ShardedLoader, linear_regression
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

# Perform linear regression using multiple processes
X = sharded_matrix.ShardedLoader('X')
Y = sharded_matrix.ShardedLoader('Y')
w = sharded_matrix.linear_regression(X, Y)
```

Also includes a C++ library for creating matrices
