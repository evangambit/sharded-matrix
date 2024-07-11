
import multiprocessing
import os
import time

import numpy as np

"""
General matrix format is

type: <4 bytes>  // type ("bool", "i8  ", "u16 ", etc.)
    <4 bytes>  // number of dimensions
    <4 bytes>  // size of dimension 1  (number of rows)
    <4 bytes>  // size of dimension 2
    ...
    [DATA]
    ...
"""

kSupportedTypes = set([np.float32, np.int16, np.int8, bool])
kType2ByteEncoding = {
  np.float32: b'f32 ',
  np.int16:   b'i16 ',
  np.int8:  b'i8  ',
  bool:     b'bool',
}
kByteEncodingToType = {
  b'f32 ': np.float32,
  b'i16 ': np.int16,
  b'i8  ': np.int8,
  b'bool': bool,
}
kType2SizeBits = {
  np.float32: 32,
  np.int16:   16,
  np.int8:  8,
  bool:     1,
}

class LoaderInterface:
  def iterator(self, skip, offset):
    raise NotImplementedError()

def path2shardname(path, i):
  return f'{path}-{str(i).rjust(5, "0")}'

def product(x):
  return np.prod(x)

kTargetedShardSizeBytes = 25_000_000  # 25 MB

def tensor2bytes(A):
  if A.dtype == bool:
    return np.packbits(A, bitorder='little').tobytes()
  else:
    return A.tobytes()

def bytes2tensor(A, dtype, shape):
  if dtype == bool:
    return np.unpackbits(np.frombuffer(A, dtype=np.uint8), bitorder='little').reshape(shape)
  else:
    return np.frombuffer(A, dtype=dtype).reshape(shape)

class ShardedWriter:
  def __init__(self, path: str, dtype: type, shape: tuple[int]) -> None:
    assert dtype in kSupportedTypes
    assert len(shape) > 0, 'At least one dimension must be provided'
    for d in shape:
      assert d > 0, 'All dimensions must be greater than 0'
    self.path = path
    self.dtype = dtype
    self.shape = shape
    self.shard_index = 0
    self.bytes_per_row = product(self.shape) * kType2SizeBits[self.dtype] // 8
    self.rows_per_shard = kTargetedShardSizeBytes // self.bytes_per_row
    self.bytes_per_shard = self.rows_per_shard * self.bytes_per_row
    assert self.rows_per_shard > 16, 'Shape is too big :('
    self._i = 0
    self._data_to_write = bytearray(self.rows_per_shard * self.bytes_per_row)

  def write(self, x: np.ndarray):
    self.write_many(x.reshape((1,) + x.shape))
  
  def __enter__(self):
    return self
  
  def __exit__(self, type, value, tb):
    self.close()

  def close(self):
    if self._i == 0:
      return
    self._dump_data(self._i // self.bytes_per_row)

  def write_many(self, x: np.ndarray):
    if x.shape[0] == 0:
      return
    assert x.shape[1:] == self.shape, f'Expected shape {self.shape}, got {x.shape[1:]}'
    x = x.astype(self.dtype)
    di = x.shape[0] * self.bytes_per_row

    if self._i + di <= self.bytes_per_shard:
      self._data_to_write[self._i:self._i+di] = tensor2bytes(x)
      self._i += di
      return
  
    bytes_left = self.bytes_per_shard - self._i
    rows_left = bytes_left // self.bytes_per_row
    assert rows_left * self.bytes_per_row == bytes_left
    self._data_to_write[self._i:] = tensor2bytes(x[:bytes_left])
    self._dump_data(self.rows_per_shard)

    self.write_many(x[rows_left:])
  
  def _dump_data(self, num_rows):
    with open(path2shardname(self.path, self.shard_index), 'wb') as f:
      f.write(kType2ByteEncoding[self.dtype])
      f.write(np.array(len(self.shape) + 1, dtype=np.int32).tobytes())
      for d in (num_rows,) + self.shape:
        f.write(np.array(d, dtype=np.int32).tobytes())
      f.write(self._data_to_write[:num_rows*self.bytes_per_row])
    self._i = 0
    self.shard_index += 1

def dump_shard(path, num_rows):
  with open(path, 'wb') as f:
      f.write(kType2ByteEncoding[self.dtype])
      f.write(np.array(len(self.shape) + 1, dtype=np.int32).tobytes())
      for d in (num_rows,) + self.shape:
        f.write(np.array(d, dtype=np.int32).tobytes())
      f.write(self._data_to_write[:num_rows*self.bytes_per_row])

def load_shard(path):
  with open(path, 'rb') as f:
    dtype = f.read(4)
    assert dtype in kByteEncodingToType, f'Unsupported type {dtype}'
    dtype = kByteEncodingToType[dtype]
    ndim = np.frombuffer(f.read(4), dtype=np.int32)[0]
    assert ndim > 0, f'Invalid number of dimensions {ndim}'
    shape = np.frombuffer(f.read(4 * ndim), dtype=np.int32)
    if dtype != bool:
      data = bytes2tensor(f.read(), dtype, shape)
    else:
      raise NotImplementedError('')
    return data.reshape(shape)

class BaseShardedLoader(LoaderInterface):
  def __init__(self, path: str):
    self._path = path
    self._last_loaded = (None, None)
    self.n = 0
    while os.path.exists(path2shardname(self._path, self.n)):
      self.n += 1
    assert self.n > 0
  
  def iterator(self, offset=0, skip=1):
    for shard_index in range(0, self.n, skip):
      path = path2shardname(self._path, shard_index)
      if not os.path.exists(path):
        raise FileExistsError(path)
      if self._last_loaded[0] == path:
        # Caching is useful when weights are derived from the same data that is being multiplied
        yield self._last_loaded[1]
      matrix = load_shard(path)
      self._last_loaded = (path, matrix)
      yield matrix

class MappingLoader(LoaderInterface):
  def __init__(self, loader, *mappings, width=None):
    self.rows_per_shard = loader.rows_per_shard
    self.width = width if width is not None else loader.width
    self.n = loader.n
    self._loader = loader
    self._mappings = mappings
  
  def iterator(self, offset=0, skip=1):
    for x in self._loader.iterator(offset=offset, skip=skip):
      for f in self._mappings:
        x = f(x)
      yield x

def _compute_innerproduct(loader1, loader2, offset):
  A = next(loader1.iterator(offset=offset))
  B = next(loader2.iterator(offset=offset))
  return A.T @ B

def _compute_weighted_innerproduct(loader1, loader2, weights_loader, offset):
  A = next(loader1.iterator(offset=offset))
  B = next(loader2.iterator(offset=offset))
  weights = next(weights_loader.iterator(offset=offset))
  A = A * weights
  B = B * weights
  return A.T @ B

def _compute_self_innerproduct(loader1, offset):
  A = next(loader1.iterator(offset=offset))
  return A.T @ A

def _compute_weighted_self_innerproduct(loader1, weights_loader, offset):
  A = next(loader1.iterator(offset=offset))
  weights = next(weights_loader.iterator(offset=offset))
  A = A * weights
  return A.T @ A

def compute_inner_product(loader1: LoaderInterface, loader2: LoaderInterface, weights_loader=None):
  assert loader1.n == loader2.n, 'Both loaders must have the same number of shards'
  shards = list(range(0, loader1.n))
  result = None
  with multiprocessing.Pool(4) as pool:
    if loader1 is loader2:
      if weights_loader is None:
        inner_products = pool.starmap(_compute_self_innerproduct, [(loader1, offset) for offset in shards])
      else:
        inner_products = pool.starmap(_compute_weighted_self_innerproduct, [(loader1, weights_loader, offset) for offset in shards])
    else:
      if weights_loader is None:
        inner_products = pool.starmap(_compute_innerproduct, [(loader1, loader2, offset) for offset in shards])
      else:
        inner_products = pool.starmap(_compute_weighted_innerproduct, [(loader1, loader2, weights_loader, offset) for offset in shards])
    return sum(inner_products)
