from torch import float16, float32

_DTYPE = {
    'torch.float16': float16,
    'torch.float32': float32,
}

def str_to_dtype(dtype_str: str) -> 'torch.dtype':
    try:
        return _DTYPE[dtype_str]
    except KeyError as e:
        raise ValueError(f'Unknown dtype string: {dtype_str!r}') from e