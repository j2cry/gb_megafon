import ctypes


def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)


def select_and_sort(df, mask, by):
    return df[mask].sort_values(by)
