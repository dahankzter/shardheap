# shardheap-malloc

`malloc`, `free`, `calloc`, `realloc`, `reallocarray`, `posix_memalign`,
`aligned_alloc`, `memalign`, `valloc`, `pvalloc`, `malloc_usable_size` and
`malloc_trim` over shardheap's `ShardAlloc`, as a shared library for
`LD_PRELOAD`. Not published; build it from this repository.

```
RUSTFLAGS='-Z tls-model=initial-exec' cargo +nightly build --release -p shardheap-malloc
LD_PRELOAD=target/release/libshardheap_malloc.so your-program
```

Nightly is needed for one flag: `-Z tls-model=initial-exec`. Without it, a
shared library reads every thread-local through a call to `__tls_get_addr`,
on every `malloc` and `free`; with it, the read is one instruction, as in
mimalloc's preload library. A library built this way must be loaded at
startup (`LD_PRELOAD`, or linked into the program), not with `dlopen`.

Semantics follow glibc: minimum alignment 16, `malloc(0)` returns a unique
block, `realloc(p, 0)` frees `p` and returns null, failures return null with
`errno` set to `ENOMEM`, `calloc` and `reallocarray` check for overflow.
`realloc` stays in place when the new size fits the block and uses at least
half of it.

`tests/run.sh` builds the library, checks that no shardheap function calls
`__tls_get_addr`, and runs `tests/api.c` (every edge case above, plus eight
threads freeing each other's blocks), `tests/smoke.cpp` and a few everyday
programs (`ls`, `sort`, `python3`, `git`) with it preloaded.

`bench/run.py` runs mimalloc-bench against several allocators; see
`bench/README.md`.
