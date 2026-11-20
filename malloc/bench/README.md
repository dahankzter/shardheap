# mimalloc-bench

[mimalloc-bench](https://github.com/daanx/mimalloc-bench) is the suite
allocator authors compare on. `run.py` runs its benchmarks with any set of
allocators loaded through `LD_PRELOAD` and prints one Markdown table.

## Setup

Outside this repository:

```
git clone --depth 1 https://github.com/daanx/mimalloc-bench
cd mimalloc-bench
./build-bench-env.sh bench mi     # the benchmarks, and mimalloc to compare with
```

`build-bench-env.sh` wants `dos2unix` (for the shbench sources) and fetches
the shbench sources, a PDF for `gs` and Lua. Other allocators can come from
it too (`je`, `tc`, `sn`, ...) or from system packages
(`/usr/lib/libjemalloc.so.2`, `/usr/lib/libtcmalloc_minimal.so.4`).

## Running

```
malloc/bench/run.py <mimalloc-bench> glibc=glibc mi=<.../libmimalloc.so> \
    shard=target/release/libshardheap_malloc.so [more name=lib.so] [--runs 3] [--only cfrac,larsonN]
```

- Command lines are bench.sh's, with `procs` = the number of CPUs.
- Each benchmark runs `--runs` times per allocator, rounds interleaved; the
  table has the median time and peak resident memory, and in brackets each
  as a ratio to `mi` (or the first allocator).
- larson, rptest, xmalloc-test and glibc-thread run for a fixed time; their
  "time" is the relative time bench.sh computes from their output (lower is
  better, as for the rest).
- A failed run is reported as failed, with the tail of its output.
- Not run: benchmarks that need extra packages or long builds (redis,
  rocksdb, lean, z3, ruby, lua).
