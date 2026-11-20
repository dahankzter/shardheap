# Phase 4: `GlobalAlloc` and a `malloc` replacement

## Intent

Make `ShardAlloc` usable as a process's allocator, two ways:

1. **Rust:** `#[global_allocator] static A: ShardAlloc = ShardAlloc::new();`
2. **C:** a shared library exporting `malloc`, `free` and the rest, loaded
   with `LD_PRELOAD`, so unmodified programs run on shardheap.

The C library exists first to run **mimalloc-bench**, the suite allocator
authors compare on (larson, xmalloc-test, cache-scratch, rptest, mstress,
cfrac, espresso, redis, lua, ...), against glibc, mimalloc, jemalloc,
tcmalloc and snmalloc. Its results decide the next steps: they are the
realistic-profile evidence the synthetic `cross` benchmarks cannot give.

## Goals

- `ShardAlloc` implements `GlobalAlloc`; a Rust program (std threads,
  collections, thread-locals with destructors) runs with it as the global
  allocator, under Miri too.
- A `free(p)` and `realloc(p, n)` path that needs no size: the C API never
  passes one.
- Re-entrancy: allocation from inside shardheap's own first-use setup
  (thread-local destructor registration can call `calloc`) must not recurse
  forever or deadlock.
- `fork()` while another thread holds the pool lock does not deadlock the
  child.
- The C library passes a smoke run of everyday programs (`ls`, `sort`,
  `python3`, `git`, a C++ program) and builds and runs every mimalloc-bench
  benchmark the other allocators run.
- A runner that produces one table: time and peak resident memory per
  benchmark per allocator, ratio to mimalloc.

## Non-goals

- macOS malloc zones, Windows; Linux x86-64 only for the C library's runs.
  aarch64 must build (a CI cross-compile check); the segment map covers 48
  bits there (8 MiB, `MAP_NORESERVE`), 47 on x86-64.
- `mallinfo`, `mallopt`, malloc hooks, `malloc_stats`; `malloc_trim` maps to
  `ShardAlloc::trim` and nothing else.
- Async-signal safety (no allocator offers it).
- Tuning for mimalloc-bench in this phase: the phase ends with results.

## Anti-goals

- Slowing the `Allocator` fast paths: the single-thread benchmarks and
  `cross` must stay where stage B left them.
- A panic unwinding across the C ABI: the library is built with
  `panic = "abort"`, and allocation failure returns null with `errno =
  ENOMEM`.

## Approaches

1. **Chosen: `GlobalAlloc` in the crate, plus a small `cdylib` crate in a
   workspace that exports the C API over a size-free path in the core.**
   Serves Rust users and the suite; the hard parts (size-free free,
   re-entrancy, fork) are shared.
2. A C library only: quickest to suite results, leaves `#[global_allocator]`
   for later.
3. Port a few mimalloc-bench workloads to Rust over `GlobalAlloc`: no C
   library needed, but no standard suite and no C/C++ programs.

## Design

### Size-free free (`free`, `realloc`, `malloc_usable_size`)

Today a free needs the `Layout`: the class comes from the size, and a large
block's header sits at `p - large_offset(align)`.

- **Small or large:** a *segment map*, one bit per 4 MiB of address space,
  set when a segment is mapped and cleared when it is unmapped: a static
  `MAP_NORESERVE` mapping of 4 MiB covers the 47-bit user address space (only
  touched pages become resident). `free(p)` reads the bit for
  `p >> 22`: set means `p` is in a segment, so a small block; clear means a
  large block. Alternatives considered: aligning large mappings to 4 MiB so
  `segment_of(p)` is always readable (breaks when `mremap` moves a grown
  mapping), and probing the segment header (unmapped for large blocks).
- **Large header:** moves to `p - PAGE` for every alignment (today it is at
  the mapping's base, `p - max(align, PAGE)`), so it is found without the
  alignment. The mapping's base stays recorded in the header.
- **Small size:** a per-slot size-class table in the segment header (one
  byte per slot, on a line of its own), written when a page is made or
  released. *Builder's call, ratified:* not the page descriptor's
  `block_size`: that line is written on every allocation, and a remote free
  reading it misses on it (the miss stage B removed from the owner check).
  A remote free needs the size too (batch budget, carrier or group).
- These give `Inner::usable_size(p)` and a size-free `ShardAlloc` free that
  reuses the owner check and remote-free path unchanged. The `Allocator` and
  `GlobalAlloc` paths keep using the `Layout` and never read the map.

### `GlobalAlloc for ShardAlloc`

`alloc`, `dealloc`, `alloc_zeroed` (fresh pages are known zero),
`realloc` (the existing resize). `os.rs`'s fallback on Miri and non-Linux
switches from `std::alloc::alloc_zeroed` (the global allocator: itself) to
`std::alloc::System`.

### Thread states and re-entrancy

A thread's `CURRENT` is one of: the empty heap (no record yet), its record,
or *torn* (after exit). First use builds a record and registers the exit
destructor; registration can allocate (glibc's `__cxa_thread_atexit_impl`
calls `calloc`, which in the C library is ours). A new state, *initializing*,
is set before that work: allocations in it take the torn path (a parked
record, under the pool lock), which never re-enters setup. Blocks so
allocated carry that record's stamp, so freeing them is an ordinary remote
free.

### Fork

`pthread_atfork` (registered with the first record): prepare takes the pool
lock, parent and child release it. `std::sync::Mutex` cannot be released
without its guard, so the pool lock becomes a small lock of our own (an
atomic flag; spin, then yield), still under Loom. In the child, other
threads' records stay active with nobody to drain them: their memory is
leaked, as in mimalloc.

### The C library

Workspace member `shardheap-malloc` (`crate-type = ["cdylib"]`,
`panic = "abort"`), exporting `malloc`, `free`, `calloc` (overflow-checked),
`realloc` (`realloc(p, 0)` frees and returns null, as glibc),
`reallocarray`, `posix_memalign`, `aligned_alloc`, `memalign`, `valloc`,
`pvalloc`, `malloc_usable_size`, `malloc_trim`. Minimum alignment 16
(`max_align_t`); `malloc(0)` returns a unique 16-byte block.

**TLS model (decided):** in a `cdylib`, stable Rust reads every
thread-local through a call to `__tls_get_addr`, on every `malloc` and
`free`. The C library alone is built on nightly with
`-Z tls-model=initial-exec`, as mimalloc's preload library is; the core
crate stays on stable (MSRV 1.100), and CI gets one nightly job for the C
library. A library built this way must be loaded at startup (`LD_PRELOAD` or
linked), not with `dlopen`.

**Public API (decided):** the core crate gains, documented and covered by
semver from 0.5.0, `ShardAlloc::usable_size(ptr) -> usize` and
`unsafe ShardAlloc::free_unsized(ptr)` (useful in Rust too, when memory
crosses to C). The C library builds on those and on `ShardAlloc`'s
`GlobalAlloc`; C `realloc` is in place when the new size fits
`usable_size`, else allocate, copy, `free_unsized`. `shardheap-malloc` is
not published (`publish = false`) for now.

### mimalloc-bench

Cloned and built outside the repository (scratch space), benchmarks only.
A runner script runs each benchmark with each allocator through
`LD_PRELOAD`, three times, unpinned as the suite does, and writes the table.
Running our own library under `LD_PRELOAD` will need a permission rule like
the one for snmalloc.

## Checking

- Unit tests: `usable_size` and the size-free free for small, large and
  over-aligned blocks, own and remote; the segment map bit set and cleared.
- `tests/global.rs`: an integration test binary with `ShardAlloc` as
  `#[global_allocator]`: threads, thread-local destructors, collections sent
  between threads.
- A unit test in the crate (it can reach the pool lock): `fork` while another
  thread holds the pool lock; the child allocates and exits (Linux only, not
  under Miri).
- Loom for the new pool lock; Miri for the rest, strict provenance.
- The smoke run, then the suite.

## Left to the builder

File layout inside the crate, names, the runner's output format, which
everyday programs make up the smoke run.
