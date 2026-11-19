# shardheap

[![crates.io](https://img.shields.io/crates/v/shardheap.svg)](https://crates.io/crates/shardheap)
[![docs.rs](https://docs.rs/shardheap/badge.svg)](https://docs.rs/shardheap)
[![CI](https://github.com/dahankzter/shardheap/actions/workflows/ci.yml/badge.svg)](https://github.com/dahankzter/shardheap/actions/workflows/ci.yml)
[![MSRV 1.100](https://img.shields.io/badge/rustc-1.100+-blue.svg)](#status)

Shared-nothing allocators for Rust's stable `Allocator` API (Rust 1.100): a
mimalloc-style heap, and bump allocators for objects that die together, all
sharing one pool of memory per thread.

```rust
use shardheap::{Arena, LocalHeap};

let heap = LocalHeap::new();                       // one per thread or executor
let mut cache: Vec<Vec<u8, &LocalHeap>> = Vec::new();

let mut arena = Arena::new_in(&heap);              // request scratch, from the heap
for request in 0..1000u32 {
    let mut body: Vec<u8, &Arena<&LocalHeap>> = Vec::new_in(&arena);
    body.extend_from_slice(b"...parse, transform, render...");
    if request % 100 == 0 {
        let mut keep = Vec::new_in(&heap);         // the few survivors are copied out
        keep.extend_from_slice(&body);
        cache.push(keep);
    }
    drop(body);
    arena.reset();                                 // the rest goes at once
}
println!("{:?}", heap.stats());                    // built-in byte accounting
```

Forgetting to copy a survivor out is a compile error at `reset`, not a dangling
pointer.

## Why

General-purpose allocators pay for things a thread-per-core program does not
need, and cannot use what Rust already knows:

- **Ownership instead of atomics.** `LocalHeap` is `!Send`, so every collection
  allocated from it is too. Every free provably happens on the owning thread:
  no atomics, no owner check.
- **The size comes with every free.** Rust passes the `Layout` to `deallocate`,
  so the size class comes from the layout rather than a metadata lookup.
- **Lifetimes instead of runtime checks.** A `Vec<T, &Arena>` borrows the arena,
  so the arena cannot be reset or dropped while anything still points into it.
- **Pick the allocator per collection.** Long-lived state goes in the heap,
  request scratch in an arena, and both draw from the same resident memory.

## Which one for which objects

| type | allocate | free | for |
|---|---|---|---|
| `LocalHeap` | free-list pop per 64 KiB page, 88 size classes | push onto the page's local list | objects with independent lifetimes |
| `Arena` (`Arena::new()` or `Arena::new_in(&heap)`) | aligned pointer bump; the newest block grows in place | nothing; `reset` frees all at once | objects that die together: a request, a pass, a frame |
| `Reap` (`heap.reap()`) | pointer bump in heap pages | decrement the page's count; empty pages return to the heap | mostly die together, survivors that cannot be copied out |
| `StackArena` | bump in a stack buffer, then a fallback allocator | newest block only | scratch collections in one function |

Any of them can sit on a heap whose memory is one registered region
(`LocalHeap::with_region`), for I/O buffers that need no copy.

## Registered I/O memory

`LocalHeap::with_region(bytes, HugePages::Auto)` takes all its memory from one
region mapped up front, so every block (and every `Arena` or `Reap` block on
it) lies in a range that can be registered once with io_uring, RDMA or a
shared-memory window:

```rust
use shardheap::{HugePages, LocalHeap};

let heap = LocalHeap::with_region(64 << 20, HugePages::Auto).unwrap();
let region = heap.region().unwrap();       // register this once
let mut payload: Vec<u8, &LocalHeap> = Vec::new_in(&heap);
payload.extend_from_slice(b"built with ordinary std APIs");
payload.resize(64 << 10, 0);
assert!(heap.in_region(payload.as_ptr())); // so: a fixed-buffer write, no copy
```

Region memory is never unmapped, purged or remapped while the heap lives
(registered pages are pinned); large blocks grow in place into free space after
them. When the region is full, the heap falls back to the OS and `in_region`
says so. In a glommio prototype this cut CPU per 1 MiB O_DIRECT write by 62%
against building a `Vec<u8>` and copying it into a registered buffer.

## Huge pages

`LocalHeap::new()` follows the kernel's transparent-huge-page policy while the
heap is small and asks for 2 MiB pages once it maps 16 MiB
(`HugePages::Auto`); `with_huge_pages` selects `Always`, `Never` or `Kernel`.
On Zen 5, huge pages made building 256 MiB of nodes 2.1x faster (98x fewer page
faults) and random pointer chasing 10-45% faster from 1 MiB to 256 MiB working
sets, for 1-3 MiB more resident memory per heap. Details in
[DESIGN.md](DESIGN.md#huge-pages).

## Returning memory

Freed memory stays resident for reuse for one purge epoch (10 ms, longer while
purged memory keeps being needed again soon after), then goes back to the OS,
including free pages inside segments still in use. Housekeeping runs from the
allocation slow path, so a heap that goes quiet needs a nudge: call
`heap.maintain()` from an event loop's idle hook, or `heap.trim()` to return
everything free at once.

After freeing 90% of 256 MiB of small objects in runs that die together, then
trimming (`cargo run --release --example fragmentation -- <allocator> clustered`):
glibc keeps 37 MiB resident, mimalloc 54, LocalHeap 61 (0.2.0 kept 186). When
survivors are scattered over every page, no size-class allocator can return
much; see [DESIGN.md](DESIGN.md#fragmentation).

## Benchmarks: one lifetime profile at a time

Median time per iteration through the same `Vec::new_in` / `Box::new_in` code
(`cargo bench` with `RUSTFLAGS='-C llvm-args=-align-loops=64'`, AMD
Threadripper PRO 9975WX, rustc 1.100). Lower is better. Arenas (bumpalo and
`Arena`) are reset inside every iteration, and the reset is part of their time.

| workload | glibc | mimalloc | mimalloc `mi_heap` | **LocalHeap** | **ShardAlloc** | bumpalo | **Arena** |
|---|---|---|---|---|---|---|---|
| alloc/free one 64 B box, 10k times | 23.6 µs | 25.0 µs | 26.8 µs | **23.6 µs** | **25.2 µs** | 8.5 µs | **8.3 µs** |
| 10k mixed-size buffers live, then freed | 503 µs | 119 µs | 118 µs | **63.2 µs** | **85.8 µs** | 15.8 µs | **14.4 µs** |
| grow a `Vec<u64>` to 8 MiB | 218 µs | 318 µs | 321 µs | **247 µs** | **249 µs** | 312 µs | **212 µs** |
| build and drop a 32k-node boxed tree | 776 µs | 250 µs | 252 µs | **227 µs** | **230 µs** | 206 µs | **198 µs** |

`ShardAlloc` finds the current thread's heap through a thread-local and checks
the owner on every free, which is what a global allocator such as mimalloc
does on every `malloc` and `free`. It ties mimalloc on the 64 B box loop and
beats it on the other three; what it pays over `&LocalHeap` there is about
0.15 ns per allocate/free pair ([DESIGN.md](DESIGN.md#thread-heaps-shardalloc)).

The arenas win by doing almost nothing: allocation is an aligned pointer bump and
freeing is free. That only works when objects die together; `LocalHeap` is the
general-purpose heap that reuses memory block by block.

Single-threaded only: these numbers say nothing yet about cross-thread frees
(phase 3) or about use as a global allocator (phase 4).

## Mixed lifetimes: most objects die with their request, a few survive

`examples/request_mix.rs`: each request fills 500 buffers of mixed sizes
(about 340 KB) and grows an index vector. A share of the buffers survives into a
cache that holds the survivors of the last *window* requests. Time per request
and resident memory at the end (best of 3 runs, one process per allocator). About
5 µs of every request is writing the buffers, the same for all.

| survivors, window | mimalloc | LocalHeap | Reap | Arena on LocalHeap, survivors copied | bumpalo, survivors copied to mimalloc |
|---|---|---|---|---|---|
| none | 7.64 µs, 6.0 MiB | 7.69 µs, 4.0 MiB | 6.34 µs, 2.0 MiB | **5.14 µs**, 2.6 MiB | 5.31 µs, 0.6 MiB |
| 0.2%, 64 | 7.54 µs, 6.0 MiB | 7.57 µs, 4.0 MiB | 6.60 µs, 8.0 MiB | **5.11 µs**, 4.6 MiB | 5.18 µs, 4.7 MiB |
| 1%, 64 | 7.55 µs, 6.1 MiB | 7.29 µs, 4.1 MiB | 7.11 µs, 16.1 MiB | **5.23 µs**, 6.7 MiB | 5.43 µs, 4.7 MiB |
| 5%, 1 | 7.88 µs, 6.1 MiB | 7.85 µs, 4.0 MiB | 6.81 µs, 2.0 MiB | **5.90 µs**, 6.7 MiB | 6.00 µs, 6.7 MiB |
| 5%, 64 | 7.85 µs, 6.1 MiB | 7.60 µs, 6.1 MiB | 7.48 µs, 24.1 MiB | **5.92 µs**, 6.6 MiB | 5.99 µs, 6.7 MiB |
| 5%, 512 | 9.34 µs, 16.6 MiB | 8.89 µs, 14.6 MiB | 17.48 µs, 176.6 MiB | **5.96 µs**, 15.1 MiB | 6.02 µs, 15.1 MiB |

- **An arena plus explicit copying of survivors wins everywhere.** Copying a few
  survivors costs far less than any per-object free. The borrow checker is what
  makes this safe: a survivor that was not copied out is a compile error at
  `reset`, not a dangling pointer.
- **The reap wins only when survivors are rare or short-lived.** Every live
  survivor pins a 64 KiB page. At 5% survival over 512 requests that is 177 MiB
  instead of 15, and the bump pointer then runs through cold memory, so it gets
  slower too. Use it where survivors cannot be copied out: they are decided deep
  in other code, or something depends on their address.
- Survivors' resident memory is the same for every allocator that copies them
  into a general heap, so the footprint is decided by where survivors live, not
  by the arena.


## Status

Phase 1 of a larger plan; [DESIGN.md](DESIGN.md) covers the layout, fast paths,
how memory goes back to the OS, the unsafe surface, and the roadmap.

- **Single-threaded for now.** Handles are `!Send`. `ShardAlloc` gives each
  thread its own heap; sending its collections across threads (phase 3, stage B)
  and use as `#[global_allocator]` (phase 4) are not implemented.
- **Rust 1.100 or newer:** the `Allocator` trait was stabilized there.
- **Linux** gets `mmap`, `mremap` and `madvise`. Other platforms, and Miri, take
  memory from the global allocator instead.
- Experimental: the API may change, and it has not run in production.

## Checking

```
cargo test
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test
cargo bench          # vs glibc, mimalloc, mimalloc first-class heaps, bumpalo
cargo run --release --example request_mix -- reap 50 64   # allocator, survivors ‰, window
cargo run --release --example thp -- auto                 # huge-page policy
cargo run --release --example fragmentation -- heap clustered  # or glibc, mimalloc; random
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
