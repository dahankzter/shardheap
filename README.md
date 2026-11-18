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

## Benchmarks: one lifetime profile at a time

Median time per iteration through the same `Vec::new_in` / `Box::new_in` code
(`cargo bench`, AMD Threadripper PRO 9975WX, rustc 1.100). Lower is better.
Arenas (bumpalo and `Arena`) are reset inside every iteration, and the reset is
part of their time.

| workload | glibc | mimalloc | mimalloc `mi_heap` | **LocalHeap** | bumpalo | **Arena** |
|---|---|---|---|---|---|---|
| alloc/free one 64 B box, 10k times | 23.5 µs | 27.2 µs | 26.9 µs | **23.7 µs** | 8.44 µs | **8.39 µs** |
| 10k mixed-size buffers live, then freed | 504 µs | 121 µs | 121 µs | **63.8 µs** | 16.1 µs | **14.4 µs** |
| grow a `Vec<u64>` to 8 MiB | 217 µs | 318 µs | 317 µs | **251 µs** | 328 µs | **212 µs** |
| build and drop a 32k-node boxed tree | 776 µs | 250 µs | 254 µs | **228 µs** | 207 µs | **196 µs** |

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

- **Single-threaded by design.** Handles are `!Send`. Cross-thread frees
  (phase 3) and use as `#[global_allocator]` (phase 4) are not implemented.
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
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
