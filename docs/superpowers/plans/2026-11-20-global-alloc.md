# Phase 4: GlobalAlloc and a malloc replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ShardAlloc` as `#[global_allocator]`, a preloadable C `malloc`
library over it, and a first mimalloc-bench comparison.

**Architecture:** A segment map (one bit per 4 MiB) and a per-slot class
table let a block be freed and measured without its size; large headers move
to `p - PAGE`. `GlobalAlloc for ShardAlloc` reuses the `Allocator` paths. The
pool lock becomes our own lock so `fork` handlers can hold it. A workspace
member `shardheap-malloc` (`cdylib`, nightly `-Z tls-model=initial-exec`)
exports the C API.

**Tech Stack:** Rust 1.100 stable (core), nightly (C library build only),
libc, Loom, Miri, a C compiler for the API test, mimalloc-bench.

**Spec:** `docs/superpowers/specs/2026-11-20-global-alloc-design.md`

## Global Constraints

- Core crate: stable, MSRV 1.100; nightly only to build `shardheap-malloc`
  with `-Z tls-model=initial-exec`.
- New public API, documented, semver from 0.5.0:
  `ShardAlloc::usable_size(ptr)` and `unsafe ShardAlloc::free_unsized(ptr)`.
- `shardheap-malloc`: `publish = false`; Linux; runs on x86-64; aarch64 must
  build (CI cross-compile check).
- Segment map: 47 bits of address space on x86-64 (4 MiB map), 48 on aarch64
  (8 MiB), `MAP_NORESERVE`.
- C semantics: minimum alignment 16; `malloc(0)` returns a unique block;
  `realloc(p, 0)` frees and returns null; failure returns null with
  `errno = ENOMEM`; `calloc`/`reallocarray` overflow-checked.
- No panic unwinds across the C ABI (`extern "C"` aborts on unwind).
- Anti-goal: the `Allocator` fast paths must not slow: `benches/alloc.rs`
  and `benches/cross.rs` stay where stage B left them (README tables).
- No tuning for mimalloc-bench in this phase: it ends with results.
- No commits unless the user asks (standing rule); tasks end uncommitted.

## Review Focus

1. `free(NULL)` and `realloc(NULL, n)`: a no-op and `malloc(n)` — `api.c`
   (Task 6).
2. A block freed with `free_unsized` after its owning thread exited (its heap
   parked): freed exactly once, no hang — Task 3 test.
3. `realloc` across the small/large boundary both ways keeps the bytes —
   `api.c` (Task 6).
4. Alignments of 4 MiB and 8 MiB (data at a 4 MiB boundary): the segment map
   says "large" and the header is found — Task 2 test.
5. In a `fork` child, freeing a block allocated before the fork by another
   thread (whose heap nobody drains now) returns and does not hang — Task 5
   test.

---

### Task 1: Large header at `p - PAGE`

**Files:**
- Modify: `src/heap.rs` (`LargeHeader`, `large_offset` users: `large_msg`,
  `large_owner`/`owner_of`, `link_large`, `free_large_header`, the large
  dealloc and resize paths, the large-mapping cache, test helper `owner_of`)

**Interfaces:**
- Produces: `pub(crate) unsafe fn large_header(p: *mut u8) -> *mut LargeHeader`
  (= `p - PAGE`); `LargeHeader` keeps `map_size`, `map_align` and gains
  nothing else (the mapping base is `header + PAGE - large_offset(map_align)`).

- [ ] **Step 1: Write the failing test** in `heap.rs` tests

```rust
#[test]
fn large_header_sits_one_page_before_data_for_every_alignment() {
    let heap = LocalHeap::new();
    for align in [8, PAGE, 2 * PAGE, 1 << 20, 4 << 20, 8 << 20] {
        let l = Layout::from_size_align(300_000, align).unwrap();
        let p = heap.allocate(l).unwrap().cast::<u8>().as_ptr();
        assert_eq!(p.addr() % align, 0);
        // SAFETY: a live large block of this heap.
        let h = unsafe { large_header(p) };
        assert_eq!(h.addr(), p.addr() - PAGE, "align {align}");
        assert_eq!(unsafe { (*h).map_align }, align.max(PAGE));
        unsafe { heap.deallocate(NonNull::new(p).unwrap(), l) };
    }
    assert_eq!(heap.stats().large_allocations, 0);
}
```

- [ ] **Step 2: Run it:** `cargo test --lib large_header_sits` — FAIL
  (`large_header` not found).
- [ ] **Step 3: Implement.** Write the header at `base + off - PAGE` in
  `link_large`; derive the base from the header wherever the mapping is
  unmapped, remapped or cached; replace every `p.sub(large_offset(align))`
  with `large_header(p)`.
- [ ] **Step 4: Run** `cargo test --release` — all suites pass.

### Task 2: Segment map

**Files:**
- Create: `src/segmap.rs`
- Modify: `src/lib.rs` (`mod segmap;`), `src/heap.rs` (`new_segment` sets;
  every path that unmaps a segment clears, before the unmap)

**Interfaces:**
- Produces, in `segmap.rs`:
  `pub(crate) fn mark(seg: *mut Segment)`,
  `pub(crate) fn unmark(seg: *mut Segment)`,
  `pub(crate) fn is_segment(p: *const u8) -> bool`.
  Bits are atomics (`fetch_or`/`fetch_and`, `Relaxed` for the bit; the
  segment is published to other threads by the remote-free path's own
  ordering). The map is mapped on first `mark` with `os::map_aligned`
  (so Miri uses its fallback); `is_segment` before any `mark`, or for an
  address beyond the covered bits, returns false.

- [ ] **Step 1: Write the failing tests** in `segmap.rs`

```rust
#[test]
fn segments_are_marked_and_large_blocks_are_not() {
    let heap = LocalHeap::new();
    let small = heap.allocate(Layout::new::<[u8; 64]>()).unwrap().cast::<u8>();
    assert!(is_segment(small.as_ptr()));
    for align in [PAGE, 4 << 20, 8 << 20] {
        let l = Layout::from_size_align(300_000, align).unwrap();
        let big = heap.allocate(l).unwrap().cast::<u8>();
        assert!(!is_segment(big.as_ptr()), "align {align}");
        unsafe { heap.deallocate(big, l) };
    }
    unsafe { heap.deallocate(small, Layout::new::<[u8; 64]>()) };
}

#[test]
fn a_released_segment_is_unmarked() {
    let heap = LocalHeap::new();
    let p = heap.allocate(Layout::new::<[u8; 64]>()).unwrap().cast::<u8>();
    let seg = crate::segment::segment_of(p.as_ptr());
    unsafe { heap.deallocate(p, Layout::new::<[u8; 64]>()) };
    heap.trim();
    assert!(!is_segment(seg.cast()));
}
```

- [ ] **Step 2: Run** `cargo test --lib segmap` — FAIL (module missing).
- [ ] **Step 3: Implement** `segmap.rs` and the two hooks. Covered bits:
  `1 << (47 - 22)` on x86-64, `1 << (48 - 22)` on aarch64.
- [ ] **Step 4: Run** `cargo test --release`, then
  `MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --lib segmap`
  — pass.

### Task 3: Size without a `Layout`: `usable_size` and `free_unsized`

**Files:**
- Modify: `src/segment.rs` (`Segment` gains `classes: [u8; SLOTS]` right
  after `owner`, so it fills the second cache line; `owner` stays at 0x38),
  `src/heap.rs` (`new_page` writes the class into every slot of the span;
  `Inner::block_of`), `src/shard.rs` (public functions)

**Interfaces:**
- Consumes: `segmap::is_segment` (Task 2), `large_header` (Task 1).
- Produces:
  - `pub(crate) unsafe fn block_of(p: *mut u8) -> BlockInfo` in `heap.rs`,
    `struct BlockInfo { owner: *mut (), size: usize, msg: Msg }` (small: size
    from `classes[slot]`, never the page descriptor; large: from the header).
  - `impl ShardAlloc { pub unsafe fn usable_size(ptr: NonNull<u8>) -> usize;
    pub unsafe fn free_unsized(ptr: NonNull<u8>) }` — own blocks freed
    locally, others through `remote_free(owner, msg, size)`, a stamp of 0
    aborts (`heap::foreign_free`). Doc comments state the contract: `ptr`
    came from `ShardAlloc` and is live.

- [ ] **Step 1: Write the failing tests** in `shard.rs` tests

```rust
#[test]
fn usable_size_is_the_block_size() {
    for (size, align) in [(1, 1), (24, 8), (1000, 16), (100_000, 64), (300_000, 8), (300_000, 1 << 20)] {
        let l = Layout::from_size_align(size, align).unwrap();
        let b = ShardAlloc::new().allocate(l).unwrap();
        let p = b.cast::<u8>();
        assert_eq!(unsafe { ShardAlloc::usable_size(p) }, b.len(), "{size}/{align}");
        unsafe { ShardAlloc::free_unsized(p) };
    }
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn free_unsized_on_another_thread_sends_the_block_home() {
    let _g = serial();
    let sizes = [64usize, 1024, 4096, 300_000];
    let blocks: Vec<Sent<u8>> = sizes.iter()
        .map(|&s| Sent(ShardAlloc::new().allocate(Layout::from_size_align(s, 8).unwrap()).unwrap().cast::<u8>().as_ptr()))
        .collect();
    std::thread::spawn(move || for b in blocks {
        unsafe { ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap()) }
    }).join().unwrap();
    ShardAlloc::maintain();
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn free_unsized_after_the_owner_exited_frees_once() {
    let _g = serial();
    let (r, b) = std::thread::spawn(|| {
        let b = ShardAlloc::new().allocate(Layout::new::<[u8; 1024]>()).unwrap();
        (current_record(), Sent(b.cast::<u8>().as_ptr()))
    }).join().unwrap();
    // Freed on a third thread: its exit flushes the batch to the parked
    // heap, which the sender drains under the pool lock.
    std::thread::spawn(move || unsafe {
        ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap())
    }).join().unwrap();
    assert_eq!(record_stats(r).allocated_bytes, 0);
}
```

(`Sent`, `serial`, `current_record`, `record_stats` are the existing test
helpers in `shard.rs`.)

- [ ] **Step 2: Run** `cargo test --lib unsized usable_size` — FAIL.
- [ ] **Step 3: Implement.** Write `classes` in `new_page` (all `span`
  slots). `free_unsized` mirrors `deallocate` with `block_of` in place of the
  `Layout`.
- [ ] **Step 4: Run** `cargo test --release`; check the `Allocator` paths
  did not change: `cargo bench --bench alloc` for `shard_alloc` and
  `shardheap` within noise of the README table.

### Task 4: `GlobalAlloc for ShardAlloc`

**Files:**
- Modify: `src/shard.rs` (`unsafe impl GlobalAlloc for ShardAlloc`;
  `init_current` sets `CURRENT` to the torn heap while it works, so any
  allocation inside setup takes the torn path; it must not hold the pool
  lock across anything that can allocate), `src/os.rs` (fallback uses
  `std::alloc::System`, not the global allocator), `src/lib.rs` docs
- Create: `tests/global.rs`

**Interfaces:**
- Produces: `unsafe impl GlobalAlloc for ShardAlloc` — `alloc`, `dealloc`,
  `alloc_zeroed`, `realloc` over the `Allocator` methods; null on failure.

- [ ] **Step 1: Write the failing test binary** `tests/global.rs`

```rust
use shardheap::ShardAlloc;
#[global_allocator]
static A: ShardAlloc = ShardAlloc::new();

#[test]
fn std_runs_on_shardalloc() {
    let mut v: Vec<String> = (0..10_000).map(|i| i.to_string()).collect();
    v.retain(|s| s.len() > 2);
    assert_eq!(v.len(), 10_000 - 100);
}

#[test]
fn threads_send_collections_both_ways() {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let h = std::thread::spawn(move || for v in rx { assert!(v.iter().all(|&b| b == 7)) });
    for n in [1, 64, 4096, 300_000] { tx.send(vec![7u8; n]).unwrap(); }
    drop(tx);
    h.join().unwrap();
}

#[test]
fn short_lived_threads_with_thread_local_destructors() {
    struct D(Vec<u8>);
    impl Drop for D { fn drop(&mut self) { let _again = vec![1u8; self.0.len()]; } }
    thread_local!(static T: std::cell::RefCell<Option<D>> = const { std::cell::RefCell::new(None) });
    for _ in 0..200 {
        std::thread::spawn(|| T.with(|t| *t.borrow_mut() = Some(D(vec![0; 1000])))).join().unwrap();
    }
}

#[test]
fn zeroed_and_realloc_keep_their_promises() {
    use std::alloc::{GlobalAlloc, Layout};
    let l = Layout::from_size_align(5000, 64).unwrap();
    unsafe {
        let p = A.alloc_zeroed(l);
        assert!(std::slice::from_raw_parts(p, 5000).iter().all(|&b| b == 0));
        p.write_bytes(9, 5000);
        let q = A.realloc(p, l, 400_000);
        assert_eq!(q.addr() % 64, 0);
        assert!(std::slice::from_raw_parts(q, 5000).iter().all(|&b| b == 9));
        A.dealloc(q, Layout::from_size_align(400_000, 64).unwrap());
    }
}
```

- [ ] **Step 2: Run** `cargo test --release --test global` — FAIL
  (`GlobalAlloc` not implemented).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --release` and
  `MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --test global`
  — pass.

### Task 5: Own pool lock and `fork`

**Files:**
- Modify: `src/remote.rs` (`Pool`'s `Mutex` becomes `RawLock`: an
  `AtomicBool`, spin with `spin_loop` then `yield_now`; Loom versions under
  `cfg(loom)`; `PoolGuard` unchanged for callers), `src/shard.rs`
  (`pthread_atfork` registered once, with the first record: prepare locks the
  pool, parent and child unlock it)

**Interfaces:**
- Produces: `impl Pool { pub(crate) unsafe fn lock_raw(&self);
  pub(crate) unsafe fn unlock_raw(&self) }` (for the fork handlers only).

- [ ] **Step 1: Write the failing tests**

In `remote.rs` Loom tests:
```rust
#[test]
fn pool_lock_excludes() {
    loom::model(|| {
        let pool = Arc::new(Pool::new());
        let n = Arc::new(loom::sync::atomic::AtomicUsize::new(0));
        let hs: Vec<_> = (0..2).map(|_| { let (p, n) = (pool.clone(), n.clone());
            loom::thread::spawn(move || { let _g = p.lock();
                let v = n.load(Relaxed); n.store(v + 1, Relaxed); }) }).collect();
        for h in hs { h.join().unwrap(); }
        assert_eq!(n.load(Relaxed), 2);
    });
}
```
In `shard.rs` tests (`#[cfg(all(target_os = "linux", not(miri)))]`):
```rust
#[test]
fn fork_while_another_thread_holds_the_pool_lock() {
    let _g = serial();
    let b = std::thread::spawn(|| Sent(ShardAlloc::new()
        .allocate(Layout::new::<[u8; 64]>()).unwrap().cast::<u8>().as_ptr()))
        .join().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || { let _l = POOL.lock(); tx.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200)); });
    rx.recv().unwrap();
    match unsafe { libc::fork() } {
        0 => unsafe {
            let v: Vec<u8, ShardAlloc> = Vec::with_capacity_in(100, ShardAlloc::new());
            drop(v);
            ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap());
            libc::_exit(0)
        },
        pid => { let mut st = 0; assert_eq!(unsafe { libc::waitpid(pid, &mut st, 0) }, pid);
            assert!(libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0); }
    }
    holder.join().unwrap();
}
```
(The child must finish in time: if it deadlocks, `waitpid` hangs and the
test times out — that is the failure.)

- [ ] **Step 2: Run** `cargo test --release --lib fork_while` — the child
  deadlocks on the pool lock without the handlers (the test hangs: run it
  with `timeout 60`). The Loom test passes on the current `Mutex`; it guards
  the replacement.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --release`;
  `LOOM_MAX_PREEMPTIONS=3 RUSTFLAGS='--cfg loom' cargo test --release --lib remote`
  — pass.

### Task 6: The C library `shardheap-malloc`

**Files:**
- Modify: `Cargo.toml` (`[workspace] members = ["malloc"]`),
  `.github/workflows/ci.yml` (jobs: `cargo test --workspace`; nightly build
  of the C library plus `malloc/tests/run.sh`; `cargo check --target
  aarch64-unknown-linux-gnu --workspace`)
- Create: `malloc/Cargo.toml` (`name = "shardheap-malloc"`,
  `publish = false`, `crate-type = ["cdylib"]`, deps `shardheap` by path,
  `libc`), `malloc/src/lib.rs`, `malloc/tests/api.c`, `malloc/tests/run.sh`,
  `malloc/README.md` (build command and `LD_PRELOAD` use)

**Interfaces:**
- Consumes: `ShardAlloc` as `GlobalAlloc` (Task 4), `usable_size`,
  `free_unsized` (Task 3).
- Produces: `#[unsafe(no_mangle)] pub unsafe extern "C" fn` `malloc`, `free`,
  `calloc`, `realloc`, `reallocarray`, `posix_memalign`, `aligned_alloc`,
  `memalign`, `valloc`, `pvalloc`, `malloc_usable_size`, `malloc_trim`.
  Build: `RUSTFLAGS='-Z tls-model=initial-exec' cargo +nightly build
  --release -p shardheap-malloc` → `target/release/libshardheap_malloc.so`.
  `malloc(n)` uses `Layout(max(n, 1), 16)`; `realloc` stays in place when
  `n <= usable_size(p)` (and `n` keeps it in the same small/large kind),
  else allocate, copy `min(old usable, n)`, `free_unsized`.

- [ ] **Step 1: Write the failing tests.** `malloc/tests/api.c` — one
  `main` with `assert`s:
  - `malloc(0)` twice: non-null, distinct, 16-aligned; `free(NULL)` returns;
    `realloc(NULL, 10)` is non-null;
  - `calloc(1 << 20, 1 << 20)` is null with `errno == ENOMEM`;
    `calloc(100, 10)` is zeroed;
  - `realloc(p, 0)` returns null;
  - `realloc` from 100 B to 1 MiB to 50 B keeps the first 50 bytes;
  - `posix_memalign` for alignments 16 to 8 MiB returns aligned memory,
    `EINVAL` for alignment 24;
  - `aligned_alloc(64, 100)`, `memalign(4096, 10)`, `valloc(10)` and
    `pvalloc(10)` are aligned;
  - `malloc_usable_size(malloc(100)) >= 100`;
  - `reallocarray` overflow is null with `errno == ENOMEM`;
  - eight `pthread`s malloc and free each other's blocks through a shared
    array, 100k each.

  `malloc/tests/run.sh` builds the library, compiles `api.c` with `cc
  -pthread`, runs it with `LD_PRELOAD`, then runs `ls -la /usr/lib`,
  `sort` of 1e6 random lines, `python3 -c "import json; print(len(json.dumps(list(range(10**6)))))"`,
  `git --version`, and a C++ program (`std::thread`, `std::vector`,
  `std::string`, `new`/`delete` across threads) under `LD_PRELOAD`, failing on
  any non-zero exit. It also checks the TLS model:
  `objdump -d --disassemble=malloc libshardheap_malloc.so | grep -c __tls_get_addr`
  must print 0.
- [ ] **Step 2: Run** `malloc/tests/run.sh` — FAIL (crate does not exist).
  (If `LD_PRELOAD` of the library is refused by the permission check, stop
  and ask the user for a rule covering `malloc/tests/run.sh`.)
- [ ] **Step 3: Implement** the crate and the CI jobs.
- [ ] **Step 4: Run** `malloc/tests/run.sh` and `cargo test --workspace
  --release` — pass; `cargo +nightly check --target
  aarch64-unknown-linux-gnu --workspace` — builds.

### Task 7: mimalloc-bench and results

**Files:**
- Create: `malloc/bench/run.sh` (the runner), `malloc/bench/README.md`
  (setup steps and how to read the table)
- Modify: `README.md` (global allocator and C library usage, the suite
  table), `DESIGN.md` (the size-free free, segment map, re-entrancy, fork,
  suite results; roadmap phase 4 done)

**Interfaces:**
- Consumes: `libshardheap_malloc.so` (Task 6).

- [ ] **Step 1:** Clone `https://github.com/daanx/mimalloc-bench` into the
  scratch directory (outside the repository); build its benchmarks and its
  `mi`, `je`, `tc`, `sn` allocators with `./build-bench-env.sh` (or the
  benchmarks only, if the script cannot install packages on this
  distribution: then use the system `libjemalloc.so.2`,
  `libtcmalloc_minimal.so.4` and build mimalloc and snmalloc's shims with
  CMake).
- [ ] **Step 2:** `malloc/bench/run.sh <bench-dir> <allocator=lib.so>...`:
  each benchmark that builds, for each allocator (glibc without preload),
  three runs under `/usr/bin/time -f '%e %M'`, median time and peak RSS,
  printed as a Markdown table with ratios to mimalloc. Benchmarks that fail
  for any allocator are reported as failed for it, not dropped.
- [ ] **Step 3:** Run it; the run must complete for every allocator.
  shardheap failures are findings: write them down with the failing
  benchmark's output, fix crashes (a crash is a bug: test first in the
  owning module), do not tune speed.
- [ ] **Step 4:** Write the docs; report the table to the user.

## Self-review notes

- Spec coverage: size-free free (Tasks 1–3), `GlobalAlloc` and `os.rs`
  fallback (4), re-entrancy (4: initializing via the torn path), fork and own
  lock (5), C library, TLS model, public API, CI, aarch64 build (6), suite
  and runner (7).
- Types: `large_header`, `segmap::{mark, unmark, is_segment}`,
  `Inner::block_of -> BlockInfo`, `ShardAlloc::{usable_size, free_unsized}`,
  `Pool::{lock_raw, unlock_raw}` are used under the same names throughout.
