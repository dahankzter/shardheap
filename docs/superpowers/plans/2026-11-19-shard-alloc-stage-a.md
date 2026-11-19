# ShardAlloc, stage A (thread heaps on one thread) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ShardAlloc`, a zero-sized `!Send` allocator handle that allocates from the current thread's own heap, and make it match or beat mimalloc on one thread, without slowing `LocalHeap`.

**Architecture:** Each thread gets a *record* (heap state with a stable address, mapped through `os.rs`, never freed, linked into a static registry) found through a const `thread_local!`. Segments and large-mapping headers are stamped with their owner record's address. `ShardAlloc`'s free checks the stamp against the current record (the check a global allocator pays; mimalloc does the same), with a cold abort on mismatch that stage B replaces with batching. `LocalHeap`'s paths are unchanged; explicit heaps stamp owner 0.

**Tech Stack:** Rust 1.100 stable (`Allocator` trait), `libc`, criterion, Miri. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-19-remote-frees-design.md` (stage A sections; stage B is out of scope).

## Global Constraints

- MSRV 1.100, edition 2024; no new dependencies.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps` clean; `cargo +1.100 test` passes.
- `MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test` passes (use `CARGO_TARGET_DIR=target-miri`).
- Every `unsafe` block carries a `// SAFETY:` comment, as in the existing code.
- `LocalHeap`, `Arena`, `Reap`, `StackArena`, `Os`: public API unchanged; no atomic and no extra branch on `LocalHeap`'s allocate/free fast paths.
- `ShardAlloc` is `!Send` and `!Sync` in stage A.
- Repo rule: **no commits, pushes, tags or publishes** unless the user asks. Work on local branch `shard-alloc`; each task ends with a green checkpoint instead of a commit.
- Targets (spec, "Performance targets"): (1) `LocalHeap` within noise of 0.3.0 on every existing benchmark; (2) `ShardAlloc` matches or beats mimalloc (`mimalloc` column) on `lifo_64B`, `batch_mixed`, `vec_growth_8MiB`, `tree_32k_nodes`. If (2) fails, make the lookup or check faster; the target does not move.

## Review Focus

1. Zero-sized layouts through `ShardAlloc`: free must not read a segment or large header for a dangling pointer. Test in Task 2.
2. Large blocks with alignment above 4096 (header at `p - align`), and blocks moving small to large and back through `grow`/`shrink`: owner found correctly every time. Test in Task 2.
3. `allocate_zeroed` returning memory previously used and freed through `ShardAlloc`: must be zero. Test in Task 2.
4. Frees and allocations from other thread-locals' destructors after the thread's exit guard ran. Test in Task 3.
5. Many threads creating records at once (registry push races). Test in Task 3 (runs under Miri's race detector).

---

### Task 1: Owner stamps and checked free in `heap.rs`

**Files:**
- Modify: `src/heap.rs` (`Inner`, `LocalHeap::new`, `LargeHeader`, `link_large`, `new_segment`, `LocalHeap::allocate_zeroed`, tests module)
- Modify: `src/segment.rs` (rename `_owner` to `owner`, doc comment)

**Interfaces:**
- Consumes: nothing new.
- Produces (all `pub(crate)` in `heap.rs`):
  - `const fn Inner::new(huge_pages: HugePages) -> Inner` (body of today's `LocalHeap::new`; `LocalHeap::new`/`with_huge_pages` call it).
  - field `Inner::owner: usize`, 0 for explicit heaps; stamped into `Segment::owner` by `new_segment` and into new field `LargeHeader::owner` by `link_large`.
  - `unsafe fn Inner::alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>` (moved from `LocalHeap::allocate_zeroed`, which now calls it).
  - `#[inline(always)] unsafe fn Inner::dealloc_owned(&mut self, p: *mut u8, layout: Layout)`: size 0 returns first; `class_for` computed once; small: `(*segment_of(p)).owner`, large: `(*(p - large_offset(align)) as LargeHeader).owner`; equal to `self.owner` frees through the existing small/large path, else calls `foreign_free()`.
  - `unsafe fn Inner::resize_owned(&mut self, p: *mut u8, old: Layout, new: Layout) -> Result<NonNull<[u8]>, AllocError>`: same owner check on `(p, old)`, then `resize`.
  - `#[cold] #[inline(never)] fn foreign_free() -> !`: writes `shardheap: block freed on a thread that does not own it` to stderr and calls `std::process::abort()`.
  - `Inner::trim`, `Inner::maintain` made `pub(crate)`.

- [ ] **Step 1: Create branch and save the 0.3.0 benchmark baseline (loop-aligned, see DESIGN.md measurement caveat)**

```bash
git checkout -b shard-alloc
RUSTFLAGS='-C llvm-args=-align-loops=64' CARGO_TARGET_DIR=target-bench-aligned \
  cargo bench --bench alloc -- --save-baseline v0.3.0
```
Expected: criterion reports all groups; baseline saved.

- [ ] **Step 2: Write failing unit tests in `src/heap.rs` `mod tests`**

```rust
#[test]
fn explicit_heaps_stamp_owner_zero() {
    let heap = LocalHeap::new();
    let small = heap.allocate(Layout::new::<[u8; 64]>()).unwrap().cast::<u8>();
    let big = Layout::from_size_align(1 << 20, 8).unwrap();
    let large = heap.allocate(big).unwrap().cast::<u8>();
    unsafe {
        assert_eq!((*segment::segment_of(small.as_ptr())).owner, 0);
        let h = large.as_ptr().sub(large_offset(8)).cast::<LargeHeader>();
        assert_eq!((*h).owner, 0);
        heap.deallocate(small, Layout::new::<[u8; 64]>());
        heap.deallocate(large, big);
    }
}

#[test]
fn owned_frees_and_resizes_pass_the_owner_check() {
    let heap = LocalHeap::new();
    unsafe { heap.inner().owner = 0x5000 };
    // small, large with align 8, large with align 1 << 16
    for l in [
        Layout::new::<[u8; 64]>(),
        Layout::from_size_align(1 << 20, 8).unwrap(),
        Layout::from_size_align(300_000, 1 << 16).unwrap(),
    ] {
        let p = heap.allocate(l).unwrap().cast::<u8>();
        unsafe {
            let owner = if class::class_for(l).is_some() {
                (*segment::segment_of(p.as_ptr())).owner
            } else {
                (*p.as_ptr().sub(large_offset(l.align())).cast::<LargeHeader>()).owner
            };
            assert_eq!(owner, 0x5000);
            let big = Layout::from_size_align(l.size() * 2, l.align()).unwrap();
            let q = heap.inner().resize_owned(p.as_ptr(), l, big).unwrap().cast::<u8>();
            heap.inner().dealloc_owned(q.as_ptr(), big);
        }
    }
    assert_eq!(heap.stats().allocated_bytes, 0);
}
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test --lib heap::tests::explicit_heaps_stamp_owner_zero heap::tests::owned_frees`
Expected: compile errors (`owner` field missing, `resize_owned`/`dealloc_owned` not found).

- [ ] **Step 4: Implement the Interfaces block above**

`new_segment`: stamp `(*seg).owner = self.owner` after `init_segment` (covers fresh and region-reused headers). `link_large`: add `owner: self.owner` to the header written. `LargeHeader` keeps `#[repr(C)]`; it still fits the 4096-byte data offset.

- [ ] **Step 5: Run the whole suite**

Run: `cargo test && cargo clippy --all-targets -- -D warnings`
Expected: all pass, including the two new tests; no warnings.

- [ ] **Step 6: Checkpoint `LocalHeap` speed (anti-goal)**

Run: `RUSTFLAGS='-C llvm-args=-align-loops=64' CARGO_TARGET_DIR=target-bench-aligned cargo bench --bench alloc -- shardheap --baseline v0.3.0`
Expected: every `*/shardheap` and `*/shardheap_arena` change within ±2% or "No change in performance detected". Changes left uncommitted.

---

### Task 2: Thread records and `ShardAlloc`

**Files:**
- Create: `src/shard.rs`
- Modify: `src/lib.rs` (`mod shard;`, `pub use shard::ShardAlloc;`, crate docs bullet)
- Test: `src/shard.rs` `mod tests`, `tests/model.rs`

**Interfaces:**
- Consumes (Task 1): `Inner::new`, `Inner::owner`, `Inner::alloc`, `Inner::alloc_zeroed`, `Inner::dealloc_owned`, `Inner::resize_owned`, `Inner::trim`, `Inner::maintain`, field `Inner::stats`; `os::map_aligned`, `os::PAGE`.
- Produces:
  - `pub(crate) struct Record { inner: UnsafeCell<Inner>, next: *mut Record }` (`next` links the registry; stage B adds pool fields).
  - `static RECORDS: AtomicPtr<Record>`: push-only registry (CAS push, `Release`/`Acquire`); keeps every record reachable from a static (Miri leak check) and is stage B's walk list.
  - `fn new_record() -> *mut Record`: maps `size_of::<Record>()` rounded up to `PAGE` with `os::map_aligned(.., PAGE)`, writes `Inner::new(HugePages::Auto)` with `owner = record address`, pushes onto `RECORDS`. Aborts (via `std::alloc::handle_alloc_error`) if mapping fails.
  - `thread_local! { static CURRENT: Cell<*mut Record> = const { Cell::new(ptr::null_mut()) }; }` (const, no destructor: a plain TLS load).
  - `#[inline(always)] fn current() -> *mut Inner`: `CURRENT` non-null → its inner; null → `#[cold] fn init_current() -> *mut Record` (creates the record, stores it in `CURRENT`; Task 3 adds the exit guard here).
  - `pub struct ShardAlloc { _not_send: PhantomData<*mut ()> }` with `#[derive(Clone, Copy, Default, Debug)]`, `pub const fn new() -> Self`, and associated functions `pub fn stats() -> Stats` (zeros without creating a record if this thread has none), `pub fn maintain()`, `pub fn trim()` (no-ops without a record).
  - `unsafe impl Allocator for ShardAlloc`: `allocate` → `alloc`; `allocate_zeroed` → `alloc_zeroed`; `deallocate` → `dealloc_owned`; `grow`, `grow_zeroed` (zero the new tail), `shrink` → `resize_owned`.

- [ ] **Step 1: Write failing tests**

In `tests/model.rs`:

```rust
#[test]
fn shard_alloc_model() {
    run_model(&ShardAlloc::new(), 31);
    run_model_with(&ShardAlloc::new(), 37, OPS / 4, true);
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn shard_alloc_zero_sized_and_zeroed() {
    let a = ShardAlloc::new();
    let z = Layout::from_size_align(0, 64).unwrap();
    let p = a.allocate(z).unwrap().cast::<u8>();
    unsafe { a.deallocate(p, z) }; // must not read any header
    let l = Layout::new::<[u8; 3000]>();
    let p = a.allocate(l).unwrap().cast::<u8>();
    fill(p, l.size(), 0xff);
    unsafe { a.deallocate(p, l) };
    let q = a.allocate_zeroed(l).unwrap().cast::<u8>();
    check(q, l.size(), 0);
    unsafe { a.deallocate(q, l) };
}

#[test]
fn shard_alloc_large_aligned_and_resized() {
    let a = ShardAlloc::new();
    let mut v: Vec<u8, ShardAlloc> = Vec::new_in(a);
    for i in 0..(3usize << 20) { v.push(i as u8); }      // small -> large through grow
    v.truncate(100);
    v.shrink_to_fit();                                     // large -> small through shrink
    assert!(v.iter().enumerate().all(|(i, &b)| b == i as u8));
    drop(v);
    let l = Layout::from_size_align(300_000, 1 << 16).unwrap();
    let p = a.allocate(l).unwrap().cast::<u8>();
    assert_eq!(p.addr().get() % (1 << 16), 0);
    unsafe { a.deallocate(p, l) };
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn shard_alloc_threads_have_their_own_heaps() {
    let a = std::thread::spawn(|| {
        let b = Box::new_in([1u8; 4000], ShardAlloc::new());
        let s = ShardAlloc::stats().allocated_bytes;
        drop(b);
        s
    }).join().unwrap();
    let mine = ShardAlloc::stats().allocated_bytes;
    let b = Box::new_in([2u8; 100], ShardAlloc::new());
    assert!(a >= 4000);
    assert_eq!(ShardAlloc::stats().allocated_bytes - mine, 112); // the 100 B class
    drop(b);
}
```

(Each `#[test]` runs on its own thread, so `ShardAlloc::stats()` sees only that test's heap. 100 B rounds to the 112 B class.)

Doc test on `ShardAlloc` (in `src/shard.rs`):

```rust
/// ```compile_fail,E0277
/// fn assert_send<T: Send>(_: T) {}
/// assert_send(Vec::<u8, _>::new_in(shardheap::ShardAlloc::new()));
/// ```
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --test model shard_alloc`
Expected: compile error, `ShardAlloc` not found.

- [ ] **Step 3: Implement `src/shard.rs` per the Interfaces block; export from `lib.rs`; add a crate-docs bullet for `ShardAlloc`**

The rustdoc for `ShardAlloc` says: allocates from the current thread's heap; `!Send` until cross-thread frees land; it is the shape of a global allocator, so on one thread it pays a thread-local lookup to allocate and an owner check to free, which `&LocalHeap` does not.

- [ ] **Step 4: Run the tests**

Run: `cargo test && cargo clippy --all-targets -- -D warnings`
Expected: all pass, including `shard_alloc_*` and the `compile_fail` doc test.

- [ ] **Step 5: Run Miri on the new tests**

Run: `MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test --test model shard_alloc`
Expected: 4 passed, no leak report, no UB. Checkpoint, uncommitted.

---

### Task 3: Thread exit and teardown

**Files:**
- Modify: `src/shard.rs`
- Test: `src/shard.rs` `mod tests`, `tests/model.rs`

**Interfaces:**
- Consumes (Task 2): `Record`, `RECORDS`, `CURRENT`, `init_current`, `ShardAlloc`; (Task 1) `Inner::trim`.
- Produces:
  - `struct ExitGuard;` with `Drop`: if `CURRENT` is non-null, `trim()` its record (a record with no live blocks ends with nothing mapped; one with live blocks keeps them). `CURRENT` stays set, so later frees and allocations on this thread (from other thread-locals' destructors) keep using the same record.
  - `thread_local! { static EXIT: ExitGuard = const { ExitGuard }; }`, touched with `EXIT.try_with(|_| ())` in `init_current` to register its destructor; an `Err` (thread already tearing down) is ignored.

- [ ] **Step 1: Write failing tests**

In `src/shard.rs` `mod tests` (they read the record directly):

```rust
fn record_stats(r: usize) -> Stats {
    unsafe { (*(*(r as *mut Record)).inner.get()).stats }
}

#[test]
fn exit_releases_an_empty_heap() {
    let r = std::thread::spawn(|| {
        let mut small: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
        small.extend_from_slice(&[1; 200]);
        let mut big: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
        big.resize(2 << 20, 2);
        drop((small, big));
        CURRENT.with(|c| c.get() as usize)
    }).join().unwrap();
    let s = record_stats(r);
    assert_eq!((s.allocated_bytes, s.mapped_bytes, s.segments), (0, 0, 0));
}

#[test]
fn exit_keeps_a_heap_with_live_blocks() {
    let (r, p) = std::thread::spawn(|| {
        let b = Box::leak(Box::new_in([7u8; 1000], ShardAlloc::new()));
        (CURRENT.with(|c| c.get() as usize), b.as_ptr() as usize)
    }).join().unwrap();
    let s = record_stats(r);
    assert!(s.allocated_bytes >= 1000 && s.mapped_bytes > 0);
    assert!(unsafe { core::slice::from_raw_parts(p as *const u8, 1000) }.iter().all(|&b| b == 7));
}

#[test]
fn records_are_distinct_and_registered_under_contention() {
    let rs: Vec<usize> = (0..8)
        .map(|_| std::thread::spawn(|| {
            drop(Box::new_in(1u64, ShardAlloc::new()));
            CURRENT.with(|c| c.get() as usize)
        }))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    let set: std::collections::HashSet<_> = rs.iter().copied().collect();
    assert_eq!(set.len(), 8);
    let mut seen = 0;
    let mut r = RECORDS.load(Ordering::Acquire);
    while !r.is_null() {
        if set.contains(&(r as usize)) { seen += 1; }
        r = unsafe { (*r).next };
    }
    assert_eq!(seen, 8);
}
```

In `tests/model.rs`:

```rust
struct LateUser(std::cell::RefCell<Option<Vec<u8, ShardAlloc>>>);
impl Drop for LateUser {
    fn drop(&mut self) {
        drop(self.0.borrow_mut().take());                  // free after the exit guard
        let mut v = Vec::with_capacity_in(64, ShardAlloc::new()); // allocate after it
        v.extend_from_slice(b"late");
        assert_eq!(&v[..], b"late");
    }
}
thread_local!(static LATE: LateUser = LateUser(std::cell::RefCell::new(None)));

#[test]
fn shard_alloc_survives_thread_teardown() {
    std::thread::spawn(|| {
        // Registered before the heap's exit guard, so (on Linux, where thread-local
        // destructors run in reverse registration order) it is destroyed after it.
        LATE.with(|_| ());
        let mut v = Vec::with_capacity_in(4096, ShardAlloc::new());
        v.resize(4096, 9u8);
        LATE.with(|l| *l.0.borrow_mut() = Some(v));
    }).join().unwrap();
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib shard::tests && cargo test --test model shard_alloc_survives`
Expected: `exit_releases_an_empty_heap` fails (mapped_bytes > 0, nothing trims at exit); the others may pass already; that is fine.

- [ ] **Step 3: Implement `ExitGuard` and register it in `init_current`**

- [ ] **Step 4: Run tests, then Miri on everything**

Run: `cargo test && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test`
Expected: all pass natively and under Miri (no data race, no leak report). Checkpoint, uncommitted.

---

### Task 4: Benchmarks, targets and docs

**Files:**
- Modify: `benches/alloc.rs` (`bench_all!`: new column `shard_alloc`)
- Modify: `README.md` (benchmark table column; a short `ShardAlloc` section; Status bullet), `DESIGN.md` (new section "Thread heaps (`ShardAlloc`)": records, registry, owner stamps, the check, exit; Roadmap: phase 3 split into stage A done / stage B next), `src/lib.rs` docs if Task 2 left gaps
- Possibly modify: `src/shard.rs`, `src/heap.rs` (only if a target is missed)

**Interfaces:**
- Consumes: `ShardAlloc` (Task 2).
- Produces: nothing for later tasks.

- [ ] **Step 1: Add the column**, after `shardheap`:

```rust
{
    let $a = &ShardAlloc::new();
    g.bench_function(BenchmarkId::from_parameter("shard_alloc"), |b| b.iter(|| $body));
}
```

- [ ] **Step 2: Measure**

Run: `RUSTFLAGS='-C llvm-args=-align-loops=64' CARGO_TARGET_DIR=target-bench-aligned cargo bench --bench alloc -- --baseline v0.3.0`
Expected, target 1: `shardheap` and `shardheap_arena` within ±2% of `v0.3.0`. Target 2: for each of the four groups, `shard_alloc` median ≤ `mimalloc` median.

- [ ] **Step 3: If target 2 misses on any group, make the path faster and re-measure**

Look first at: generated code of `allocate`/`deallocate` (`cargo asm` or `perf annotate` on the bench binary): `CURRENT` must compile to one `%fs`-relative load with the null check out of line; `class_for` must run once per free; the owner read must be the only extra load. Record what was changed and the before/after numbers in the DESIGN section. The target does not move; if it cannot be met, stop and report the gap with the evidence instead of shipping.

- [ ] **Step 4: Docs**

README benchmark table gains a `ShardAlloc` column with the measured medians and one sentence: what it pays over `LocalHeap` and why (lookup and owner check, the work a global allocator does). DESIGN gets the section and roadmap change above.

- [ ] **Step 5: Final verification**

Run:
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings \
  && RUSTDOCFLAGS='-D warnings' cargo doc --no-deps && cargo test && cargo +1.100 test \
  && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test
```
Expected: all clean and passing. Leave everything uncommitted on `shard-alloc` and report to the user: target results per group, any optimizations made, and the proposed version (0.4.0) for when they ask to release.
