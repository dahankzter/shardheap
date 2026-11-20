# ShardAlloc, stage B (frees across threads) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ShardAlloc` `Send + Sync`: a block freed on another thread is batched by the freeing thread and sent to its owner's inbox (one atomic operation per batch), exited threads' heaps are parked in a pool and adopted later, and cross-thread handoff beats mimalloc.

**Architecture:** A small concurrency core (`src/remote.rs`: inbox, state, pool, park/push handshake, Loom-tested in isolation) is embedded in each thread record. The freeing side (batch cache, flush, remote grow) and the owning side (draining messages as local frees) are single-threaded code in `src/shard.rs` and `src/heap.rs`. `LocalHeap` stays local-only.

**Tech Stack:** Rust 1.100 stable, `libc`, criterion, Miri, `loom` 0.7 (new dev-dependency, only under `--cfg loom`).

**Spec:** `docs/superpowers/specs/2026-11-19-remote-frees-design.md` (stage B: "Messages", "Draining", "Parking and the pool", "After thread-local teardown", "Transport trait", "Public API", "Module layout", "Checking"). Stage A shipped as 0.4.0; its code is the starting point.

## Global Constraints

- MSRV 1.100, edition 2024. Only new dependency: `loom = "0.7"` as a dev-dependency (`[target.'cfg(loom)'.dev-dependencies]`); declare `cfg(loom)` in `[lints.rust] unexpected_cfgs`.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`, `cargo test`, `cargo +1.100 test`, `MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test` all pass; plus the Loom suite: `RUSTFLAGS='--cfg loom' CARGO_TARGET_DIR=target-loom cargo test --release --lib remote`.
- Every `unsafe` block carries `// SAFETY:`.
- Anti-goals (spec): no atomic and no extra branch on `LocalHeap`'s allocate/free fast paths; no global lock on any common path (the pool lock only at thread start and exit, and for a remote free into a parked heap); the batch cache's held bytes are bounded.
- `Inner` stays the first field of a `#[repr(C)]` `Record` (the owner stamp is the record's address); records stay at `RECORD_OFFSET` in their page; the const assert that a record fits after `RECORD_OFFSET` must keep holding.
- Values from the spec: batch cache of **8** slots with LRU eviction, byte budget **64 KiB** (tunable in Task 6); bounded drain of **256** blocks on the slow path and on a flush, remainder kept owner-local; `maintain()` and parking drain everything.
- Orderings: push compare-and-swap (success), state store, state load after push, and the inbox swap are all `SeqCst` (spec: the store-load handshake).
- Repo rule: no commits, pushes or publishes unless the user asks; work on local branch `remote-frees`; each task ends with a green, uncommitted checkpoint.
- Performance targets: (1) `LocalHeap` within noise of 0.4.0; (2) `ShardAlloc` on one thread still ties or beats mimalloc on the four existing benchmarks; (3) `ShardAlloc` beats mimalloc on producer/consumer and many-to-one, 64 B to 64 KiB. Measure with `RUSTFLAGS='-C llvm-args=-align-loops=64'` and interleaved runs (DESIGN.md measurement caveat). If (3) cannot be met, stop and report with evidence.

## Review Focus

1. A remote free racing the owner's thread exit (push while the owner parks): the batch is freed exactly once, by the owner or by the sender under the pool lock. Real-thread test in Task 4.
2. A block sent to another thread and grown or shrunk there: the new block comes from the receiving thread's heap, contents are kept, the old block goes home. Test in Task 3.
3. A thread that exits with batches still in its cache: they reach their owners at exit. Test in Task 4.
4. Zero-sized blocks and large blocks (alignment 64 KiB) freed on another thread. Test in Task 3.
5. An owner that never allocates again after receiving remote frees: `ShardAlloc::maintain()` on it drains them, and its byte count returns to zero. Test in Task 3.

---

### Task 1: The concurrency core, `src/remote.rs`

**Files:**
- Create: `src/remote.rs`
- Modify: `src/lib.rs` (`mod remote;`; gate `mod shard;` and `pub use shard::ShardAlloc;` behind `#[cfg(not(loom))]`), `Cargo.toml` (loom dev-dependency, `unexpected_cfgs` lint), `.github/workflows/ci.yml` (Loom job)

**Interfaces:**
- Consumes: nothing.
- Produces (all `pub(crate)`):
  - `type Msg = *mut u8;` — an opaque message; its link word is reached through `L: Link`.
  - `trait Link { unsafe fn next(m: Msg) -> Msg; unsafe fn set_next(m: Msg, next: Msg); }`
  - `struct Remote { inbox: AtomicPtr<u8>, state: AtomicU8, pool_next: UnsafeCell<*mut Remote> }` with `const fn new() -> Remote` (state `ACTIVE`), consts `ACTIVE: u8 = 0`, `PARKED: u8 = 1`.
  - `unsafe fn Remote::push<L: Link>(&self, head: Msg, tail: Msg) -> bool`: CAS loop linking `tail` to the old head (`SeqCst` success), then returns `self.state.load(SeqCst) == PARKED` (the caller must then lock the pool and drain if still parked).
  - `fn Remote::take_all(&self) -> Msg` (swap with null, `SeqCst`).
  - `fn Remote::is_parked(&self) -> bool`.
  - `struct Pool { lock: Mutex<PoolList> }`, `struct PoolList { head: *mut Remote }` (`unsafe impl Send`), `const fn Pool::new()` (std only), `fn Pool::lock(&self) -> PoolGuard<'_>`; on the guard: `unsafe fn park(&mut self, r: *mut Remote)` (store `PARKED` `SeqCst`, link in), `fn pop_where(&mut self, f: impl FnMut(*mut Remote) -> bool) -> Option<*mut Remote>` (unlink, store `ACTIVE`), `fn contains(&self, r: *mut Remote) -> bool`.
  - `trait Transport { unsafe fn send<L: Link>(dest: &Remote, head: Msg, tail: Msg) -> bool; fn receive(dest: &Remote) -> Msg; }` and `struct StackTransport;` implementing it with `push`/`take_all` (spec: internal trait, one implementation).
  - Under `cfg(loom)`, atomics, `UnsafeCell` and `Mutex` come from `loom`; otherwise from `core`/`std`.

- [ ] **Step 1: Create branch; write failing unit tests in `src/remote.rs` (std threads, also run under Miri)**

Test helper: `struct Node { next: *mut u8, id: usize }` boxed; `impl Link for NodeLink` reads/writes `next`.

```rust
#[test]
fn senders_batches_all_arrive_once() {
    // 3 threads each push 2 batches of 3 nodes; owner takes all after join.
    // Collect ids: exactly 0..18, each once.
}

#[test]
fn push_reports_parked() {
    let r = Remote::new();
    assert!(!unsafe { r.push::<NodeLink>(n1, n1) });
    let pool = Pool::new();
    unsafe { pool.lock().park(&r as *const _ as *mut _) };
    assert!(unsafe { r.push::<NodeLink>(n2, n2) });
    assert!(pool.lock().pop_where(|_| true).is_some());
    assert!(!r.is_parked());
}
```

Loom tests, `#[cfg(all(test, loom))] mod loom_tests`:

```rust
#[test] fn loom_two_senders_one_drain()    // both batches seen exactly once by take_all after joins
#[test] fn loom_park_versus_push()          // owner: lock; park; take_all. sender: push; if parked { lock; if contains(r) { take_all } }.
                                            // every node drained exactly once, none lost (count == pushed)
#[test] fn loom_adopt_versus_push()         // r parked; adopter: lock; pop_where; then take_all. sender: push; if parked { lock; if contains { take_all } }.
                                            // every node drained exactly once
```

- [ ] **Step 2: Run to verify they fail**

Run: `git checkout -b remote-frees && cargo test --lib remote`
Expected: compile error (module missing).

- [ ] **Step 3: Implement `src/remote.rs` per Interfaces; add loom dev-dependency, `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(loom)'] }`, and the CI job (`RUSTFLAGS='--cfg loom' cargo test --release --lib remote`, `LOOM_MAX_PREEMPTIONS=3`)**

- [ ] **Step 4: Run tests natively, under Miri, and under Loom**

Run: `cargo test --lib remote && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test --lib remote && RUSTFLAGS='--cfg loom' CARGO_TARGET_DIR=target-loom cargo test --release --lib remote`
Expected: all pass; Loom explores without failure. Checkpoint, uncommitted.

---

### Task 2: Freeing a message on its owner, `src/heap.rs`

**Files:**
- Modify: `src/heap.rs`
- Test: `src/heap.rs` `mod tests`

**Interfaces:**
- Consumes: `remote::Msg`.
- Produces (`pub(crate)`):
  - `LargeHeader` gains `remote_next: *mut u8` (the link word of a large message).
  - `const LARGE_MSG: usize = 1;` tag bit; `unsafe fn small_msg(p: *mut u8) -> Msg` (`p` itself; link = the block's first word) and `unsafe fn large_msg(p: *mut u8, align: usize) -> Msg` (header address `| LARGE_MSG`).
  - `struct MsgLink; impl remote::Link for MsgLink` (first word of a small block, `remote_next` of a large header).
  - `unsafe fn Inner::free_msg(&mut self, m: Msg)`: small: page from the segment, freed exactly like `free_small` (class from the page; `local_free` push, `used`, stats, slow transitions); large: `free_large_header(h)` (refactor `free_large(p, layout)` into finding the header plus `free_large_header`, size and alignment from the header).
  - `unsafe fn Inner::free_msgs(&mut self, list: Msg, limit: usize) -> Msg`: frees up to `limit` messages along the list, returns the rest (null when done).

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn messages_free_like_local_frees() {
    let heap = LocalHeap::new();
    // 300 small blocks of mixed classes and 2 large ones (align 8 and 1 << 16),
    // each turned into a message and linked into one list through MsgLink.
    // free_msgs(list, 256) returns a non-null rest; free_msgs(rest, 256) returns null.
    // stats().allocated_bytes == 0 and large_allocations == 0 afterwards.
}

#[test]
fn freed_message_blocks_are_reused() {
    // free one 64 B block via free_msg, then allocate 64 B: same address.
}
```

- [ ] **Step 2: Run to verify they fail** — `cargo test --lib heap::tests::messages` → compile error.

- [ ] **Step 3: Implement per Interfaces.** `free_small` and `free_msg` share one body that takes the page (no class argument; `free_small` keeps its debug assert on the class).

- [ ] **Step 4: Run** `cargo test && cargo clippy --all-targets -- -D warnings`, then the `LocalHeap` checkpoint: `RUSTFLAGS='-C llvm-args=-align-loops=64' CARGO_TARGET_DIR=target-bench-aligned cargo bench --bench alloc -- '/shardheap$'` interleaved against a 0.4.0 build (`git worktree` of `v0.4.0`) — Expected: within noise. Checkpoint.

---

### Task 3: Remote frees through `ShardAlloc`, and `Send + Sync`

**Files:**
- Modify: `src/shard.rs`, `src/heap.rs` (`dealloc_owned`/`resize_owned` foreign branch)
- Test: `src/shard.rs` `mod tests`, `tests/cross.rs` (new integration test file)

**Interfaces:**
- Consumes: Task 1 (`Remote`, `StackTransport`, `Link`), Task 2 (`small_msg`, `large_msg`, `MsgLink`, `free_msg`, `free_msgs`).
- Produces:
  - `Record` gains, after `inner`: `remote: Remote`, `batch: BatchCache`, `remainder: Msg` (owner-local, undrained messages).
  - `struct BatchCache { slots: [Slot; 8], bytes: usize, clock: u64 }`, `Slot { dest: *mut Record, head: Msg, tail: Msg, bytes: usize, used: u64 }`; `fn add(&mut self, dest: *mut Record, m: Msg, bytes: usize)` (LRU slot; evicting flushes that slot), `fn flush_all(&mut self)`; budget `BATCH_BUDGET: usize = 64 << 10` (flush all when exceeded). Flushing a slot: `StackTransport::send`, and if it reports parked, `drain_parked(dest)` (Task 4 fills in; Task 3 stub: debug-assert not parked).
  - `Inner::dealloc_owned(this, p, layout, foreign: impl FnOnce(usize))`: the mismatch branch calls `foreign(owner)` instead of aborting; `resize_owned` likewise returns through `foreign` handling in shard.rs (owner mismatch: allocate from `this`'s heap, copy `min(old, new)` bytes, remote-free the old block).
  - `#[cold] fn remote_free(owner: usize, p: *mut u8, layout: Layout)`: owner 0 (explicit heap) aborts via `foreign_free`; if the thread is on the empty heap, makes a fresh empty record first (spec: never adopt one with outstanding blocks); adds the message to the batch cache under `owner as *mut Record`; when the cache flushes, drains its own inbox, bounded.
  - Draining: `fn drain(r: *mut Record, limit: usize)` (remainder first, then `take_all`), called bounded (256) from `allocate_slow` and after a flush, fully from `ShardAlloc::maintain()` (which also flushes the batch cache first).
  - `ShardAlloc` drops its `PhantomData<*mut ()>`: `Send + Sync` (doc test flips to `assert_send`/`assert_sync` compiling).

- [ ] **Step 1: Failing tests** (`tests/cross.rs`)

```rust
#[test] fn blocks_freed_on_another_thread_go_home()
    // owner thread allocates 10_000 Box<[u8; 64], ShardAlloc> (tagged), sends them over a channel;
    // consumer checks tags and drops them; owner then calls ShardAlloc::maintain()
    // and asserts ShardAlloc::stats().allocated_bytes == 0.
#[test] fn many_senders_one_owner()
    // owner allocates 4 x 2_000 mixed-size Vec<u8, ShardAlloc>, hands 2_000 to each of 4 threads that drop them;
    // after joins, owner maintain() -> allocated_bytes == 0.
#[test] fn sent_vec_grows_and_shrinks_on_the_receiver()
    // Vec of 1000 bytes sent; receiver extends to 300_000 (large), checks prefix, shrinks to 10, checks,
    // drops; owner maintain() -> 0; receiver's own stats return to 0 after its drop.
#[test] fn zero_sized_and_large_aligned_blocks_freed_remotely()
    // ZST Box and a 300_000-byte block with align 1 << 16, freed on another thread; owner maintain() -> 0.
#[test] fn owner_that_never_allocates_again_drains_on_maintain()
    // (Review Focus 5) owner allocates, sends all away, waits for the frees, never allocates;
    // stats show bytes still allocated before maintain(), 0 after.
```

And a positive doc test on `ShardAlloc`: `fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<Vec<u8, ShardAlloc>>();`.

- [ ] **Step 2: Run to verify they fail** — `cargo test --test cross` → compile error (`Vec<_, ShardAlloc>` not `Send`).

- [ ] **Step 3: Implement per Interfaces.** Keep `ShardAlloc::deallocate`'s local path exactly as fast as 0.4.0: the foreign closure must compile to a cold out-of-line call.

- [ ] **Step 4: Run** `cargo test && cargo clippy --all-targets -- -D warnings && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test --test cross` (shrink counts under `cfg!(miri)`). Expected: all pass, no data race. Then the target 2 checkpoint: `lifo_64B`, `batch_mixed`, `vec_growth_8MiB`, `tree_32k_nodes` for `shard_alloc` interleaved with `mimalloc` — Expected: ties or beats on each. Checkpoint.

---

### Task 4: Parking, adoption and teardown

**Files:**
- Modify: `src/shard.rs`, `src/heap.rs` (remove stage A's `exited` and `trim_if_exited`)
- Test: `src/shard.rs` `mod tests`, `tests/cross.rs`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces:
  - `static POOL: Pool`.
  - `ExitGuard::drop`: flush the batch cache (outside the lock), then under the pool lock: drain the remainder, `park`, drain the inbox fully, `trim`, and if `allocated_bytes == 0` release everything; then set `CURRENT` to `TORN`.
  - `static TORN: Empty` (a second read-only empty heap): after teardown, allocation misses into `allocate_slow`, which under the pool lock allocates from a parked record (`pop_where(|_| true)`, or a new one), then parks it again; a free on `TORN` goes to `remote_free`, which sends a single-message batch to the owner without a batch cache (and drains under the lock if the owner is parked).
  - `fn drain_parked(dest: *mut Record)`: lock; if `contains(dest)`: drain fully, `maintain()` its heap, release everything if `allocated_bytes == 0`.
  - Adoption in `init_current` (first use is an allocation): `pop_where(|_| true)`, else `new_record()`. First use is a free: `pop_where(|r| allocated_bytes(r) == 0)`, else `new_record()`.

- [ ] **Step 1: Failing tests**

```rust
// src/shard.rs
#[test] fn exited_thread_heap_is_parked_and_adopted()
    // thread A allocates and frees (record rA), exits; rA in POOL and parked;
    // thread B allocates: CURRENT == rA (adopted), rA active, not in POOL.
#[test] fn remote_frees_into_a_parked_heap_release_it()
    // thread A allocates 1 MiB + 1_000 small blocks, sends them to B, exits (rA parked, bytes > 0);
    // B drops them all and calls maintain() (flushes); then rA.allocated_bytes == 0 and rA.mapped_bytes == 0.
#[test] fn exiting_thread_flushes_its_batches()
    // (Review Focus 3) owner O allocates 500 blocks, sends to T; T drops them (batched, under budget) and exits;
    // O maintain() -> allocated_bytes == 0.
#[test] fn remote_free_racing_owner_exit_frees_once()
    // (Review Focus 1) 20 rounds: owner allocates 1_000 blocks, sends to 3 freeing threads, exits at once
    // while they free; after all joins and a maintain() from a fresh thread that adopts nothing:
    // every round's record ends at allocated_bytes == 0 (no lost or double frees; tags checked on drop).

// tests/model.rs — existing teardown tests keep passing:
//   shard_alloc_survives_thread_teardown, frees_after_the_exit_guard_still_release_an_emptied_heap (now via the pool).
```

- [ ] **Step 2: Run to verify they fail** — `cargo test --lib shard && cargo test --test cross` → parking tests fail (records not in a pool).

- [ ] **Step 3: Implement per Interfaces;** delete `Inner::exited` and `trim_if_exited` (the parked-heap path replaces them).

- [ ] **Step 4: Run** `cargo test && cargo clippy --all-targets -- -D warnings`, then full Miri and Loom:
`MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test && RUSTFLAGS='--cfg loom' CARGO_TARGET_DIR=target-loom cargo test --release --lib remote`
Expected: all pass, no data race. Checkpoint.

---

### Task 5: Cross-thread model test

**Files:**
- Test: `tests/cross.rs`

**Interfaces:**
- Consumes: `ShardAlloc` (`Send + Sync`).
- Produces: nothing.

- [ ] **Step 1: Write the test**

```rust
#[test]
fn cross_thread_model() {
    // 4 worker threads in a ring, each with an mpsc receiver from its left neighbour.
    // Each runs OPS (20_000; 500 under Miri) random steps: allocate (sizes 0..=300_000,
    // alignments 1..=65_536) a tagged Vec<u8, ShardAlloc>; grow/shrink a held one (check tags);
    // send a held one to the right; drop a held one (check tag); keep received ones.
    // At the end each thread drops what it holds, joins, and then every thread's record
    // (collected through a test-only accessor) reaches allocated_bytes == 0 after one maintain
    // pass per live thread and drains of parked records.
}
```

- [ ] **Step 2: Run it natively and under Miri**

Run: `cargo test --release --test cross cross_thread_model && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test --test cross cross_thread_model`
Expected: pass. A failure here is a bug in Tasks 2-4: debug it with superpowers:systematic-debugging, add a focused regression test to the owning task's tests, then fix. Checkpoint.

---

### Task 6: Cross-thread benchmarks, targets, docs

**Files:**
- Create: `benches/cross.rs` (criterion, `harness = false`, add `[[bench]]` to `Cargo.toml`)
- Modify: `README.md` (`ShardAlloc` across threads; results table), `DESIGN.md` (stage B section: messages, inbox, batch cache, draining, pool, teardown, results; Roadmap: stage B done), `src/lib.rs` docs, `src/shard.rs` docs
- Possibly modify: `src/shard.rs` (`BATCH_BUDGET`, slot count) when tuning

**Interfaces:**
- Consumes: `ShardAlloc`.

- [ ] **Step 1: Write the benchmarks**

Groups `producer_consumer/<size>` and `many_to_one/<size>` for sizes 64, 1024, 16384, 65536 bytes; allocators `system`, `mimalloc` (as in `benches/alloc.rs`), `shard_alloc`. `iter_custom`: producer thread allocates 100_000 filled `Vec<u8, A>` and sends them over `std::sync::mpsc::sync_channel(1024)` to a consumer that drops them (`many_to_one`: 4 producers, 25_000 each, one consumer that drops; the producers are the owners). Time from first allocation to the consumer's last drop, threads spawned outside the timed region and released by a barrier.

- [ ] **Step 2: Measure**

Run: `RUSTFLAGS='-C llvm-args=-align-loops=64' CARGO_TARGET_DIR=target-bench-aligned cargo bench --bench cross`, then three interleaved rounds per group with the bench binary (as in stage A).
Expected (target 3): `shard_alloc` median below `mimalloc` in every group.

- [ ] **Step 3: If target 3 misses, tune and re-measure:** batch budget (16 KiB to 256 KiB) and slot count first, then drain limit; then profile (`perf stat`: `cycles`, `instructions`, `ls_bad_status2.stli_other`, cache misses). The target does not move; if it cannot be met, stop and report with the evidence.

- [ ] **Step 4: Re-check targets 1 and 2** (interleaved against a `v0.4.0` worktree build and mimalloc). Expected: unchanged.

- [ ] **Step 5: Docs** with the measured numbers; roadmap marks stage B done only if target 3 holds.

- [ ] **Step 6: Final verification**

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings \
  && RUSTDOCFLAGS='-D warnings' cargo doc --no-deps && cargo test && cargo +1.100 test \
  && MIRIFLAGS=-Zmiri-strict-provenance CARGO_TARGET_DIR=target-miri cargo +nightly miri test \
  && RUSTFLAGS='--cfg loom' CARGO_TARGET_DIR=target-loom cargo test --release --lib remote
```
Expected: all clean. Leave uncommitted on `remote-frees`; report targets per group and the proposed version (0.5.0).
