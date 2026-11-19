# Phase 3: remote frees and a `Send` heap handle

Status: approved approaches; design for planning.
Crate: shardheap 0.3.0 → next minor (stage A), then the one after (stage B).

## Stages

Built and released in two stages; stage B starts only after stage A meets its
targets.

- **Stage A: thread heaps and the handle, on one thread.** Thread records, the
  thread-local lookup, the zero-sized handle, owner fields in segments and large
  headers, the owner check on free. The handle is **not** `Send` in stage A: a
  `Send` handle without remote frees would let safe code free a block on the
  wrong thread. This stage carries the same-thread costs a global allocator
  pays (lookup on allocate, owner check on free), so it is where shardheap
  meets mimalloc on mimalloc's own terms.
- **Stage B: across threads.** Batch cache, inbox, draining, the pool and
  parking, the transport trait, Loom; the handle becomes `Send + Sync`.

## Performance targets

1. **Explicit `LocalHeap`:** within noise of 0.3.0 on every existing benchmark
   (stage A and B).
2. **Handle on one thread:** match or beat mimalloc's `malloc`/`free` on every
   existing benchmark (`lifo_64B`, `batch_mixed`, `vec_growth_8MiB`,
   `tree_32k_nodes`), which do the same work: a thread-local lookup to
   allocate and an owner check to free. Stage A is done when this holds. If it
   does not, the lookup or the check gets a faster path; the target does not
   move.
3. **Handle across threads** (stage B, set when stage B starts): beat mimalloc
   on producer/consumer and many-to-one, 64 B to 64 KiB.

## Intent

Let a block allocated by one thread be freed on another, so that buffers can
move between shards (glommio executors handing buffers to each other), and lay
the groundwork for phase 4 (`#[global_allocator]`). The owner's own allocation
and free paths must not pay for this: the point of shardheap is that a shard's
memory is touched by its shard without atomics.

The model is snmalloc's message passing: a thread that frees someone else's
block batches it locally and sends whole batches to the owner, one atomic
operation per batch. Not mimalloc's compare-and-swap per free, and not
Seastar's allocator-level atomic push per free.

## Goals

- A zero-sized allocator handle (`Copy`; in stage B also `Send + Sync`), so
  `Vec<T, Handle>` / `Box<T, Handle>` are `Send` when `T` is, and can be grown
  and freed on any thread.
- Remote frees cost about as much as a local free, plus a share of one atomic
  operation per batch.
- Memory of exited threads is never stuck forever: it is reused by threads
  that come later, and remote frees into it complete without waiting for one.
- Concurrency verified with Loom; whole-crate behaviour verified with Miri
  running real threads (it detects data races).
- A cross-thread benchmark (producer allocates, consumer frees) against
  mimalloc and glibc.

## Non-goals (phase 4 or later)

- `GlobalAlloc` and `#[global_allocator]`; allocation before the thread-local
  heap exists; re-entrancy; `fork`.
- Remote frees into explicit `LocalHeap`s, arenas, reaps, stack arenas or
  region heaps. They stay local-only, and their API does not change.
- Configuring thread heaps (huge-page policy, regions): they use the defaults.
- Seastar-style per-pair queues. They belong to the glommio integration
  (phase 2), behind the transport trait below.

## Anti-goals (failure even if it works)

- Any atomic operation, or any extra branch, on the free and allocate fast
  paths of an explicit `LocalHeap`. Its benchmarks must stay within noise.
- A global lock on any common path. The pool lock is taken only when a thread
  starts or exits, and when a remote free targets a parked heap.
- Unbounded memory held in a freeing thread's batch cache.

## Decided approaches

Each was chosen over the listed alternative.

1. **Thread exit: park the heap in a global pool** (snmalloc). Rejected:
   mimalloc-style abandoned segments adopted one by one (finer reclamation,
   much more concurrent code to verify); leaking outstanding blocks (wrong for
   glommio, where executors come and go).
2. **Batch flushing: byte budget plus housekeeping.** A batch cache keyed by
   owning heap, flushed when it passes a byte budget (start at 64 KiB, tune with
   the benchmark), on the allocation slow path, in `maintain()`, and at thread
   exit. Rejected: sending every free at once (one atomic per free).
3. **Inbox: a lock-free stack of batches.** Pushing a batch is one
   compare-and-swap of the stack head; the owner takes the whole stack with one
   swap. Rejected: per-pair single-producer queues (n² queues, a registry of all
   threads); that shape is left to the glommio integration.
4. **Handle: zero-sized, current-thread heap** (the allocator value a
   collection carries, `Vec<T, ShardAlloc>`; phase 4's `GlobalAlloc` is this
   handle in a `static`). It allocates from the current
   thread's heap and frees by finding the owner from the block. Rejected: a
   handle pinned to one heap through an `Arc` (more state, and growth on a
   foreign thread cannot use that heap anyway).
5. **Scope:** thread heaps are new; everything that exists stays local-only.

## Design

### Thread heaps

A *thread heap* is a heap record with a stable address for the life of the
process: a leaked allocation holding

- the heap state (`Inner`, as in `LocalHeap` today) behind an `UnsafeCell`,
- the inbox (an atomic pointer, the head of the stack of batches),
- an atomic state: `Active` (owned by a thread) or `Parked` (in the pool),
- the batch cache this heap uses for frees it sends to *other* heaps.

Records are never freed, so a block's owner pointer is always valid: no
reference counting, no use-after-free on a late remote free.

A thread-local holds a pointer to the current thread's record, created on
first use. Records are mapped through `os.rs`, not taken from the global
allocator, and the pool links them through a field of the record, not a `Vec`,
so phase 4 can reuse this layer without recursing into itself (decided).

- First use is an allocation: adopt a parked record from the pool if there is
  one, else map a new one.
- First use is a free: map a fresh, empty record (or take an empty parked one).
  A thread that only frees never runs the allocation slow path, so it must not
  adopt a record with outstanding blocks that would then never drain
  (decided).

The thread-local's destructor parks the record (below).

### Finding the owner

- Small blocks: the segment header's reserved `_owner` field becomes the owning
  record's address (null for segments of explicit `LocalHeap`s, which never see
  remote frees). Found by masking the block address, as today.
- Large blocks: `LargeHeader` gains an owner field. The handle's `deallocate`
  gets the `Layout`, so it knows a block is large and where its header is
  (`p - large_offset(align)`), exactly as `free_large` does today.

### The handle

In stage A the handle is `!Send` and `!Sync`, so a block is always freed on
the thread that allocated it, and the owner check always finds the current
record. The check still runs (its cost is what target 2 measures); its foreign
branch is a cold path that aborts with a message in stage A and becomes the
batching below in stage B.

`deallocate(p, layout)` on thread T:

1. Look up T's record through the thread-local.
2. Read the owner (segment header or large header).
3. Owner is T's record: free locally, through the existing paths.
4. Otherwise: put the block in T's batch cache under that owner; flush if the
   cache passes its byte budget.

`allocate` uses T's record like an explicit `LocalHeap`. `grow` and `shrink`
on a foreign block allocate from T's heap, copy, and remote-free the old one.
They never resize in place on another thread's heap.

### Messages

A message is one freed block. The first word of a freed block links it into a
list, so batches need no allocation. A batch is a singly linked list with head,
tail and byte count, kept in a slot of the batch cache. A large block is sent
as its header address with the low bit set, so the owner can tell the kinds
apart; small blocks are at least word-aligned, so the bit is free.

Batch cache: a few slots (start with 8), each holding one destination. A free
for a destination without a slot evicts (flushes) the least recently used slot.
A total byte budget across slots bounds what the cache holds.

Flushing a batch: one compare-and-swap loop links the batch's tail to the
current inbox head and installs the batch's head (release ordering). Batches
from different senders interleave as whole lists.

### Draining

The owner swaps the inbox head with null (acquire), walks the lists, and frees
each block as a local free: small blocks into their page's `local_free` (the
page and class come from the segment, as in `free_small`; no `Layout` needed),
large blocks through the large path (size and alignment come from the header).
Byte accounting is updated as for local frees.

The owner drains on its allocation slow path (next to the heartbeat), when it
flushes its own batch cache, in `maintain()`, and before parking. The fast
paths do not look at the inbox.

Draining on the slow path and on a flush is bounded: at most a fixed number of
blocks (builder tunes it; start at 256), the rest kept in an owner-local
remainder list drained first next time, so one allocation never walks
megabytes of frees on a reactor thread. `maintain()` and parking drain
everything, remainder included (decided).

### Thread exit in stage A

Without the pool, a record whose thread exits releases its memory if it holds
no live blocks, and otherwise stays mapped (its blocks can only be freed by
thread-local destructors still running on that thread, which reach it through
the owner field). Stage B replaces this with parking.

### Parking and the pool (stage B)

The pool is a mutex-protected list of parked records.

At thread exit the owner flushes its own batch cache, then, under the pool
lock:

1. drains its remainder list,
2. sets its state to `Parked`,
3. drains its inbox,
4. trims (empty segments and free slots go back to the OS),
5. puts the record in the pool.

A sender that flushes a batch to a record checks the record's state *after*
pushing. If it is `Parked`, the sender takes the pool lock and, if the record
is still parked, drains it fully, acting as its owner under the lock, then runs
the record's `maintain()` housekeeping (counting as a heartbeat, so memory free
for a whole purge epoch goes back to the OS). When a parked record's
`allocated_bytes` reaches zero, the drainer releases all its memory (segments
and large cache unmapped; the record itself stays). So a remote free into a
parked heap completes without waiting for an adopter, and an exited thread's
memory returns as its blocks are freed (decided). System calls under the pool
lock happen only on this path.

The owner flushes its batch cache *before* taking the pool lock at exit, so a
flush that finds another parked record can take the lock itself.

The push-then-check (sender) and set-then-drain (owner) pair is a store-load
handshake: either the owner's drain sees the batch, or the sender sees
`Parked`. Both pairs need sequentially consistent ordering. This is the core
property the Loom tests must cover.

Adoption, under the pool lock: take a record, set `Active`, register it in the
new thread's thread-local; its next slow path drains whatever arrived.

### After thread-local teardown

Other thread-locals' destructors may allocate or free through the handle after
this thread's record has been parked. When the thread-local is gone,
`allocate` and `deallocate` take the pool lock and act on a parked record (for
a free, the block's owner if parked, else push a single-block batch to the
active owner). This is slow but rare, and correct.

### Transport trait

Flushing and draining go through a small internal trait (push a batch to a
record; take all batches of a record), so the glommio integration can later
carry batches over its own channels between executors. Only the default (the
lock-free stack) is implemented in this phase; the trait is not public yet.

### Public API

- `ShardAlloc`: the zero-sized handle. Implements `Allocator`, `Copy`,
  `Default`; `Send + Sync` from stage B.
- Associated functions on it for the current thread's record: `stats()`,
  `maintain()` (also flushes the batch cache and drains the inbox), `trim()`.
- `LocalHeap`, `Arena`, `Reap`, `StackArena`, `Os`: unchanged.

### Module layout

Concurrency lives in one small module (`remote.rs`: inbox, state, pool,
handshake) written against `loom`-swappable atomics and mutex
(`#[cfg(loom)]`). The heap-side parts (batch cache, drain-as-local-free, owner
fields) stay in `heap.rs`/`segment.rs` and are single-threaded code.

## Checking

Stage A: unit and model tests through the handle on one thread (including
blocks freed by other thread-locals' destructors during teardown), Miri, and
the handle as a new column in `cargo bench`. Stage B adds the rest:

- **Loom** (`RUSTFLAGS="--cfg loom"`): push/drain with 2-3 senders; the
  park-versus-push handshake (no lost batch, no double drain); adoption racing
  a remote free into a parked record.
- **Miri** with real threads: producer/consumer handoff, many senders to one
  owner, a thread that exits with blocks still live elsewhere, then a new thread
  adopting its record; strict provenance as today.
- **Model test across threads:** random allocate/free/grow with blocks handed
  between threads over channels, checking tags and that byte accounting
  returns to zero once every thread has flushed and drained.
- **Benchmarks:** existing `cargo bench` for `LocalHeap` must stay within noise
  (anti-goal). New: producer/consumer (one thread allocates N-byte buffers,
  another frees them) and many-to-one, against mimalloc and glibc, sizes 64 B
  to 64 KiB.
- CI: add a Loom job.

## Left to the builder

- Module names; the batch cache's slot count and byte budget (tuned
  with the benchmark).
- How the thread-local is declared (const-initialized pointer plus a
  destructor guard is expected; avoid lazy-init cost on the fast path).
- Whether the owner also drains when an inbox byte counter passes a threshold
  (would need a counter updated per batch).
- Where the link word goes in a large block's message (the header has room).
- Exact memory orderings beyond the handshake, as long as Loom passes.

## Known limits accepted

- An owner that stops allocating and is not maintained drains nothing; its
  remote-freed memory stays in its inbox until it does. glommio will call
  `maintain()` from the reactor's idle path (phase 2).
- A parked record keeps its segments mapped (trimmed) until adopted; with
  fewer threads than before, its memory comes back only through remote frees
  draining into it and trim.
