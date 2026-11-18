# shardheap design

A mimalloc-style heap for Rust's stable `Allocator` trait (Rust 1.100), built
shared-nothing first. Phase 1 is a single-threaded heap whose handle cannot leave
its thread.

## Layout

- **Segment:** 4 MiB, aligned to 4 MiB, from `mmap`. Slot 0 holds the header:
  segment links, a 64-bit slot bitmap, and one page descriptor per slot. The
  header of any block is `addr & !(4 MiB - 1)`.
- **Slot:** 64 KiB. A page spans 1 to 16 contiguous slots, found first-fit in the
  bitmap. Every slot descriptor stores `head_offset`, the distance back to its
  span's first slot.
- **Page:** blocks of one size class. Holds `free`, `local_free`, a bump index
  over never-used blocks, a `used` count, and queue links.
- **Size classes:** 16 to 128 bytes in steps of 16, then 8 classes per power of
  two up to 128 KiB (88 classes, at most 12.5% apart). Pages spanning more than one
  slot hold at least 8 blocks.
- **Large:** requests above 128 KiB, or aligned above 64 KiB, get their own
  mapping with a one-page header that links all large mappings (for `reset` and
  `Drop`). Freed page-aligned mappings go into a small cache (8 entries, 64 MiB)
  and are reused, best fit within 2x, by later large allocations and by growth.
  Their pages stay resident, so reuse costs no page faults. Without a cached fit,
  growing uses `mremap`, which moves the pages without copying.

## Arena

`Arena` is a separate type rather than a mode of `LocalHeap`, so `LocalHeap`'s
fast path carries no extra branch.

- **Chunks:** 64 KiB, doubling to 4 MiB; a bigger request gets a chunk of its own
  size. A chunk header links all chunks for `reset` and `Drop`.
- **Chunk source:** any `Allocator`. The default, `Os`, maps each chunk;
  `Arena::new_in(&heap)` takes chunks from a `LocalHeap` instead (64 and 128 KiB
  from size classes, bigger ones from its large path and large-mapping cache). The
  arena then borrows the heap, chunks freed by `reset` stay resident in the heap
  for its other allocations, and `heap.stats()` counts everything. Only the slow
  path changes.
- **Bumps upward.** That costs one alignment computation more than bumpalo's
  downward bump, but it lets the most recent block grow in place without copying.
  bumpalo has to `memmove` it.
- **No "last block" field.** Free and resize receive the `Layout`, so
  `p + size == ptr` identifies the most recent block. A block in an older chunk
  can never end at the current chunk's `ptr`, which lies past that chunk's header.
- **No counter on the fast path:** bytes in use are derived from the bump
  pointers, plus the bytes used in chunks that are no longer current.
- **Reset:** if the round fit in one chunk, it is kept with its pages resident. If
  it spilled over several, they are replaced by a single chunk as large as all of
  them together, so the next round of the same shape needs no system call and
  takes no page faults.

What the benchmarks showed along the way:
- Capping chunks at 4 MiB and keeping only the largest chunk on reset made
  every round of a 6.8 MB batch map fresh chunks: 3x slower than bumpalo.
  Coalescing on reset fixed it.
- A `last` field and a whole-struct counter update per operation made the
  alloc/free loop 40% slower than bumpalo. Removing both made it a tie.

## Reap

`heap.reap()` returns a `Reap`, a `Copy` handle that borrows only the heap.
Berger, Zorn and McKinley's "reap" (OOPSLA 2002): regions that allow frees.

- **Pages:** single 64 KiB slots claimed from the heap's segments, marked with
  class `REAP_CLASS` and kept out of every class queue. The heap holds one
  current bump page.
- **Allocate:** bump in the current page and increment its `used`. A request
  above 8 KiB or aligned above 4 KiB goes to the heap's size classes; the choice
  depends only on the layout, so free takes the same path.
- **Free:** find the page by masking the address, decrement `used`. If it reaches
  zero, the page goes back to its segment, or, for the current page, the bump
  pointer rewinds to its start. Freeing or shrinking the most recent block gives
  its bytes back, so a `Vec` grows in place.
- **No reset, no scope.** Blocks live as long as the heap. That is the point over
  an arena (survivors need no copy) and the cost: one live block keeps 64 KiB
  resident.

What `examples/request_mix.rs` showed (see the README table): with no survivors
the reap is 18% faster than `LocalHeap`, but each live survivor pins a page, so
memory grows with survival rate times survivor lifetime: 177 MiB against 15 MiB
at 5% over 512 requests, and slower, because the bump pointer runs through cold
memory. An `Arena` on the same heap, with survivors copied into the heap, was the
fastest at every setting, with about the heap's footprint. Smaller reap pages would
reduce the pinning; they would need sub-slot page descriptors.

## Fast paths

- **Allocate:** `class_for(layout)` (a shift and a table lookup), then pop the
  `free` list of the first page in the class queue. If `free` is empty but
  `local_free` is not, they are swapped inline; without that, an allocate/free loop
  on one block took the slow path every time and ran 1.6x slower than mimalloc.
- **Free:** compute the class from the `Layout`, mask the address to find the
  segment, read `head_offset`, push onto the page's `local_free`. No owner check:
  the heap is `!Send`, so the free provably happens on the owning thread.
- **Slow path** (`free` empty): swap `local_free` into `free`; else bump; else
  unlink the full page and try the next; else make a page. Each slow-path call
  advances a tick that drives housekeeping.

## Memory return

- An empty page goes straight back to its segment, unless it is its class's only
  page. That page is kept for `RETIRE_DELAY` ticks so a workload crossing a page
  boundary does not thrash.
- Cached large mappings idle for `PURGE_DELAY` ticks are unmapped.
- An empty segment is cached (at most `MAX_CACHED_SEGMENTS`). After
  `PURGE_DELAY` ticks its slots are `madvise(MADV_DONTNEED)`ed; the mapping and
  header stay. Further empty segments are unmapped.
- `trim()` releases everything idle now; `reset(&mut self)` frees all blocks at
  once (an arena reset) and keeps a few segments mapped for reuse.

## What Rust allows that C does not

- **The size comes with every free.** The class, and the small/large decision,
  come from the `Layout` rather than a metadata read. The allocator contract
  allows any size between the requested size and the returned length on free,
  and every such size maps to the same class.
- **No thread-local lookup:** the heap is passed as the allocator value.
- **No owner check on free:** `!Send` enforced by the compiler.
- **Lifetimes replace runtime checks:** `Vec<T, &LocalHeap>` borrows the heap, so
  the heap cannot be dropped or reset while blocks are live.

## Unsafe surface and checking

All raw-memory logic is in `heap.rs`, `segment.rs` and `os.rs`. Free-list links
are stored as pointers, so provenance travels with them, and the segment mask
uses `map_addr`. That is compatible with strict provenance. Under Miri, `os.rs`
maps memory with the global allocator, so Miri checks everything above the
syscalls:

```
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test
```

`tests/model.rs` runs random allocate, free, grow and shrink sequences across
every size range and alignment. It fills each block with a tag, checks the tag on
every free and resize, asserts live blocks are pairwise disjoint, and checks the
byte accounting returns to zero.

## Roadmap

1. **Phase 1 (this crate):** `LocalHeap`, `Arena` (on the OS or on a heap),
   `Reap` and `StackArena`.
2. **Phase 2:** optional glommio integration: a heap per executor, byte
   accounting per task queue feeding the scheduler, allocation sampling on the
   slow path.
3. **Phase 3:** remote frees. `Page::_thread_free` and `Segment::_owner` are
   reserved. The candidate design is snmalloc-style message batching to the
   owner, rather than mimalloc's per-free atomic. This gives a `Send` heap handle,
   checked with Loom.
4. **Phase 4:** `GlobalAlloc`: thread-local lookup, abandoned segments of exited
   threads, re-entrancy, `fork`.

## Known limits (phase 1)

- Free slots inside a segment that is still partly in use are never purged;
  only whole empty segments are.
- Housekeeping is driven by slow-path calls, so a heap that stops allocating keeps
  its cached memory until `trim()`.
- Linux only for real memory (`mmap`); elsewhere `os.rs` uses the global allocator.
