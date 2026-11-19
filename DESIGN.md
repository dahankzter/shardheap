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

## Region-backed heaps

`LocalHeap::with_region(bytes, huge_pages)` maps one region up front (rounded up
to 4 MiB, 4 MiB-aligned) and takes all memory from it, falling back to the OS
only when it is full. Built for io_uring registered buffers: register
`heap.region()` once, then `heap.in_region(p)` decides between a fixed-buffer
and a regular operation. The same holds for an `Arena` or `Reap` on the heap.

- **Granules:** 64 KiB, tracked in a bitmap (one bit each, 2 KiB of bitmap per
  GiB). Segments take 64 granules at a 4 MiB-aligned position, large blocks a
  first-fit run. Alignment above 4 MiB goes to the OS.
- **Pinned memory is never discarded.** Region memory is not unmapped,
  `madvise(MADV_DONTNEED)`ed or moved with `mremap` while the heap lives:
  registered pages are pinned, and replacing them would leave the device writing
  to the old physical pages. Freed granules only return to the bitmap.
- **Growth without mremap:** a large block grows in place when the granules
  after it are free; otherwise it is copied.
- **Reused memory is not zero:** segment headers are zeroed on reuse, and
  `allocate_zeroed` zeroes reused large blocks.
- **Huge pages:** `Auto` and `Always` advise the whole region `MADV_HUGEPAGE`.
  A registered region is resident anyway, so they cost no memory, and the
  region is pinned as 2 MiB pages. (The effect on pinning time was not
  measured.)

In glommio (`registered-vec` prototype), a `Vec<u8, IoAlloc>` on such a heap,
built with `extend_from_slice` and written with `WRITE_FIXED`, took 62% less
CPU per 1 MiB O_DIRECT write than building a `Vec<u8>` and copying it into a
registered `DmaBuffer`. See glommio's `docs/investigations/registered-vec`.

## Huge pages

Segments are 4 MiB and 4 MiB-aligned, so each is two 2 MiB transparent huge
pages. `LocalHeap::with_huge_pages` picks the policy; `new()` uses `Auto`:

- **Auto:** while the heap is small, the kernel's policy applies (no advice).
  Once it maps 16 MiB, new segments get `MADV_HUGEPAGE`, existing ones are
  re-advised so `khugepaged` collapses them in the background, and large
  mappings of at least 4 MiB are always advised. It never refuses huge pages.
  `MADV_COLLAPSE` is avoided on purpose: it collapses synchronously and would
  stall a reactor thread.
- **Always** / **Never:** `MADV_HUGEPAGE` / `MADV_NOHUGEPAGE` on every mapping.
- **Kernel:** no advice at all.

Measured with `examples/thp.rs` (64-byte boxed nodes linked in one random
cycle; build time includes the page faults) on a Zen 5 Threadripper, kernel
THP policy `always`, best of 3, pinned to one core:

| nodes | build, Never | build, Auto | chase, Never | chase, Auto | RSS, Never | RSS, Auto |
|---|---|---|---|---|---|---|
| 1 MiB | 0.1 ms, 324 faults | 0.1 ms, 68 faults | 4.7 ns/hop | **2.6 ns/hop** | 3 MiB | 4 MiB |
| 16 MiB | 2.5 ms, 4110 faults | **1.1 ms**, 525 faults | 10.6 ns/hop | **8.7 ns/hop** | 23 MiB | 25 MiB |
| 256 MiB | 51.4 ms, 66213 faults | **24.8 ms**, 677 faults | 125.3 ns/hop | **113.9 ns/hop** | 339 MiB | 344 MiB |

The cost is resident memory in 2 MiB steps. In `examples/request_mix.rs`
(`LocalHeap`), RSS with huge pages vs `Never`: 4.0 vs 1.0 MiB with no survivors,
6.1 vs 2.4 MiB at 5% survivors over 64 requests, 14.6 vs 11.6 MiB over 512.
Time per request was unchanged; warm microbenchmarks (`cargo bench`) were also
unchanged, because they reuse memory that is already faulted in.

So huge pages win even for a 1 MiB working set (the L1 TLB covers only 384 KiB
of 4 KiB pages), and the price is 1-3 MiB per heap. `Auto` therefore leaves
small heaps to the deployment's kernel policy instead of refusing, and
guarantees huge pages for big heaps even on `madvise` kernels (Ubuntu's
default). Deployments with many small heaps and tight memory limits should use
`Never`. Not measured: a kernel whose policy is `madvise` (no root on the test
machine); there `Auto` below 16 MiB behaves like the `Never` column.

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
- An empty segment is cached (at most `MAX_CACHED_SEGMENTS`); further empty
  segments are unmapped.
- **Free slots go back to the OS after a whole idle epoch,** in empty and partly
  used segments alike (the mapping and header stay). Each segment keeps three
  slot bitmaps: *dirty* (freed this epoch), *stale* (freed in an earlier epoch
  and still free) and *purged*. At the end of an epoch, stale slots are
  `madvise(MADV_DONTNEED)`ed and dirty ones become stale. Claiming a slot clears
  it from all three, so a slot is purged only after being free for at least one
  whole epoch.
- **Epochs are wall time:** 10 ms, the order of mimalloc's purge delay, read
  from `CLOCK_MONOTONIC_COARSE` (vDSO, no syscall) once per heartbeat. Counting
  slow-path calls instead tied purging to the allocation rate: 1024 calls pass
  in under a millisecond in a busy loop, and `batch_mixed` ran 2.1x slower,
  refaulting memory it had purged moments earlier.
- **Epochs adapt.** Claiming a purged slot is a refault: the purge saved nothing
  and cost a page fault per 4 KiB. An epoch with a refault doubles the epoch
  length, up to 64x (640 ms); four clean epochs in a row halve it. Halving after
  a single clean epoch oscillated, because the epoch after a refault purges
  nothing new and is always clean.
- **Huge pages:** purging 64 KiB inside a 2 MiB page splits it, and `khugepaged`
  collapses ranges with up to `max_ptes_none` (default 511 of 512) empty
  entries again, refilling the hole with zeroes. So while segments ask for huge
  pages (`Auto` past 16 MiB, `Always`), a segment in use is purged only when its
  whole upper half is free; the lower half holds the header. Empty segments are
  purged whole.
- `maintain()` runs housekeeping on demand. It otherwise runs only from the
  allocation slow path, so a heap that goes quiet would keep what it freed; an
  event loop calls `maintain()` from its idle hook. `trim()` releases everything
  free now, splitting huge pages if need be. `reset(&mut self)` frees all blocks
  at once (an arena reset) and keeps a few segments mapped for reuse.

### Fragmentation

`examples/fragmentation.rs`:
1. Fill 256 MiB with 16-1024 B objects.
2. Free 90% of them. Under *random*, survivors are scattered over every page.
   Under *clustered*, runs of 4096 consecutive allocations (about 2 MiB) die
   together.
3. Add 256 MiB of 2-8 KiB objects.
4. Churn: 20 rounds, each replacing half of a 128 MiB live set with sizes from
   a range that drifts every round.

Figures are resident MiB above the baseline, with the best churn time of 3.
Trim means `LocalHeap::trim`, `mi_collect(true)` or `malloc_trim(0)`. AMD
Threadripper PRO 9975WX, THP `always`.

| | glibc | mimalloc | LocalHeap 0.2.0 | LocalHeap `Auto` | LocalHeap `Never` |
|---|---|---|---|---|---|
| clustered, thinned to 33 MiB, then trim | 37 | 54 | 186 | 61 | 58 |
| random, thinned to 26 MiB, then trim | 160 | 283 | 273 | 273 | 268 |
| clustered, 289 MiB live after step 3 | 295 | 338 | 343 | 340 | 329 |
| random, end of churn (128 MiB live) | 242 | 421 | 376 | 376 | 334 |
| random, end of churn, then trim | 193 | 335 | 376 | 326 | 321 |
| random, churn time | 678 ms | 237 ms | 240 ms | 237 ms | 267 ms |

- **When data dies together, its memory now goes back.** 0.2.0 returned only
  wholly empty segments, so after the clustered thinning it kept 186 MiB for
  33 MiB live. It now keeps 61 MiB, close to mimalloc's 54.
- **`Never` returns idle memory as it goes:** 376 to 334 MiB at the end of
  churn without a trim. It pays 4-11% churn time (247 ms in 0.2.0, 256-267 ms
  now, across runs), because purged slots fault back in when reused.
- **`Auto` keeps huge pages whole.** It runs at full speed and returns the rest
  on `trim`.
- **Scattered survivors defeat size classes.** Every page keeps a survivor, so
  neither mimalloc nor `LocalHeap` can return or reuse the space for other
  sizes. glibc coalesces neighbouring free chunks, ends far more compact, and
  runs churn 2.8x slower.

Measurement caveat: `vec_growth_8MiB` swings 16% (248 against 294 µs) between
builds that differ only in unrelated code. With
`-C llvm-args=-align-loops=64`, both run at 247 µs: the swing is the alignment
of the benchmark's push loop, not the heap.

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

1. **Phase 1 (this crate):** `LocalHeap` (with a huge-page policy, optionally
   over a fixed region), `Arena` (on the OS or on a heap), `Reap` and
   `StackArena`.
2. **Phase 2:** optional glommio integration: a heap per executor,
   `maintain()` from the reactor's idle path, byte accounting per task queue
   feeding the scheduler, allocation sampling on the slow path.
3. **Phase 3:** remote frees. `Page::_thread_free` and `Segment::_owner` are
   reserved. The candidate design is snmalloc-style message batching to the
   owner, rather than mimalloc's per-free atomic. This gives a `Send` heap handle,
   checked with Loom.
4. **Phase 4:** `GlobalAlloc`: thread-local lookup, abandoned segments of exited
   threads, re-entrancy, `fork`.

## Known limits (phase 1)

- Segregated size classes cannot reuse a hole for a block of another class.
  When survivors are scattered over every page (`fragmentation random`), freed
  memory stays resident until those pages empty, as with mimalloc; glibc
  coalesces neighbouring free chunks and does better there.
- Housekeeping runs from the allocation slow path and `maintain()`; a heap that
  neither allocates nor is maintained keeps its free memory until `trim()`.
- Linux only for real memory (`mmap`); elsewhere `os.rs` uses the global allocator.
