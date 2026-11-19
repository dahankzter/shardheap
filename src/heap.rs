//! [`LocalHeap`]: a single-threaded heap implementing [`Allocator`].

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::fmt;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use crate::class::{self, NUM_CLASSES};
use crate::os::{self, PAGE};
use crate::region::Region;
use crate::segment::{self, Block, MAGIC, Page, SEGMENT_SIZE, SLOT_SIZE, Segment};

/// Empty segments kept mapped for reuse; further empty segments are unmapped.
const MAX_CACHED_SEGMENTS: usize = 1;
/// Segments kept by [`LocalHeap::reset`].
const MAX_RESET_SEGMENTS: usize = 4;
/// Slow-path events between purge checks.
const HEARTBEAT: u64 = 256;
/// Slow-path events a cached empty segment stays resident before its memory is
/// returned to the OS.
const PURGE_DELAY: u64 = 1024;
/// Slow-path events the last empty page of a class is kept before its slots go
/// back to the segment.
const RETIRE_DELAY: u64 = 256;
/// Freed large mappings kept for reuse (page-aligned ones only).
const LARGE_CACHE_SLOTS: usize = 8;
/// Upper bound on the bytes held in the large-mapping cache.
const LARGE_CACHE_MAX_BYTES: usize = 64 << 20;

/// Under [`HugePages::Auto`], segments ask for huge pages once the heap maps
/// this much. Huge pages cost about 1-3 MiB of extra resident memory per heap
/// (partly used 2 MiB pages), which is noise from here on.
const HUGE_AFTER: usize = 16 << 20;
/// Under [`HugePages::Auto`], large mappings of at least this size ask for huge
/// pages.
const HUGE_LARGE_MIN: usize = 4 << 20;

/// How a [`LocalHeap`] uses transparent huge pages (2 MiB pages on x86-64 Linux;
/// elsewhere every setting behaves like `Kernel`).
///
/// Huge pages cut page faults by up to 512x and TLB misses, but memory becomes
/// resident in 2 MiB steps, so a heap holds 1-3 MiB more than it uses. Measured
/// on Zen 5 (see `DESIGN.md`): building 256 MiB of nodes 2.1x faster, random
/// pointer chasing 10-45% faster from 1 MiB to 256 MiB working sets, and a
/// request workload using 1 MiB holding 4 MiB instead of 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HugePages {
    /// Follow the kernel's policy while the heap is small; once it maps 16 MiB,
    /// ask for huge pages even where the policy is `madvise`: new segments
    /// immediately, existing ones when `khugepaged` gets to them (in the
    /// background; nothing is collapsed synchronously). Large mappings of at
    /// least 4 MiB always ask. Never refuses huge pages the kernel would give.
    #[default]
    Auto,
    /// Every segment and large mapping asks for huge pages (`MADV_HUGEPAGE`),
    /// including on kernels whose policy is `madvise`.
    Always,
    /// Never use huge pages (`MADV_NOHUGEPAGE`), even where the kernel's policy
    /// is `always`: the leanest choice for many small heaps.
    Never,
    /// Leave the system policy (`/sys/kernel/mm/transparent_hugepage/enabled`)
    /// in charge.
    Kernel,
}

/// A freed large mapping kept mapped (and its pages resident) for reuse.
#[derive(Clone, Copy)]
struct CachedMap {
    base: *mut u8,
    size: usize,
    idle_since: u64,
}

const NO_MAP: CachedMap = CachedMap {
    base: ptr::null_mut(),
    size: 0,
    idle_since: 0,
};

/// Counters kept by a [`LocalHeap`]. Updating them is one add on the fast path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Bytes currently handed out: block sizes plus large mappings.
    pub allocated_bytes: usize,
    /// Bytes currently mapped from the OS.
    pub mapped_bytes: usize,
    /// Segments currently mapped.
    pub segments: usize,
    /// Mapped segments with no pages in use.
    pub empty_segments: usize,
    /// Live large allocations.
    pub large_allocations: usize,
    /// Pages held by the [`Reap`](crate::Reap): the current bump page plus every
    /// earlier one that still has a live block.
    pub reap_pages: usize,
    /// Bytes taken from the heap's region (see [`LocalHeap::with_region`]),
    /// in 64 KiB granules: segments, large blocks and cached large blocks.
    pub region_bytes: usize,
}

/// Header at the start of every large mapping. Large mappings form an intrusive
/// list so [`LocalHeap::reset`] and `Drop` can release them.
#[repr(C)]
struct LargeHeader {
    next: *mut LargeHeader,
    prev: *mut LargeHeader,
    map_size: usize,
    map_align: usize,
}

/// Distance from the start of a large mapping to the data.
#[inline(always)]
fn large_offset(align: usize) -> usize {
    align.max(PAGE)
}

pub(crate) struct Inner {
    /// Per class: head of the list of pages that are not full.
    queues: [*mut Page; NUM_CLASSES],
    segments: *mut Segment,
    large: *mut LargeHeader,
    large_cache: [CachedMap; LARGE_CACHE_SLOTS],
    large_cache_bytes: usize,
    tick: u64,
    pub(crate) stats: Stats,
    /// The reap's bump page, or null; see `reap.rs`.
    pub(crate) reap_page: *mut Page,
    /// Next free byte in `reap_page`.
    pub(crate) reap_ptr: *mut u8,
    /// End of `reap_page`.
    pub(crate) reap_end: *mut u8,
    huge_pages: HugePages,
    /// Whether the heap has crossed `HUGE_AFTER` under `HugePages::Auto`.
    huge_now: bool,
    /// Memory to take segments and large blocks from before the OS.
    region: Option<Region>,
}

/// A shared-nothing heap: one per thread (or executor), no atomics, no locks.
///
/// `LocalHeap` is `!Send` and `!Sync`, and so is every `Vec<T, &LocalHeap>` or
/// `Box<T, &LocalHeap>`. The compiler therefore guarantees that every block is
/// freed on the thread that allocated it.
///
/// ```
/// use shardheap::LocalHeap;
///
/// let heap = LocalHeap::new();
/// let mut v: Vec<u64, &LocalHeap> = Vec::new_in(&heap);
/// v.extend(0..1000);
/// let b = Box::new_in([0u8; 100], &heap);
/// assert_eq!(v.iter().sum::<u64>(), 499_500);
/// drop((v, b));
/// assert_eq!(heap.stats().allocated_bytes, 0);
/// ```
///
/// Neither the heap nor its collections can cross threads:
///
/// ```compile_fail,E0277
/// fn assert_send<T: Send>(_: T) {}
/// let heap = shardheap::LocalHeap::new();
/// assert_send(Vec::<u8, _>::new_in(&heap));
/// ```
///
/// And the heap cannot be dropped while anything still uses it:
///
/// ```compile_fail,E0505
/// let heap = shardheap::LocalHeap::new();
/// let v: Vec<u8, &shardheap::LocalHeap> = Vec::with_capacity_in(10, &heap);
/// drop(heap);
/// drop(v);
/// ```
pub struct LocalHeap {
    inner: UnsafeCell<Inner>,
    _not_send: PhantomData<*mut ()>,
}

impl Default for LocalHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for LocalHeap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalHeap")
            .field("stats", &self.stats())
            .finish()
    }
}

impl LocalHeap {
    /// An empty heap. Maps nothing until the first allocation.
    pub const fn new() -> Self {
        LocalHeap {
            inner: UnsafeCell::new(Inner {
                queues: [ptr::null_mut(); NUM_CLASSES],
                segments: ptr::null_mut(),
                large: ptr::null_mut(),
                large_cache: [NO_MAP; LARGE_CACHE_SLOTS],
                large_cache_bytes: 0,
                tick: 0,
                stats: Stats {
                    allocated_bytes: 0,
                    mapped_bytes: 0,
                    segments: 0,
                    empty_segments: 0,
                    large_allocations: 0,
                    reap_pages: 0,
                    region_bytes: 0,
                },
                reap_page: ptr::null_mut(),
                reap_ptr: ptr::null_mut(),
                reap_end: ptr::null_mut(),
                huge_pages: HugePages::Auto,
                huge_now: false,
                region: None,
            }),
            _not_send: PhantomData,
        }
    }

    /// An empty heap with the given huge-page policy. [`new`](Self::new) uses
    /// [`HugePages::Auto`].
    pub const fn with_huge_pages(policy: HugePages) -> Self {
        let mut heap = Self::new();
        heap.inner.get_mut().huge_pages = policy;
        heap
    }

    /// A heap that takes all its memory from one region of at least `bytes`
    /// (rounded up to 4 MiB), mapped now and owned by the heap, and from the OS
    /// only once the region is full. `None` if the region cannot be mapped.
    ///
    /// Every block of the heap, and of an [`Arena`](crate::Arena) or
    /// [`Reap`](crate::Reap) on it, then lies in one known range, which is what
    /// io_uring registered buffers (`IORING_REGISTER_BUFFERS`), RDMA memory
    /// regions or a shared-memory window need. Register [`region`](Self::region)
    /// once, then check each buffer with [`in_region`](Self::in_region): inside
    /// means a fixed-buffer operation, outside (the OS fallback) a regular one.
    ///
    /// Region memory is never unmapped, purged or moved with `mremap` while the
    /// heap lives, because registered memory is pinned. Huge pages follow
    /// `huge_pages` (`Auto` asks for them: a registered region is resident
    /// anyway, so they cost nothing).
    ///
    /// ```
    /// use shardheap::{HugePages, LocalHeap};
    ///
    /// let heap = LocalHeap::with_region(8 << 20, HugePages::Auto).unwrap();
    /// let region = heap.region().unwrap();
    /// assert_eq!(region.len(), 8 << 20);
    /// // ... register `region` with io_uring ...
    ///
    /// let mut payload: Vec<u8, &LocalHeap> = Vec::new_in(&heap);
    /// payload.extend_from_slice(b"header");
    /// payload.resize(64 << 10, 0);
    /// assert!(heap.in_region(payload.as_ptr())); // submit with WRITE_FIXED
    /// ```
    pub fn with_region(bytes: usize, huge_pages: HugePages) -> Option<Self> {
        let advice = match huge_pages {
            HugePages::Auto | HugePages::Always => Some(true),
            HugePages::Never => Some(false),
            HugePages::Kernel => None,
        };
        let region = Region::map(bytes, advice)?;
        let mut heap = Self::with_huge_pages(huge_pages);
        heap.inner.get_mut().region = Some(region);
        Some(heap)
    }

    /// The heap's region (see [`with_region`](Self::with_region)), if it has one.
    pub fn region(&self) -> Option<NonNull<[u8]>> {
        // SAFETY: see `stats`.
        unsafe { (*self.inner.get()).region.as_ref().map(Region::range) }
    }

    /// Whether `p` lies in the heap's region. A block allocated from the heap
    /// lies entirely inside or entirely outside it.
    #[inline]
    pub fn in_region(&self, p: *const u8) -> bool {
        // SAFETY: see `stats`.
        unsafe { (*self.inner.get()).region.as_ref() }.is_some_and(|r| r.contains(p))
    }

    /// Current counters.
    pub fn stats(&self) -> Stats {
        // SAFETY: single-threaded (`!Sync`), and no method keeps a reference to
        // `Inner` across a call, so this read does not alias a live `&mut`.
        unsafe { (*self.inner.get()).stats }
    }

    /// Unmaps every empty segment now, instead of waiting for the purge delay.
    pub fn trim(&self) {
        // SAFETY: see `inner`.
        unsafe { self.inner().trim() }
    }

    /// Frees everything at once, like an arena reset. Keeps a few segments mapped
    /// for reuse. Taking `&mut self` proves no collection still borrows the heap.
    pub fn reset(&mut self) {
        // SAFETY: `&mut self`: no blocks are reachable any more.
        unsafe { self.inner.get_mut().reset() }
    }

    /// # Safety
    /// The returned reference must not outlive the current method call, and no
    /// other reference to `Inner` may exist. Holds because `LocalHeap` is `!Sync`,
    /// its methods never re-enter one another, and none calls user code.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn inner(&self) -> &mut Inner {
        // SAFETY: see above.
        unsafe { &mut *self.inner.get() }
    }
}

impl Drop for LocalHeap {
    fn drop(&mut self) {
        // SAFETY: `&mut self`: nothing borrows the heap any more, so no block is live.
        unsafe { self.inner.get_mut().release_all() }
    }
}

#[inline(always)]
pub(crate) fn slice(p: *mut u8, len: usize) -> NonNull<[u8]> {
    // SAFETY: callers pass pointers to blocks they just allocated, never null.
    NonNull::slice_from_raw_parts(unsafe { NonNull::new_unchecked(p) }, len)
}

#[inline(always)]
pub(crate) fn dangling(align: usize) -> NonNull<[u8]> {
    // SAFETY: an alignment is never zero.
    let p = unsafe { NonNull::new_unchecked(ptr::without_provenance_mut::<u8>(align)) };
    NonNull::slice_from_raw_parts(p, 0)
}

impl Inner {
    // ---- allocation --------------------------------------------------------

    #[inline(always)]
    pub(crate) unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling(layout.align()));
        }
        match class::class_for(layout) {
            // SAFETY: `c` comes from `class_for`.
            Some(c) => unsafe { self.alloc_small(c) },
            // SAFETY: `layout` is nonzero and takes the large path.
            None => unsafe { self.alloc_large(layout).map(|(b, _)| b) },
        }
    }

    /// # Safety
    /// `c < NUM_CLASSES`.
    #[inline(always)]
    unsafe fn alloc_small(&mut self, c: usize) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `c < NUM_CLASSES`; queue pages are live pages of this heap.
        unsafe {
            let page = *self.queues.get_unchecked(c);
            if !page.is_null() {
                let mut b = (*page).free;
                if b.is_null() {
                    // Collect our own frees inline: an alloc/free loop on one block
                    // would otherwise take the slow path every time.
                    b = ptr::replace(&raw mut (*page).local_free, ptr::null_mut());
                }
                if !b.is_null() {
                    (*page).free = (*b).next;
                    (*page).used += 1;
                    let bs = (*page).block_size;
                    self.stats.allocated_bytes += bs;
                    return Ok(slice(b.cast(), bs));
                }
            }
            self.alloc_small_slow(c)
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn alloc_small_slow(&mut self, c: usize) -> Result<NonNull<[u8]>, AllocError> {
        self.heartbeat();
        // SAFETY: every page reached through the queue is a live page of this heap.
        unsafe {
            loop {
                let page = self.queues[c];
                if page.is_null() {
                    break;
                }
                if let Some(b) = self.take_block(page) {
                    return Ok(b);
                }
                // Full: leave the queue until one of its blocks is freed.
                self.unlink(page);
            }
            let page = self.new_page(c)?;
            Ok(self.take_block(page).expect("a fresh page has free blocks"))
        }
    }

    /// Takes a block from `local_free` (swapped into `free`) or the unused tail.
    unsafe fn take_block(&mut self, page: *mut Page) -> Option<NonNull<[u8]>> {
        // SAFETY: `page` is a live page of this heap.
        unsafe {
            if (*page).free.is_null() {
                (*page).free = ptr::replace(&raw mut (*page).local_free, ptr::null_mut());
            }
            let bs = (*page).block_size;
            let b = (*page).free;
            let p = if !b.is_null() {
                (*page).free = (*b).next;
                b.cast::<u8>()
            } else if (*page).bump < (*page).capacity {
                let p = (*page).start.add((*page).bump as usize * bs);
                (*page).bump += 1;
                p
            } else {
                return None;
            };
            (*page).used += 1;
            self.stats.allocated_bytes += bs;
            Some(slice(p, bs))
        }
    }

    unsafe fn new_page(&mut self, c: usize) -> Result<*mut Page, AllocError> {
        // SAFETY: `claim_span` returns a fresh page descriptor of this heap.
        unsafe {
            let page = self.claim_span(class::span_slots(c))?;
            (*page).block_size = class::size_of(c);
            (*page).capacity = class::capacity(c) as u32;
            (*page).class = c as u8;
            self.push_front(page);
            Ok(page)
        }
    }

    /// Claims `k` contiguous free slots (first fit, mapping a segment if none
    /// fits) and returns their head page with `start` and `span` set and every
    /// other field zero. The page is in no queue.
    pub(crate) unsafe fn claim_span(&mut self, k: usize) -> Result<*mut Page, AllocError> {
        // SAFETY: the segment list holds live segments of this heap; a found span
        // is free, and descriptors of free slots are zeroed by `release_page`.
        unsafe {
            let mut seg = self.segments;
            let (seg, slot) = loop {
                if seg.is_null() {
                    let seg = self.new_segment()?;
                    let slot = segment::find_span((*seg).used_slots, k)
                        .expect("an empty segment fits any page");
                    break (seg, slot);
                }
                if let Some(slot) = segment::find_span((*seg).used_slots, k) {
                    break (seg, slot);
                }
                seg = (*seg).next;
            };
            if (*seg).used_slots == 1 {
                self.stats.empty_segments -= 1;
                (*seg).purged = false;
            }
            (*seg).used_slots |= segment::span_mask(slot, k);
            let pages = (&raw mut (*seg).pages).cast::<Page>();
            for i in 0..k {
                (*pages.add(slot + i)).head_offset = i as u8;
            }
            let page = pages.add(slot);
            ptr::write_bytes(page, 0, 1);
            (*page).start = seg.cast::<u8>().add(slot * SLOT_SIZE);
            (*page).span = k as u8;
            Ok(page)
        }
    }

    unsafe fn new_segment(&mut self) -> Result<*mut Segment, AllocError> {
        // SAFETY: valid size and alignment; the mapping is fresh and zeroed.
        unsafe {
            let (base, zeroed) = self.map(SEGMENT_SIZE, SEGMENT_SIZE).ok_or(AllocError)?;
            if zeroed {
                if !self.huge_now && self.stats.mapped_bytes >= HUGE_AFTER {
                    self.switch_to_huge();
                }
                self.advise(base, SEGMENT_SIZE, self.huge_now);
            } else {
                // Reused region memory: `init_segment` needs a zeroed header.
                ptr::write_bytes(base.as_ptr(), 0, core::mem::size_of::<Segment>());
            }
            let seg = segment::init_segment(base);
            (*seg).next = self.segments;
            if !self.segments.is_null() {
                (*self.segments).prev = seg;
            }
            self.segments = seg;
            self.stats.segments += 1;
            self.stats.empty_segments += 1; // `init_page` takes it out again
            self.stats.mapped_bytes += SEGMENT_SIZE;
            Ok(seg)
        }
    }

    /// The heap has grown past `HUGE_AFTER`: from now on segments ask for huge
    /// pages, and existing ones are re-advised so `khugepaged` can collapse them.
    #[cold]
    fn switch_to_huge(&mut self) {
        self.huge_now = true;
        if self.huge_pages != HugePages::Auto {
            return;
        }
        let mut seg = self.segments;
        while !seg.is_null() {
            // SAFETY: the segment list holds live segments of this heap.
            unsafe {
                if !self.in_region(seg.cast()) {
                    os::advise_huge(NonNull::new_unchecked(seg.cast()), SEGMENT_SIZE, true);
                }
                seg = (*seg).next;
            }
        }
    }

    /// Takes `size` bytes aligned to `align` from the region, or else from the
    /// OS; returns whether the memory is zeroed (fresh OS memory is).
    ///
    /// # Safety
    /// `size` is a nonzero multiple of `PAGE`; `align` a power of two `>= PAGE`.
    unsafe fn map(&mut self, size: usize, align: usize) -> Option<(NonNull<u8>, bool)> {
        if let Some(r) = &mut self.region
            && let Some(p) = r.alloc(size, align)
        {
            self.stats.region_bytes = r.used_bytes;
            return Some((p, false));
        }
        // SAFETY: caller's contract.
        unsafe { os::map_aligned(size, align) }.map(|p| (p, true))
    }

    /// Returns memory from `map` to the region or the OS.
    ///
    /// # Safety
    /// `p`, `size`, `align` are exactly as passed to and returned by `map`, and
    /// nothing uses the memory any more.
    unsafe fn unmap(&mut self, p: NonNull<u8>, size: usize, align: usize) {
        if let Some(r) = &mut self.region
            && r.contains(p.as_ptr())
        {
            r.free(p, size);
            self.stats.region_bytes = r.used_bytes;
            return;
        }
        // SAFETY: caller's contract.
        unsafe { os::unmap(p, size, align) }
    }

    fn in_region(&self, p: *const u8) -> bool {
        self.region.as_ref().is_some_and(|r| r.contains(p))
    }

    /// Applies the huge-page policy to a fresh mapping; `auto` is the decision
    /// under [`HugePages::Auto`].
    ///
    /// # Safety
    /// The range is a mapping of this heap.
    unsafe fn advise(&self, p: NonNull<u8>, size: usize, auto: bool) {
        let huge = match self.huge_pages {
            HugePages::Auto if auto => true,
            HugePages::Auto => return,
            HugePages::Always => true,
            HugePages::Never => false,
            HugePages::Kernel => return,
        };
        // SAFETY: caller's contract.
        unsafe { os::advise_huge(p, size, huge) }
    }

    // ---- deallocation ------------------------------------------------------

    #[inline(always)]
    pub(crate) unsafe fn dealloc(&mut self, p: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        match class::class_for(layout) {
            // SAFETY: caller's contract: `p` was allocated here with a fitting layout.
            Some(c) => unsafe { self.free_small(p, c) },
            None => unsafe { self.free_large(p, layout) },
        }
    }

    #[inline(always)]
    unsafe fn free_small(&mut self, p: *mut u8, c: usize) {
        let seg = segment::segment_of(p);
        // SAFETY: `p` is a live block of this heap, so `seg` is its live segment.
        unsafe {
            debug_assert_eq!((*seg).magic, MAGIC, "pointer not from a LocalHeap");
            let page = segment::page_of(seg, p);
            debug_assert_eq!(
                (*page).class as usize,
                c,
                "deallocated with a different layout"
            );
            let _ = c;
            let b = p.cast::<Block>();
            (*b).next = (*page).local_free;
            (*page).local_free = b;
            (*page).used -= 1;
            self.stats.allocated_bytes -= (*page).block_size;
            if !(*page).in_queue || (*page).used == 0 {
                self.free_small_slow(page);
            }
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn free_small_slow(&mut self, page: *mut Page) {
        // SAFETY: `page` is a live page of this heap.
        unsafe {
            if !(*page).in_queue {
                self.push_front(page); // was full, has room again
            }
            if (*page).used == 0 {
                self.retire(page);
            }
        }
    }

    /// Returns an empty page's slots to its segment. The only page of its class is
    /// kept for `RETIRE_DELAY` instead, which avoids thrashing when a workload
    /// keeps crossing a page boundary.
    unsafe fn retire(&mut self, page: *mut Page) {
        // SAFETY: `page` is a live, empty page of this heap.
        unsafe {
            let c = (*page).class as usize;
            if self.queues[c] == page && (*page).next.is_null() {
                (*page).idle_since = self.tick;
                return;
            }
            self.release_page(page);
        }
    }

    /// Unlinks an empty page (if queued) and returns its slots to the segment.
    pub(crate) unsafe fn release_page(&mut self, page: *mut Page) {
        // SAFETY: `page` is a live, empty page of this heap.
        unsafe {
            debug_assert_eq!((*page).used, 0);
            if (*page).in_queue {
                self.unlink(page);
            }
            let seg = segment::segment_of((*page).start);
            let slot = ((*page).start.addr() - seg.addr()) / SLOT_SIZE;
            let k = (*page).span as usize;
            ptr::write_bytes(page, 0, 1);
            (*seg).used_slots &= !segment::span_mask(slot, k);
            if (*seg).used_slots == 1 {
                self.segment_became_empty(seg);
            }
        }
    }

    unsafe fn segment_became_empty(&mut self, seg: *mut Segment) {
        // SAFETY: `seg` is a live, empty segment of this heap.
        unsafe {
            (*seg).idle_since = self.tick;
            self.stats.empty_segments += 1;
            if self.stats.empty_segments > MAX_CACHED_SEGMENTS {
                self.release_segment(seg);
            }
        }
    }

    unsafe fn release_segment(&mut self, seg: *mut Segment) {
        // SAFETY: `seg` is a live, empty segment of this heap; nothing points into it.
        unsafe {
            let (prev, next) = ((*seg).prev, (*seg).next);
            if prev.is_null() {
                self.segments = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
            if (*seg).used_slots == 1 {
                self.stats.empty_segments -= 1;
            }
            self.unmap(
                NonNull::new_unchecked(seg.cast()),
                SEGMENT_SIZE,
                SEGMENT_SIZE,
            );
            self.stats.segments -= 1;
            self.stats.mapped_bytes -= SEGMENT_SIZE;
        }
    }

    // ---- page queues -------------------------------------------------------

    unsafe fn push_front(&mut self, page: *mut Page) {
        // SAFETY: `page` is a live page not currently in a queue.
        unsafe {
            let c = (*page).class as usize;
            let head = self.queues[c];
            (*page).prev = ptr::null_mut();
            (*page).next = head;
            if !head.is_null() {
                (*head).prev = page;
            }
            self.queues[c] = page;
            (*page).in_queue = true;
        }
    }

    unsafe fn unlink(&mut self, page: *mut Page) {
        // SAFETY: `page` is a live page currently in its class queue.
        unsafe {
            let c = (*page).class as usize;
            let (prev, next) = ((*page).prev, (*page).next);
            if prev.is_null() {
                self.queues[c] = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
            (*page).prev = ptr::null_mut();
            (*page).next = ptr::null_mut();
            (*page).in_queue = false;
        }
    }

    // ---- housekeeping ------------------------------------------------------

    pub(crate) fn heartbeat(&mut self) {
        self.tick += 1;
        if self.tick.is_multiple_of(HEARTBEAT) {
            // SAFETY: walks only this heap's live segments.
            unsafe { self.purge_idle() }
        }
    }

    /// Releases long-idle kept pages, then returns the memory of long-idle empty
    /// segments to the OS (the mapping stays).
    unsafe fn purge_idle(&mut self) {
        // SAFETY: queues and the segment list hold live pages/segments of this heap.
        unsafe {
            self.release_kept_pages(RETIRE_DELAY);
            self.evict_large_cache(PURGE_DELAY);
            let mut seg = self.segments;
            while !seg.is_null() {
                if (*seg).used_slots == 1
                    && !(*seg).purged
                    && self.tick - (*seg).idle_since >= PURGE_DELAY
                    && !self.in_region(seg.cast())
                {
                    let slots = NonNull::new_unchecked(seg.cast::<u8>().add(SLOT_SIZE));
                    os::purge(slots, SEGMENT_SIZE - SLOT_SIZE);
                    (*seg).purged = true;
                }
                seg = (*seg).next;
            }
        }
    }

    /// Releases every kept empty page idle for at least `delay` ticks. Empty pages
    /// other than a class's only page are released immediately by `retire`, so
    /// only queue heads need checking.
    unsafe fn release_kept_pages(&mut self, delay: u64) {
        // SAFETY: queue heads are live pages of this heap.
        unsafe {
            for c in 0..NUM_CLASSES {
                let page = self.queues[c];
                if !page.is_null()
                    && (*page).used == 0
                    && (*page).next.is_null()
                    && self.tick - (*page).idle_since >= delay
                {
                    self.release_page(page);
                }
            }
        }
    }

    unsafe fn trim(&mut self) {
        // SAFETY: queues and the segment list hold live pages/segments of this heap.
        unsafe {
            self.release_kept_pages(0);
            self.release_idle_reap_page();
            self.evict_large_cache(0);
            let mut seg = self.segments;
            while !seg.is_null() {
                let next = (*seg).next;
                if (*seg).used_slots == 1 {
                    self.release_segment(seg);
                }
                seg = next;
            }
        }
    }

    unsafe fn reset(&mut self) {
        // SAFETY: no blocks are live (caller holds `&mut LocalHeap`).
        unsafe {
            self.release_large();
            self.queues = [ptr::null_mut(); NUM_CLASSES];
            self.forget_reap();
            let mut kept = 0;
            let mut seg = self.segments;
            while !seg.is_null() {
                let next = (*seg).next;
                if kept < MAX_RESET_SEGMENTS {
                    if (*seg).used_slots != 1 {
                        ptr::write_bytes(&raw mut (*seg).pages, 0, 1);
                        (*seg).used_slots = 1;
                        (*seg).idle_since = self.tick;
                        self.stats.empty_segments += 1;
                    }
                    kept += 1;
                } else {
                    if (*seg).used_slots != 1 {
                        (*seg).used_slots = 1; // so `release_segment` counts it as empty
                        self.stats.empty_segments += 1;
                    }
                    self.release_segment(seg);
                }
                seg = next;
            }
            self.stats.allocated_bytes = 0;
        }
    }

    unsafe fn release_all(&mut self) {
        // SAFETY: no blocks are live (the heap is being dropped).
        unsafe {
            self.release_large();
            let mut seg = self.segments;
            while !seg.is_null() {
                let next = (*seg).next;
                self.unmap(
                    NonNull::new_unchecked(seg.cast()),
                    SEGMENT_SIZE,
                    SEGMENT_SIZE,
                );
                seg = next;
            }
            self.segments = ptr::null_mut();
        }
    }

    // ---- large allocations -------------------------------------------------

    /// Returns the block and whether it is a fresh (zeroed) mapping.
    unsafe fn alloc_large(&mut self, layout: Layout) -> Result<(NonNull<[u8]>, bool), AllocError> {
        let off = large_offset(layout.align());
        let need = off
            .checked_add(layout.size())
            .and_then(|s| s.checked_next_multiple_of(PAGE))
            .ok_or(AllocError)?;
        let map_align = layout.align().max(PAGE);
        // SAFETY: nonzero multiple of PAGE, power-of-two alignment >= PAGE; cached
        // mappings are unused mappings of this heap.
        unsafe {
            let (base, map_size, fresh) = match (map_align == PAGE)
                .then(|| self.take_cached(need))
                .flatten()
            {
                Some(m) => (NonNull::new_unchecked(m.base), m.size, false),
                None => {
                    let (base, zeroed) = self.map(need, map_align).ok_or(AllocError)?;
                    if zeroed {
                        self.advise(base, need, need >= HUGE_LARGE_MIN);
                    }
                    self.stats.mapped_bytes += need;
                    (base, need, zeroed)
                }
            };
            Ok((self.link_large(base, map_size, map_align, off), fresh))
        }
    }

    /// Writes a large header at `base`, links it, and returns the data block.
    unsafe fn link_large(
        &mut self,
        base: NonNull<u8>,
        map_size: usize,
        map_align: usize,
        off: usize,
    ) -> NonNull<[u8]> {
        // SAFETY: `base` is an unused mapping of `map_size` bytes owned by this heap.
        unsafe {
            let h = base.as_ptr().cast::<LargeHeader>();
            h.write(LargeHeader {
                next: self.large,
                prev: ptr::null_mut(),
                map_size,
                map_align,
            });
            if !self.large.is_null() {
                (*self.large).prev = h;
            }
            self.large = h;
            self.stats.large_allocations += 1;
            self.stats.allocated_bytes += map_size;
            slice(base.as_ptr().add(off), map_size - off)
        }
    }

    /// Best-fitting cached mapping of at least `need` bytes and at most twice that.
    fn take_cached(&mut self, need: usize) -> Option<CachedMap> {
        let limit = need.saturating_mul(2);
        let i = (0..LARGE_CACHE_SLOTS)
            .filter(|&i| {
                let m = self.large_cache[i];
                !m.base.is_null() && m.size >= need && m.size <= limit
            })
            .min_by_key(|&i| self.large_cache[i].size)?;
        let m = std::mem::replace(&mut self.large_cache[i], NO_MAP);
        self.large_cache_bytes -= m.size;
        Some(m)
    }

    /// Keeps a freed page-aligned mapping for reuse, evicting the oldest entries
    /// to stay within the cache limits. Returns `false` if it should be unmapped.
    unsafe fn put_cached(&mut self, base: *mut u8, size: usize) -> bool {
        if size > LARGE_CACHE_MAX_BYTES {
            return false;
        }
        loop {
            let free_slot = self.large_cache.iter().position(|m| m.base.is_null());
            match free_slot {
                Some(i) if self.large_cache_bytes + size <= LARGE_CACHE_MAX_BYTES => {
                    self.large_cache[i] = CachedMap {
                        base,
                        size,
                        idle_since: self.tick,
                    };
                    self.large_cache_bytes += size;
                    return true;
                }
                _ => {
                    let oldest = (0..LARGE_CACHE_SLOTS)
                        .filter(|&i| !self.large_cache[i].base.is_null())
                        .min_by_key(|&i| self.large_cache[i].idle_since)
                        .expect("a full cache has entries");
                    // SAFETY: cached entries are unused mappings of this heap.
                    unsafe { self.unmap_cached(oldest) };
                }
            }
        }
    }

    unsafe fn unmap_cached(&mut self, i: usize) {
        let m = std::mem::replace(&mut self.large_cache[i], NO_MAP);
        self.large_cache_bytes -= m.size;
        self.stats.mapped_bytes -= m.size;
        // SAFETY: an unused, page-aligned mapping of this heap.
        unsafe { self.unmap(NonNull::new_unchecked(m.base), m.size, PAGE) };
    }

    /// Unmaps cached mappings idle for at least `delay` ticks.
    unsafe fn evict_large_cache(&mut self, delay: u64) {
        for i in 0..LARGE_CACHE_SLOTS {
            let m = self.large_cache[i];
            if !m.base.is_null() && self.tick - m.idle_since >= delay {
                // SAFETY: cached entries are unused mappings of this heap.
                unsafe { self.unmap_cached(i) };
            }
        }
    }

    unsafe fn unlink_large(&mut self, h: *mut LargeHeader) {
        // SAFETY: `h` is a live large header of this heap.
        unsafe {
            let (prev, next) = ((*h).prev, (*h).next);
            if prev.is_null() {
                self.large = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
        }
    }

    unsafe fn free_large(&mut self, p: *mut u8, layout: Layout) {
        // SAFETY: `p` was returned by `alloc_large` (or `grow_large`) with this alignment.
        unsafe {
            let h = p.sub(large_offset(layout.align())).cast::<LargeHeader>();
            self.unlink_large(h);
            let (size, align) = ((*h).map_size, (*h).map_align);
            self.stats.large_allocations -= 1;
            self.stats.allocated_bytes -= size;
            if align != PAGE || !self.put_cached(h.cast(), size) {
                self.stats.mapped_bytes -= size;
                self.unmap(NonNull::new_unchecked(h.cast()), size, align);
            }
        }
    }

    unsafe fn release_large(&mut self) {
        // SAFETY: called only when no blocks are live.
        unsafe {
            let mut h = self.large;
            while !h.is_null() {
                let next = (*h).next;
                let (size, align) = ((*h).map_size, (*h).map_align);
                self.stats.allocated_bytes -= size;
                self.stats.mapped_bytes -= size;
                self.unmap(NonNull::new_unchecked(h.cast()), size, align);
                h = next;
            }
            self.large = ptr::null_mut();
            self.stats.large_allocations = 0;
            self.evict_large_cache(0);
        }
    }

    /// Resizes a large allocation in place with `mremap` (no copy) where possible.
    /// Both layouts take the large path.
    unsafe fn resize_large(
        &mut self,
        p: *mut u8,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `p` is a live large allocation made with `old`.
        unsafe {
            if old.align() == new.align() && new.align() <= PAGE {
                let h = p.sub(PAGE).cast::<LargeHeader>();
                let new_map = PAGE
                    .checked_add(new.size())
                    .and_then(|s| s.checked_next_multiple_of(PAGE))
                    .ok_or(AllocError)?;
                let old_map = (*h).map_size;
                if new_map == old_map || (new_map < old_map && new.size() >= old.size()) {
                    // Already fits (a cached mapping can be larger than asked for).
                    return Ok(slice(p, old_map - PAGE));
                }
                if new_map > old_map
                    && let Some(r) = &mut self.region
                    && r.contains(h.cast())
                    && r.try_grow(NonNull::new_unchecked(h.cast()), old_map, new_map)
                {
                    // Region memory cannot be remapped, but the granules after it
                    // were free: grown in place, no copy.
                    self.stats.region_bytes = r.used_bytes;
                    (*h).map_size = new_map;
                    self.stats.allocated_bytes = self.stats.allocated_bytes - old_map + new_map;
                    self.stats.mapped_bytes = self.stats.mapped_bytes - old_map + new_map;
                    return Ok(slice(p, new_map - PAGE));
                }
                if new_map > old_map
                    && let Some(m) = self.take_cached(new_map)
                {
                    // Resident memory beats `mremap` plus page faults on the tail.
                    let base = NonNull::new_unchecked(m.base);
                    let n = self.link_large(base, m.size, PAGE, PAGE);
                    ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), old.size());
                    self.free_large(p, old);
                    return Ok(n);
                }
                // Never move region memory: it may be registered and pinned.
                if !self.in_region(h.cast())
                    && let Some(nb) = os::remap(NonNull::new_unchecked(h.cast()), old_map, new_map)
                {
                    // The advice moves with the mapping; re-decide for the new size.
                    self.advise(nb, new_map, new_map >= HUGE_LARGE_MIN);
                    let nh = nb.as_ptr().cast::<LargeHeader>();
                    (*nh).map_size = new_map;
                    let (prev, next) = ((*nh).prev, (*nh).next);
                    if prev.is_null() {
                        self.large = nh;
                    } else {
                        (*prev).next = nh;
                    }
                    if !next.is_null() {
                        (*next).prev = nh;
                    }
                    self.stats.allocated_bytes = self.stats.allocated_bytes - old_map + new_map;
                    self.stats.mapped_bytes = self.stats.mapped_bytes - old_map + new_map;
                    return Ok(slice(nb.as_ptr().add(PAGE), new_map - PAGE));
                }
            }
            self.realloc_move(p, old, new)
        }
    }

    /// Allocate, copy, free.
    unsafe fn realloc_move(
        &mut self,
        p: *mut u8,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `p` is live with layout `old`; the new block is disjoint from it.
        unsafe {
            let n = self.alloc(new)?;
            ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), old.size().min(new.size()));
            self.dealloc(p, old);
            Ok(n)
        }
    }

    pub(crate) unsafe fn resize(
        &mut self,
        p: *mut u8,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old.size() == 0 {
            // SAFETY: nothing to copy or free.
            return unsafe { self.alloc(new) };
        }
        if new.size() == 0 {
            // SAFETY: `p` is live with layout `old`.
            unsafe { self.dealloc(p, old) };
            return Ok(dangling(new.align()));
        }
        // SAFETY: `p` is live with layout `old`.
        unsafe {
            match (class::class_for(old), class::class_for(new)) {
                // Same class: the block already fits.
                (Some(a), Some(b)) if a == b => Ok(slice(p, class::size_of(a))),
                (None, None) => self.resize_large(p, old, new),
                _ => self.realloc_move(p, old, new),
            }
        }
    }
}

// SAFETY:
// - Blocks are disjoint: each comes from exactly one page slot range or one
//   large mapping, and is handed out once until freed.
// - A block stays valid until it is deallocated or the heap is dropped/reset;
//   dropping and resetting need ownership/`&mut`, which no live collection allows.
// - Nothing unwinds: OS failures return `AllocError`; the `expect`s are
//   unreachable invariants.
// - `deallocate`/`grow`/`shrink` accept any layout that fits the block: the class
//   (and with it the large/small decision) is the same for every size between the
//   requested size and the returned length.
unsafe impl Allocator for LocalHeap {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: see `inner`.
        unsafe { self.inner().alloc(layout) }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: see `inner`.
        let inner = unsafe { self.inner() };
        if layout.size() != 0 && class::class_for(layout).is_none() {
            // SAFETY: nonzero layout on the large path.
            let (b, fresh) = unsafe { inner.alloc_large(layout)? };
            if !fresh {
                // SAFETY: a reused mapping of `b.len()` bytes; fresh ones are already zero.
                unsafe { ptr::write_bytes(b.as_ptr().cast::<u8>(), 0, b.len()) };
            }
            return Ok(b);
        }
        // SAFETY: as `allocate`.
        let b = unsafe { inner.alloc(layout)? };
        // SAFETY: `b` is a fresh block of `b.len()` bytes.
        unsafe { ptr::write_bytes(b.as_ptr().cast::<u8>(), 0, b.len()) };
        Ok(b)
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: caller's contract; see `inner`.
        unsafe { self.inner().dealloc(ptr.as_ptr(), layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; see `inner`.
        unsafe { self.inner().resize(ptr.as_ptr(), old, new) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; see `inner`.
        unsafe { self.inner().resize(ptr.as_ptr(), old, new) }
    }
}
