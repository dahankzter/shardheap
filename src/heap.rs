//! [`LocalHeap`]: a single-threaded heap implementing [`Allocator`].

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::fmt;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use crate::class::{self, NUM_CLASSES};
use crate::os::{self, PAGE};
use crate::region::Region;
use crate::remote::{Link, Msg};
use crate::segmap;
use crate::segment::{self, Block, MAGIC, Page, SEGMENT_SIZE, SLOT_SIZE, Segment};

/// Empty segments kept mapped for reuse; further empty segments are unmapped.
const MAX_CACHED_SEGMENTS: usize = 1;
/// Segments kept by [`LocalHeap::reset`].
const MAX_RESET_SEGMENTS: usize = 4;
/// Slow-path events between purge checks.
const HEARTBEAT: u64 = 256;
/// Slow-path events a cached large mapping is kept before it is unmapped.
const PURGE_DELAY: u64 = 1024;
/// Slow-path events the last empty page of a class is kept before its slots go
/// back to the segment.
const RETIRE_DELAY: u64 = 256;
/// Purge epoch for free slots of segments, in milliseconds: a slot free for a
/// whole epoch is returned to the OS at its end; the mapping stays. Wall time rather
/// than slow-path events, which pass in microseconds under heavy allocation.
/// The same order as mimalloc's purge delay.
const EPOCH_MS: u64 = 10;
/// Longest purge epoch, as a power-of-two multiple of `EPOCH_MS`. The epoch doubles whenever purged slots are claimed again
/// within it (the purge cost a refault and saved nothing) and halves after
/// `CLEAN_EPOCHS` epochs in a row without that.
const MAX_EPOCH_SHIFT: u32 = 6;
/// See `MAX_EPOCH_SHIFT`. Halving after a single clean epoch oscillates: the
/// epoch after a refault purges nothing new, so it is always clean.
const CLEAN_EPOCHS: u32 = 4;
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

/// The purge clock; unit tests freeze it to run epochs by hand.
#[cfg(not(test))]
use os::now_ms;
#[cfg(test)]
fn now_ms() -> u64 {
    let _ = os::now_ms; // used outside tests
    0
}

/// Header at the start of every large mapping. Large mappings form an intrusive
/// list so [`LocalHeap::reset`] and `Drop` can release them.
#[repr(C)]
pub(crate) struct LargeHeader {
    next: *mut LargeHeader,
    prev: *mut LargeHeader,
    map_size: usize,
    map_align: usize,
    /// As `Segment::owner`.
    owner: *mut (),
    /// Link word while the block travels as a remote-free message.
    remote_next: *mut u8,
}

/// Pops a block from `page`'s `free` list, swapping in `local_free` when it is
/// empty, or returns null if both are. Collecting our own frees inline matters:
/// an alloc/free loop on one block would otherwise take the slow path every time.
///
/// # Safety
/// `page` is a live page whose heap is in use by this thread only.
#[inline(always)]
unsafe fn pop_free(page: *mut Page) -> *mut Block {
    // SAFETY: caller's contract.
    unsafe {
        let mut b = (*page).free;
        if b.is_null() {
            b = ptr::replace(&raw mut (*page).local_free, ptr::null_mut());
        }
        if !b.is_null() {
            // The free list holds pointers as they were freed, with whatever
            // provenance their last user's references narrowed them to; the
            // block handed out carries the page's (the segment's) instead.
            b = page.cast::<Block>().with_addr(b.addr());
            let n = (*b).next;
            prefetch(n.cast());
            (*page).free = n;
            (*page).used += 1;
        }
        b
    }
}

/// Starts loading the line at `p` for writing: the next block to hand out,
/// whose line another thread may hold since it freed the block. Never faults.
#[inline(always)]
fn prefetch(p: *const u8) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    // SAFETY: a prefetch is a hint; it never faults, whatever `p` is.
    unsafe {
        core::arch::asm!("prefetchw [{}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = p;
}

/// Tag bit of a large block's message: the header address with this bit set
/// (headers are page-aligned; small blocks are at least word-aligned, so the bit
/// is free in both).
pub(crate) const LARGE_MSG: usize = 1;

/// The message for the small block at `p`: the block itself, linked through
/// its first word.
///
/// # Safety
/// `p` is a live small block.
#[inline(always)]
pub(crate) unsafe fn small_msg(p: *mut u8) -> Msg {
    p
}

/// The small block at `p` with its segment's provenance instead of its
/// caller's, for a block handed back to a caller (an in-place resize). A
/// pointer a global allocator receives comes from std's references, narrowed
/// to the value they held (one byte for a `Box<u8>`), and the caller's new use
/// needs the whole block. A block being freed is written through the caller's
/// own pointer instead: during the free, the caller's tag is protected, and a
/// write through any other provenance would invalidate it (undefined behaviour
/// under both Stacked and Tree Borrows).
#[inline(always)]
pub(crate) fn own_small(p: *mut u8) -> *mut u8 {
    segment::segment_of(p).cast::<u8>().with_addr(p.addr())
}

/// Tag bit of a group message: several small blocks of one page, sent as one.
/// Blocks are 16-aligned, so the bit is free.
pub(crate) const GROUP_MSG: usize = 2;

/// Tag bit of a carrier message: a small block holding a list of other small
/// blocks of the same heap, sent as one. The sender writes only the carrier;
/// the listed blocks' lines stay where they are until their owner reuses them.
pub(crate) const CARRIER_MSG: usize = 4;

/// Tag bits a message may carry.
const MSG_TAGS: usize = LARGE_MSG | GROUP_MSG | CARRIER_MSG;

/// Words of a carrier before its list: the link, the count.
const CARRIER_HEADER: usize = 2;

/// How many blocks a carrier of `bytes` bytes (at most its block size) lists.
#[inline(always)]
pub(crate) fn carrier_capacity(bytes: usize) -> usize {
    (bytes / size_of::<usize>()).saturating_sub(CARRIER_HEADER)
}

/// The `i`th entry of the list in carrier `c`.
///
/// # Safety
/// `c` is a small block owned by the caller, with `i < carrier_capacity` of
/// its size.
#[inline(always)]
pub(crate) unsafe fn carrier_slot(c: *mut u8, i: usize) -> *mut *mut u8 {
    // SAFETY: caller's contract: inside the block.
    unsafe { c.cast::<*mut u8>().add(CARRIER_HEADER + i) }
}

/// The message for carrier `c` listing `count` blocks (`carrier_slot`): small,
/// untagged messages for blocks of the heap that owns `c`.
///
/// # Safety
/// As `carrier_slot`; entries `0..count` are written.
#[inline]
pub(crate) unsafe fn carrier_msg(c: *mut u8, count: usize) -> Msg {
    // SAFETY: caller's contract: the count word is inside the block.
    unsafe { *c.cast::<usize>().add(1) = count };
    c.map_addr(|a| a | CARRIER_MSG)
}

/// The batching key of a small message: blocks with equal keys lie in one
/// slot, hence one page, and may travel as one group (`group_msg`). `None` for
/// a large message.
///
/// # Safety
/// `m` is an untagged small message or a large message.
#[inline(always)]
pub(crate) unsafe fn group_key(m: Msg) -> Option<usize> {
    (m.addr() & LARGE_MSG == 0).then(|| m.addr() / segment::SLOT_SIZE)
}

/// The message for `count` small blocks of one page chained from `first` to
/// `last` through their first words (`last`'s is free): `last`, carrying the
/// count and the offset back to `first` in its second word, so the owner
/// splices the whole chain into the page touching `last`'s line only.
/// A chain of one is a plain small message.
///
/// # Safety
/// The blocks are live, owned by the caller, and share one `group_key`;
/// `1 <= count`, and `count == 1` exactly when `first == last`.
#[inline]
pub(crate) unsafe fn group_msg(first: *mut u8, last: *mut u8, count: usize) -> Msg {
    if count == 1 {
        return last;
    }
    // Within one 64 KiB slot: the offset fits in 16 signed bits counted in
    // 16-byte units, the count (at most 4096 blocks) in the 16 bits above.
    let units = ((first.addr() as isize - last.addr() as isize) / 16) as i16;
    let meta = (count << 16) | units as u16 as usize;
    // SAFETY: caller's contract: `last` is at least 16 bytes, its second word
    // is free.
    unsafe { *last.cast::<usize>().add(1) = meta };
    last.map_addr(|a| a | GROUP_MSG)
}

/// The message for the large block at `p`: its header, tagged, linked through
/// `LargeHeader::remote_next`.
///
/// # Safety
/// `p` is a live large block.
#[inline(always)]
pub(crate) unsafe fn large_msg(p: *mut u8) -> Msg {
    // SAFETY: caller's contract.
    unsafe { large_header(p).cast::<u8>().map_addr(|a| a | LARGE_MSG) }
}

/// The link word of a message: a small block's first word, a large header's
/// `remote_next`.
pub(crate) struct MsgLink;

impl MsgLink {
    #[inline(always)]
    fn word(m: Msg) -> *mut *mut u8 {
        if m.addr() & LARGE_MSG != 0 {
            let h = m.map_addr(|a| a & !LARGE_MSG).cast::<LargeHeader>();
            // SAFETY: no dereference; a field address inside the header.
            unsafe { &raw mut (*h).remote_next }
        } else {
            m.map_addr(|a| a & !MSG_TAGS).cast()
        }
    }
}

impl Link for MsgLink {
    unsafe fn next(m: Msg) -> Msg {
        // SAFETY: caller's contract: `m` is a live message.
        unsafe { *Self::word(m) }
    }
    unsafe fn set_next(m: Msg, next: Msg) {
        // SAFETY: caller's contract: `m` is a live message owned by the caller.
        unsafe { *Self::word(m) = next }
    }
}

/// The owner stamp of the large block at `p`.
///
/// # Safety
/// `p` is a live large block.
#[inline(always)]
unsafe fn large_owner(p: *mut u8) -> *mut () {
    // SAFETY: caller's contract.
    unsafe { (*large_header(p)).owner }
}

/// The header of the large block at `p`: the page right before the data,
/// whatever the alignment, so it is found without the `Layout`.
///
/// # Safety
/// `p` is a live large block.
#[inline(always)]
pub(crate) unsafe fn large_header(p: *mut u8) -> *mut LargeHeader {
    // SAFETY: caller's contract: the data starts at least a page into the
    // mapping (`large_offset`).
    let _ = p;
    ptr::with_exposed_provenance_mut(p.addr() - PAGE)
}

/// Start of the mapping whose header is `h`.
///
/// # Safety
/// `h` is a live large header.
#[inline(always)]
unsafe fn large_base(h: *mut LargeHeader) -> *mut u8 {
    // SAFETY: caller's contract: the data, a page after `h`, sits
    // `large_offset(map_align)` into the mapping.
    unsafe { h.cast::<u8>().add(PAGE).sub(large_offset((*h).map_align)) }
}

/// A block reached a heap that cannot take it: a `LocalHeap` block freed
/// through `ShardAlloc`. Safe code cannot get here, so this is memory
/// corruption or misuse of unsafe code: stop before a heap is damaged.
#[cold]
#[inline(never)]
pub(crate) fn foreign_free() -> ! {
    eprintln!("shardheap: block freed on a thread that does not own it");
    std::process::abort()
}

/// What a free without a size needs to know about a block (`block_of`).
pub(crate) struct BlockInfo {
    /// Owner stamp: the owning heap's address, or 0 for an explicit heap.
    pub(crate) owner: *mut (),
    /// Usable size in bytes.
    pub(crate) size: usize,
    /// The block's size class; `None` for a large block.
    pub(crate) class: Option<usize>,
    /// The block as a remote-free message.
    pub(crate) msg: Msg,
}

/// Owner, size and kind of the live block at `p`, from metadata alone: the
/// segment map tells small from large, then the segment's class table or the
/// large header gives the rest.
///
/// # Safety
/// `p` is a live block from a nonzero-size allocation of a heap whose segments
/// are in the segment map.
pub(crate) unsafe fn block_of(p: *mut u8) -> BlockInfo {
    // SAFETY: caller's contract: the segment header or the large header of a
    // live block is readable.
    unsafe {
        if segmap::is_segment(p) {
            let seg = segment::segment_of(p);
            let c = (*seg).classes[(p.addr() - seg.addr()) / SLOT_SIZE] as usize;
            BlockInfo {
                owner: (*seg).owner,
                size: class::size_of(c),
                class: Some(c),
                msg: small_msg(p),
            }
        } else {
            let h = large_header(p);
            BlockInfo {
                owner: (*h).owner,
                size: (*h).map_size - large_offset((*h).map_align),
                class: None,
                msg: large_msg(p),
            }
        }
    }
}

/// Distance from the start of a large mapping to the data. The header is the
/// last page of it (`large_header`).
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
    /// `os::now_ms` when the current purge epoch began; see `end_epoch`.
    epoch_start: u64,
    /// The purge epoch is `EPOCH_MS << epoch_shift` milliseconds.
    epoch_shift: u32,
    /// Whether purged slots were claimed again in the current epoch.
    refaulted: bool,
    /// Epochs in a row without a refault, up to `CLEAN_EPOCHS`.
    clean_epochs: u32,
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
    /// For a thread heap (`ShardAlloc`), this heap's own address, which never
    /// changes; null for an explicit heap. Stamped into every segment and large
    /// header.
    pub(crate) owner: *mut (),
    /// A thread heap's remote-free mail: sends its batches and drains its
    /// inbox, returning whether that freed any block. Called on every heartbeat
    /// that runs housekeeping, and before a size class takes a fresh page, so
    /// blocks freed elsewhere are reused while still warm. `None` for explicit
    /// heaps; only those two rare paths look at it.
    pub(crate) mail: Option<unsafe fn(&mut Inner) -> bool>,
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
            inner: UnsafeCell::new(Inner::new(HugePages::Auto)),
            _not_send: PhantomData,
        }
    }

    /// An empty heap with the given huge-page policy. [`new`](Self::new) uses
    /// [`HugePages::Auto`].
    pub const fn with_huge_pages(policy: HugePages) -> Self {
        LocalHeap {
            inner: UnsafeCell::new(Inner::new(policy)),
            _not_send: PhantomData,
        }
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

    /// Returns cached memory to the OS now, instead of waiting for the purge
    /// delay: unmaps every empty segment and the large-mapping cache, and purges
    /// the free slots of segments still in use (splitting huge pages if need be).
    /// Region memory stays mapped and resident.
    pub fn trim(&self) {
        // SAFETY: see `inner`.
        unsafe { self.inner().trim() }
    }

    /// Runs the heap's housekeeping now, as the allocation slow path does from
    /// time to time: memory free for a whole purge epoch (10 ms, longer while
    /// purged memory keeps being reused soon after) goes back to the OS, and
    /// counts as a burst of slow-path activity for the other delays.
    ///
    /// Housekeeping otherwise runs only while the heap allocates, so a heap that
    /// goes quiet keeps what it freed. Call this when the thread goes idle, for
    /// example from an event loop's idle hook. Unlike [`trim`](Self::trim), it
    /// leaves recently freed memory resident for reuse.
    pub fn maintain(&self) {
        // SAFETY: see `inner`.
        unsafe { self.inner().maintain() }
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
    /// An empty heap with the given huge-page policy, owned by no thread record.
    pub(crate) const fn new(huge_pages: HugePages) -> Inner {
        Inner {
            queues: [ptr::null_mut(); NUM_CLASSES],
            segments: ptr::null_mut(),
            large: ptr::null_mut(),
            large_cache: [NO_MAP; LARGE_CACHE_SLOTS],
            large_cache_bytes: 0,
            tick: 0,
            epoch_start: 0,
            epoch_shift: 0,
            refaulted: false,
            clean_epochs: 0,
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
            huge_pages,
            huge_now: false,
            region: None,
            owner: ptr::null_mut(),
            mail: None,
        }
    }

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
                let b = pop_free(page);
                if !b.is_null() {
                    let bs = (*page).block_size;
                    self.stats.allocated_bytes += bs;
                    return Ok(slice(b.cast(), bs));
                }
            }
            self.alloc_small_slow(c)
        }
    }

    /// The small-block fast path through a raw pointer, for a heap that may be
    /// the shared, read-only empty heap of `ShardAlloc`: `this` is only read
    /// until a page in its queue yields a block, and the empty heap's queues
    /// are all null. `None` means: take the slow path (on a real heap).
    ///
    /// # Safety
    /// `this` is a live heap with no other reference in use, or a heap whose
    /// queues are all null.
    #[inline(always)]
    pub(crate) unsafe fn alloc_fast(this: *mut Inner, layout: Layout) -> Option<NonNull<[u8]>> {
        if layout.size() == 0 {
            return Some(dangling(layout.align()));
        }
        let c = class::class_for(layout)?;
        // SAFETY: caller's contract; `c < NUM_CLASSES`; queue pages are live
        // pages of `this`, so a non-null one means `this` is a real heap.
        unsafe {
            let page = *(&raw const (*this).queues).cast::<*mut Page>().add(c);
            if page.is_null() {
                return None;
            }
            let b = pop_free(page);
            if b.is_null() {
                return None;
            }
            let bs = (*page).block_size;
            (*this).stats.allocated_bytes += bs;
            Some(slice(b.cast(), bs))
        }
    }

    /// `alloc` after `alloc_fast` returned `None` on this heap: straight to the
    /// slow path, without repeating the fast-path attempt.
    ///
    /// # Safety
    /// `layout` is nonzero.
    #[inline(always)]
    pub(crate) unsafe fn alloc_miss(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: nonzero layout; `c` comes from `class_for`.
        unsafe {
            match class::class_for(layout) {
                Some(c) => self.alloc_small_slow(c),
                None => self.alloc_large(layout).map(|(b, _)| b),
            }
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn alloc_small_slow(&mut self, c: usize) -> Result<NonNull<[u8]>, AllocError> {
        self.heartbeat();
        // SAFETY: every page reached through the queue is a live page of this heap.
        unsafe {
            let mut mail = self.mail;
            loop {
                let page = self.queues[c];
                if page.is_null() {
                    // Out of pages: collect remote frees before fresh memory.
                    match mail.take() {
                        Some(f) if f(self) => continue,
                        _ => break,
                    }
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
                // As in `pop_free`: the page's provenance, not the last user's.
                let b = page.cast::<Block>().with_addr(b.addr());
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
            let seg = segment::segment_of((*page).start);
            let first = ((*page).start.addr() - seg.addr()) / SLOT_SIZE;
            for slot in first..first + (*page).span as usize {
                (*seg).classes[slot] = c as u8;
            }
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
            }
            let mask = segment::span_mask(slot, k);
            (*seg).used_slots |= mask;
            (*seg).dirty_slots &= !mask;
            (*seg).stale_slots &= !mask;
            if (*seg).purged_slots & mask != 0 {
                self.refaulted = true;
                (*seg).purged_slots &= !mask;
            }
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
            (*seg).owner = self.owner;
            segmap::mark(seg);
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

    /// `alloc` with the block zeroed; fresh large mappings are not cleared twice.
    pub(crate) unsafe fn alloc_zeroed(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: `layout` takes the path matching its class, as in `alloc`.
        unsafe {
            if layout.size() != 0 && class::class_for(layout).is_none() {
                let (b, fresh) = self.alloc_large(layout)?;
                if !fresh {
                    // A reused mapping; fresh ones are already zero.
                    ptr::write_bytes(b.as_ptr().cast::<u8>(), 0, b.len());
                }
                return Ok(b);
            }
            let b = self.alloc(layout)?;
            ptr::write_bytes(b.as_ptr().cast::<u8>(), 0, b.len());
            Ok(b)
        }
    }

    // ---- deallocation ------------------------------------------------------

    /// Frees the live block at `p` without its size: like `dealloc_owned`, with
    /// the block's owner, size and kind from `block_of`. `foreign` receives the
    /// owner, the message and the block's size.
    ///
    /// # Safety
    /// As `dealloc_owned`; `p` came from a nonzero-size allocation of a heap
    /// whose segments are in the segment map.
    pub(crate) unsafe fn dealloc_unsized(
        this: *mut Inner,
        p: *mut u8,
        foreign: impl FnOnce(*mut (), Msg, usize),
    ) {
        // SAFETY: caller's contract; `this` is dereferenced only once it
        // matches the block's owner stamp.
        unsafe {
            let b = block_of(p);
            if b.owner != this.cast() {
                return foreign(b.owner, b.msg, b.size);
            }
            match b.class {
                Some(c) => (*this).free_small(p, c),
                None => (*this).free_large(p),
            }
        }
    }

    /// `dealloc` for a heap reached without proof of ownership (`ShardAlloc`).
    /// A thread heap's owner stamp is its own address, so the check compares the
    /// block's stamp with `this` before touching `this`. A block of another heap
    /// (or any block, if `this` is an empty heap that owns nothing) goes to
    /// `foreign(owner, message)` instead, as a remote-free message.
    ///
    /// # Safety
    /// As `dealloc`, except that the block may belong to another heap; `this` is
    /// a live heap whose `owner` is its own address, or a heap that owns nothing.
    #[inline(always)]
    pub(crate) unsafe fn dealloc_owned(
        this: *mut Inner,
        p: *mut u8,
        layout: Layout,
        foreign: impl FnOnce(*mut (), Msg),
    ) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: caller's contract: `p` is a live block allocated with `layout`
        // by some heap, so its page or large header is readable; `this` is
        // dereferenced only once it matches the block's owner stamp.
        unsafe {
            match class::class_for(layout) {
                Some(c) => {
                    // The segment header's stamp: a line the owner rarely writes,
                    // unlike the page descriptor (its free list and counts).
                    let owner = (*segment::segment_of(p)).owner;
                    if owner != this.cast() {
                        return foreign(owner, small_msg(p));
                    }
                    (*this).free_small(p, c);
                }
                None => {
                    let owner = large_owner(p);
                    if owner != this.cast() {
                        return foreign(owner, large_msg(p));
                    }
                    (*this).free_large(p);
                }
            }
        }
    }

    /// The owner stamp of the live block at `p`, allocated with `layout`, and its
    /// remote-free message.
    ///
    /// # Safety
    /// `p` is a live block of some heap, allocated with nonzero `layout`.
    pub(crate) unsafe fn owner_of(p: *mut u8, layout: Layout) -> (*mut (), Msg) {
        // SAFETY: caller's contract.
        unsafe {
            match class::class_for(layout) {
                Some(_) => ((*segment::segment_of(p)).owner, small_msg(p)),
                None => (large_owner(p), large_msg(p)),
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn dealloc(&mut self, p: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        match class::class_for(layout) {
            // SAFETY: caller's contract: `p` was allocated here with a fitting layout.
            Some(c) => unsafe { self.free_small(p, c) },
            None => unsafe { self.free_large(p) },
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
            self.free_block(page, p);
        }
    }

    /// Frees `p` into its page: `free_small` once the page is known.
    ///
    /// # Safety
    /// `p` is a live small block of this heap, in `page`.
    #[inline(always)]
    unsafe fn free_block(&mut self, page: *mut Page, p: *mut u8) {
        // SAFETY: caller's contract.
        unsafe {
            // The caller's pointer, not the page's: see `own_small`.
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

    /// Frees one message sent by another thread, as a local free: the page and
    /// class of a small block come from its segment, the size and alignment of
    /// a large one from its header.
    ///
    /// # Safety
    /// `m` is a message (`small_msg`/`large_msg`) for a live block of this heap,
    /// no longer used by anyone.
    pub(crate) unsafe fn free_msg(&mut self, m: Msg) {
        // SAFETY: caller's contract.
        unsafe {
            if m.addr() & LARGE_MSG != 0 {
                self.free_large_header(m.map_addr(|a| a & !LARGE_MSG).cast());
            } else if m.addr() & GROUP_MSG != 0 {
                self.free_group(m.map_addr(|a| a & !GROUP_MSG));
            } else if m.addr() & CARRIER_MSG != 0 {
                self.free_carrier(m.map_addr(|a| a & !CARRIER_MSG));
            } else {
                let page = segment::page_of(segment::segment_of(m), m);
                self.free_block(page, m);
            }
        }
    }

    /// Frees a carrier (`carrier_msg`): every block it lists, then itself. The
    /// listed blocks' lines are requested all at once first, so their misses
    /// overlap.
    ///
    /// # Safety
    /// As `free_msg`; `c` is an untagged carrier message.
    unsafe fn free_carrier(&mut self, c: *mut u8) {
        // SAFETY: caller's contract: `c` lists `count` live blocks of this
        // heap; the list is read before `c` itself is freed.
        unsafe {
            let count = *c.cast::<usize>().add(1);
            for i in 0..count {
                prefetch(*carrier_slot(c, i));
            }
            for i in 0..count {
                let p = *carrier_slot(c, i);
                self.free_block(segment::page_of(segment::segment_of(p), p), p);
            }
            self.free_block(segment::page_of(segment::segment_of(c), c), c);
        }
    }

    /// Frees a group message's chain (`group_msg`) in one splice: `last`'s
    /// first word takes the page's local free list, the chain's first block
    /// becomes its head.
    ///
    /// # Safety
    /// As `free_msg`; `last` is an untagged group message.
    unsafe fn free_group(&mut self, last: *mut u8) {
        // SAFETY: caller's contract: `last` heads a well-formed chain of
        // `count` blocks of one page of this heap.
        unsafe {
            let meta = *last.cast::<usize>().add(1);
            let count = meta >> 16;
            let first = last.wrapping_offset(meta as u16 as i16 as isize * 16);
            let page = segment::page_of(segment::segment_of(last), last);
            (*last.cast::<Block>()).next = (*page).local_free;
            (*page).local_free = first.cast();
            (*page).used -= count as u32;
            self.stats.allocated_bytes -= count * (*page).block_size;
            if !(*page).in_queue || (*page).used == 0 {
                self.free_small_slow(page);
            }
        }
    }

    /// Frees up to `limit` messages along the list at `m` (linked through
    /// `MsgLink`) and returns the rest, null when it is all freed.
    ///
    /// # Safety
    /// Every message of the list is as for `free_msg`.
    pub(crate) unsafe fn free_msgs(&mut self, mut m: Msg, limit: usize) -> Msg {
        let mut n = 0;
        while !m.is_null() && n < limit {
            // SAFETY: caller's contract; the link is read before the free
            // overwrites it.
            unsafe {
                let next = MsgLink::next(m);
                self.free_msg(m);
                m = next;
            }
            n += 1;
        }
        m
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
            let mask = segment::span_mask(slot, k);
            (*seg).used_slots &= !mask;
            if !self.in_region(seg.cast()) {
                // Region memory is pinned and never purged.
                (*seg).dirty_slots |= mask;
            }
            if (*seg).used_slots == 1 {
                self.segment_became_empty(seg);
            }
        }
    }

    unsafe fn segment_became_empty(&mut self, seg: *mut Segment) {
        // SAFETY: `seg` is a live, empty segment of this heap.
        unsafe {
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
            segmap::unmark(seg);
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
            // SAFETY: walks only this heap's live segments; `mail` is set only
            // by `ShardAlloc` for its own thread heaps.
            unsafe {
                self.purge_idle();
                if let Some(f) = self.mail {
                    f(self);
                }
            }
        }
    }

    pub(crate) unsafe fn maintain(&mut self) {
        self.tick += HEARTBEAT;
        // SAFETY: as `heartbeat`.
        unsafe { self.purge_idle() }
    }

    /// Releases long-idle kept pages and cached large mappings, and ends the
    /// purge epoch if it is over.
    unsafe fn purge_idle(&mut self) {
        // SAFETY: queues and the segment list hold live pages/segments of this heap.
        unsafe {
            self.release_kept_pages(RETIRE_DELAY);
            self.evict_large_cache(PURGE_DELAY);
            let now = now_ms();
            if now.wrapping_sub(self.epoch_start) >= EPOCH_MS << self.epoch_shift {
                self.epoch_start = now;
                self.end_epoch();
            }
        }
    }

    /// Ends a purge epoch: adapts its length, then purges the free slots that
    /// were idle for all of it (in empty segments too: their mapping stays) and
    /// marks those freed during it stale.
    unsafe fn end_epoch(&mut self) {
        if self.refaulted {
            self.epoch_shift = (self.epoch_shift + 1).min(MAX_EPOCH_SHIFT);
            self.clean_epochs = 0;
        } else {
            self.clean_epochs += 1;
            if self.clean_epochs == CLEAN_EPOCHS {
                self.epoch_shift = self.epoch_shift.saturating_sub(1);
                self.clean_epochs = 0;
            }
        }
        self.refaulted = false;
        // SAFETY: the segment list holds live segments of this heap; dirty and
        // stale slots are free and outside the region.
        unsafe {
            let mut seg = self.segments;
            while !seg.is_null() {
                let purged = self.purge_free(seg, (*seg).stale_slots, false);
                (*seg).purged_slots |= purged;
                (*seg).stale_slots = ((*seg).stale_slots & !purged) | (*seg).dirty_slots;
                (*seg).dirty_slots = 0;
                seg = (*seg).next;
            }
        }
    }

    /// Returns free slots among `candidates` of a segment to the OS and returns
    /// the mask purged. With huge pages, a partial purge of a segment in use
    /// would split a 2 MiB page (and `khugepaged` may fill the hole again when it
    /// collapses the range), so unless `split` allows that, only a wholly free
    /// upper half is purged; the lower half holds the header. An empty segment
    /// is purged whole: only its header is still in use.
    ///
    /// # Safety
    /// `seg` is a live segment of this heap; `candidates` are free slots of it,
    /// outside the region.
    unsafe fn purge_free(&mut self, seg: *mut Segment, candidates: u64, split: bool) -> u64 {
        const UPPER: u64 = !0 << (segment::SLOTS / 2);
        // SAFETY: caller's contract; free slots hold nothing.
        unsafe {
            let purge = if split || (*seg).used_slots == 1 || !self.segments_huge() {
                candidates
            } else if (*seg).used_slots & UPPER == 0 && candidates & UPPER != 0 {
                UPPER
            } else {
                0
            };
            segment::for_each_run(purge, |first, n| {
                let p = NonNull::new_unchecked(seg.cast::<u8>().add(first * SLOT_SIZE));
                os::purge(p, n * SLOT_SIZE);
            });
            purge
        }
    }

    /// Whether segments are (or may be made of) huge pages by this heap's advice.
    fn segments_huge(&self) -> bool {
        match self.huge_pages {
            HugePages::Auto => self.huge_now,
            HugePages::Always => true,
            HugePages::Never | HugePages::Kernel => false,
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

    /// What an empty heap keeps for the thread that adopts it: its pages back
    /// in their segments (so at most one empty segment stays mapped) and
    /// nothing else; cached large mappings and the reap page are released.
    pub(crate) unsafe fn keep_warm(&mut self) {
        // SAFETY: as `trim`.
        unsafe {
            self.release_kept_pages(0);
            self.release_idle_reap_page();
            self.evict_large_cache(0);
        }
    }

    pub(crate) unsafe fn trim(&mut self) {
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
                } else {
                    let free = (*seg).dirty_slots | (*seg).stale_slots;
                    // Not recorded in `purged_slots`: reuse after an explicit trim
                    // says nothing about the purge epoch.
                    let purged = self.purge_free(seg, free, true);
                    (*seg).dirty_slots &= !purged;
                    (*seg).stale_slots &= !purged;
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
                        if !self.in_region(seg.cast()) {
                            (*seg).dirty_slots |= (*seg).used_slots & !1;
                        }
                        (*seg).used_slots = 1;
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
                segmap::unmark(seg);
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

    /// Writes a large header a page before the data (`off` into the mapping at
    /// `base`), links it, and returns the data block.
    unsafe fn link_large(
        &mut self,
        base: NonNull<u8>,
        map_size: usize,
        map_align: usize,
        off: usize,
    ) -> NonNull<[u8]> {
        // SAFETY: `base` is an unused mapping of `map_size` bytes owned by this heap.
        unsafe {
            let h = base.as_ptr().add(off - PAGE).cast::<LargeHeader>();
            let _ = h.expose_provenance();
            h.write(LargeHeader {
                next: self.large,
                prev: ptr::null_mut(),
                map_size,
                map_align,
                owner: self.owner,
                remote_next: ptr::null_mut(),
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

    unsafe fn free_large(&mut self, p: *mut u8) {
        // SAFETY: `p` was returned by `alloc_large` (or `grow_large`).
        unsafe { self.free_large_header(large_header(p)) }
    }

    /// Frees the large mapping whose header is `h`.
    ///
    /// # Safety
    /// `h` heads a live large mapping of this heap that nothing uses any more.
    unsafe fn free_large_header(&mut self, h: *mut LargeHeader) {
        // SAFETY: caller's contract.
        unsafe {
            self.unlink_large(h);
            let (size, align, base) = ((*h).map_size, (*h).map_align, large_base(h));
            self.stats.large_allocations -= 1;
            self.stats.allocated_bytes -= size;
            if align != PAGE || !self.put_cached(base, size) {
                self.stats.mapped_bytes -= size;
                self.unmap(NonNull::new_unchecked(base), size, align);
            }
        }
    }

    unsafe fn release_large(&mut self) {
        // SAFETY: called only when no blocks are live.
        unsafe {
            let mut h = self.large;
            while !h.is_null() {
                let next = (*h).next;
                let (size, align, base) = ((*h).map_size, (*h).map_align, large_base(h));
                self.stats.allocated_bytes -= size;
                self.stats.mapped_bytes -= size;
                self.unmap(NonNull::new_unchecked(base), size, align);
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
                let h = large_header(p);
                // The block with the mapping's provenance, not the caller's.
                let p = h.cast::<u8>().add(PAGE);
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
                    self.free_large(p);
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
                (Some(a), Some(b)) if a == b => Ok(slice(own_small(p), class::size_of(a))),
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
        unsafe { self.inner().alloc_zeroed(layout) }
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

#[cfg(test)]
mod tests {
    use super::*;

    const L: Layout = Layout::new::<[u8; 1024]>();

    /// A heap with four pages of 1 KiB blocks (the fourth keeps a page of the
    /// class in the queue, so emptied pages are released at once), and the
    /// blocks of the second page freed. Returns the heap, the live blocks and
    /// the slot mask of the freed page.
    fn heap_with_a_free_page() -> (LocalHeap, Vec<NonNull<u8>>, *mut Segment, u64) {
        let heap = LocalHeap::with_huge_pages(HugePages::Never);
        let c = class::class_for(L).unwrap();
        let per_page = class::capacity(c);
        let mut blocks: Vec<_> = (0..3 * per_page + 1)
            .map(|i| {
                let p = heap.allocate(L).unwrap().cast::<u8>();
                // SAFETY: fresh block of 1 KiB.
                unsafe { p.as_ptr().write_bytes(i as u8, L.size()) };
                p
            })
            .collect();
        let seg = segment::segment_of(blocks[0].as_ptr());
        let slot = (blocks[per_page].addr().get() - seg.addr()) / SLOT_SIZE;
        let mask = segment::span_mask(slot, class::span_slots(c));
        for b in blocks.drain(per_page..2 * per_page) {
            // SAFETY: allocated above with `L`.
            unsafe { heap.deallocate(b, L) };
        }
        (heap, blocks, seg, mask)
    }

    fn end_epoch(heap: &LocalHeap) {
        // SAFETY: no other reference to the inner heap is live.
        unsafe { heap.inner().end_epoch() }
    }

    fn slots(seg: *mut Segment) -> (u64, u64, u64) {
        // SAFETY: a live segment of the test's heap.
        unsafe { ((*seg).dirty_slots, (*seg).stale_slots, (*seg).purged_slots) }
    }

    fn epoch_shift(heap: &LocalHeap) -> u32 {
        // SAFETY: no other reference to the inner heap is live.
        unsafe { heap.inner().epoch_shift }
    }

    #[test]
    fn free_slots_are_purged_after_a_whole_idle_epoch() {
        let (heap, blocks, seg, mask) = heap_with_a_free_page();
        assert_eq!(slots(seg), (mask, 0, 0), "released slots start dirty");
        end_epoch(&heap);
        assert_eq!(slots(seg), (0, mask, 0), "then stale for one epoch");
        end_epoch(&heap);
        assert_eq!(slots(seg), (0, 0, mask), "then purged");
        assert_eq!(epoch_shift(&heap), 0);

        // Claiming the purged slots again is a refault: the epoch doubles, then
        // shrinks back after an epoch without one.
        let again: Vec<_> = (0..class::capacity(class::class_for(L).unwrap()))
            .map(|_| heap.allocate(L).unwrap().cast::<u8>())
            .collect();
        assert_eq!(slots(seg), (0, 0, 0));
        end_epoch(&heap);
        assert_eq!(epoch_shift(&heap), 1);
        for _ in 1..CLEAN_EPOCHS {
            end_epoch(&heap);
            assert_eq!(epoch_shift(&heap), 1, "one clean epoch is not enough");
        }
        end_epoch(&heap);
        assert_eq!(epoch_shift(&heap), 0);

        for b in blocks.into_iter().chain(again) {
            // SAFETY: allocated with `L`.
            unsafe { heap.deallocate(b, L) };
        }
    }

    #[test]
    fn claimed_slots_leave_every_set() {
        let (heap, blocks, seg, mask) = heap_with_a_free_page();
        end_epoch(&heap);
        assert_eq!(slots(seg), (0, mask, 0));
        // The fourth page takes all but one; the rest needs a new page, which
        // takes the lowest free slots: the freed page's.
        let again: Vec<_> = (0..class::capacity(class::class_for(L).unwrap()))
            .map(|_| heap.allocate(L).unwrap().cast::<u8>())
            .collect();
        assert_eq!(slots(seg), (0, 0, 0), "the free page's slots were claimed");
        assert!(
            // SAFETY: no other reference to the inner heap is live.
            !unsafe { heap.inner().refaulted },
            "stale slots were still resident"
        );
        for b in blocks.into_iter().chain(again) {
            // SAFETY: allocated with `L`.
            unsafe { heap.deallocate(b, L) };
        }
    }

    /// The owner stamp of the block at `p`, allocated with `l`.
    fn owner_of(p: NonNull<u8>, l: Layout) -> *mut () {
        // SAFETY: a live block of a heap, allocated with `l`.
        unsafe {
            if class::class_for(l).is_some() {
                (*segment::segment_of(p.as_ptr())).owner
            } else {
                (*p.as_ptr().sub(PAGE).cast::<LargeHeader>()).owner
            }
        }
    }

    const OWNER_LAYOUTS: [(usize, usize); 3] = [(64, 8), (1 << 20, 8), (300_000, 1 << 16)];

    #[test]
    fn explicit_heaps_stamp_no_owner() {
        let heap = LocalHeap::new();
        for (size, align) in OWNER_LAYOUTS {
            let l = Layout::from_size_align(size, align).unwrap();
            let p = heap.allocate(l).unwrap().cast::<u8>();
            assert!(owner_of(p, l).is_null());
            // SAFETY: allocated above with `l`.
            unsafe { heap.deallocate(p, l) };
        }
    }

    #[test]
    fn owned_frees_and_resizes_pass_the_owner_check() {
        let heap = LocalHeap::new();
        let this = heap.inner.get();
        // SAFETY: no other reference to the inner heap is live; the heap does
        // not move during the test, as a thread heap never moves.
        unsafe { (*this).owner = this.cast() };
        for (size, align) in OWNER_LAYOUTS {
            let l = Layout::from_size_align(size, align).unwrap();
            let p = heap.allocate(l).unwrap().cast::<u8>();
            assert_eq!(owner_of(p, l), this.cast());
            // SAFETY: `p` is live with `l`; no other reference to the inner heap.
            assert_eq!(unsafe { Inner::owner_of(p.as_ptr(), l) }.0, this.cast());
            let big = Layout::from_size_align(size * 2, align).unwrap();
            // SAFETY: as above.
            unsafe {
                let q = (*this).resize(p.as_ptr(), l, big).unwrap();
                assert_eq!(owner_of(q.cast(), big), this.cast());
                Inner::dealloc_owned(this, q.as_ptr().cast(), big, |_, _| {
                    panic!("an own block is not foreign")
                });
            }
        }
        assert_eq!(heap.stats().allocated_bytes, 0);
    }

    /// Turns live blocks into one message list, linked through `MsgLink`.
    fn message_list(blocks: &[(NonNull<u8>, Layout)]) -> Msg {
        let msgs: Vec<Msg> = blocks
            .iter()
            .map(|&(p, l)| {
                // SAFETY: live blocks of the test's heap.
                unsafe {
                    if class::class_for(l).is_some() {
                        small_msg(p.as_ptr())
                    } else {
                        large_msg(p.as_ptr())
                    }
                }
            })
            .collect();
        for w in msgs.windows(2) {
            // SAFETY: each message is owned here.
            unsafe { <MsgLink as crate::remote::Link>::set_next(w[0], w[1]) };
        }
        // SAFETY: as above.
        unsafe {
            <MsgLink as crate::remote::Link>::set_next(*msgs.last().unwrap(), ptr::null_mut())
        };
        msgs[0]
    }

    #[test]
    fn messages_free_like_local_frees() {
        let heap = LocalHeap::new();
        let mut blocks = Vec::new();
        for i in 0..300 {
            let l = Layout::from_size_align(8 + (i * 37) % 2000, 8).unwrap();
            blocks.push((heap.allocate(l).unwrap().cast::<u8>(), l));
        }
        for (size, align) in [(1 << 20, 8), (300_000, 1 << 16)] {
            let l = Layout::from_size_align(size, align).unwrap();
            blocks.push((heap.allocate(l).unwrap().cast::<u8>(), l));
        }
        let list = message_list(&blocks);
        // SAFETY: the list holds live blocks of this heap; nothing else uses them.
        unsafe {
            let rest = heap.inner().free_msgs(list, 256);
            assert!(!rest.is_null(), "bounded: 302 messages, 256 freed");
            assert!(heap.stats().allocated_bytes > 0);
            assert!(heap.inner().free_msgs(rest, 256).is_null());
        }
        let s = heap.stats();
        assert_eq!((s.allocated_bytes, s.large_allocations), (0, 0));
    }

    #[test]
    fn freed_message_blocks_are_reused() {
        let heap = LocalHeap::new();
        let l = Layout::new::<[u8; 64]>();
        let keep = heap.allocate(l).unwrap(); // keeps the page in use
        let p = heap.allocate(l).unwrap().cast::<u8>();
        // SAFETY: `p` is a live block of this heap.
        unsafe { heap.inner().free_msg(small_msg(p.as_ptr())) };
        assert_eq!(heap.allocate(l).unwrap().cast::<u8>(), p);
        // SAFETY: allocated above with `l`.
        unsafe {
            heap.deallocate(p, l);
            heap.deallocate(keep.cast(), l);
        }
    }

    #[test]
    fn large_header_sits_one_page_before_data_for_every_alignment() {
        let heap = LocalHeap::new();
        for align in [8, PAGE, 2 * PAGE, 1 << 20, 4 << 20, 8 << 20] {
            let l = Layout::from_size_align(300_000, align).unwrap();
            let p = heap.allocate(l).unwrap().cast::<u8>().as_ptr();
            assert_eq!(p.addr() % align, 0);
            // SAFETY: a live large block of this heap.
            unsafe {
                let h = large_header(p);
                assert_eq!(h.addr(), p.addr() - PAGE, "align {align}");
                assert_eq!((*h).map_align, align.max(PAGE));
                heap.deallocate(NonNull::new(p).unwrap(), l);
            }
        }
        assert_eq!(heap.stats().large_allocations, 0);
    }

    #[test]
    fn a_carrier_message_frees_every_block_it_lists() {
        let heap = LocalHeap::new();
        let l = Layout::new::<[u8; 1024]>();
        let keep = heap.allocate(l).unwrap(); // keeps the page in use
        let c = heap.allocate(l).unwrap().cast::<u8>().as_ptr();
        let listed: Vec<*mut u8> = (0..8)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        assert!(carrier_capacity(l.size()) >= listed.len());
        // SAFETY: the blocks are live blocks of this heap, owned here; `c`
        // has room for the list.
        unsafe {
            for (i, &p) in listed.iter().enumerate() {
                *carrier_slot(c, i) = p;
            }
            let m = carrier_msg(c, listed.len());
            <MsgLink as crate::remote::Link>::set_next(m, ptr::null_mut());
            assert!(heap.inner().free_msgs(m, 1).is_null(), "one message");
        }
        assert_eq!(heap.stats().allocated_bytes, l.size());
        let mut again: Vec<*mut u8> = (0..9)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        let mut want = listed.clone();
        want.push(c);
        again.sort();
        want.sort();
        assert_eq!(again, want, "the carrier and its listed blocks are reused");
        // SAFETY: allocated above with `l`.
        unsafe {
            for b in again {
                heap.deallocate(NonNull::new(b).unwrap(), l);
            }
            heap.deallocate(keep.cast(), l);
        }
        assert_eq!(heap.stats().allocated_bytes, 0);
    }

    #[test]
    fn a_group_message_frees_its_whole_chain() {
        let heap = LocalHeap::new();
        let l = Layout::new::<[u8; 1024]>();
        let keep = heap.allocate(l).unwrap(); // keeps the page in use
        let blocks: Vec<*mut u8> = (0..8)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        // SAFETY: live small blocks of this heap.
        let key = |b| unsafe { group_key(small_msg(b)) };
        assert!(key(blocks[0]).is_some());
        assert!(blocks.iter().all(|&b| key(b) == key(blocks[0])));
        // SAFETY: the blocks are owned here; the chain runs from `blocks[7]`
        // down to `blocks[0]`, linked through their first words.
        unsafe {
            for w in blocks.windows(2) {
                *w[1].cast::<*mut u8>() = w[0];
            }
            let m = group_msg(blocks[7], blocks[0], 8);
            <MsgLink as crate::remote::Link>::set_next(m, ptr::null_mut());
            assert!(heap.inner().free_msgs(m, 1).is_null(), "one message");
        }
        assert_eq!(heap.stats().allocated_bytes, l.size());
        let mut again: Vec<*mut u8> = (0..8)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        let mut want = blocks.clone();
        again.sort();
        want.sort();
        assert_eq!(again, want, "the chain's blocks are reused");
        // SAFETY: allocated above with `l`.
        unsafe {
            for b in again {
                heap.deallocate(NonNull::new(b).unwrap(), l);
            }
            heap.deallocate(keep.cast(), l);
        }
        assert_eq!(heap.stats().allocated_bytes, 0);
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn trim_returns_free_slots_of_a_segment_in_use() {
        fn resident_pages(p: *mut u8, len: usize) -> usize {
            let mut v = vec![0u8; len / PAGE];
            // SAFETY: a mapped, page-aligned range; `v` has one byte per page.
            assert_eq!(unsafe { libc::mincore(p.cast(), len, v.as_mut_ptr()) }, 0);
            v.iter().filter(|&&b| b & 1 != 0).count()
        }
        let (heap, blocks, seg, mask) = heap_with_a_free_page();
        let first = mask.trailing_zeros() as usize;
        let n = mask.count_ones() as usize;
        // SAFETY: slots of the live segment.
        let free = unsafe { seg.cast::<u8>().add(first * SLOT_SIZE) };
        assert!(resident_pages(free, n * SLOT_SIZE) > 0);
        heap.trim();
        assert_eq!(heap.stats().segments, 1, "the segment stays mapped");
        assert_eq!(resident_pages(free, n * SLOT_SIZE), 0, "free slots purged");
        assert_eq!(
            slots(seg),
            (0, 0, 0),
            "an explicit trim is not a purge epoch"
        );
        let per_page = class::capacity(class::class_for(L).unwrap());
        for (i, b) in blocks.into_iter().enumerate() {
            let tag = if i < per_page { i } else { i + per_page } as u8;
            // SAFETY: a live block of 1 KiB.
            let s = unsafe { core::slice::from_raw_parts(b.as_ptr(), L.size()) };
            assert!(
                s.iter().all(|&x| x == tag),
                "live blocks keep their contents"
            );
            // SAFETY: allocated with `L`.
            unsafe { heap.deallocate(b, L) };
        }
    }
}
