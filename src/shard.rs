//! [`ShardAlloc`]: an allocator handle for the current thread's own heap.
//!
//! Every thread that uses `ShardAlloc` gets a *record*: heap state at a stable
//! address, mapped from the OS (not from the global allocator) and never freed,
//! so a block's owner stamp always names a valid record. Records are linked into
//! a static registry, and a const thread-local points at the current thread's.

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::{Cell, UnsafeCell};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::heap::{HugePages, Inner, Stats};
use crate::os::{self, PAGE};

/// A thread's heap, at an address that stays valid for the life of the process.
/// `repr(C)` with the heap first, so a record's address is its heap's address
/// (`UnsafeCell` is `repr(transparent)`): the thread-local value is the owner
/// stamp itself.
#[repr(C)]
pub(crate) struct Record {
    inner: UnsafeCell<Inner>,
    /// Next record in `RECORDS`.
    next: *mut Record,
}

/// Every record ever made, newest first. Push-only; keeps records reachable from
/// a static, and is the list stage B walks.
static RECORDS: AtomicPtr<Record> = AtomicPtr::new(ptr::null_mut());

/// The empty heap: every thread's `CURRENT` until its first allocation. Its
/// queues are all null, so allocation through it falls into the slow path,
/// which makes the thread's record; no block is stamped with its address, so a
/// free through it fails the owner check. Nothing ever writes it, nor takes a
/// `&mut` to it: it is only read, through raw pointers, from any thread. That
/// saves the null check on the allocation fast path (mimalloc's empty heap).
#[repr(transparent)]
struct Empty(Record);

// SAFETY: never written after const initialization; only read.
unsafe impl Sync for Empty {}

static EMPTY: Empty = Empty(Record {
    inner: UnsafeCell::new(Inner::new(HugePages::Auto)),
    next: ptr::null_mut(),
});

#[inline(always)]
fn empty() -> *mut Record {
    (&raw const EMPTY.0).cast_mut()
}

thread_local! {
    /// The current thread's record, or the empty heap before its first use.
    /// Const and without a destructor, so reading it is a plain thread-local
    /// load.
    static CURRENT: Cell<*mut Record> = const { Cell::new((&raw const EMPTY.0).cast_mut()) };

    /// Destroyed at thread exit; registered on the thread's first use.
    static EXIT: ExitGuard = const { ExitGuard };
}

/// Returns the exiting thread's free memory to the OS. A heap with no live
/// blocks ends with nothing mapped; one with live blocks keeps them, until a
/// later thread-local destructor frees the last one, which trims it. `CURRENT`
/// stays set: other thread-locals' destructors may still allocate and free on
/// this thread after this one ran, and they keep using the same record.
struct ExitGuard;

impl Drop for ExitGuard {
    fn drop(&mut self) {
        if let Some(h) = existing() {
            // SAFETY: the current thread's record; nothing else references it now.
            unsafe {
                (*h).exited = true;
                (*h).trim();
            }
        }
    }
}

/// Offset of a record inside its mapping. The first block of every page is
/// 64 KiB-aligned, and a small class's hottest block sits in the first KiB of
/// a 4 KiB page; a page-aligned record put its queue heads at the same low 12
/// address bits, so every store to that block stalled the next load of the
/// heap (4K aliasing; `lifo_64B` 29 µs against 26.7 at this offset).
const RECORD_OFFSET: usize = 0x840;

const _: () = assert!(
    RECORD_OFFSET >= 1024
        && RECORD_OFFSET.is_multiple_of(64)
        && RECORD_OFFSET + size_of::<Record>() <= PAGE
);

/// Maps, initializes and registers a new record.
fn new_record() -> *mut Record {
    // SAFETY: PAGE is a nonzero multiple of itself; PAGE alignment.
    let Some(base) = (unsafe { os::map_aligned(PAGE, PAGE) }) else {
        std::alloc::handle_alloc_error(Layout::new::<Record>())
    };
    // SAFETY: the record fits in the page after `RECORD_OFFSET` (checked above).
    let r = unsafe { base.as_ptr().add(RECORD_OFFSET) }.cast::<Record>();
    let mut inner = Inner::new(HugePages::Auto);
    inner.owner = r.addr(); // the heap's own address: `inner` is the first field
    // SAFETY: inside a fresh mapping, 64-byte aligned, with room for a record.
    unsafe {
        r.write(Record {
            inner: UnsafeCell::new(inner),
            next: ptr::null_mut(),
        })
    };
    let mut head = RECORDS.load(Ordering::Relaxed);
    loop {
        // SAFETY: `r` is not yet shared; only this thread writes it.
        unsafe { (*r).next = head };
        match RECORDS.compare_exchange_weak(head, r, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return r,
            Err(h) => head = h,
        }
    }
}

#[cold]
#[inline(never)]
fn init_current() -> *mut Record {
    let r = new_record();
    CURRENT.with(|c| c.set(r));
    // Registers the exit guard's destructor. Fails only if this thread is
    // already tearing down; then there is no exit left to guard.
    let _ = EXIT.try_with(|_| ());
    r
}

/// The current thread's heap, made on first use. Off the hot paths: those read
/// `CURRENT` themselves and leave the first-use branch as one cold call, so
/// they stay small enough to inline (`<&ShardAlloc as Allocator>` included).
fn current() -> *mut Inner {
    let mut r = CURRENT.with(Cell::get);
    if r == empty() {
        r = init_current();
    }
    // SAFETY: records are never freed.
    unsafe { (*r).inner.get() }
}

/// The current thread's heap, or the empty heap if it has none yet (no live
/// block's owner stamp matches the empty heap).
#[inline(always)]
fn owner_this() -> *mut Inner {
    CURRENT.with(Cell::get).cast::<Inner>()
}

/// `allocate` after the fast path on `this` found no block: the heap's slow
/// path, with the heap made first if the thread is still on the empty heap.
#[inline(always)]
fn allocate_slow(mut this: *mut Inner, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    if this == empty().cast() {
        this = current();
    }
    // SAFETY: the current thread's heap; no other reference is live. The fast
    // path returns a block for zero-sized layouts, so `layout` is nonzero.
    unsafe { (*this).alloc_miss(layout) }
}

/// The current thread's heap, if it has one.
fn existing() -> Option<*mut Inner> {
    let r = CURRENT.with(Cell::get);
    // SAFETY: records are never freed.
    (r != empty()).then(|| unsafe { (*r).inner.get() })
}

/// Allocates from the current thread's own heap.
///
/// A zero-sized handle: `Vec<T, ShardAlloc>` costs no space for its allocator.
/// Each thread that uses it gets its own heap, made on first use.
///
/// This is the shape of a global allocator, and it pays what one pays on a
/// single thread: a thread-local lookup to allocate and an owner check to free.
/// [`&LocalHeap`](crate::LocalHeap) pays neither, because the heap is a value and
/// the compiler proves the free happens on the owning thread.
///
/// `ShardAlloc` is `!Send` until cross-thread frees land, so its collections stay
/// on the thread that made them:
///
/// ```compile_fail,E0277
/// fn assert_send<T: Send>(_: T) {}
/// assert_send(Vec::<u8, _>::new_in(shardheap::ShardAlloc::new()));
/// ```
///
/// ```
/// use shardheap::ShardAlloc;
///
/// let mut v: Vec<u64, ShardAlloc> = Vec::new_in(ShardAlloc::new());
/// v.extend(0..1000);
/// assert!(ShardAlloc::stats().allocated_bytes >= 8000);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ShardAlloc {
    _not_send: PhantomData<*mut ()>,
}

impl ShardAlloc {
    /// The handle. All handles on a thread share that thread's heap.
    pub const fn new() -> Self {
        ShardAlloc {
            _not_send: PhantomData,
        }
    }

    /// Byte counters of the current thread's heap (all zero if it has none yet).
    pub fn stats() -> Stats {
        // SAFETY: the current thread's record; nothing else references it now.
        existing().map_or_else(Stats::default, |h| unsafe { (*h).stats })
    }

    /// [`LocalHeap::maintain`](crate::LocalHeap::maintain) for the current
    /// thread's heap.
    pub fn maintain() {
        if let Some(h) = existing() {
            // SAFETY: as `stats`.
            unsafe { (*h).maintain() }
        }
    }

    /// [`LocalHeap::trim`](crate::LocalHeap::trim) for the current thread's heap.
    pub fn trim() {
        if let Some(h) = existing() {
            // SAFETY: as `stats`.
            unsafe { (*h).trim() }
        }
    }
}

// SAFETY: forwards to the current thread's heap, which upholds the allocator
// contract. A block is always freed on its allocating thread (`!Send`), so it
// reaches the heap that made it; the owner check aborts otherwise. Records live
// for the process, so blocks stay valid however long the handle is kept.
unsafe impl Allocator for ShardAlloc {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the current thread's heap with no other reference in use, or
        // the empty heap, whose queues are all null.
        let this = owner_this();
        match unsafe { Inner::alloc_fast(this, layout) } {
            Some(b) => Ok(b),
            None => allocate_slow(this, layout),
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: as `allocate`.
        unsafe { (*current()).alloc_zeroed(layout) }
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // The owner check compares the block's stamp with this value first, and
        // the empty heap (no record) never matches.
        let this = owner_this();
        // SAFETY: caller's contract; `this` is the empty heap or the current
        // thread's heap, whose owner stamp is its address.
        unsafe { Inner::dealloc_owned(this, ptr.as_ptr(), layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; the current thread's heap (made if needed).
        unsafe { Inner::resize_owned(current(), ptr.as_ptr(), old, new) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; the block holds `b.len() >= old.size()` bytes.
        unsafe {
            let b = Inner::resize_owned(current(), ptr.as_ptr(), old, new)?;
            let tail = b.as_ptr().cast::<u8>().add(old.size());
            ptr::write_bytes(tail, 0, b.len() - old.size());
            Ok(b)
        }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; the current thread's heap (made if needed).
        unsafe { Inner::resize_owned(current(), ptr.as_ptr(), old, new) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pointer handed back from a thread that has exited.
    struct Sent<T>(*const T);
    // SAFETY: only read after the thread that made it is joined.
    unsafe impl<T> Send for Sent<T> {}

    fn record_stats(r: Sent<Record>) -> Stats {
        // SAFETY: records are never freed; the owning thread has exited.
        unsafe { (*(*r.0).inner.get()).stats }
    }

    fn current_record() -> Sent<Record> {
        Sent(CURRENT.with(Cell::get))
    }

    #[test]
    fn exit_releases_an_empty_heap() {
        let r = std::thread::spawn(|| {
            let mut small: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
            small.extend_from_slice(&[1; 200]);
            let mut big: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
            big.resize(2 << 20, 2);
            drop((small, big));
            current_record()
        })
        .join()
        .unwrap();
        let s = record_stats(r);
        assert_eq!((s.allocated_bytes, s.mapped_bytes, s.segments), (0, 0, 0));
    }

    #[test]
    fn exit_keeps_a_heap_with_live_blocks() {
        let (r, p) = std::thread::spawn(|| {
            let b = Box::leak(Box::new_in([7u8; 1000], ShardAlloc::new()));
            (current_record(), Sent(b.as_ptr()))
        })
        .join()
        .unwrap();
        let s = record_stats(r);
        assert!(s.allocated_bytes >= 1000 && s.mapped_bytes > 0);
        // SAFETY: the leaked block stays mapped with its record.
        let bytes = unsafe { core::slice::from_raw_parts(p.0, 1000) };
        assert!(bytes.iter().all(|&b| b == 7));
    }

    struct LateVec(core::cell::RefCell<Option<Vec<u8, ShardAlloc>>>);
    impl Drop for LateVec {
        fn drop(&mut self) {
            drop(self.0.borrow_mut().take());
        }
    }
    thread_local!(static LATE_SMALL: LateVec = const { LateVec(core::cell::RefCell::new(None)) });
    thread_local!(static LATE_LARGE: LateVec = const { LateVec(core::cell::RefCell::new(None)) });

    #[test]
    fn frees_after_the_exit_guard_still_release_an_emptied_heap() {
        let r = std::thread::spawn(|| {
            // Registered before the exit guard, so destroyed after it (Linux).
            LATE_SMALL.with(|_| ());
            LATE_LARGE.with(|_| ());
            let mut small = Vec::new_in(ShardAlloc::new());
            small.extend_from_slice(&[1u8; 200]);
            let mut large = Vec::new_in(ShardAlloc::new());
            large.resize(1 << 20, 2u8);
            LATE_SMALL.with(|l| *l.0.borrow_mut() = Some(small));
            LATE_LARGE.with(|l| *l.0.borrow_mut() = Some(large));
            current_record()
        })
        .join()
        .unwrap();
        let s = record_stats(r);
        assert_eq!((s.allocated_bytes, s.mapped_bytes, s.segments), (0, 0, 0));
    }

    #[test]
    fn a_fresh_thread_starts_on_the_empty_heap() {
        std::thread::spawn(|| {
            let empty = (&raw const EMPTY.0).cast_mut();
            assert_eq!(CURRENT.with(Cell::get), empty, "no record before first use");
            assert_eq!(ShardAlloc::stats(), Stats::default());
            drop(Box::new_in(7u32, ShardAlloc::new()));
            let r = CURRENT.with(Cell::get);
            assert_ne!(r, empty, "the first allocation made a record");
            assert_eq!(r.addr() % PAGE, RECORD_OFFSET);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn record_heaps_sit_clear_of_page_start_blocks() {
        // The first block of every page starts 64 KiB-aligned; small classes put
        // their hottest block in the first KiB of a 4 KiB page. The heap's queue
        // heads and counters must not share those low 12 address bits, or each
        // store to that block stalls the next load of the heap (4K aliasing).
        drop(Box::new_in(0u8, ShardAlloc::new()));
        let off = current_record().0.addr() % PAGE;
        assert!(off >= 1024, "record heap at page offset {off:#x}");
        assert!(
            off + size_of::<Record>() <= PAGE,
            "record heap crosses a page"
        );
    }

    #[test]
    fn records_are_distinct_and_registered_under_contention() {
        let rs: Vec<usize> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    drop(Box::new_in(1u64, ShardAlloc::new()));
                    current_record().0.addr()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect();
        let set: std::collections::HashSet<_> = rs.iter().copied().collect();
        assert_eq!(set.len(), 8);
        let mut seen = 0;
        let mut r = RECORDS.load(Ordering::Acquire);
        while !r.is_null() {
            if set.contains(&r.addr()) {
                seen += 1;
            }
            // SAFETY: registered records are never freed.
            r = unsafe { (*r).next };
        }
        assert_eq!(seen, 8);
    }
}
