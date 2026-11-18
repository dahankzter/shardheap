//! [`Reap`]: bump allocation from heap pages that are recycled when their last
//! block dies.
//!
//! The reap's state lives in the heap (`Inner::reap_*`): one current bump page.
//! A reap page is a single 64 KiB slot whose descriptor has class `REAP_CLASS`
//! and is never in a class queue. `used` counts its live blocks; nothing else in
//! the descriptor is used.

use core::alloc::{AllocError, Allocator, Layout};
use core::fmt;
use core::ptr::{self, NonNull};

use crate::heap::{Inner, LocalHeap, dangling, slice};
use crate::os::PAGE;
use crate::segment::{self, Page, SLOT_SIZE};

/// Largest request served from reap pages; bigger ones go to the heap. Keeps the
/// space lost at the end of a page under 1/8 of it.
const REAP_MAX_SIZE: usize = 8 << 10;
/// Largest alignment served from reap pages.
const REAP_MAX_ALIGN: usize = PAGE;
/// Class value marking a reap page descriptor.
pub(crate) const REAP_CLASS: u8 = u8::MAX;

/// Whether `layout` is served from reap pages. Depends only on the layout, so
/// allocation and free always agree.
#[inline(always)]
fn eligible(layout: Layout) -> bool {
    layout.size() <= REAP_MAX_SIZE && layout.align() <= REAP_MAX_ALIGN
}

/// Arena-speed allocation with individual frees, for objects that mostly die
/// together while a few outlive the rest.
///
/// Blocks are bump-allocated from 64 KiB pages of a [`LocalHeap`]. Freeing a
/// block only decrements its page's live count. When the count reaches zero, the
/// page goes back to the heap, or, if it is the current bump page, the bump
/// pointer rewinds to its start. There is no `reset` and no lifetime tying blocks
/// to a scope: a block lives as long as it is needed, and a straggler pins only
/// its own page rather than a whole arena. The trade-off is the reverse of an
/// arena's: one live block keeps its whole 64 KiB page resident.
///
/// Requests above 8 KiB, or aligned above 4 KiB, go to the heap's normal size
/// classes, so a `Vec` that keeps growing moves there once it is large.
///
/// The handle is `Copy`, one pointer wide, and borrows only the heap; it is
/// `!Send` like the heap.
///
/// ```
/// use shardheap::{LocalHeap, Reap};
///
/// let heap = LocalHeap::new();
/// let reap = heap.reap();
/// let mut cache: Vec<Box<[u8; 100], Reap<'_>>> = Vec::new();
/// for request in 0..1000u32 {
///     let scratch: Vec<Box<[u8; 100], Reap<'_>>> =
///         (0..50).map(|i| Box::new_in([i as u8; 100], reap)).collect();
///     if request % 20 == 0 {
///         cache.push(Box::new_in([1; 100], reap)); // a straggler, freed whenever
///     }
///     drop(scratch); // pages whose blocks all died are recycled
/// }
/// assert!(heap.stats().reap_pages <= cache.len() + 1);
/// ```
///
/// The design is the "reap" of Berger, Zorn and McKinley, *Reconsidering Custom
/// Memory Allocation* (OOPSLA 2002).
#[derive(Clone, Copy)]
pub struct Reap<'h> {
    heap: &'h LocalHeap,
}

impl fmt::Debug for Reap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reap")
            .field("reap_pages", &self.heap.stats().reap_pages)
            .finish()
    }
}

impl LocalHeap {
    /// A [`Reap`] allocating from this heap. All handles share one bump page.
    pub fn reap(&self) -> Reap<'_> {
        Reap { heap: self }
    }
}

impl Inner {
    /// Bump-allocates from the current reap page, or `None` if it does not fit.
    /// `layout` is eligible and nonzero.
    #[inline(always)]
    fn reap_bump(&mut self, layout: Layout) -> Option<NonNull<[u8]>> {
        let p = self.reap_ptr;
        let pad = p.addr().wrapping_neg() & (layout.align() - 1);
        let room = self.reap_end.addr().wrapping_sub(p.addr());
        // `pad < 4096` and `size <= 8 KiB`: no overflow. Before the first page
        // both pointers are null and `room == 0`.
        if pad + layout.size() > room {
            return None;
        }
        let start = p.wrapping_add(pad);
        self.reap_ptr = start.wrapping_add(layout.size());
        // SAFETY: something fit, so `room > 0` and `reap_page` is the live page.
        unsafe { (*self.reap_page).used += 1 };
        self.stats.allocated_bytes += layout.size();
        Some(slice(start, layout.size()))
    }

    #[inline(always)]
    pub(crate) unsafe fn reap_alloc(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling(layout.align()));
        }
        if !eligible(layout) {
            // SAFETY: forwarded unchanged; `reap_free` forwards the same layouts.
            return unsafe { self.alloc(layout) };
        }
        match self.reap_bump(layout) {
            Some(b) => Ok(b),
            // SAFETY: eligible, nonzero layout.
            None => unsafe { self.reap_alloc_slow(layout) },
        }
    }

    /// The bump page is full (or there is none yet): start a fresh one. The full
    /// page keeps its live blocks and is released when the last one is freed. It
    /// cannot be empty: an empty bump page is rewound in `reap_free`, and then
    /// any eligible request fits.
    #[cold]
    #[inline(never)]
    unsafe fn reap_alloc_slow(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.heartbeat();
        // SAFETY: `claim_span` returns a fresh, zeroed descriptor of this heap.
        unsafe {
            let page = self.claim_span(1)?;
            (*page).class = REAP_CLASS;
            let old = self.reap_page;
            debug_assert!(old.is_null() || (*old).used > 0);
            self.reap_page = page;
            self.reap_ptr = (*page).start;
            self.reap_end = (*page).start.add(SLOT_SIZE);
            self.stats.reap_pages += 1;
        }
        Ok(self
            .reap_bump(layout)
            .expect("an empty page fits any reap block"))
    }

    /// Whether `p..p + size` is the most recent block of the current page. A
    /// block of an older page can end exactly where the current page starts (the
    /// next slot), so the start is checked too.
    #[inline(always)]
    fn reap_is_last(&self, p: *mut u8, size: usize) -> bool {
        p.wrapping_add(size) == self.reap_ptr
            && p.addr() >= self.reap_end.addr().wrapping_sub(SLOT_SIZE)
    }

    /// # Safety
    /// `p` was allocated by `reap_alloc`/`reap_resize` with a layout that fits it.
    #[inline(always)]
    pub(crate) unsafe fn reap_free(&mut self, p: *mut u8, layout: Layout) {
        let size = layout.size();
        if size == 0 {
            return;
        }
        if !eligible(layout) {
            // SAFETY: allocated by the heap with this layout (see `reap_alloc`).
            return unsafe { self.dealloc(p, layout) };
        }
        self.stats.allocated_bytes -= size;
        // SAFETY: `p` is a live reap block, so it lies in a live reap page.
        unsafe {
            let page = segment::page_of(segment::segment_of(p), p);
            debug_assert_eq!((*page).class, REAP_CLASS, "not a reap block");
            (*page).used -= 1;
            if page == self.reap_page {
                if (*page).used == 0 {
                    self.reap_ptr = (*page).start; // all died: reuse the whole page
                } else if p.wrapping_add(size) == self.reap_ptr {
                    self.reap_ptr = p; // most recent block: give its bytes back
                }
            } else if (*page).used == 0 {
                self.release_reap_page(page);
            }
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn release_reap_page(&mut self, page: *mut Page) {
        self.stats.reap_pages -= 1;
        // SAFETY: an empty reap page of this heap, in no queue.
        unsafe { self.release_page(page) }
    }

    /// `trim`: releases the current bump page if nothing in it is live.
    pub(crate) unsafe fn release_idle_reap_page(&mut self) {
        let page = self.reap_page;
        // SAFETY: a non-null `reap_page` is a live reap page of this heap.
        unsafe {
            if !page.is_null() && (*page).used == 0 {
                self.reap_page = ptr::null_mut();
                self.reap_ptr = ptr::null_mut();
                self.reap_end = ptr::null_mut();
                self.release_reap_page(page);
            }
        }
    }

    /// `reset`: every page is being recycled wholesale; drop the bump page.
    pub(crate) fn forget_reap(&mut self) {
        self.reap_page = ptr::null_mut();
        self.reap_ptr = ptr::null_mut();
        self.reap_end = ptr::null_mut();
        self.stats.reap_pages = 0;
    }

    /// # Safety
    /// `p` is a live block of `reap_alloc` with a layout fitting `old`.
    pub(crate) unsafe fn reap_resize(
        &mut self,
        p: *mut u8,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract for `p`/`old`.
        unsafe {
            if old.size() == 0 {
                return self.reap_alloc(new);
            }
            if new.size() == 0 {
                self.reap_free(p, old);
                return Ok(dangling(new.align()));
            }
            match (eligible(old), eligible(new)) {
                (false, false) => return self.resize(p, old, new),
                (true, true) if p.addr() & (new.align() - 1) == 0 => {
                    // `reap_is_last` first: then `p` lies in the current page and
                    // `reap_end - p` cannot underflow.
                    if self.reap_is_last(p, old.size())
                        && new.size() <= self.reap_end.addr() - p.addr()
                    {
                        self.reap_ptr = p.add(new.size());
                        self.stats.allocated_bytes =
                            self.stats.allocated_bytes - old.size() + new.size();
                        return Ok(slice(p, new.size()));
                    }
                    if new.size() <= old.size() {
                        // The tail stays unused until the page is recycled.
                        self.stats.allocated_bytes -= old.size() - new.size();
                        return Ok(slice(p, new.size()));
                    }
                }
                _ => {}
            }
            let n = self.reap_alloc(new)?;
            ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), old.size().min(new.size()));
            self.reap_free(p, old);
            Ok(n)
        }
    }
}

// SAFETY:
// - Blocks are disjoint: the bump pointer only moves back over bytes of blocks
//   that were just freed (the most recent one, or a whole page with no live
//   block), and a page is reused only once its live count is zero.
// - A block stays valid until it is freed or the heap is reset or dropped, both
//   of which need `&mut`/ownership of the heap, which a live `Reap` prevents.
// - Every copy of the handle refers to the same heap state.
// - Which path serves a block depends only on its layout, and the reap returns
//   exactly the requested length, so free and resize find the same path.
unsafe impl Allocator for Reap<'_> {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: see `LocalHeap::inner`.
        unsafe { self.heap.inner().reap_alloc(layout) }
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: caller's contract; see `LocalHeap::inner`.
        unsafe { self.heap.inner().reap_free(ptr.as_ptr(), layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; see `LocalHeap::inner`.
        unsafe { self.heap.inner().reap_resize(ptr.as_ptr(), old, new) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; see `LocalHeap::inner`.
        unsafe { self.heap.inner().reap_resize(ptr.as_ptr(), old, new) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straggler_pins_only_its_page() {
        let heap = LocalHeap::new();
        let reap = heap.reap();
        let l = Layout::from_size_align(1024, 8).unwrap();
        // 64 blocks of 1 KiB fill a page; 200 blocks take 4 pages.
        let blocks: Vec<_> = (0..200).map(|_| reap.allocate(l).unwrap()).collect();
        assert_eq!(heap.stats().reap_pages, 4);
        for b in &blocks[1..] {
            // SAFETY: live reap blocks allocated with `l`.
            unsafe { reap.deallocate(b.cast(), l) };
        }
        // The straggler's page, plus the current (now empty, rewound) bump page.
        assert_eq!(heap.stats().reap_pages, 2);
        let again = reap.allocate(l).unwrap();
        assert_eq!(
            again.cast::<u8>(),
            blocks[192].cast::<u8>(),
            "bump page rewound"
        );
        // SAFETY: as above.
        unsafe {
            reap.deallocate(again.cast(), l);
            reap.deallocate(blocks[0].cast(), l);
        }
        assert_eq!(heap.stats().reap_pages, 1);
        assert_eq!(heap.stats().allocated_bytes, 0);
        heap.trim();
        assert_eq!((heap.stats().reap_pages, heap.stats().segments), (0, 0));
    }

    #[test]
    fn vec_grows_in_place_then_moves_to_the_heap() {
        let heap = LocalHeap::new();
        let mut v: Vec<u64, Reap<'_>> = Vec::with_capacity_in(1, heap.reap());
        let p0 = v.as_ptr();
        v.extend(0..1000); // 8000 bytes: still a reap block, the most recent one
        assert_eq!(v.as_ptr(), p0, "grew in place");
        v.extend(0..1000); // above 8 KiB: moves to the heap's size classes
        assert_ne!(v.as_ptr(), p0);
        assert_eq!(heap.stats().reap_pages, 1);
        drop(v);
        assert_eq!(heap.stats().allocated_bytes, 0);
    }
}
