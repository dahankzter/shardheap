//! The segment map: one bit per 4 MiB of address space, set while a segment
//! lives there. It tells a small block (inside a segment) from a large one
//! (its own mapping) by address alone, for frees that come without a size
//! (`ShardAlloc::free_unsized`, C `free`).
//!
//! The map covers the user address space (47 bits on x86-64, 48 on aarch64):
//! 4 or 8 MiB reserved on first use, of which only the pages holding bits for
//! segments ever become resident. Bits are atomics: segments are made and
//! released by many threads at once. A segment's bit is set before any of its
//! blocks exists and cleared after the last one is gone, so for a live block
//! the bit is stable; it reaches a thread that frees the block by whatever
//! handed the block over.

use core::ptr::{self, NonNull};
use core::sync::atomic::AtomicPtr;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};

use crate::os;
use crate::segment::{SEGMENT_SIZE, Segment};

/// Address bits the map covers.
#[cfg(target_arch = "aarch64")]
const ADDRESS_BITS: u32 = 48;
#[cfg(not(target_arch = "aarch64"))]
const ADDRESS_BITS: u32 = 47;

const SEGMENTS: usize = 1 << (ADDRESS_BITS - SEGMENT_SIZE.trailing_zeros());
const WORDS: usize = SEGMENTS / 64;
const MAP_BYTES: usize = WORDS * size_of::<u64>();

static MAP: AtomicPtr<AtomicU64> = AtomicPtr::new(ptr::null_mut());

/// The word holding `p`'s bit and the bit's mask, if the map covers `p`.
#[inline(always)]
fn slot(p: *const u8) -> Option<(usize, u64)> {
    let i = p.addr() / SEGMENT_SIZE;
    (i < SEGMENTS).then(|| (i / 64, 1u64 << (i % 64)))
}

/// The map, reserved on first use.
fn map() -> *mut AtomicU64 {
    let m = MAP.load(Acquire);
    if !m.is_null() {
        return m;
    }
    // SAFETY: a nonzero multiple of the page size.
    let fresh = unsafe { os::reserve(MAP_BYTES) }
        .unwrap_or_else(|| std::process::abort())
        .as_ptr()
        .cast::<AtomicU64>();
    match MAP.compare_exchange(ptr::null_mut(), fresh, AcqRel, Acquire) {
        Ok(_) => fresh,
        Err(won) => {
            // SAFETY: our own reservation, never published.
            unsafe { os::unreserve(NonNull::new_unchecked(fresh.cast()), MAP_BYTES) };
            won
        }
    }
}

/// Marks the segment at `seg` as live.
pub(crate) fn mark(seg: *mut Segment) {
    let Some((w, bit)) = slot(seg.cast()) else {
        debug_assert!(false, "segment beyond the mapped address space");
        return;
    };
    // SAFETY: `w < WORDS`; the map is zero-initialized atomics.
    unsafe { (*map().add(w)).fetch_or(bit, Relaxed) };
}

/// Clears the mark of the segment at `seg`, before its memory goes away.
pub(crate) fn unmark(seg: *mut Segment) {
    let m = MAP.load(Acquire);
    if let (false, Some((w, bit))) = (m.is_null(), slot(seg.cast())) {
        // SAFETY: `w < WORDS`.
        unsafe { (*m.add(w)).fetch_and(!bit, Relaxed) };
    }
}

/// Whether `p` lies in a live segment.
#[inline]
pub(crate) fn is_segment(p: *const u8) -> bool {
    let m = MAP.load(Acquire);
    match (m.is_null(), slot(p)) {
        // SAFETY: `w < WORDS`.
        (false, Some((w, bit))) => unsafe { (*m.add(w)).load(Relaxed) & bit != 0 },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LocalHeap;
    use crate::os::PAGE;
    use core::alloc::{Allocator, Layout};

    #[test]
    fn segments_are_marked_and_large_blocks_are_not() {
        let heap = LocalHeap::new();
        let l64 = Layout::new::<[u8; 64]>();
        let small = heap.allocate(l64).unwrap().cast::<u8>();
        assert!(is_segment(small.as_ptr()));
        for align in [PAGE, 4 << 20, 8 << 20] {
            let l = Layout::from_size_align(300_000, align).unwrap();
            let big = heap.allocate(l).unwrap().cast::<u8>();
            assert!(!is_segment(big.as_ptr()), "align {align}");
            // SAFETY: allocated above with `l`.
            unsafe { heap.deallocate(big, l) };
        }
        // SAFETY: allocated above with `l64`.
        unsafe { heap.deallocate(small, l64) };
    }

    #[test]
    fn a_released_segment_is_unmarked() {
        let heap = LocalHeap::new();
        let l64 = Layout::new::<[u8; 64]>();
        let p = heap.allocate(l64).unwrap().cast::<u8>();
        let seg = crate::segment::segment_of(p.as_ptr());
        // SAFETY: allocated above with `l64`.
        unsafe { heap.deallocate(p, l64) };
        heap.trim();
        assert!(!is_segment(seg.cast()));
    }
}
