//! A fixed memory region a heap takes all its memory from, for memory that must
//! stay in one known range (io_uring registered buffers, RDMA, shared memory).
//!
//! The region is carved into 64 KiB granules tracked by a bitmap. Segments take
//! 64 granules at a 4 MiB-aligned position; large blocks take a run of granules.
//! Freed memory goes back to the bitmap only: it is never unmapped or purged while
//! the region lives, because registered memory is pinned and the device may still
//! hold references to the physical pages.

use core::ptr::NonNull;

use crate::os::{self, PAGE};
use crate::segment::{SEGMENT_SIZE, SLOT_SIZE};

/// Allocation granule.
const GRANULE: usize = SLOT_SIZE;

pub(crate) struct Region {
    /// Start of the mapping (4 MiB-aligned) and its length in bytes.
    base: NonNull<u8>,
    len: usize,
    /// Bit `i` set: granule `i` is in use.
    used: Box<[u64]>,
    granules: usize,
    /// Bytes currently handed out from the region.
    pub(crate) used_bytes: usize,
}

impl Region {
    /// Maps a region of at least `len` bytes (rounded up to 4 MiB), asking for
    /// huge pages when `huge` is true.
    pub(crate) fn map(len: usize, huge: Option<bool>) -> Option<Region> {
        let len = len.max(1).checked_next_multiple_of(SEGMENT_SIZE)?;
        // SAFETY: nonzero multiple of PAGE; power-of-two alignment >= PAGE.
        let base = unsafe { os::map_aligned(len, SEGMENT_SIZE)? };
        if let Some(huge) = huge {
            // SAFETY: the mapping was just created.
            unsafe { os::advise_huge(base, len, huge) };
        }
        let granules = len / GRANULE;
        Some(Region {
            base,
            len,
            used: vec![0u64; granules.div_ceil(64)].into_boxed_slice(),
            granules,
            used_bytes: 0,
        })
    }

    pub(crate) fn range(&self) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(self.base, self.len)
    }

    #[inline]
    pub(crate) fn contains(&self, p: *const u8) -> bool {
        p.addr().wrapping_sub(self.base.addr().get()) < self.len
    }

    fn is_free(&self, g: usize) -> bool {
        self.used[g / 64] & (1 << (g % 64)) == 0
    }

    fn mark(&mut self, first: usize, n: usize, used: bool) {
        for g in first..first + n {
            if used {
                self.used[g / 64] |= 1 << (g % 64);
            } else {
                self.used[g / 64] &= !(1 << (g % 64));
            }
        }
    }

    /// Takes `size` bytes (rounded up to granules) starting at an address aligned
    /// to `align`, first fit. The memory is not zeroed.
    pub(crate) fn alloc(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if align > SEGMENT_SIZE {
            return None; // the base is only 4 MiB-aligned: the OS serves these
        }
        let n = size.checked_next_multiple_of(GRANULE)? / GRANULE;
        let step = (align / GRANULE).max(1); // the base is 4 MiB-aligned
        let mut g = 0;
        while g + n <= self.granules {
            // Whole words of used granules are skipped quickly.
            if step == 1 && g % 64 == 0 && self.used[g / 64] == !0 {
                g += 64;
                continue;
            }
            match (g..g + n).rev().find(|&i| !self.is_free(i)) {
                None => {
                    self.mark(g, n, true);
                    self.used_bytes += n * GRANULE;
                    // SAFETY: `g + n <= granules`, so the range is inside the mapping.
                    return Some(unsafe { self.base.add(g * GRANULE) });
                }
                // Restart past the used granule, at the next aligned position.
                Some(busy) => g = (busy + 1).next_multiple_of(step),
            }
        }
        None
    }

    /// Extends the block at `p` from `old` to `new` bytes in place if the
    /// granules after it are free. `old` and `p` are as passed to and returned
    /// by `alloc`; on success, free the block with `new` from then on.
    pub(crate) fn try_grow(&mut self, p: NonNull<u8>, old: usize, new: usize) -> bool {
        let first = (p.addr().get() - self.base.addr().get()) / GRANULE;
        let (n0, n1) = (old.div_ceil(GRANULE), new.div_ceil(GRANULE));
        if n1 <= n0 {
            return true;
        }
        if first + n1 > self.granules || !(first + n0..first + n1).all(|g| self.is_free(g)) {
            return false;
        }
        self.mark(first + n0, n1 - n0, true);
        self.used_bytes += (n1 - n0) * GRANULE;
        true
    }

    /// Returns `size` bytes at `p` (as passed to `alloc`) to the region.
    pub(crate) fn free(&mut self, p: NonNull<u8>, size: usize) {
        debug_assert!(self.contains(p.as_ptr()));
        let n = size.div_ceil(GRANULE);
        let first = (p.addr().get() - self.base.addr().get()) / GRANULE;
        debug_assert!((first..first + n).all(|g| !self.is_free(g)));
        self.mark(first, n, false);
        self.used_bytes -= n * GRANULE;
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `map`; the heap released every block first.
        unsafe { os::unmap(self.base, self.len, SEGMENT_SIZE) }
    }
}

const _: () = assert!(SEGMENT_SIZE.is_multiple_of(GRANULE) && GRANULE.is_multiple_of(PAGE));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_alignment_and_reuse() {
        let mut r = Region::map(12 << 20, None).unwrap();
        let a = r.alloc(100_000, PAGE).unwrap(); // 2 granules
        let s = r.alloc(SEGMENT_SIZE, SEGMENT_SIZE).unwrap();
        assert_eq!(s.addr().get() % SEGMENT_SIZE, 0);
        assert_ne!(s, r.base, "granules 0-1 are taken, so the segment moves on");
        let b = r.alloc(GRANULE, PAGE).unwrap();
        assert_eq!(b.addr().get(), a.addr().get() + 2 * GRANULE, "first fit");
        r.free(a, 100_000);
        assert_eq!(r.alloc(GRANULE, PAGE), Some(a), "freed granules are reused");
        assert!(r.alloc(SEGMENT_SIZE, SEGMENT_SIZE).is_some());
        assert!(r.alloc(SEGMENT_SIZE, SEGMENT_SIZE).is_none(), "full");
        assert!(r.contains(b.as_ptr()) && !r.contains(core::ptr::null()));
    }

    #[test]
    fn grows_in_place_into_free_granules() {
        let mut r = Region::map(4 << 20, None).unwrap();
        let a = r.alloc(GRANULE, PAGE).unwrap();
        assert!(r.try_grow(a, GRANULE, 3 * GRANULE));
        let b = r.alloc(GRANULE, PAGE).unwrap();
        assert_eq!(b.addr().get(), a.addr().get() + 3 * GRANULE);
        assert!(!r.try_grow(a, 3 * GRANULE, 4 * GRANULE), "blocked by b");
        assert!(r.try_grow(b, GRANULE, 61 * GRANULE), "up to the end");
        assert!(!r.try_grow(b, 61 * GRANULE, 62 * GRANULE), "past the end");
        r.free(a, 3 * GRANULE);
        r.free(b, 61 * GRANULE);
        assert_eq!(r.used_bytes, 0);
    }
}
