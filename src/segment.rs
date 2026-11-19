//! Segments and pages.
//!
//! A segment is a 4 MiB region aligned to 4 MiB, split into 64 slots of 64 KiB.
//! Slot 0 holds the [`Segment`] header with one [`Page`] descriptor per slot. A page
//! is a span of 1 to 16 slots holding blocks of one size class. Because segments
//! are aligned, the header of any block is found by masking its address.

use core::ptr::{self, NonNull};

/// Segment size and alignment.
pub(crate) const SEGMENT_SIZE: usize = 4 << 20;
/// Slot size; pages are whole slots.
pub(crate) const SLOT_SIZE: usize = 64 << 10;
/// Slots per segment, including the header slot.
pub(crate) const SLOTS: usize = SEGMENT_SIZE / SLOT_SIZE;

pub(crate) const MAGIC: usize = 0x5348_4152_4448_4550; // "SHARDHEP"

/// A free block; its first word links to the next free block.
#[repr(C)]
pub(crate) struct Block {
    pub(crate) next: *mut Block,
}

/// Page descriptor. Only the span's first slot (its head) carries page state;
/// every slot carries `head_offset` to find the head.
#[repr(C)]
pub(crate) struct Page {
    /// Blocks to allocate from.
    pub(crate) free: *mut Block,
    /// Blocks freed by the owning heap; swapped into `free` when it runs dry.
    pub(crate) local_free: *mut Block,
    /// First block of the page.
    pub(crate) start: *mut u8,
    /// Neighbours in the heap's queue for this class.
    pub(crate) next: *mut Page,
    pub(crate) prev: *mut Page,
    pub(crate) block_size: usize,
    pub(crate) capacity: u32,
    /// Blocks currently handed out.
    pub(crate) used: u32,
    /// Blocks `bump..capacity` have never been handed out.
    pub(crate) bump: u32,
    pub(crate) class: u8,
    pub(crate) span: u8,
    /// Distance from this slot back to the span's head slot.
    pub(crate) head_offset: u8,
    /// Whether the page is linked into its class queue (not full).
    pub(crate) in_queue: bool,
    /// Heap tick at which the page last became empty while kept for reuse.
    pub(crate) idle_since: u64,
    /// Phase 3: atomic list of blocks freed by other threads.
    pub(crate) _thread_free: usize,
}

#[repr(C)]
pub(crate) struct Segment {
    pub(crate) magic: usize,
    pub(crate) next: *mut Segment,
    pub(crate) prev: *mut Segment,
    /// Bit `i` set: slot `i` is in use. Bit 0 (the header) is always set.
    pub(crate) used_slots: u64,
    /// Free slots released in the current purge epoch; their memory may be
    /// resident. A subset of the zero bits of `used_slots`.
    pub(crate) dirty_slots: u64,
    /// Free slots released in an earlier epoch and not yet purged: idle for at
    /// least one whole epoch. Disjoint from `dirty_slots`.
    pub(crate) stale_slots: u64,
    /// Free slots returned to the OS by a purge while the segment was in use.
    pub(crate) purged_slots: u64,
    /// Phase 3: identity of the owning heap.
    pub(crate) _owner: usize,
    pub(crate) pages: [Page; SLOTS],
}

const _: () = assert!(SLOTS == 64);
const _: () = assert!(core::mem::size_of::<Segment>() <= SLOT_SIZE);

/// The segment containing `p`.
#[inline(always)]
pub(crate) fn segment_of(p: *mut u8) -> *mut Segment {
    p.map_addr(|a| a & !(SEGMENT_SIZE - 1)).cast()
}

/// The head page of the span containing `p`.
///
/// # Safety
/// `p` points into a slot of the live segment `seg`.
#[inline(always)]
pub(crate) unsafe fn page_of(seg: *mut Segment, p: *mut u8) -> *mut Page {
    let slot = (p.addr() - seg.addr()) / SLOT_SIZE;
    debug_assert!(slot > 0 && slot < SLOTS);
    // SAFETY: `slot < SLOTS` because `p` lies within the 4 MiB segment.
    unsafe {
        let pages = (&raw mut (*seg).pages).cast::<Page>();
        let back = (*pages.add(slot)).head_offset as usize;
        pages.add(slot - back)
    }
}

/// First slot of a run of `k` free slots, if any.
#[inline]
pub(crate) fn find_span(used_slots: u64, k: usize) -> Option<usize> {
    let mut m = !used_slots;
    for _ in 1..k {
        m &= m >> 1;
    }
    (m != 0).then(|| m.trailing_zeros() as usize)
}

/// Bit mask for slots `slot..slot + k`.
#[inline]
pub(crate) fn span_mask(slot: usize, k: usize) -> u64 {
    let bits = if k >= 64 { !0 } else { (1u64 << k) - 1 };
    bits << slot
}

/// Calls `f(first, n)` for every maximal run of set bits in `mask`.
#[inline]
pub(crate) fn for_each_run(mut mask: u64, mut f: impl FnMut(usize, usize)) {
    while mask != 0 {
        let first = mask.trailing_zeros() as usize;
        let n = (!(mask >> first)).trailing_zeros() as usize;
        f(first, n);
        mask &= !span_mask(first, n);
    }
}

/// Initializes a fresh, zeroed segment header.
///
/// # Safety
/// `base` is a zeroed, `SEGMENT_SIZE`-aligned mapping of `SEGMENT_SIZE` bytes.
pub(crate) unsafe fn init_segment(base: NonNull<u8>) -> *mut Segment {
    let seg = base.as_ptr().cast::<Segment>();
    // SAFETY: the header fits in slot 0 of the mapping; zero is a valid value for
    // every other field.
    unsafe {
        (*seg).magic = MAGIC;
        (*seg).used_slots = 1;
        (*seg).next = ptr::null_mut();
        (*seg).prev = ptr::null_mut();
    }
    seg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans() {
        assert_eq!(find_span(1, 1), Some(1));
        assert_eq!(find_span(1, 63), Some(1));
        assert_eq!(find_span(1, 64), None);
        assert_eq!(find_span(0b1011, 2), Some(4));
        assert_eq!(find_span(!0 ^ (0b11 << 62), 2), Some(62));
        assert_eq!(find_span(!0 ^ (1 << 63), 2), None);
        assert_eq!(span_mask(3, 2), 0b11000);
        assert_eq!(span_mask(0, 64), !0);
    }

    #[test]
    fn runs() {
        let mut v = Vec::new();
        for_each_run(0b1110_0110 | 1 << 63, |a, n| v.push((a, n)));
        assert_eq!(v, [(1, 2), (5, 3), (63, 1)]);
        v.clear();
        for_each_run(!0, |a, n| v.push((a, n)));
        assert_eq!(v, [(0, 64)]);
    }
}
