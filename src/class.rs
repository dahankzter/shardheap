//! Size classes.
//!
//! Classes 0..8 are 16, 32, ..., 128 bytes. Above that, every power-of-two range
//! `[2^k, 2^(k+1))` is split into 8 classes, so a block wastes at most 12.5% of
//! its size. The largest small class is [`MAX_SMALL`]. Every class size is a
//! multiple of 16.

use core::alloc::Layout;

use crate::segment::SLOT_SIZE;

/// Largest block served from pages; anything bigger takes the large path.
pub(crate) const MAX_SMALL: usize = 128 * 1024;

/// Number of size classes.
pub(crate) const NUM_CLASSES: usize = 88;

/// Minimum number of blocks a multi-slot page should hold.
const MIN_BLOCKS_PER_PAGE: usize = 8;

const fn compute_size(c: usize) -> usize {
    if c < 8 {
        16 * (c + 1)
    } else {
        let k = 7 + (c - 8) / 8;
        let j = (c - 8) % 8;
        (1 << k) + (j + 1) * (1 << (k - 3))
    }
}

const SIZES: [usize; NUM_CLASSES] = {
    let mut t = [0; NUM_CLASSES];
    let mut c = 0;
    while c < NUM_CLASSES {
        t[c] = compute_size(c);
        c += 1;
    }
    t
};

const _: () = assert!(SIZES[NUM_CLASSES - 1] == MAX_SMALL);

/// Block size of class `c`.
#[inline(always)]
pub(crate) fn size_of(c: usize) -> usize {
    debug_assert!(c < NUM_CLASSES);
    // SAFETY: every caller passes a class from `class_of`/`class_for`, which are < NUM_CLASSES.
    unsafe { *SIZES.get_unchecked(c) }
}

/// Smallest class whose block size is `>= size`. Requires `1 <= size <= MAX_SMALL`.
#[inline(always)]
pub(crate) fn class_of(size: usize) -> usize {
    debug_assert!((1..=MAX_SMALL).contains(&size));
    if size <= 128 {
        size.div_ceil(16) - 1
    } else {
        let s = size - 1;
        let k = (usize::BITS - 1 - s.leading_zeros()) as usize;
        let j = (s >> (k - 3)) & 7;
        8 + (k - 7) * 8 + j
    }
}

/// The class serving `layout`, or `None` for the large path.
#[inline(always)]
pub(crate) fn class_for(layout: Layout) -> Option<usize> {
    let (size, align) = (layout.size(), layout.align());
    if size > MAX_SMALL || align > SLOT_SIZE {
        return None;
    }
    let size = size.max(1);
    if align <= 16 {
        // Every class size is a multiple of 16 and pages are slot-aligned.
        return Some(class_of(size));
    }
    aligned_class(size, align)
}

/// Smallest class `>= size` whose block size is a multiple of `align`. Blocks
/// start at `page_start + i * block_size` with slot-aligned pages, so that
/// makes every block `align`-aligned.
#[cold]
fn aligned_class(size: usize, align: usize) -> Option<usize> {
    let size = size.checked_next_multiple_of(align)?;
    if size > MAX_SMALL {
        return None;
    }
    (class_of(size)..NUM_CLASSES).find(|&c| SIZES[c].is_multiple_of(align))
}

/// Number of 64 KiB slots a page of class `c` spans.
pub(crate) fn span_slots(c: usize) -> usize {
    (size_of(c) * MIN_BLOCKS_PER_PAGE)
        .div_ceil(SLOT_SIZE)
        .max(1)
}

/// Number of blocks in a page of class `c`.
pub(crate) fn capacity(c: usize) -> usize {
    span_slots(c) * SLOT_SIZE / size_of(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_increasing_multiples_of_16() {
        for c in 0..NUM_CLASSES {
            assert_eq!(SIZES[c] % 16, 0);
            if c > 0 {
                assert!(SIZES[c] > SIZES[c - 1]);
                if SIZES[c - 1] >= 128 {
                    // at most 12.5% apart
                    assert!(SIZES[c] - SIZES[c - 1] <= SIZES[c - 1] / 8);
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 128K iterations; covered natively
    fn class_of_is_the_smallest_fitting_class() {
        for size in 1..=MAX_SMALL {
            let c = class_of(size);
            assert!(SIZES[c] >= size, "size {size} -> class {c}");
            assert!(c == 0 || SIZES[c - 1] < size, "size {size} -> class {c}");
        }
    }

    #[test]
    fn aligned_classes_are_aligned_and_minimal() {
        for align in [32, 64, 128, 4096, 65536] {
            for size in [1, 17, 100, 1000, 5000, 70_000, MAX_SMALL] {
                let l = Layout::from_size_align(size, align).unwrap();
                match class_for(l) {
                    Some(c) => {
                        assert_eq!(SIZES[c] % align, 0);
                        assert!(SIZES[c] >= size);
                    }
                    None => assert!(size.next_multiple_of(align) > MAX_SMALL),
                }
            }
        }
        let l = Layout::from_size_align(8, 2 * SLOT_SIZE).unwrap();
        assert!(class_for(l).is_none());
    }

    #[test]
    fn pages_hold_enough_blocks() {
        for c in 0..NUM_CLASSES {
            assert!(span_slots(c) <= 16);
            assert!(capacity(c) >= 1);
            if span_slots(c) > 1 {
                assert!(capacity(c) >= MIN_BLOCKS_PER_PAGE);
            }
        }
    }
}
