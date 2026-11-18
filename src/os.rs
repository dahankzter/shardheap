//! Operating-system memory: map, unmap, remap, purge.
//!
//! On Linux this is `mmap`/`munmap`/`mremap`/`madvise`. Under Miri, and on other
//! platforms, it falls back to the global allocator so the rest of the crate can be
//! checked by Miri. Mapped memory is always zeroed.

use core::ptr::NonNull;

/// OS page size assumed for alignment and rounding.
pub(crate) const PAGE: usize = 4096;

#[cfg(all(target_os = "linux", not(miri)))]
mod imp {
    use super::PAGE;
    use core::ptr::{self, NonNull};

    /// # Safety
    /// `size` is a nonzero multiple of `PAGE`; `align` is a power of two `>= PAGE`.
    pub(crate) unsafe fn map_aligned(size: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(
            size.is_multiple_of(PAGE) && size > 0 && align.is_power_of_two() && align >= PAGE
        );
        let total = size.checked_add(align - PAGE)?;
        // SAFETY: anonymous private mapping, no aliasing with existing memory.
        let raw = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return None;
        }
        let raw = raw.cast::<u8>();
        let head = (align - (raw.addr() & (align - 1))) & (align - 1);
        let tail = total - head - size;
        // SAFETY: both trimmed ranges lie inside the mapping created above.
        unsafe {
            if head > 0 {
                libc::munmap(raw.cast(), head);
            }
            if tail > 0 {
                libc::munmap(raw.add(head + size).cast(), tail);
            }
        }
        NonNull::new(raw.wrapping_add(head))
    }

    /// # Safety
    /// `p`/`size` describe a mapping returned by `map_aligned` or `remap`.
    pub(crate) unsafe fn unmap(p: NonNull<u8>, size: usize, _align: usize) {
        // SAFETY: caller guarantees the range is one of our mappings.
        unsafe { libc::munmap(p.as_ptr().cast(), size) };
    }

    /// Resizes a mapping, possibly moving it. Page alignment only.
    ///
    /// # Safety
    /// `p`/`old` describe a mapping from `map_aligned`/`remap`; `new` is a nonzero
    /// multiple of `PAGE`. On `None` the old mapping is untouched.
    pub(crate) unsafe fn remap(p: NonNull<u8>, old: usize, new: usize) -> Option<NonNull<u8>> {
        // SAFETY: caller guarantees the range is one of our mappings.
        let r = unsafe { libc::mremap(p.as_ptr().cast(), old, new, libc::MREMAP_MAYMOVE) };
        if r == libc::MAP_FAILED {
            None
        } else {
            NonNull::new(r.cast())
        }
    }

    /// Returns the physical pages to the OS; the range reads as zero afterwards.
    ///
    /// # Safety
    /// The range lies inside one of our mappings and holds no live data.
    pub(crate) unsafe fn purge(p: NonNull<u8>, size: usize) {
        // SAFETY: caller guarantees the range is ours and unused.
        unsafe { libc::madvise(p.as_ptr().cast(), size, libc::MADV_DONTNEED) };
    }
}

#[cfg(not(all(target_os = "linux", not(miri))))]
mod imp {
    use core::ptr::NonNull;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    pub(crate) unsafe fn map_aligned(size: usize, align: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, align).ok()?;
        // SAFETY: `size > 0`.
        NonNull::new(unsafe { alloc_zeroed(layout) })
    }

    pub(crate) unsafe fn unmap(p: NonNull<u8>, size: usize, align: usize) {
        // SAFETY: allocated by `map_aligned` with this exact layout.
        unsafe { dealloc(p.as_ptr(), Layout::from_size_align_unchecked(size, align)) }
    }

    pub(crate) unsafe fn remap(_p: NonNull<u8>, _old: usize, _new: usize) -> Option<NonNull<u8>> {
        None
    }

    pub(crate) unsafe fn purge(_p: NonNull<u8>, _size: usize) {}
}

pub(crate) use imp::{map_aligned, purge, remap, unmap};

use core::alloc::{AllocError, Allocator, Layout};

/// Memory straight from the operating system: every allocation is its own
/// mapping (`mmap` on Linux), rounded up to whole pages and zeroed.
///
/// Every call is a system call, so this is a chunk source for
/// [`Arena`](crate::Arena) (its default), not an allocator for objects.
#[derive(Clone, Copy, Debug, Default)]
pub struct Os;

/// The mapping size and alignment for `layout`. Every size between a request and
/// the length returned for it rounds to the same mapping, as `deallocate` needs.
#[inline]
fn os_layout(layout: Layout) -> Option<(usize, usize)> {
    let size = layout.size().max(1).checked_next_multiple_of(PAGE)?;
    Some((size, layout.align().max(PAGE)))
}

// SAFETY: each block is a fresh mapping of its own, valid until deallocated;
// `deallocate` unmaps exactly the mapping `allocate` made, because every layout
// that fits the block rounds to the same size.
unsafe impl Allocator for Os {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (size, align) = os_layout(layout).ok_or(AllocError)?;
        // SAFETY: nonzero multiple of PAGE; power-of-two alignment >= PAGE.
        let p = unsafe { map_aligned(size, align) }.ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(p, size))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate(layout) // fresh mappings are zeroed
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // `allocate` succeeded with a layout that rounds the same way.
        if let Some((size, align)) = os_layout(layout) {
            // SAFETY: caller's contract: `ptr` is a mapping made by `allocate`.
            unsafe { unmap(ptr, size, align) }
        }
    }
}

/// Whether `remap` can ever succeed on this platform.
#[allow(dead_code)]
pub(crate) const CAN_REMAP: bool = cfg!(all(target_os = "linux", not(miri)));

#[allow(dead_code)]
fn _assert_signatures() {
    let _: unsafe fn(usize, usize) -> Option<NonNull<u8>> = map_aligned;
}
