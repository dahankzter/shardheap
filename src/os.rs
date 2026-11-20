//! Operating-system memory: map, unmap, remap, purge.
//!
//! On Linux this is `mmap`/`munmap`/`mremap`/`madvise`. Under Miri, and on other
//! platforms, it falls back to the system allocator so the rest of the crate can
//! be checked by Miri. Mapped memory is always zeroed.

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

    /// Reserves `size` bytes of zeroed address space without committing it
    /// (`MAP_NORESERVE`): pages become resident as they are touched.
    ///
    /// # Safety
    /// `size` is a nonzero multiple of `PAGE`.
    pub(crate) unsafe fn reserve(size: usize) -> Option<NonNull<u8>> {
        // SAFETY: anonymous private mapping, no aliasing with existing memory.
        let raw = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        (raw != libc::MAP_FAILED).then(|| NonNull::new(raw.cast()))?
    }

    /// # Safety
    /// `p` and `size` are a reservation from `reserve`, no longer used.
    pub(crate) unsafe fn unreserve(p: NonNull<u8>, size: usize) {
        // SAFETY: caller's contract.
        unsafe { libc::munmap(p.as_ptr().cast(), size) };
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

    /// Asks for (`true`) or refuses (`false`) transparent huge pages on a range.
    ///
    /// # Safety
    /// The range is one of our mappings.
    pub(crate) unsafe fn advise_huge(p: NonNull<u8>, size: usize, huge: bool) {
        let advice = if huge {
            libc::MADV_HUGEPAGE
        } else {
            libc::MADV_NOHUGEPAGE
        };
        // SAFETY: caller guarantees the range is ours. Failure (no THP support)
        // leaves the kernel's policy in place, which is harmless.
        unsafe { libc::madvise(p.as_ptr().cast(), size, advice) };
    }

    /// Returns the physical pages to the OS; the range reads as zero afterwards.
    ///
    /// # Safety
    /// The range lies inside one of our mappings and holds no live data.
    pub(crate) unsafe fn purge(p: NonNull<u8>, size: usize) {
        // SAFETY: caller guarantees the range is ours and unused.
        unsafe { libc::madvise(p.as_ptr().cast(), size, libc::MADV_DONTNEED) };
    }

    /// Milliseconds on a monotonic clock. The coarse clock is read from the
    /// vDSO without a syscall, in a few nanoseconds; its resolution (a few
    /// milliseconds) is plenty for purge epochs.
    pub(crate) fn now_ms() -> u64 {
        let mut t = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: plain clock read into a local.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_COARSE, &mut t) };
        t.tv_sec as u64 * 1000 + t.tv_nsec as u64 / 1_000_000
    }
}

#[cfg(not(all(target_os = "linux", not(miri))))]
mod imp {
    use core::ptr::NonNull;
    // `System`, not the global allocator: that may be shardheap itself.
    use std::alloc::{GlobalAlloc, Layout, System};

    pub(crate) unsafe fn map_aligned(size: usize, align: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, align).ok()?;
        // SAFETY: `size > 0`.
        NonNull::new(unsafe { System.alloc_zeroed(layout) })
    }

    pub(crate) unsafe fn unmap(p: NonNull<u8>, size: usize, align: usize) {
        // SAFETY: allocated by `map_aligned` with this exact layout.
        unsafe { System.dealloc(p.as_ptr(), Layout::from_size_align_unchecked(size, align)) }
    }

    pub(crate) unsafe fn reserve(size: usize) -> Option<NonNull<u8>> {
        // SAFETY: caller's contract.
        unsafe { map_aligned(size, super::PAGE) }
    }

    pub(crate) unsafe fn unreserve(p: NonNull<u8>, size: usize) {
        // SAFETY: caller's contract: a reservation from `reserve`.
        unsafe { unmap(p, size, super::PAGE) }
    }

    pub(crate) unsafe fn remap(_p: NonNull<u8>, _old: usize, _new: usize) -> Option<NonNull<u8>> {
        None
    }

    pub(crate) unsafe fn purge(_p: NonNull<u8>, _size: usize) {}

    pub(crate) unsafe fn advise_huge(_p: NonNull<u8>, _size: usize, _huge: bool) {}

    pub(crate) fn now_ms() -> u64 {
        static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        BASE.get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis() as u64
    }
}

pub(crate) use imp::{advise_huge, map_aligned, now_ms, purge, remap, reserve, unmap, unreserve};

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
