//! [`StackArena`]: bump allocation from a caller-provided buffer, typically on the
//! stack, with a fallback allocator once the buffer is full.

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};
use std::alloc::Global;

/// A bump arena over a borrowed buffer. Scratch collections live in the buffer
/// (usually a local array) with no heap traffic; requests that do not fit go to the
/// fallback allocator, because `Vec` aborts on allocation failure.
///
/// The borrow of the buffer ties every collection to it: nothing allocated here can
/// outlive the stack frame that owns the buffer.
///
/// The most recent allocation can be freed or grown in place (LIFO), which is the
/// common pattern for a scratch `Vec` that keeps pushing.
///
/// ```
/// use std::mem::MaybeUninit;
/// use shardheap::StackArena;
///
/// let mut buf = [MaybeUninit::<u8>::uninit(); 4096];
/// let arena = StackArena::new(&mut buf);
/// let mut v: Vec<u32, &StackArena> = Vec::new_in(&arena);
/// v.extend(0..100);            // grows in place inside `buf`
/// assert!(arena.contains(v.as_ptr().cast()));
/// v.extend(0..10_000);         // too big: moves to the global allocator
/// assert!(!arena.contains(v.as_ptr().cast()));
/// ```
///
/// ```compile_fail,E0597
/// use std::mem::MaybeUninit;
/// let v = {
///     let mut buf = [MaybeUninit::<u8>::uninit(); 64];
///     let arena = shardheap::StackArena::new(&mut buf);
///     let v: Vec<u8, shardheap::StackArena<'_>> = Vec::with_capacity_in(8, arena);
///     v
/// };
/// ```
pub struct StackArena<'buf, A: Allocator = Global> {
    start: *mut u8,
    end: *mut u8,
    /// Next free byte.
    cur: Cell<*mut u8>,
    /// Start of the most recent allocation still live at the top, or null.
    last: Cell<*mut u8>,
    fallback: A,
    _buf: PhantomData<&'buf mut [MaybeUninit<u8>]>,
}

impl<'buf> StackArena<'buf> {
    /// An arena over `buf`, falling back to the global allocator.
    pub fn new(buf: &'buf mut [MaybeUninit<u8>]) -> Self {
        Self::with_fallback(buf, Global)
    }
}

impl<'buf, A: Allocator> StackArena<'buf, A> {
    /// An arena over `buf`, falling back to `fallback`.
    pub fn with_fallback(buf: &'buf mut [MaybeUninit<u8>], fallback: A) -> Self {
        let start = buf.as_mut_ptr().cast::<u8>();
        let end = start.wrapping_add(buf.len());
        StackArena {
            start,
            end,
            cur: Cell::new(start),
            last: Cell::new(ptr::null_mut()),
            fallback,
            _buf: PhantomData,
        }
    }

    /// Whether `p` points into the arena's buffer.
    pub fn contains(&self, p: *const u8) -> bool {
        (self.start.addr()..self.end.addr()).contains(&p.addr())
    }

    /// Bytes still free in the buffer (ignoring alignment padding).
    pub fn remaining(&self) -> usize {
        self.end.addr() - self.cur.get().addr()
    }

    /// Bump-allocates from the buffer, or `None` if it does not fit.
    fn bump(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        let cur = self.cur.get();
        let pad = (layout.align() - (cur.addr() & (layout.align() - 1))) & (layout.align() - 1);
        let avail = self.end.addr() - cur.addr();
        let needed = pad.checked_add(layout.size())?;
        if needed > avail {
            return None;
        }
        let p = cur.wrapping_add(pad);
        self.cur.set(p.wrapping_add(layout.size()));
        self.last.set(p);
        // SAFETY: `p` lies in the buffer, which is never null.
        Some(NonNull::slice_from_raw_parts(
            unsafe { NonNull::new_unchecked(p) },
            layout.size(),
        ))
    }

    fn dangling(align: usize) -> NonNull<[u8]> {
        // SAFETY: an alignment is never zero.
        let p = unsafe { NonNull::new_unchecked(ptr::without_provenance_mut::<u8>(align)) };
        NonNull::slice_from_raw_parts(p, 0)
    }

    fn is_top(&self, p: *mut u8, size: usize) -> bool {
        self.last.get() == p && p.wrapping_add(size) == self.cur.get()
    }
}

// SAFETY:
// - Buffer blocks are disjoint: `cur` only moves forward, except when the top block
//   is freed, which makes its bytes available again.
// - Buffer blocks stay valid while the arena exists, and the arena cannot outlive
//   the borrowed buffer; fallback blocks are the fallback's responsibility.
// - Nothing here unwinds.
unsafe impl<A: Allocator> Allocator for StackArena<'_, A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(Self::dangling(layout.align()));
        }
        match self.bump(layout) {
            Some(b) => Ok(b),
            None => self.fallback.allocate(layout),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let p = ptr.as_ptr();
        if layout.size() == 0 {
            // Zero-sized blocks are dangling and owned by nobody.
        } else if self.contains(p) {
            if self.is_top(p, layout.size()) {
                self.cur.set(p);
                self.last.set(ptr::null_mut());
            }
        } else {
            // SAFETY: a nonzero block outside the buffer came from the fallback.
            unsafe { self.fallback.deallocate(ptr, layout) }
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old.size() == 0 {
            return self.allocate(new);
        }
        let p = ptr.as_ptr();
        if self.contains(p) {
            let fits =
                p.addr() & (new.align() - 1) == 0 && new.size() <= self.end.addr() - p.addr();
            if self.is_top(p, old.size()) && fits {
                self.cur.set(p.wrapping_add(new.size()));
                return Ok(NonNull::slice_from_raw_parts(ptr, new.size()));
            }
            let n = self.allocate(new)?;
            // SAFETY: `n` is a fresh block disjoint from `p` (a new bump region or a
            // fallback block); copy the old contents.
            unsafe {
                ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), old.size());
                self.deallocate(ptr, old);
            }
            return Ok(n);
        }
        // SAFETY: not ours, so it came from the fallback with layout `old`.
        unsafe { self.fallback.grow(ptr, old, new) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if new.size() == 0 {
            // SAFETY: caller's contract for `ptr`/`old`.
            unsafe { self.deallocate(ptr, old) };
            return Ok(Self::dangling(new.align()));
        }
        let p = ptr.as_ptr();
        if self.contains(p) {
            if p.addr() & (new.align() - 1) == 0 {
                if self.is_top(p, old.size()) {
                    self.cur.set(p.wrapping_add(new.size()));
                }
                return Ok(NonNull::slice_from_raw_parts(ptr, new.size()));
            }
            let n = self.allocate(new)?;
            // SAFETY: fresh disjoint block; keep the first `new.size()` bytes.
            unsafe {
                ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), new.size());
                self.deallocate(ptr, old);
            }
            return Ok(n);
        }
        // SAFETY: not ours, so it came from the fallback with layout `old`.
        unsafe { self.fallback.shrink(ptr, old, new) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_lifo_and_fallback() {
        let mut buf = [MaybeUninit::<u8>::uninit(); 256];
        let a = StackArena::new(&mut buf);
        let l = Layout::from_size_align(100, 8).unwrap();
        let x = a.allocate(l).unwrap();
        assert!(a.contains(x.as_ptr().cast()));
        let before = a.remaining();
        // SAFETY: `x` came from `a` with `l` and is the top block.
        unsafe { a.deallocate(x.cast(), l) };
        assert_eq!(a.remaining(), before + 100);
        let big = Layout::from_size_align(1000, 8).unwrap();
        let y = a.allocate(big).unwrap();
        assert!(!a.contains(y.as_ptr().cast()));
        // SAFETY: from the fallback with `big`.
        unsafe { a.deallocate(y.cast(), big) };
    }

    #[test]
    fn vec_grows_in_place_then_spills() {
        let mut buf = [MaybeUninit::<u8>::uninit(); 1024];
        let a = StackArena::new(&mut buf);
        let mut v: Vec<u64, &StackArena> = Vec::with_capacity_in(4, &a);
        let p0 = v.as_ptr();
        v.extend(0..64); // 512 bytes: fits, grows in place
        assert_eq!(v.as_ptr(), p0);
        v.extend(0..1000); // spills
        assert!(!a.contains(v.as_ptr().cast()));
        assert_eq!(
            v.iter().sum::<u64>(),
            (0..64).sum::<u64>() + (0..1000).sum::<u64>()
        );
    }

    #[test]
    fn alignment_padding() {
        let mut buf = [MaybeUninit::<u8>::uninit(); 512];
        let a = StackArena::new(&mut buf);
        let _ = a.allocate(Layout::from_size_align(3, 1).unwrap()).unwrap();
        let b = a
            .allocate(Layout::from_size_align(64, 64).unwrap())
            .unwrap();
        assert_eq!(b.as_ptr().cast::<u8>().addr() % 64, 0);
    }
}
