//! [`Arena`]: bump allocation with bulk reset, for objects that die together.

use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::fmt;
use core::ptr::{self, NonNull};

use crate::os::{Os, PAGE};

/// First chunk size; later chunks double up to [`MAX_CHUNK`].
const FIRST_CHUNK: usize = 64 << 10;
/// Largest chunk made by doubling. Bigger requests get a chunk of their own size.
const MAX_CHUNK: usize = 4 << 20;

/// Header at the start of every chunk; chunks form a list, newest first.
#[repr(C)]
struct Chunk {
    prev: *mut Chunk,
    size: usize,
}

const HEADER: usize = core::mem::size_of::<Chunk>();

/// Counters kept by an [`Arena`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Bytes bumped since the last reset (including alignment padding), minus
    /// bytes given back by freeing or shrinking the most recent block. Derived from
    /// the bump pointers, so the fast path keeps no counter.
    pub allocated_bytes: usize,
    /// Bytes held in chunks, taken from the backing allocator.
    pub chunk_bytes: usize,
    /// Chunks currently held.
    pub chunks: usize,
}

/// A bump arena: allocation is an aligned pointer increment, freeing is free, and
/// [`reset`](Arena::reset) releases everything at once.
///
/// Use it for allocations that die together: a request, a compiler pass, a frame,
/// a batch. Memory is not reused until `reset`, so it is the wrong choice for
/// objects with independent lifetimes; use [`LocalHeap`](crate::LocalHeap) there.
///
/// It bumps upward, so the most recent block can be grown in place without
/// copying: a `Vec` that keeps pushing in an arena never moves while it is the
/// latest allocation and fits in its chunk. After `reset` the arena keeps one
/// chunk big enough for the previous round, so steady-state workloads stop
/// touching the OS entirely.
///
/// Like `LocalHeap` it is `!Send` and `!Sync`, and the borrow checker keeps every
/// collection from outliving it or surviving a reset.
///
/// Chunks come from the backing allocator `A`: by default [`Os`] (fresh mappings),
/// or a [`LocalHeap`](crate::LocalHeap) with [`Arena::new_in`], so short-lived
/// arena memory and long-lived heap memory share one pool of resident pages and
/// one set of counters.
///
/// ```
/// use shardheap::{Arena, LocalHeap};
///
/// let heap = LocalHeap::new();
/// let mut arena = Arena::new_in(&heap);
/// let mut kept: Vec<Vec<u8, &LocalHeap>> = Vec::new();
/// for request in 0..3u8 {
///     let mut scratch: Vec<u8, &Arena<&LocalHeap>> = Vec::new_in(&arena);
///     scratch.extend((0..100).map(|b| b + request));
///     // The few survivors are copied into the heap; the compiler rejects any
///     // that would still point into the arena after `reset`.
///     let mut survivor = Vec::new_in(&heap);
///     survivor.extend_from_slice(&scratch[..2]);
///     kept.push(survivor);
///     drop(scratch);
///     arena.reset();
/// }
/// assert_eq!(kept[2][1], 3);
/// ```
///
/// ```
/// use shardheap::Arena;
///
/// let mut arena = Arena::new();
/// for request in 0..3 {
///     let mut v: Vec<u32, &Arena> = Vec::new_in(&arena);
///     v.extend(0..10_000); // grows in place: no copies
///     let b = Box::new_in(request, &arena);
///     assert_eq!(*b, request);
///     drop((v, b));
///     arena.reset(); // everything at once
/// }
/// ```
///
/// ```compile_fail,E0502
/// let mut arena = shardheap::Arena::new();
/// let v: Vec<u8, &shardheap::Arena> = Vec::with_capacity_in(8, &arena);
/// arena.reset();
/// drop(v);
/// ```
pub struct Arena<A: Allocator = Os> {
    /// Next free byte in the current chunk.
    ptr: Cell<*mut u8>,
    /// End of the current chunk.
    end: Cell<*mut u8>,
    /// Current chunk; all chunks are linked through `prev`.
    chunk: Cell<*mut Chunk>,
    next_size: Cell<usize>,
    /// Bytes used in chunks that are no longer current.
    retired: Cell<usize>,
    /// Chunk counters (`allocated_bytes` is derived in `stats`).
    stats: Cell<ArenaStats>,
    backing: A,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator> fmt::Debug for Arena<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("stats", &self.stats())
            .finish()
    }
}

#[inline(always)]
fn dangling(align: usize) -> NonNull<[u8]> {
    // SAFETY: an alignment is never zero.
    let p = unsafe { NonNull::new_unchecked(ptr::without_provenance_mut::<u8>(align)) };
    NonNull::slice_from_raw_parts(p, 0)
}

#[inline(always)]
fn block(p: *mut u8, len: usize) -> NonNull<[u8]> {
    // SAFETY: callers pass pointers into a chunk, never null.
    NonNull::slice_from_raw_parts(unsafe { NonNull::new_unchecked(p) }, len)
}

impl Arena {
    /// An empty arena that maps its chunks from the OS. Maps nothing until the
    /// first allocation.
    pub const fn new() -> Self {
        Self::new_in(Os)
    }
}

/// The layout every chunk is requested and returned with.
#[inline]
fn chunk_layout(size: usize) -> Layout {
    // A chunk size is a nonzero multiple of PAGE that was successfully requested
    // before (or is about to be, after the overflow checks in `alloc_slow`).
    Layout::from_size_align(size, PAGE).expect("chunk sizes fit a Layout")
}

impl<A: Allocator> Arena<A> {
    /// An empty arena that takes its chunks from `backing`. Takes nothing until
    /// the first allocation.
    pub const fn new_in(backing: A) -> Self {
        Arena {
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunk: Cell::new(ptr::null_mut()),
            next_size: Cell::new(FIRST_CHUNK),
            retired: Cell::new(0),
            stats: Cell::new(ArenaStats {
                allocated_bytes: 0,
                chunk_bytes: 0,
                chunks: 0,
            }),
            backing,
        }
    }

    /// Current counters.
    pub fn stats(&self) -> ArenaStats {
        let c = self.chunk.get();
        let current = if c.is_null() {
            0
        } else {
            self.ptr.get().addr() - (c.addr() + HEADER)
        };
        ArenaStats {
            allocated_bytes: self.retired.get() + current,
            ..self.stats.get()
        }
    }

    fn update(&self, f: impl FnOnce(&mut ArenaStats)) {
        let mut s = self.stats.get();
        f(&mut s);
        self.stats.set(s);
    }

    /// Frees every allocation at once. Taking `&mut self` proves no collection still
    /// borrows the arena.
    ///
    /// If the round fit in one chunk, that chunk is kept (its pages stay resident).
    /// If it spilled over several, they are replaced by a single chunk as large as
    /// all of them together, so the next round of the same shape fits in one chunk
    /// and never touches the OS.
    pub fn reset(&mut self) {
        // SAFETY: `&mut self`: no block is reachable, so every chunk is unused.
        unsafe {
            let (mut count, mut total) = (0usize, 0usize);
            let mut c = self.chunk.get();
            while !c.is_null() {
                count += 1;
                total = total.saturating_add((*c).size);
                c = (*c).prev;
            }
            if count > 1 {
                self.release_chunks();
                if let Ok(b) = self.backing.allocate(chunk_layout(total)) {
                    self.install(b.cast(), b.len());
                }
            }
            match NonNull::new(self.chunk.get()) {
                None => {
                    self.ptr.set(ptr::null_mut());
                    self.end.set(ptr::null_mut());
                }
                Some(keep) => {
                    let keep = keep.as_ptr();
                    self.ptr.set(keep.cast::<u8>().add(HEADER));
                    self.end.set(keep.cast::<u8>().add((*keep).size));
                    self.next_size
                        .set(self.next_size.get().max((*keep).size).min(MAX_CHUNK));
                }
            }
            self.retired.set(0);
        }
    }

    /// Makes the fresh block `base` of `size` bytes the current chunk.
    ///
    /// # Safety
    /// `base` is a block of `size >= HEADER` bytes just allocated from `backing`
    /// with `chunk_layout(n)` for some `n <= size`.
    unsafe fn install(&self, base: NonNull<u8>, size: usize) {
        let old = self.chunk.get();
        if !old.is_null() {
            let used = self.ptr.get().addr() - (old.addr() + HEADER);
            self.retired.set(self.retired.get() + used);
        }
        // SAFETY: caller's contract.
        unsafe {
            let c = base.as_ptr().cast::<Chunk>();
            c.write(Chunk {
                prev: self.chunk.get(),
                size,
            });
            self.chunk.set(c);
            self.ptr.set(base.as_ptr().add(HEADER));
            self.end.set(base.as_ptr().add(size));
        }
        self.update(|s| {
            s.chunk_bytes += size;
            s.chunks += 1;
        });
    }

    /// Returns every chunk to the backing allocator.
    ///
    /// # Safety
    /// No block of this arena is live.
    unsafe fn release_chunks(&self) {
        let mut c = self.chunk.replace(ptr::null_mut());
        while !c.is_null() {
            // SAFETY: `c` is one of our chunks, allocated from `backing` with a
            // layout that fits its recorded size; caller's contract: it is unused.
            unsafe {
                let (prev, size) = ((*c).prev, (*c).size);
                self.update(|s| {
                    s.chunk_bytes -= size;
                    s.chunks -= 1;
                });
                self.backing
                    .deallocate(NonNull::new_unchecked(c.cast()), chunk_layout(size));
                c = prev;
            }
        }
    }

    /// Bump-allocates from the current chunk, or `None` if it does not fit.
    #[inline(always)]
    fn bump(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        let p = self.ptr.get();
        let pad = p.addr().wrapping_neg() & (layout.align() - 1);
        let room = self.end.get().addr().wrapping_sub(p.addr());
        // Unsigned `room` is garbage only before the first chunk, where both
        // pointers are null and `room == 0`.
        if pad.checked_add(layout.size())? > room {
            return None;
        }
        let start = p.wrapping_add(pad);
        self.ptr.set(start.wrapping_add(layout.size()));
        Some(block(start, layout.size()))
    }

    #[cold]
    #[inline(never)]
    fn alloc_slow(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let need = HEADER
            .checked_add(layout.align())
            .and_then(|n| n.checked_add(layout.size()))
            .and_then(|n| n.checked_next_multiple_of(PAGE))
            .ok_or(AllocError)?;
        let size = need.max(self.next_size.get());
        let chunk = Layout::from_size_align(size, PAGE).map_err(|_| AllocError)?;
        let b = self.backing.allocate(chunk)?;
        // SAFETY: a fresh block of `b.len() >= size` bytes from `backing`.
        unsafe { self.install(b.cast(), b.len()) };
        if size == self.next_size.get() {
            self.next_size.set((size * 2).min(MAX_CHUNK).max(size));
        }
        Ok(self.bump(layout).expect("a fresh chunk fits the request"))
    }

    /// Whether `p..p + size` (nonempty) is the most recent block. The `Layout`
    /// passed to free and resize carries the size, so the end address identifies
    /// the block: no block from an older chunk can end at the current chunk's
    /// `ptr`, which lies past that chunk's header. After a free, the block below
    /// becomes the most recent one if no padding separates them.
    #[inline(always)]
    fn is_last(&self, p: *mut u8, size: usize) -> bool {
        p.wrapping_add(size) == self.ptr.get()
    }
}

impl<A: Allocator> Drop for Arena<A> {
    fn drop(&mut self) {
        // SAFETY: dropping: no block is reachable.
        unsafe { self.release_chunks() }
    }
}

// SAFETY:
// - Blocks are disjoint: `ptr` only moves forward, except when the most recent
//   block is freed or shrunk, which returns exactly its own bytes.
// - Blocks stay valid until the arena is reset or dropped, both of which need
//   `&mut`/ownership, which no live collection allows.
// - Nothing unwinds: OS failure returns `AllocError`.
unsafe impl<A: Allocator> Allocator for Arena<A> {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(dangling(layout.align()));
        }
        match self.bump(layout) {
            Some(b) => Ok(b),
            None => self.alloc_slow(layout),
        }
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let p = ptr.as_ptr();
        if layout.size() != 0 && self.is_last(p, layout.size()) {
            self.ptr.set(p);
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let p = ptr.as_ptr();
        if old.size() == 0 {
            return self.allocate(new);
        }
        // `is_last` first: only then is `p` inside the current chunk, so the room
        // computation below cannot underflow.
        if self.is_last(p, old.size())
            && p.addr() & (new.align() - 1) == 0
            && new.size() <= self.end.get().addr() - p.addr()
        {
            // Extend in place: the bytes after the most recent block are free.
            self.ptr.set(p.wrapping_add(new.size()));
            return Ok(block(p, new.size()));
        }
        let n = self.allocate(new)?;
        // SAFETY: `n` is a fresh block disjoint from `p`; `p` holds `old.size()` bytes.
        unsafe { ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), old.size()) };
        Ok(n)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let p = ptr.as_ptr();
        if new.size() == 0 {
            // SAFETY: caller's contract for `ptr`/`old`.
            unsafe { self.deallocate(ptr, old) };
            return Ok(dangling(new.align()));
        }
        if p.addr() & (new.align() - 1) == 0 {
            if self.is_last(p, old.size()) {
                self.ptr.set(p.wrapping_add(new.size()));
            }
            return Ok(block(p, new.size()));
        }
        let n = self.allocate(new)?;
        // SAFETY: fresh disjoint block; keep the first `new.size()` bytes.
        unsafe { ptr::copy_nonoverlapping(p, n.as_ptr().cast::<u8>(), new.size()) };
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_grows_in_place() {
        let arena = Arena::new();
        let mut v: Vec<u64, &Arena> = Vec::with_capacity_in(1, &arena);
        let p0 = v.as_ptr();
        v.extend(0..5_000); // 40 KB, within the first 64 KiB chunk
        assert_eq!(v.as_ptr(), p0, "the latest block grew without moving");
        assert_eq!(arena.stats().chunks, 1);
    }

    #[test]
    fn lifo_free_and_oversized_requests() {
        let arena = Arena::new();
        let l = Layout::from_size_align(100, 8).unwrap();
        let a = arena.allocate(l).unwrap();
        let before = arena.stats().allocated_bytes;
        // SAFETY: most recent block, allocated with `l`.
        unsafe { arena.deallocate(a.cast(), l) };
        assert_eq!(arena.stats().allocated_bytes, before - 100);
        let b = arena.allocate(l).unwrap();
        assert_eq!(a.cast::<u8>(), b.cast::<u8>(), "freed bytes reused");
        let big = Layout::from_size_align(10 << 20, 4096).unwrap();
        let c = arena.allocate(big).unwrap();
        assert_eq!(c.cast::<u8>().as_ptr().addr() % 4096, 0);
        assert_eq!(arena.stats().chunks, 2);
    }

    #[test]
    fn reset_coalesces_into_one_chunk() {
        let mut arena = Arena::new();
        for _ in 0..3 {
            for _ in 0..200 {
                let _ = arena
                    .allocate(Layout::from_size_align(8 << 10, 8).unwrap())
                    .unwrap();
            }
            let _ = arena
                .allocate(Layout::from_size_align(6 << 20, 8).unwrap())
                .unwrap();
            arena.reset();
            let s = arena.stats();
            assert_eq!((s.chunks, s.allocated_bytes), (1, 0));
        }
        // The single replacement chunk holds a whole round.
        for _ in 0..200 {
            let _ = arena
                .allocate(Layout::from_size_align(8 << 10, 8).unwrap())
                .unwrap();
        }
        assert_eq!(arena.stats().chunks, 1);
    }
}
