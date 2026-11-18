//! Allocator comparison through the stable `Allocator` trait. Every competitor is
//! driven by the same `Vec::new_in` / `Box::new_in` code.
//!
//! - `system`: glibc malloc (`std::alloc::System`).
//! - `mimalloc`: mimalloc's default (thread-local) heap, via its `GlobalAlloc`.
//! - `mi_heap`: a first-class mimalloc heap (`mi_heap_new`), the closest C analogue
//!   of `LocalHeap`.
//! - `bumpalo`: a bump arena, reset after every iteration, using bumpalo's own
//!   free/grow/shrink of its most recent block (via `allocator-api2`).
//! - `shardheap`: `LocalHeap`.
//! - `shardheap_arena`: `Arena`, reset after every iteration like `bumpalo`.

use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::ptr::NonNull;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use shardheap::{Arena, LocalHeap};

// ---- competitors ------------------------------------------------------------------

struct MiGlobal;

// SAFETY: forwards to mimalloc, which upholds the allocator contract.
unsafe impl Allocator for MiGlobal {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if l.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(NonNull::<u8>::dangling(), 0));
        }
        // SAFETY: nonzero size.
        let p = unsafe { GlobalAlloc::alloc(&mimalloc::MiMalloc, l) };
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, l.size()))
            .ok_or(AllocError)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        if l.size() != 0 {
            // SAFETY: allocated above with `l`.
            unsafe { GlobalAlloc::dealloc(&mimalloc::MiMalloc, p.as_ptr(), l) }
        }
    }
    unsafe fn grow(
        &self,
        p: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old.size() == 0 || old.align() != new.align() {
            let n = self.allocate(new)?;
            // SAFETY: disjoint blocks.
            unsafe {
                std::ptr::copy_nonoverlapping(p.as_ptr(), n.as_ptr().cast(), old.size());
                self.deallocate(p, old);
            }
            return Ok(n);
        }
        // SAFETY: `p` allocated with `old`.
        let q = unsafe { GlobalAlloc::realloc(&mimalloc::MiMalloc, p.as_ptr(), old, new.size()) };
        NonNull::new(q)
            .map(|q| NonNull::slice_from_raw_parts(q, new.size()))
            .ok_or(AllocError)
    }
}

struct MiHeap(*mut libmimalloc_sys::mi_heap_t);

impl MiHeap {
    fn new() -> Self {
        // SAFETY: creates a heap owned by this thread.
        MiHeap(unsafe { libmimalloc_sys::mi_heap_new() })
    }
}

impl Drop for MiHeap {
    fn drop(&mut self) {
        // SAFETY: heap created by `mi_heap_new`; everything in it is dead.
        unsafe { libmimalloc_sys::mi_heap_destroy(self.0) }
    }
}

// SAFETY: forwards to mimalloc's heap API.
unsafe impl Allocator for MiHeap {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: valid heap; mimalloc handles size 0.
        let p = unsafe { libmimalloc_sys::mi_heap_malloc_aligned(self.0, l.size(), l.align()) };
        NonNull::new(p.cast::<u8>())
            .map(|p| NonNull::slice_from_raw_parts(p, l.size()))
            .ok_or(AllocError)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, _l: Layout) {
        // SAFETY: allocated from this heap.
        unsafe { libmimalloc_sys::mi_free(p.as_ptr().cast()) }
    }
    unsafe fn grow(
        &self,
        p: NonNull<u8>,
        _old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: allocated from this heap.
        let q = unsafe {
            libmimalloc_sys::mi_heap_realloc_aligned(
                self.0,
                p.as_ptr().cast(),
                new.size(),
                new.align(),
            )
        };
        NonNull::new(q.cast::<u8>())
            .map(|q| NonNull::slice_from_raw_parts(q, new.size()))
            .ok_or(AllocError)
    }
}

struct Bump(bumpalo::Bump);

/// Forwards to bumpalo's own allocator implementation (via `allocator-api2`), so
/// it frees and grows its most recent block itself rather than through defaults.
// SAFETY: bumpalo upholds the allocator contract; blocks live until `reset`.
unsafe impl Allocator for Bump {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocator_api2::alloc::Allocator::allocate(&&self.0, l).map_err(|_| AllocError)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        // SAFETY: allocated by this bump with `l`.
        unsafe { allocator_api2::alloc::Allocator::deallocate(&&self.0, p, l) }
    }
    unsafe fn grow(
        &self,
        p: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract.
        unsafe { allocator_api2::alloc::Allocator::grow(&&self.0, p, old, new) }
            .map_err(|_| AllocError)
    }
    unsafe fn shrink(
        &self,
        p: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract.
        unsafe { allocator_api2::alloc::Allocator::shrink(&&self.0, p, old, new) }
            .map_err(|_| AllocError)
    }
}

// ---- workloads --------------------------------------------------------------------

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Allocate one 64-byte box and free it, repeatedly: the hottest fast path.
fn lifo<A: Allocator>(a: &A, n: usize) {
    for i in 0..n {
        let b = Box::new_in([i as u8; 64], a);
        black_box(&b);
    }
}

/// Allocate `sizes.len()` buffers of mixed small sizes, all live at once, then free
/// them in allocation order: like a batch of tasks or requests.
fn batch<A: Allocator>(a: &A, sizes: &[usize]) {
    let mut live: Vec<Vec<u8, &A>> = Vec::with_capacity(sizes.len());
    for &s in sizes {
        let mut v = Vec::with_capacity_in(s, a);
        v.push(1);
        live.push(v);
    }
    black_box(&live);
    drop(live);
}

/// Grow one vector to 8 MiB by pushing: reallocation through every size class and
/// then the large path.
fn growth<A: Allocator>(a: &A, n: usize) {
    let mut v: Vec<u64, &A> = Vec::new_in(a);
    for i in 0..n as u64 {
        v.push(i);
    }
    black_box(&v);
}

/// Build a binary tree of boxed nodes, sum it, drop it.
type Child<'a, A> = Box<Node<'a, A>, &'a A>;

struct Node<'a, A: Allocator> {
    val: u64,
    kids: Option<(Child<'a, A>, Child<'a, A>)>,
}

fn tree<'a, A: Allocator>(a: &'a A, depth: u32) -> Box<Node<'a, A>, &'a A> {
    let kids = (depth > 0).then(|| (tree(a, depth - 1), tree(a, depth - 1)));
    Box::new_in(
        Node {
            val: depth as u64,
            kids,
        },
        a,
    )
}

fn sum<A: Allocator>(n: &Node<'_, A>) -> u64 {
    n.val + n.kids.as_ref().map_or(0, |(l, r)| sum(l) + sum(r))
}

// ---- harness ----------------------------------------------------------------------

/// Runs `$body` against every allocator. Persistent allocators stay warm across
/// iterations; the bump arena is reset inside each iteration, which is how an arena
/// is used and part of its cost.
macro_rules! bench_all {
    ($c:expr, $group:expr, $elems:expr, |$a:ident| $body:expr) => {{
        let mut g = $c.benchmark_group($group);
        g.throughput(Throughput::Elements($elems));
        {
            let $a = &System;
            g.bench_function(BenchmarkId::from_parameter("system"), |b| b.iter(|| $body));
        }
        {
            let $a = &MiGlobal;
            g.bench_function(BenchmarkId::from_parameter("mimalloc"), |b| {
                b.iter(|| $body)
            });
        }
        {
            let heap = MiHeap::new();
            let $a = &heap;
            g.bench_function(BenchmarkId::from_parameter("mi_heap"), |b| b.iter(|| $body));
        }
        {
            let mut bump = Bump(bumpalo::Bump::new());
            g.bench_function(BenchmarkId::from_parameter("bumpalo"), |b| {
                b.iter(|| {
                    {
                        let $a = &bump;
                        $body;
                    }
                    bump.0.reset();
                })
            });
        }
        {
            let heap = LocalHeap::new();
            let $a = &heap;
            g.bench_function(BenchmarkId::from_parameter("shardheap"), |b| {
                b.iter(|| $body)
            });
        }
        {
            let mut arena = Arena::new();
            g.bench_function(BenchmarkId::from_parameter("shardheap_arena"), |b| {
                b.iter(|| {
                    {
                        let $a = &arena;
                        $body;
                    }
                    arena.reset();
                })
            });
        }
        g.finish();
    }};
}

fn benches(c: &mut Criterion) {
    let n = 10_000;
    bench_all!(c, "lifo_64B", n as u64, |a| lifo(a, n));

    let mut r = rng(7);
    let sizes: Vec<usize> = (0..10_000)
        .map(|_| match r() % 100 {
            0..=59 => 16 + (r() % 112) as usize,
            60..=89 => 128 + (r() % 896) as usize,
            _ => 1024 + (r() % 7168) as usize,
        })
        .collect();
    bench_all!(c, "batch_mixed", sizes.len() as u64, |a| batch(a, &sizes));

    let pushes = 1 << 20;
    bench_all!(c, "vec_growth_8MiB", pushes as u64, |a| growth(a, pushes));

    let depth = 14;
    bench_all!(c, "tree_32k_nodes", (1u64 << (depth + 1)) - 1, |a| {
        black_box(sum(&tree(a, depth)))
    });
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30)
}

criterion_group! {
    name = alloc;
    config = config();
    targets = benches
}
criterion_main!(alloc);
