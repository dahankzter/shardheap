//! Request-shaped workload: objects that mostly die together, with a few
//! survivors that outlive their request.
//!
//! Each request allocates 500 filled buffers of mixed sizes and grows one index
//! vector, then ends. A fraction of the buffers survives into a FIFO cache that
//! keeps the survivors of the last `window` requests, so survivors die later and
//! in a different order than they were allocated.
//!
//! ```text
//! cargo run --release --example request_mix -- <allocator> [survive_permille] [window]
//! ```
//!
//! Allocators:
//! - `glibc`, `mimalloc`: everything from one general-purpose allocator.
//! - `heap`: everything from a `LocalHeap`.
//! - `reap`: everything from `LocalHeap::reap`; survivors stay where they are.
//! - `arena`: request objects in an `Arena` backed by the `LocalHeap`, reset after
//!   every request; survivors are copied into the heap.
//! - `bumpalo`: request objects in bumpalo, reset after every request; survivors
//!   are copied into mimalloc. The common setup today.
//!
//! Run one allocator per process so the resident-memory figures are its own.

use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout, System};
use std::collections::VecDeque;
use std::hint::black_box;
use std::ptr::NonNull;
use std::time::Instant;

use shardheap::{Arena, LocalHeap};

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

struct Bump(bumpalo::Bump);

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
}

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const OBJECTS: usize = 500;
const INDEX_LEN: u32 = 2_000;
const WARMUP: usize = 2_000;
const REQUESTS: usize = 20_000;

struct Workload {
    sizes: Vec<usize>,
    /// Per object: whether it survives its request.
    survives: Vec<bool>,
    window: usize,
}

impl Workload {
    fn new(survive_permille: u64, window: usize) -> Self {
        let mut r = rng(42);
        let n = OBJECTS * 64; // cycle through 64 request shapes
        let sizes = (0..n)
            .map(|_| match r() % 100 {
                0..=59 => 16 + (r() % 112) as usize,
                60..=89 => 128 + (r() % 896) as usize,
                _ => 1024 + (r() % 7168) as usize,
            })
            .collect();
        let survives = (0..n).map(|_| r() % 1000 < survive_permille).collect();
        Workload {
            sizes,
            survives,
            window,
        }
    }

    /// One request: `scratch` serves the request's objects, `promote` turns a
    /// survivor into a `Vec` in the long-lived allocator (a move or a copy).
    fn request<A: Allocator + Copy, S: Allocator>(
        &self,
        n: usize,
        scratch: A,
        promote: &impl Fn(Vec<u8, A>) -> Vec<u8, S>,
        cache: &mut VecDeque<Vec<Vec<u8, S>>>,
    ) {
        let base = (n % 64) * OBJECTS;
        let mut objects: Vec<Vec<u8, A>> = Vec::with_capacity(OBJECTS);
        let mut index: Vec<u32, A> = Vec::new_in(scratch);
        for i in 0..OBJECTS {
            let s = self.sizes[base + i];
            let mut v = Vec::with_capacity_in(s, scratch);
            v.resize(s, i as u8);
            objects.push(v);
            index.extend((0..INDEX_LEN / OBJECTS as u32).map(|k| k + i as u32));
        }
        black_box((&objects, &index));
        drop(index);
        let mut kept = Vec::new();
        for (i, v) in objects.into_iter().enumerate() {
            if self.survives[base + i] {
                kept.push(promote(v));
            }
        }
        cache.push_back(kept);
        if cache.len() > self.window {
            cache.pop_front();
        }
    }
}

fn rss_kib() -> (usize, usize) {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let resident: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // SAFETY: plain syscall filling a zeroed struct.
    let peak = unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        u.ru_maxrss as usize
    };
    (resident * 4, peak)
}

/// Runs warm-up plus measured requests; `step(n)` runs request `n`. Returns
/// ns per request and the resident set (KiB) at the end, while everything the
/// allocator holds is still alive.
fn drive(mut step: impl FnMut(usize)) -> (f64, usize) {
    for n in 0..WARMUP {
        step(n);
    }
    let t = Instant::now();
    for n in WARMUP..WARMUP + REQUESTS {
        step(n);
    }
    let ns = t.elapsed().as_nanos() as f64 / REQUESTS as f64;
    (ns, rss_kib().0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("reap");
    let permille: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let window: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let w = Workload::new(permille, window);
    let (rss0, _) = rss_kib();

    let mut extra = String::new();
    let (ns, rss) = match which {
        "glibc" => {
            let mut cache = VecDeque::new();
            drive(|n| w.request(n, &System, &|v| v, &mut cache))
        }
        "mimalloc" => {
            let mut cache = VecDeque::new();
            drive(|n| w.request(n, &MiGlobal, &|v| v, &mut cache))
        }
        "heap" => {
            let heap = LocalHeap::new();
            let mut cache = VecDeque::new();
            let r = drive(|n| w.request(n, &heap, &|v| v, &mut cache));
            extra = format!("heap mapped {} KiB", heap.stats().mapped_bytes >> 10);
            r
        }
        "reap" => {
            let heap = LocalHeap::new();
            let reap = heap.reap();
            let mut cache = VecDeque::new();
            let r = drive(|n| w.request(n, reap, &|v| v, &mut cache));
            let s = heap.stats();
            extra = format!(
                "heap mapped {} KiB, reap pages {}",
                s.mapped_bytes >> 10,
                s.reap_pages
            );
            r
        }
        "arena" => {
            let heap = LocalHeap::new();
            let mut arena = Arena::new_in(&heap);
            let mut cache = VecDeque::new();
            let r = drive(|n| {
                let promote = |v: Vec<u8, &Arena<&LocalHeap>>| {
                    let mut k = Vec::with_capacity_in(v.len(), &heap);
                    k.extend_from_slice(&v);
                    k
                };
                w.request(n, &arena, &promote, &mut cache);
                arena.reset();
            });
            extra = format!(
                "heap mapped {} KiB, arena chunk {} KiB",
                heap.stats().mapped_bytes >> 10,
                arena.stats().chunk_bytes >> 10
            );
            r
        }
        "bumpalo" => {
            let mut bump = Bump(bumpalo::Bump::new());
            let mut cache = VecDeque::new();
            drive(|n| {
                let promote = |v: Vec<u8, &Bump>| {
                    let mut k = Vec::with_capacity_in(v.len(), &MiGlobal);
                    k.extend_from_slice(&v);
                    k
                };
                w.request(n, &bump, &promote, &mut cache);
                bump.0.reset();
            })
        }
        other => panic!("unknown allocator {other}"),
    };
    let peak = rss_kib().1;
    println!(
        "{which:>8}  survive {permille:>3}‰  window {window:>4}  {:>8.2} µs/request  \
         rss +{:>6} KiB  peak {:>6} KiB  {extra}",
        ns / 1000.0,
        rss.saturating_sub(rss0),
        peak,
    );
}
