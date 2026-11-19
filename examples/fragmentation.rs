//! Fragmentation: resident memory against live bytes after objects die in
//! patterns that leave holes.
//!
//! Phases, each followed by a resident-memory reading (and one after the
//! allocator's trim call: `LocalHeap::trim`, `mi_collect`, `malloc_trim`):
//!
//! 1. **fill:** 256 MiB of small objects, 16-1024 bytes.
//! 2. **thin:** free 90% of them. `random` frees at random, so survivors are
//!    scattered over every page; `clustered` frees whole runs of 4096
//!    consecutive allocations, about 2 MiB (data that was born together, such
//!    as one request's, dies together).
//! 3. **refill:** 256 MiB of objects of other sizes, 2-8 KiB. Holes left by
//!    phase 2 can only be reused across size classes by allocators that
//!    coalesce or that give whole empty pages back to the segment.
//! 4. **drain:** free everything from phase 3.
//! 5. **churn:** 20 rounds, each replacing half the live objects with ones of
//!    a size range that drifts every round; the live set stays near its size.
//!
//! ```text
//! cargo run --release --example fragmentation -- <glibc|mimalloc|heap> [random|clustered] [huge_pages]
//! ```
//!
//! `huge_pages` (`auto`, `always`, `never`, `kernel`) sets the `LocalHeap` policy.
//!
//! Run one allocator per process so the resident figures are its own.

use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::time::Instant;

use shardheap::{HugePages, LocalHeap};

const MIB: usize = 1 << 20;
const FILL: usize = 256 * MIB;
const RUN: usize = 4096;

struct Global<G: GlobalAlloc>(G);

// SAFETY: forwards to a global allocator, which upholds the contract; sizes
// here are never zero.
unsafe impl<G: GlobalAlloc> Allocator for Global<G> {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: nonzero size.
        let p = unsafe { self.0.alloc(l) };
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, l.size()))
            .ok_or(AllocError)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        // SAFETY: allocated above with `l`.
        unsafe { self.0.dealloc(p.as_ptr(), l) }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }
}

fn rss() -> usize {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap();
    let pages: usize = s.split_whitespace().nth(1).unwrap().parse().unwrap();
    pages * 4096
}

struct Blocks {
    v: Vec<(NonNull<u8>, Layout)>,
    live: usize,
}

impl Blocks {
    fn alloc(&mut self, a: &impl Allocator, size: usize) {
        let l = Layout::from_size_align(size, 8).unwrap();
        let p = a.allocate(l).unwrap().cast::<u8>();
        // SAFETY: fresh block of `size` bytes; touched so it is resident.
        unsafe { p.as_ptr().write_bytes(0xa5, size) };
        self.v.push((p, l));
        self.live += size;
    }
    fn free(&mut self, a: &impl Allocator, i: usize) {
        let (p, l) = self.v.swap_remove(i);
        // SAFETY: allocated by `a` with `l`, freed once.
        unsafe { a.deallocate(p, l) };
        self.live -= l.size();
    }
    fn free_where(&mut self, a: &impl Allocator, mut f: impl FnMut(usize) -> bool) {
        let mut kept = Vec::with_capacity(self.v.len());
        for (i, &(p, l)) in self.v.iter().enumerate() {
            if f(i) {
                // SAFETY: as in `free`.
                unsafe { a.deallocate(p, l) };
                self.live -= l.size();
            } else {
                kept.push((p, l));
            }
        }
        self.v = kept;
    }
}

fn run(a: &impl Allocator, trim: &dyn Fn(), clustered: bool) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut small = Blocks {
        v: Vec::new(),
        live: 0,
    };
    let mut big = Blocks {
        v: Vec::new(),
        live: 0,
    };
    // Bookkeeping comes from the system allocator in every run and is reserved
    // before the baseline, so it does not count against anyone.
    small.v.reserve(FILL / 16);
    big.v.reserve(FILL / 2048);
    let base = rss();

    while small.live < FILL {
        let s = rng.range(2, 128) * 8;
        small.alloc(a, s);
    }
    let r = |phase: &str, live: usize| {
        let before = rss() - base;
        trim();
        let after = rss() - base;
        println!(
            "{phase:>8}  live {:>6.1} MiB  resident {:>6.1} MiB ({:>5.2}x)  after trim {:>6.1} MiB ({:>5.2}x)",
            live as f64 / MIB as f64,
            before as f64 / MIB as f64,
            before as f64 / live.max(1) as f64,
            after as f64 / MIB as f64,
            after as f64 / live.max(1) as f64,
        );
    };
    r("fill", small.live);

    if clustered {
        let mut keep = Vec::new();
        for _ in 0..small.v.len().div_ceil(RUN) {
            keep.push(rng.next().is_multiple_of(10));
        }
        small.free_where(a, |i| !keep[i / RUN]);
    } else {
        small.free_where(a, |_| !rng.next().is_multiple_of(10));
    }
    r("thin", small.live);

    while big.live < FILL {
        let s = rng.range(256, 1024) * 8;
        big.alloc(a, s);
    }
    r("refill", small.live + big.live);

    big.free_where(a, |_| true);
    r("drain", small.live);

    // Churn: live set held near 128 MiB while the size range drifts.
    while small.live < FILL / 2 {
        let s = rng.range(2, 128) * 8;
        small.alloc(a, s);
    }
    let mut peak = 0;
    let t = Instant::now();
    for round in 0..20 {
        let (lo, hi) = [(2, 32), (16, 128), (64, 512), (8, 64), (128, 1024)][round % 5];
        let target = small.live / 2;
        let mut freed = 0;
        while freed < target {
            let i = rng.range(0, small.v.len() - 1);
            freed += small.v[i].1.size();
            small.free(a, i);
        }
        while freed > 0 {
            let s = rng.range(lo, hi) * 8;
            small.alloc(a, s);
            freed = freed.saturating_sub(s);
        }
        peak = peak.max(rss() - base);
    }
    println!(
        "{:>8}  live {:>6.1} MiB  peak resident {:>6.1} MiB ({:>5.2}x)  {:.0} ms",
        "churn",
        small.live as f64 / MIB as f64,
        peak as f64 / MIB as f64,
        peak as f64 / small.live as f64,
        t.elapsed().as_secs_f64() * 1e3,
    );
    r("end", small.live);
    small.free_where(a, |_| true);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let which = args.next().unwrap_or_else(|| "heap".into());
    let clustered = args.next().as_deref() == Some("clustered");
    let policy = match args.next().as_deref() {
        None | Some("auto") => HugePages::Auto,
        Some("always") => HugePages::Always,
        Some("never") => HugePages::Never,
        Some("kernel") => HugePages::Kernel,
        Some(p) => panic!("unknown huge-page policy {p}"),
    };
    println!(
        "{which}, {} survivors",
        if clustered { "clustered" } else { "random" }
    );
    match which.as_str() {
        "glibc" => run(
            &Global(System),
            &|| unsafe {
                libc::malloc_trim(0);
            },
            clustered,
        ),
        "mimalloc" => run(
            &Global(mimalloc::MiMalloc),
            &|| unsafe {
                libmimalloc_sys::mi_collect(true);
            },
            clustered,
        ),
        "heap" => {
            let heap = LocalHeap::with_huge_pages(policy);
            run(&&heap, &|| heap.trim(), clustered);
            let s = heap.stats();
            println!("stats after: {s:?}");
        }
        _ => panic!("unknown allocator {which}"),
    }
}
