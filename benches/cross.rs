//! Buffers that cross threads: allocated by one thread, freed by another.
//!
//! - `producer_consumer/<size>`: one thread allocates filled `Vec<u8, A>`s and
//!   sends them over a bounded ring to a thread that drops them. The ring
//!   waits by spinning (`common/ring.rs`): with a blocking channel, futex
//!   wake-ups of a consumer that keeps up dominated the timings.
//! - `many_to_one/<size>`: four producers, one consumer.
//!
//! Timed from the first producer's start after a barrier (threads are spawned
//! outside the timed region) to the consumer's last drop. Per iteration about 64 MiB of buffers,
//! between 1_000 and 100_000 of them.

use std::alloc::{AllocError, Allocator, GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::ptr::NonNull;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use shardheap::ShardAlloc;

#[path = "common/ring.rs"]
mod ring;
use ring::Ring;

/// mimalloc as an `Allocator` value, as its global allocator is used.
#[derive(Clone, Copy)]
struct MiGlobal;

// SAFETY: forwards to mimalloc, which upholds the allocator contract; sizes
// here are never zero.
unsafe impl Allocator for MiGlobal {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: nonzero size.
        let p = unsafe { GlobalAlloc::alloc(&mimalloc::MiMalloc, l) };
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, l.size()))
            .ok_or(AllocError)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        // SAFETY: allocated above with `l`.
        unsafe { GlobalAlloc::dealloc(&mimalloc::MiMalloc, p.as_ptr(), l) }
    }
}

const SIZES: [usize; 4] = [64, 1024, 16 << 10, 64 << 10];

fn count(size: usize) -> usize {
    ((64 << 20) / size).clamp(1_000, 100_000)
}

/// One run: `producers` threads allocate `n` buffers in total and send them
/// to one consumer, which drops them. Returns the time from start to the last
/// drop.
fn run<A>(a: A, producers: usize, size: usize, n: usize) -> Duration
where
    A: Allocator + Copy + Send + Sync + 'static,
{
    let ring = Arc::new(Ring::<Vec<u8, A>>::new(1024));
    let start = Arc::new(Barrier::new(producers + 2));
    let consumer = {
        let (ring, start) = (ring.clone(), start.clone());
        thread::spawn(move || {
            start.wait();
            for _ in 0..n {
                black_box(ring.recv());
            }
            Instant::now()
        })
    };
    let workers: Vec<_> = (0..producers)
        .map(|p| {
            let (tx, start) = (ring.clone(), start.clone());
            thread::spawn(move || {
                start.wait();
                let t0 = Instant::now();
                for i in (p..n).step_by(producers) {
                    let mut v = Vec::with_capacity_in(size, a);
                    v.resize(size, i as u8);
                    tx.send(v);
                }
                t0
            })
        })
        .collect();
    start.wait();
    // From the first producer's start: this thread may wake late from the
    // barrier.
    let t0 = workers
        .into_iter()
        .map(|w| w.join().unwrap())
        .min()
        .unwrap();
    consumer.join().unwrap() - t0
}

fn bench_one<A>(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: &str,
    a: A,
    producers: usize,
    size: usize,
) where
    A: Allocator + Copy + Send + Sync + 'static,
{
    let n = count(size);
    g.bench_function(BenchmarkId::new(id, size), |b| {
        b.iter_custom(|iters| (0..iters).map(|_| run(a, producers, size, n)).sum())
    });
}

fn benches(c: &mut Criterion) {
    for (name, producers) in [("producer_consumer", 1), ("many_to_one", 4)] {
        let mut g = c.benchmark_group(name);
        for size in SIZES {
            g.throughput(Throughput::Elements(count(size) as u64));
            bench_one(&mut g, "system", System, producers, size);
            bench_one(&mut g, "mimalloc", MiGlobal, producers, size);
            bench_one(&mut g, "shard_alloc", ShardAlloc::new(), producers, size);
        }
        g.finish();
    }
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(20)
}

criterion_group! {
    name = cross;
    config = config();
    targets = benches
}
criterion_main!(cross);
