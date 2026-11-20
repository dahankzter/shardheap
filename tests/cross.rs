//! `ShardAlloc` across threads: blocks freed on one thread go home to the heap
//! of the thread that allocated them.

use std::alloc::{Allocator, Layout};
use std::sync::mpsc;
use std::thread;

use shardheap::ShardAlloc;

/// Scaled down under Miri.
const fn n(native: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { native }
}

type Buf = Vec<u8, ShardAlloc>;

fn tagged(len: usize, tag: u8) -> Buf {
    let mut v = Vec::with_capacity_in(len, ShardAlloc::new());
    fill(&mut v, len, tag);
    v
}

/// `v.resize(len, tag)` with a bulk write for the new bytes: Miri interprets a
/// per-byte loop slowly. Growing goes through `reserve`, so through
/// `Allocator::grow`, as `resize` does.
fn fill(v: &mut Buf, len: usize, tag: u8) {
    if len <= v.len() {
        v.truncate(len);
        return;
    }
    let old = v.len();
    v.reserve(len - old);
    // SAFETY: `reserve` made room for `len` bytes; the new ones are written
    // before `set_len` exposes them.
    unsafe {
        v.as_mut_ptr().add(old).write_bytes(tag, len - old);
        v.set_len(len);
    }
}

/// Every byte natively; under Miri about 64 bytes spread over the block, and
/// the last one.
fn check(v: &[u8], tag: u8) {
    let step = if cfg!(miri) { (v.len() / 64).max(1) } else { 1 };
    let ok = v.iter().step_by(step).all(|&b| b == tag) && v.last().is_none_or(|&b| b == tag);
    assert!(ok, "contents survive the trip");
}

/// This thread's allocated bytes after its first allocation: a thread may adopt
/// an exited thread's heap that still holds blocks living elsewhere.
fn baseline() -> usize {
    drop(Box::new_in(0u8, ShardAlloc::new()));
    ShardAlloc::maintain();
    ShardAlloc::stats().allocated_bytes
}

/// Runs `f` on a fresh thread that owns the blocks it allocates, and returns
/// what it returns.
fn on_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    thread::spawn(f).join().unwrap()
}

#[test]
fn blocks_freed_on_another_thread_go_home() {
    on_thread(|| {
        let base = baseline();
        let count = n(10_000, 200);
        let (tx, rx) = mpsc::channel::<Box<[u8; 64], ShardAlloc>>();
        let consumer = thread::spawn(move || {
            for (i, b) in rx.iter().enumerate() {
                check(&b[..], i as u8);
            }
        });
        for i in 0..count {
            tx.send(Box::new_in([i as u8; 64], ShardAlloc::new()))
                .unwrap();
        }
        drop(tx);
        consumer.join().unwrap();
        ShardAlloc::maintain();
        assert_eq!(ShardAlloc::stats().allocated_bytes, base);
    });
}

#[test]
fn many_senders_one_owner() {
    on_thread(|| {
        let base = baseline();
        let per = n(2_000, 50);
        let mut workers = Vec::new();
        for w in 0..4u8 {
            let bufs: Vec<Buf> = (0..per).map(|i| tagged(1 + (i * 97) % 5000, w)).collect();
            workers.push(thread::spawn(move || {
                for b in bufs {
                    check(&b, w);
                }
            }));
        }
        for w in workers {
            w.join().unwrap();
        }
        ShardAlloc::maintain();
        assert_eq!(ShardAlloc::stats().allocated_bytes, base);
    });
}

#[test]
fn sent_vec_grows_and_shrinks_on_the_receiver() {
    on_thread(|| {
        let base = baseline();
        let v = tagged(1000, 3);
        let receiver = thread::spawn(move || {
            let base = baseline();
            let mut v = v;
            fill(&mut v, n(300_000, 140_000), 4); // grows past the small classes
            check(&v[..1000], 3);
            check(&v[1000..], 4);
            v.truncate(10);
            v.shrink_to_fit();
            check(&v, 3);
            drop(v);
            ShardAlloc::maintain();
            ShardAlloc::stats().allocated_bytes - base
        });
        assert_eq!(
            receiver.join().unwrap(),
            0,
            "the grown block was the receiver's"
        );
        ShardAlloc::maintain();
        assert_eq!(
            ShardAlloc::stats().allocated_bytes,
            base,
            "the original went home"
        );
    });
}

#[test]
fn zero_sized_and_large_aligned_blocks_freed_remotely() {
    struct Sent(*mut u8);
    // SAFETY: handed to one other thread, which frees it.
    unsafe impl Send for Sent {}
    on_thread(|| {
        let base = baseline();
        let a = ShardAlloc::new();
        let zst = Box::new_in((), a);
        let l = Layout::from_size_align(300_000, 1 << 16).unwrap();
        let big = Sent(a.allocate(l).unwrap().cast::<u8>().as_ptr());
        assert_eq!(big.0.addr() % (1 << 16), 0);
        on_thread(move || {
            drop(zst);
            let big = big;
            // SAFETY: allocated above with `l`, freed once.
            unsafe { ShardAlloc::new().deallocate(std::ptr::NonNull::new(big.0).unwrap(), l) };
            ShardAlloc::maintain();
        });
        ShardAlloc::maintain();
        assert_eq!(ShardAlloc::stats().allocated_bytes, base);
        assert_eq!(
            ShardAlloc::stats().large_allocations,
            0,
            "no large block left"
        );
    });
}

#[test]
fn owner_that_never_allocates_again_drains_on_maintain() {
    on_thread(|| {
        let base = baseline();
        let bufs: Vec<Buf> = (0..n(1_000, 40)).map(|i| tagged(100, i as u8)).collect();
        let before = ShardAlloc::stats().allocated_bytes;
        on_thread(move || {
            drop(bufs);
            ShardAlloc::maintain(); // sends this thread's batches
        });
        assert_eq!(
            ShardAlloc::stats().allocated_bytes,
            before,
            "remote frees wait in the inbox until the owner drains"
        );
        ShardAlloc::maintain();
        assert_eq!(ShardAlloc::stats().allocated_bytes, base);
    });
}

#[test]
fn owner_reuses_remotely_freed_blocks_before_taking_fresh_memory() {
    struct Sent(Vec<std::ptr::NonNull<u8>>);
    // SAFETY: blocks handed to one other thread, which frees them.
    unsafe impl Send for Sent {}
    on_thread(|| {
        let a = ShardAlloc::new();
        let l = Layout::from_size_align(64 << 10, 8).unwrap();
        let blocks: Vec<_> = (0..16)
            .map(|_| a.allocate(l).unwrap().cast::<u8>())
            .collect();
        let addrs: Vec<usize> = blocks.iter().map(|p| p.addr().get()).collect();
        let sent = Sent(blocks);
        on_thread(move || {
            let sent = sent;
            for p in sent.0 {
                // SAFETY: allocated above with `l`, freed once.
                unsafe { ShardAlloc::new().deallocate(p, l) };
            }
            ShardAlloc::maintain(); // sends the batch home
        });
        // Far fewer allocations than a heartbeat: the inbox is drained because
        // the size class ran out, before a fresh page is taken.
        let p = a.allocate(l).unwrap().cast::<u8>();
        assert!(
            addrs.contains(&p.addr().get()),
            "a block freed remotely is reused before fresh memory"
        );
        // SAFETY: allocated just above with `l`.
        unsafe { a.deallocate(p, l) };
        ShardAlloc::maintain();
    });
}

// ---- Model: random allocate / resize / send / free on four threads ---------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A block held by a worker: a raw allocation with an arbitrary layout, or a
/// `Vec` that can be resized; every byte is `tag`.
enum Item {
    Raw(std::ptr::NonNull<u8>, Layout, u8),
    Vec(Buf, u8),
}

// SAFETY: a `Raw` block is owned by whoever holds the item; `ShardAlloc` frees
// it on any thread.
unsafe impl Send for Item {}

impl Item {
    fn check(&self) {
        match self {
            Item::Raw(p, l, t) => {
                // SAFETY: a live block of `l.size()` bytes, all written.
                check(
                    unsafe { std::slice::from_raw_parts(p.as_ptr(), l.size()) },
                    *t,
                )
            }
            Item::Vec(v, t) => check(v, *t),
        }
    }
}

impl Drop for Item {
    fn drop(&mut self) {
        self.check();
        if let Item::Raw(p, l, _) = *self {
            // SAFETY: allocated with `l` through `ShardAlloc`, freed once.
            unsafe { ShardAlloc::new().deallocate(p, l) };
        }
    }
}

fn size(rng: &mut Rng) -> usize {
    match rng.below(100) {
        0..=69 => rng.below(2048),
        70..=94 => rng.below(64 << 10),
        _ => rng.below(n(300_000, 140_000)),
    }
}

#[test]
fn cross_thread_model() {
    const WORKERS: usize = 4;
    let ops = n(20_000, 300);
    let (txs, rxs): (Vec<_>, Vec<_>) = (0..WORKERS).map(|_| mpsc::channel::<Item>()).unzip();
    let phase = std::sync::Arc::new(std::sync::Barrier::new(WORKERS));
    let mut rxs: Vec<_> = rxs.into_iter().map(Some).collect();
    let workers: Vec<_> = (0..WORKERS)
        .map(|w| {
            let right = txs[(w + 1) % WORKERS].clone();
            let rx = rxs[w].take().unwrap();
            let phase = phase.clone();
            thread::spawn(move || {
                let base = baseline();
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((w as u64 + 1) * 0x1234_5678));
                let mut held: Vec<Item> = Vec::new();
                let a = ShardAlloc::new();
                for step in 0..ops {
                    held.extend(rx.try_iter());
                    let tag = (step * WORKERS + w) as u8;
                    match rng.below(100) {
                        0..=29 => {
                            let l = Layout::from_size_align(size(&mut rng), 1 << rng.below(17))
                                .unwrap();
                            let p = a.allocate(l).unwrap().cast::<u8>();
                            assert_eq!(p.addr().get() % l.align(), 0);
                            // SAFETY: a fresh block of `l.size()` bytes.
                            unsafe { p.as_ptr().write_bytes(tag, l.size()) };
                            held.push(Item::Raw(p, l, tag));
                        }
                        30..=49 => held.push(Item::Vec(tagged(size(&mut rng), tag), tag)),
                        50..=64 if !held.is_empty() => {
                            let i = rng.below(held.len());
                            if let Item::Vec(v, t) = &mut held[i] {
                                let len = size(&mut rng);
                                fill(v, len, *t); // grows or shrinks, maybe a foreign block
                                if rng.below(2) == 0 {
                                    v.shrink_to_fit();
                                }
                            }
                            held[i].check();
                        }
                        65..=79 if !held.is_empty() => {
                            let i = rng.below(held.len());
                            right.send(held.swap_remove(i)).unwrap();
                        }
                        80..=99 if !held.is_empty() => {
                            let i = rng.below(held.len());
                            drop(held.swap_remove(i));
                        }
                        _ => {}
                    }
                }
                drop(right);
                phase.wait(); // every send is done
                held.extend(rx.iter());
                drop(held);
                phase.wait(); // every block is freed (some batched)
                ShardAlloc::maintain(); // send the batches home
                phase.wait();
                ShardAlloc::maintain(); // drain what came home
                assert_eq!(ShardAlloc::stats().allocated_bytes, base, "worker {w}");
            })
        })
        .collect();
    drop(txs);
    for w in workers {
        w.join().unwrap();
    }
}
