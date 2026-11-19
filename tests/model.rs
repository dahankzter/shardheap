//! Model-based tests: random operation sequences checked against a simple model of
//! the live allocations.

use std::alloc::{Allocator, Layout};
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use shardheap::{Arena, HugePages, LocalHeap, ShardAlloc, StackArena};

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const OPS: usize = if cfg!(miri) { 1_500 } else { 100_000 };

struct Live {
    ptr: NonNull<u8>,
    layout: Layout,
    tag: u8,
}

fn random_layout(r: &mut impl FnMut() -> u64, big: bool) -> Layout {
    let size = match r() % 100 {
        0..=69 => 1 + (r() % 256) as usize,          // small
        70..=89 => 257 + (r() % 8_000) as usize,     // medium
        90..=97 => 8_000 + (r() % 130_000) as usize, // multi-slot pages and the boundary
        _ if cfg!(miri) || !big => 200_000,
        _ => 130_000 + (r() % 1_000_000) as usize, // large
    };
    let size = if r().is_multiple_of(50) { 0 } else { size };
    let align = match r() % 20 {
        0 => 64,
        1 => 4096,
        2 => 1 << 16,
        3 if !cfg!(miri) => 1 << 17, // beyond a slot: large path
        _ => 1 << (r() % 4),
    };
    Layout::from_size_align(size, align).unwrap()
}

fn fill(p: NonNull<u8>, len: usize, tag: u8) {
    // SAFETY: `p` is a live block of at least `len` bytes.
    unsafe { std::ptr::write_bytes(p.as_ptr(), tag, len) }
}

fn check(p: NonNull<u8>, len: usize, tag: u8) {
    // Miri interprets every byte read, so it samples; native runs check every byte.
    let step = if cfg!(miri) { 509 } else { 1 };
    // SAFETY: `p` is a live block of at least `len` initialized bytes.
    let s = unsafe { std::slice::from_raw_parts(p.as_ptr(), len) };
    if step == 1 {
        if let Some(i) = s.iter().position(|&b| b != tag) {
            panic!("corruption at byte {i} of {len}: {:#x} != {tag:#x}", s[i]);
        }
        return;
    }
    let last = len.checked_sub(1).into_iter();
    for i in (0..len).step_by(step).chain(last) {
        assert_eq!(s[i], tag, "corruption at byte {i} of {len}");
    }
}

/// Asserts the live non-empty blocks are pairwise disjoint.
fn assert_disjoint(live: &[Live]) {
    let mut ranges = BTreeMap::new();
    for l in live.iter().filter(|l| l.layout.size() > 0) {
        let a = l.ptr.as_ptr().addr();
        ranges.insert(a, a + l.layout.size());
    }
    let mut prev_end = 0;
    for (&start, &end) in &ranges {
        assert!(start >= prev_end, "overlapping blocks at {start:#x}");
        prev_end = end;
    }
}

fn run_model<A: Allocator>(a: &A, seed: u64) {
    run_model_with(a, seed, OPS, true);
}

/// `big`: include large allocations up to ~1 MiB (an arena never reuses memory,
/// so its model run keeps them small).
fn run_model_with<A: Allocator>(a: &A, seed: u64, ops: usize, big: bool) {
    let mut r = rng(seed);
    let mut live: Vec<Live> = Vec::new();
    let mut tag = 1u8;
    for op in 0..ops {
        tag = tag.wrapping_add(1).max(1);
        match r() % 10 {
            // allocate (zeroed sometimes)
            0..=4 => {
                let layout = random_layout(&mut r, big);
                let zeroed = r().is_multiple_of(8);
                let b = if zeroed {
                    a.allocate_zeroed(layout)
                } else {
                    a.allocate(layout)
                }
                .unwrap();
                assert!(b.len() >= layout.size());
                let p = b.cast::<u8>();
                assert_eq!(
                    p.as_ptr().addr() % layout.align(),
                    0,
                    "misaligned {layout:?}"
                );
                if zeroed {
                    check(p, layout.size(), 0);
                }
                fill(p, layout.size(), tag);
                live.push(Live {
                    ptr: p,
                    layout,
                    tag,
                });
            }
            // free a random block
            5..=7 if !live.is_empty() => {
                let l = live.swap_remove((r() as usize) % live.len());
                check(l.ptr, l.layout.size(), l.tag);
                // SAFETY: live block from `a` with this layout.
                unsafe { a.deallocate(l.ptr, l.layout) };
            }
            // grow or shrink a random block
            8..=9 if !live.is_empty() => {
                let i = (r() as usize) % live.len();
                let old = live[i].layout;
                let new_size = if r().is_multiple_of(2) {
                    old.size() + (r() % (old.size() as u64 + 64)) as usize
                } else {
                    old.size() / (1 + (r() % 4) as usize)
                };
                let new = Layout::from_size_align(new_size, old.align()).unwrap();
                let kept = old.size().min(new.size());
                check(live[i].ptr, old.size(), live[i].tag);
                // SAFETY: live block from `a` with layout `old`; sizes ordered as required.
                let b = unsafe {
                    if new.size() >= old.size() {
                        a.grow(live[i].ptr, old, new)
                    } else {
                        a.shrink(live[i].ptr, old, new)
                    }
                }
                .unwrap();
                let p = b.cast::<u8>();
                assert_eq!(p.as_ptr().addr() % new.align(), 0);
                check(p, kept, live[i].tag);
                fill(p, new.size(), tag);
                live[i] = Live {
                    ptr: p,
                    layout: new,
                    tag,
                };
            }
            _ => {}
        }
        if op % 997 == 0 {
            assert_disjoint(&live);
        }
    }
    assert_disjoint(&live);
    for l in live {
        check(l.ptr, l.layout.size(), l.tag);
        // SAFETY: live block from `a` with this layout.
        unsafe { a.deallocate(l.ptr, l.layout) };
    }
}

#[test]
fn local_heap_model() {
    let heap = LocalHeap::new();
    run_model(&heap, 1);
    assert_eq!(heap.stats().allocated_bytes, 0);
    assert_eq!(heap.stats().large_allocations, 0);
    run_model(&heap, 2); // reuse a warm heap
    assert_eq!(heap.stats().allocated_bytes, 0);
    heap.trim();
    assert_eq!(heap.stats().segments, 0);
    assert_eq!(heap.stats().mapped_bytes, 0);
}

#[test]
fn every_huge_page_policy() {
    for (i, policy) in [
        HugePages::Auto,
        HugePages::Always,
        HugePages::Never,
        HugePages::Kernel,
    ]
    .into_iter()
    .enumerate()
    {
        let heap = LocalHeap::with_huge_pages(policy);
        // Past the Auto threshold (16 MiB), so the switch is exercised too.
        let mut v: Vec<u64, &LocalHeap> = Vec::new_in(&heap);
        v.extend(0..if cfg!(miri) { 10_000 } else { 3_000_000 });
        let boxes: Vec<Box<[u8; 1000], &LocalHeap>> = (0..if cfg!(miri) { 50 } else { 20_000 })
            .map(|j| Box::new_in([j as u8; 1000], &heap))
            .collect();
        run_model_with(&heap, 20 + i as u64, OPS / 10, true);
        assert!(boxes.iter().enumerate().all(|(j, b)| b[999] == j as u8));
        drop((v, boxes));
        assert_eq!(heap.stats().allocated_bytes, 0);
    }
}

#[test]
fn region_heap_model() {
    // 8 MiB: two segments, so the model also overflows to the OS fallback.
    let heap = LocalHeap::with_region(8 << 20, HugePages::Auto).unwrap();
    run_model(&heap, 30);
    run_model(&heap.reap(), 31);
    {
        let mut arena = Arena::new_in(&heap);
        run_model_with(&arena, 32, OPS / 5, false);
        arena.reset();
    }
    assert_eq!(heap.stats().allocated_bytes, 0);
    heap.trim();
    let s = heap.stats();
    assert_eq!((s.segments, s.region_bytes, s.mapped_bytes), (0, 0, 0));
}

#[test]
fn region_blocks_are_inside_until_it_is_full() {
    let heap = LocalHeap::with_region(4 << 20, HugePages::Kernel).unwrap();
    let region = heap.region().unwrap();
    let mut small = Vec::new();
    // One segment: its 63 usable slots hold 63 * 64 KiB of 4 KiB blocks.
    for _ in 0..63 * 16 {
        small.push(Box::new_in([7u8; 4096], &heap));
    }
    assert!(small.iter().all(|b| heap.in_region(b.as_ptr())));
    let spill = Box::new_in([8u8; 4096], &heap);
    assert!(!heap.in_region(spill.as_ptr()), "region full: OS fallback");
    let big: Vec<u8, &LocalHeap> = Vec::with_capacity_in(1 << 20, &heap);
    assert!(!heap.in_region(big.as_ptr()));
    assert_eq!(heap.stats().region_bytes, region.len());
    drop((small, spill, big));
    heap.trim();
    assert_eq!(heap.stats().region_bytes, 0);
    // Freed region memory is reused, and handed out zeroed where asked.
    let l = Layout::from_size_align(300_000, 8).unwrap();
    let z = heap.allocate_zeroed(l).unwrap();
    let p = z.cast::<u8>();
    assert!(heap.in_region(p.as_ptr()));
    check(p, l.size(), 0);
    // SAFETY: allocated above with `l`.
    unsafe { heap.deallocate(p, l) };
}

#[test]
fn stack_arena_model() {
    let mut buf = vec![MaybeUninit::<u8>::uninit(); 64 << 10];
    let arena = StackArena::new(&mut buf);
    run_model(&arena, 3);
}

#[test]
fn stack_arena_over_local_heap() {
    let heap = LocalHeap::new();
    let mut buf = vec![MaybeUninit::<u8>::uninit(); 16 << 10];
    let arena = StackArena::with_fallback(&mut buf, &heap);
    run_model(&arena, 4);
    assert_eq!(heap.stats().allocated_bytes, 0);
}

#[test]
fn arena_model_and_reset() {
    let mut arena = Arena::new();
    for seed in 5..8 {
        run_model_with(&arena, seed, if cfg!(miri) { 800 } else { 20_000 }, false);
        arena.reset();
        assert_eq!(arena.stats().allocated_bytes, 0);
        assert_eq!(arena.stats().chunks, 1);
    }
}

#[test]
fn arena_over_local_heap() {
    let heap = LocalHeap::new();
    {
        let mut arena = Arena::new_in(&heap);
        for seed in 8..11 {
            run_model_with(&arena, seed, if cfg!(miri) { 800 } else { 20_000 }, false);
            arena.reset();
            assert_eq!(arena.stats().chunks, 1);
            // The chunk is the heap's only allocation (plus a large-mapping header).
            let (h, c) = (heap.stats().allocated_bytes, arena.stats().chunk_bytes);
            assert!(h >= c && h <= c + 4096, "{h} vs {c}");
        }
    }
    assert_eq!(heap.stats().allocated_bytes, 0);
    heap.trim();
    assert_eq!(heap.stats().mapped_bytes, 0);
}

#[test]
fn reap_model() {
    let heap = LocalHeap::new();
    run_model(&heap.reap(), 11);
    assert_eq!(heap.stats().allocated_bytes, 0);
    // Only the current bump page can remain, empty.
    assert!(heap.stats().reap_pages <= 1);
    run_model(&heap.reap(), 12);
    // Mixed with plain heap traffic on the same heap.
    let mut v: Vec<u32, &LocalHeap> = Vec::new_in(&heap);
    v.extend(0..10_000);
    run_model(&heap.reap(), 13);
    drop(v);
    assert_eq!(heap.stats().allocated_bytes, 0);
    heap.trim();
    let s = heap.stats();
    assert_eq!((s.reap_pages, s.segments, s.mapped_bytes), (0, 0, 0));
}

#[test]
fn reap_survivors_outlive_their_batch() {
    let heap = LocalHeap::new();
    let reap = heap.reap();
    let mut survivors = std::collections::VecDeque::new();
    for round in 0..200u32 {
        let batch: Vec<Box<[u32; 30], _>, _> = (0..300)
            .map(|i| Box::new_in([round ^ i; 30], reap))
            .collect_in(reap);
        survivors.push_back(Box::new_in(batch[7][0], reap));
        if survivors.len() > 16 {
            survivors.pop_front();
        }
        drop(batch);
        // Each live survivor pins at most one page, plus the bump page.
        assert!(heap.stats().reap_pages <= survivors.len() + 1);
    }
    assert!(
        survivors
            .iter()
            .enumerate()
            .all(|(i, b)| **b == (184 + i as u32) ^ 7)
    );
}

#[test]
fn collections() {
    let heap = LocalHeap::new();
    {
        let mut v: Vec<String, &LocalHeap> = Vec::new_in(&heap);
        for i in 0..2_000 {
            v.push(format!("item {i}"));
        }
        let mut big: Vec<u64, &LocalHeap> = Vec::new_in(&heap);
        big.extend(0..if cfg!(miri) { 40_000 } else { 1_000_000 }); // large path, grown by remap
        let boxes: Vec<Box<[u8; 300], &LocalHeap>, &LocalHeap> = (0..500)
            .map(|i| Box::new_in([i as u8; 300], &heap))
            .collect_in(&heap);
        assert_eq!(v[1999], "item 1999");
        assert!(boxes.iter().enumerate().all(|(i, b)| b[299] == i as u8));
        assert_eq!(big.iter().copied().max(), Some(big.len() as u64 - 1));
        assert!(heap.stats().allocated_bytes > 0);
    }
    assert_eq!(heap.stats().allocated_bytes, 0);
}

#[test]
fn reset_frees_everything_and_reuses() {
    let mut heap = LocalHeap::new();
    for round in 0..3 {
        let mut kept = Vec::new_in(&heap);
        for i in 0..5_000u32 {
            kept.push(Box::new_in(i, &heap));
        }
        let large = Box::new_in([round as u8; 300_000], &heap);
        assert_eq!(large[299_999], round as u8);
        // Leak deliberately: reset reclaims it all.
        std::mem::forget((kept, large));
        assert!(heap.stats().allocated_bytes > 0);
        heap.reset();
        let s = heap.stats();
        assert_eq!((s.allocated_bytes, s.large_allocations), (0, 0));
        assert!(s.segments <= 4 && s.empty_segments == s.segments);
    }
}

#[test]
fn same_class_resize_stays_in_place() {
    let heap = LocalHeap::new();
    let old = Layout::from_size_align(100, 8).unwrap();
    let b = heap.allocate(old).unwrap();
    let new = Layout::from_size_align(b.len(), 8).unwrap();
    // SAFETY: live block, `new.size() >= old.size()`.
    let g = unsafe { heap.grow(b.cast(), old, new) }.unwrap();
    assert_eq!(g.cast::<u8>(), b.cast::<u8>());
    // SAFETY: live block with layout `new`.
    unsafe { heap.deallocate(g.cast(), new) };
}

trait CollectIn<T> {
    fn collect_in<A: Allocator>(self, a: A) -> Vec<T, A>;
}
impl<T, I: Iterator<Item = T>> CollectIn<T> for I {
    fn collect_in<A: Allocator>(self, a: A) -> Vec<T, A> {
        let mut v = Vec::new_in(a);
        v.extend(self);
        v
    }
}

/// An idle heap returns memory through `maintain` once it has been free for
/// whole purge epochs, without `trim`.
#[cfg(all(target_os = "linux", not(miri)))]
#[test]
fn maintain_purges_idle_memory() {
    fn resident(blocks: &[NonNull<u8>]) -> usize {
        let mut v = [0u8; 1];
        blocks
            .iter()
            .filter(|b| {
                let page = b.as_ptr().map_addr(|a| a & !4095);
                // SAFETY: a mapped page of the heap; `v` holds its one byte.
                assert_eq!(
                    unsafe { libc::mincore(page.cast(), 4096, v.as_mut_ptr()) },
                    0
                );
                v[0] & 1 != 0
            })
            .count()
    }
    let heap = LocalHeap::with_huge_pages(HugePages::Never);
    let l = Layout::new::<[u8; 4096]>();
    let blocks: Vec<_> = (0..4096)
        .map(|_| {
            let b = heap.allocate(l).unwrap().cast::<u8>();
            fill(b, l.size(), 7);
            b
        })
        .collect();
    // Keep every 64th block: no page empties into a whole free segment, but
    // most pages do empty.
    let (keep, free): (Vec<_>, Vec<_>) = blocks
        .into_iter()
        .enumerate()
        .partition(|(i, _)| i % 64 == 0);
    for &(_, b) in &free {
        // SAFETY: allocated above with `l`.
        unsafe { heap.deallocate(b, l) };
    }
    // Blocks sharing a 64 KiB slot with a kept block stay resident; the
    // others are in pages that emptied. (4 KiB blocks: one-slot pages.)
    let slot = |b: &NonNull<u8>| b.addr().get() >> 16;
    let pinned: std::collections::HashSet<_> = keep.iter().map(|(_, b)| slot(b)).collect();
    let free: Vec<_> = free
        .into_iter()
        .map(|(_, b)| b)
        .filter(|b| !pinned.contains(&slot(b)))
        .collect();
    let before = resident(&free);
    assert!(
        before > free.len() / 2,
        "freed memory is still resident: {before}"
    );
    for _ in 0..4 {
        std::thread::sleep(std::time::Duration::from_millis(25));
        heap.maintain();
    }
    let after = resident(&free);
    assert!(
        after < free.len() / 16,
        "idle memory was purged: {before} -> {after}"
    );
    for (_, b) in keep {
        check(b, l.size(), 7);
        // SAFETY: allocated above with `l`.
        unsafe { heap.deallocate(b, l) };
    }
}

// ---- ShardAlloc (each #[test] runs on its own thread, so its heap is its own) ----

#[test]
fn shard_alloc_model() {
    run_model(&ShardAlloc::new(), 31);
    run_model_with(&ShardAlloc::new(), 37, OPS / 4, true);
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn shard_alloc_zero_sized_and_zeroed() {
    let a = ShardAlloc::new();
    let z = Layout::from_size_align(0, 64).unwrap();
    let p = a.allocate(z).unwrap().cast::<u8>();
    // SAFETY: allocated above with `z`; must not read any header.
    unsafe { a.deallocate(p, z) };
    let l = Layout::new::<[u8; 3000]>();
    let p = a.allocate(l).unwrap().cast::<u8>();
    fill(p, l.size(), 0xff);
    // SAFETY: allocated above with `l`.
    unsafe { a.deallocate(p, l) };
    let q = a.allocate_zeroed(l).unwrap().cast::<u8>();
    check(q, l.size(), 0);
    // SAFETY: allocated above with `l`.
    unsafe { a.deallocate(q, l) };
}

#[test]
fn shard_alloc_large_aligned_and_resized() {
    let a = ShardAlloc::new();
    let mut v: Vec<u8, ShardAlloc> = Vec::new_in(a);
    // Past the largest small class (128 KiB), so growth takes the large path.
    let n = if cfg!(miri) { 300_000 } else { 3 << 20 };
    for i in 0..n {
        v.push(i as u8); // small to large through `grow`
    }
    v.truncate(100);
    v.shrink_to_fit(); // large to small through `shrink`
    assert!(v.iter().enumerate().all(|(i, &b)| b == i as u8));
    drop(v);
    let l = Layout::from_size_align(300_000, 1 << 16).unwrap();
    let p = a.allocate(l).unwrap().cast::<u8>();
    assert_eq!(p.addr().get() % (1 << 16), 0);
    // SAFETY: allocated above with `l`.
    unsafe { a.deallocate(p, l) };
    assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
}

#[test]
fn shard_alloc_threads_have_their_own_heaps() {
    let other = std::thread::spawn(|| {
        let b = Box::new_in([1u8; 4000], ShardAlloc::new());
        let s = ShardAlloc::stats().allocated_bytes;
        drop(b);
        s
    })
    .join()
    .unwrap();
    let mine = ShardAlloc::stats().allocated_bytes;
    let b = Box::new_in([2u8; 100], ShardAlloc::new());
    assert!(other >= 4000);
    assert_eq!(ShardAlloc::stats().allocated_bytes - mine, 112); // the 112 B class
    drop(b);
}

struct LateUser(std::cell::RefCell<Option<Vec<u8, ShardAlloc>>>);

impl Drop for LateUser {
    fn drop(&mut self) {
        drop(self.0.borrow_mut().take()); // a free after the heap's exit guard
        let mut v = Vec::with_capacity_in(64, ShardAlloc::new()); // an allocation after it
        v.extend_from_slice(b"late");
        assert_eq!(&v[..], b"late");
    }
}

thread_local!(static LATE: LateUser = const { LateUser(std::cell::RefCell::new(None)) });

#[test]
fn shard_alloc_survives_thread_teardown() {
    std::thread::spawn(|| {
        // Registered before the heap's exit guard, so (on Linux, where thread-local
        // destructors run in reverse order of registration) destroyed after it.
        LATE.with(|_| ());
        let mut v = Vec::with_capacity_in(4096, ShardAlloc::new());
        v.resize(4096, 9u8);
        LATE.with(|l| *l.0.borrow_mut() = Some(v));
    })
    .join()
    .unwrap();
}
