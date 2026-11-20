//! `ShardAlloc` as the global allocator: the whole test binary, std included,
//! allocates through it.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::RefCell;

use shardheap::ShardAlloc;

#[global_allocator]
static A: ShardAlloc = ShardAlloc::new();

#[test]
fn std_runs_on_shardalloc() {
    let mut v: Vec<String> = (0..10_000).map(|i| i.to_string()).collect();
    v.retain(|s| s.len() > 2);
    assert_eq!(v.len(), 10_000 - 100);
}

#[test]
fn threads_send_collections_both_ways() {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let (back_tx, back_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let h = std::thread::spawn(move || {
        for v in rx {
            assert!(v.iter().all(|&b| b == 7));
            back_tx.send(vec![3u8; v.len()]).unwrap();
        }
    });
    for n in [1, 64, 4096, 300_000] {
        tx.send(vec![7u8; n]).unwrap();
        assert_eq!(back_rx.recv().unwrap().len(), n);
    }
    drop(tx);
    h.join().unwrap();
}

#[test]
fn short_lived_threads_with_thread_local_destructors() {
    struct D(Vec<u8>);
    impl Drop for D {
        fn drop(&mut self) {
            // Allocates while the thread is exiting.
            let again = vec![1u8; self.0.len()];
            assert_eq!(again.len(), self.0.len());
        }
    }
    thread_local!(static T: RefCell<Option<D>> = const { RefCell::new(None) });
    let n = if cfg!(miri) { 4 } else { 200 };
    for _ in 0..n {
        std::thread::spawn(|| T.with(|t| *t.borrow_mut() = Some(D(vec![0; 1000]))))
            .join()
            .unwrap();
    }
}

#[test]
fn zeroed_and_realloc_keep_their_promises() {
    let l = Layout::from_size_align(5000, 64).unwrap();
    // SAFETY: nonzero layouts; each pointer is used within its block and freed
    // with the layout it has.
    unsafe {
        let p = A.alloc_zeroed(l);
        assert!(!p.is_null());
        assert!(std::slice::from_raw_parts(p, 5000).iter().all(|&b| b == 0));
        p.write_bytes(9, 5000);
        let q = A.realloc(p, l, 400_000);
        assert_eq!(q.addr() % 64, 0);
        assert!(std::slice::from_raw_parts(q, 5000).iter().all(|&b| b == 9));
        let r = A.realloc(q, Layout::from_size_align(400_000, 64).unwrap(), 100);
        assert!(std::slice::from_raw_parts(r, 100).iter().all(|&b| b == 9));
        A.dealloc(r, Layout::from_size_align(100, 64).unwrap());
    }
}

#[test]
fn small_boxes_free_through_their_narrow_pointers() {
    // A `Box<u8>`'s pointer is narrowed to one byte; freeing writes a link
    // word into the block.
    drop(Box::new(1u8));
    let boxes: Vec<Box<u32>> = (0..64).map(Box::new).collect();
    std::thread::spawn(move || drop(boxes)).join().unwrap();
}

#[test]
fn an_in_place_realloc_hands_back_a_usable_pointer() {
    let mut v = vec![1u8, 2, 3].into_boxed_slice().into_vec();
    v.push(4); // grows in place: the block has room
    assert_eq!(v, [1, 2, 3, 4]);
}
