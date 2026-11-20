//! # shardheap
//!
//! A shared-nothing heap for Rust's stable [`Allocator`](core::alloc::Allocator) API
//! (Rust 1.100), built on mimalloc's design:
//!
//! - 4 MiB segments aligned to their size, so a block's metadata is one address
//!   mask away; no per-object header.
//! - Free lists sharded per 64 KiB page instead of per size class, with a separate
//!   `local_free` list swapped in when `free` runs dry.
//! - 88 size classes at most 12.5% apart, up to 128 KiB; larger blocks get their
//!   own mapping, grown in place with `mremap` on Linux.
//! - Empty segments are cached, then their memory is returned to the OS after a
//!   delay.
//!
//! What it changes for Rust:
//!
//! - [`LocalHeap`] is `!Send`, so its collections are too. Every free provably
//!   happens on the owning thread: no atomics, no owner check on free.
//! - Deallocation receives the `Layout`, so the size class (and the small/large
//!   split) comes from the layout instead of a metadata lookup.
//! - The heap is an explicit value, not a thread-local lookup.
//! - Built-in byte accounting ([`Stats`]) and arena-style [`LocalHeap::reset`].
//! - Optionally all memory from one fixed region ([`LocalHeap::with_region`]),
//!   for io_uring registered buffers, and a huge-page policy ([`HugePages`]).
//!
//! For objects with other lifetime profiles:
//!
//! - [`Arena`] bump-allocates with bulk [`reset`](Arena::reset), for objects that
//!   die together; the most recent block grows in place. Its chunks come from the
//!   OS ([`Os`]) or from a `LocalHeap` ([`Arena::new_in`]).
//! - [`Reap`] ([`LocalHeap::reap`]) bump-allocates from heap pages and recycles a
//!   page when its last block is freed, for objects that mostly die together
//!   while a few outlive the rest.
//! - [`StackArena`] bump-allocates scratch collections from a stack buffer, with a
//!   fallback allocator.
//!
//! Without passing a heap around:
//!
//! - [`ShardAlloc`] is a zero-sized handle for the current thread's own heap,
//!   made on first use: the shape of a global allocator, paying a thread-local
//!   lookup to allocate and an owner check to free. It is `Send + Sync`: a
//!   block freed on another thread is batched there and sent home to the heap
//!   that made it.
//!
//! `LocalHeap` and the arenas are `!Send`; `ShardAlloc` is the way to share
//! collections between threads. It is also a [`GlobalAlloc`](core::alloc::GlobalAlloc):
//!
//! ```
//! #[global_allocator]
//! static A: shardheap::ShardAlloc = shardheap::ShardAlloc::new();
//! # fn main() { let v = vec![1u8; 100]; assert_eq!(v.len(), 100); }
//! ```
//!
//! See `DESIGN.md`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod arena;
mod class;
mod heap;
mod os;
mod reap;
mod region;
mod remote;
mod segmap;
mod segment;
#[cfg(not(loom))]
mod shard;
mod stack;

pub use arena::{Arena, ArenaStats};
pub use heap::{HugePages, LocalHeap, Stats};
pub use os::Os;
pub use reap::Reap;
#[cfg(not(loom))]
pub use shard::ShardAlloc;
pub use stack::StackArena;
