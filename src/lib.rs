//! Kraph page-table engine.
//!
//! The crate owns the CPU page-table walker and the per-arch PTE encoding
//! for x86_64, aarch64, and riscv64. Designed for downstream pluggability
//! — an IOMMU consumer (in-kraph now, future `kiommu` crate) supplies its
//! own PTE types implementing [`PageTableEntry`] and reuses the walker.
//!
//! # Crate structure
//!
//! - Traits: [`PageTableEntry`], [`PagingMetaData`], [`FrameAllocator`],
//!   [`AddrSpaceActivation`]
//! - Flags: [`AccessFlags`], [`MemoryAttributes`] — shared access/attr model
//! - Mapping surface: [`Mapping`], [`MapBacking`], [`MappingFlags`]
//! - Walker: [`PageTableWalker`] with internal batching for whole-range maps
//! - Per-arch PTE impls: `arch::x86_64`, `arch::aarch64`, `arch::riscv64`
//!
//! # Why not `page_table_multiarch`?
//!
//! Upstream's `GenericPTE::new_table(paddr) -> Self` has no level
//! parameter. AMD-Vi intermediate PDEs encode the child table's level in
//! bits 11:9 — the cursor knows the level during walk but doesn't plumb
//! it through. Rather than fork upstream for one trait-method change, we
//! own the walker substrate and expose `new_table(paddr, level)` day one,
//! unlocking AMD-Vi and future per-level encodings uniformly.

#![cfg_attr(not(test), no_std)]

pub mod arch;

mod error;
mod mapping;
mod meta;
mod paging;
mod walker;

pub use error::{PagingError, PagingResult};
pub use mapping::{IntoMapBacking, MapBacking, Mapping, MappingContiguity, MappingFlags};
pub use meta::{AccessFlags, CachePolicy, Coherency, MemoryAttributes, PageSize, Shareability};
pub use paging::{
    AddrSpaceActivation, AddrSpaceToken, FrameAllocator, NoFlush, PageTableEntry,
    PageTableEntryKind, PagingMetaData, TlbInvalidation,
};
pub use walker::{PageTable, PageTableWalker};
