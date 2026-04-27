//! Kore Memory: a crate for page-table management and IOMMU support in Rust.
//!
//! The crate owns the CPU page-table walker and the per-arch PTE encoding
//! for x86_64, aarch64, and riscv64. Designed for downstream pluggability
//! — an IOMMU consumer (in-kraph now, future `kiommu` crate) supplies its
//! own PTE types implementing [`PageTableEntry`] and reuses the walker.

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
