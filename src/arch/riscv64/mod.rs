//! riscv64 PTE + metadata.
//!
//! RISC-V Sv39/Sv48/Sv57 all use the same 4 KiB table geometry: 9-bit table
//! indices, a 4 KiB base page, and leaf entries at any translation level. This
//! module provides explicit concrete metadata types for each VA width.
//!
//! # Cache encoding
//!
//! Base RISC-V page tables do not encode cache policy in PTEs. [`Rv64Pte`]
//! therefore preserves all [`crate::CachePolicy`] inputs as architectural
//! default/PMA memory and decodes them as `Writeback`.
//!
//! Systems with Svpbmt may opt into [`Rv64SvpbmtPte`], which encodes PBMT bits:
//!
//! | PBMT | Policy       |
//! |------|--------------|
//! | 0    | Writeback    |
//! | 1    | WriteCombine |
//! | 2    | Uncached     |
//!
//! `WriteThrough` has no direct Svpbmt representation and is clamped to the
//! default/PMA policy.

mod activation;
mod entry;
mod tlb;

pub use activation::{Rv64SatpActivation, Rv64SatpControls, Rv64SatpToken};
pub use entry::{
    RV64_PBMT_IO, RV64_PBMT_MASK, RV64_PBMT_NC, RV64_PBMT_PMA, Rv64Flags, Rv64Meta39, Rv64Meta48,
    Rv64Meta57, Rv64Pte, Rv64SvpbmtPte,
};
pub use tlb::Rv64Tlb;

/// RISC-V Sv39 page table.
pub type Rv64PageTable39<Alloc> = crate::PageTableWalker<Rv64Meta39, Rv64Pte, Alloc>;

/// RISC-V Sv48 page table.
pub type Rv64PageTable48<Alloc> = crate::PageTableWalker<Rv64Meta48, Rv64Pte, Alloc>;

/// RISC-V Sv57 page table.
pub type Rv64PageTable57<Alloc> = crate::PageTableWalker<Rv64Meta57, Rv64Pte, Alloc>;

/// RISC-V Sv39 page table with Svpbmt memory-type bits enabled.
pub type Rv64SvpbmtPageTable39<Alloc> = crate::PageTableWalker<Rv64Meta39, Rv64SvpbmtPte, Alloc>;

/// RISC-V Sv48 page table with Svpbmt memory-type bits enabled.
pub type Rv64SvpbmtPageTable48<Alloc> = crate::PageTableWalker<Rv64Meta48, Rv64SvpbmtPte, Alloc>;

/// RISC-V Sv57 page table with Svpbmt memory-type bits enabled.
pub type Rv64SvpbmtPageTable57<Alloc> = crate::PageTableWalker<Rv64Meta57, Rv64SvpbmtPte, Alloc>;
