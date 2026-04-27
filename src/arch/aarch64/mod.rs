//! AArch64 PTE + metadata.
//!
//! Concrete metadata/PTE types are provided for 4 KiB, 16 KiB, and 64 KiB
//! translation granules across 48-bit and 52-bit address widths. There is no
//! default `A64Pte`/`A64Meta` alias; runtime detection should select one of the
//! explicit concrete variants.
//!
//! # Cache encoding
//!
//! AArch64 descriptors carry a MAIR_EL1 attribute index. This module exposes
//! [`A64_MAIR_EL1_VALUE`] and [`install_a64_mair_el1`] for the MAIR layout used
//! by the concrete PTE types:
//!
//! | AttrIndx | Policy       | MAIR byte |
//! |----------|--------------|-----------|
//! | 0        | Writeback    | `0xff`    |
//! | 1        | Uncached     | `0x00`    |
//! | 2        | WriteCombine | `0x44`    |
//! | 3        | WriteThrough | `0xbb`    |
//!
//! `WriteCombine` is represented as Normal Non-cacheable memory. It is the
//! portable AArch64 analogue for framebuffer/device-buffer mappings, not a
//! literal x86 PAT WC type.
//!
//! The active TCR_EL1 granule, address-size, and IPS/DS settings must match
//! the selected `A64PageTable{4K,16K,64K}{48,52}` type.

mod activation;
mod entry;
mod tlb;

pub use activation::{
    A64PagingActivation, A64PagingControls, A64PagingToken, A64Ttbr, A64TtbrConfig,
};
pub use entry::{
    A64_MAIR_ATTR_DEVICE_NGNRNE, A64_MAIR_ATTR_NORMAL_NC, A64_MAIR_ATTR_WRITEBACK,
    A64_MAIR_ATTR_WRITETHROUGH, A64_MAIR_DEVICE_INDEX, A64_MAIR_EL1_VALUE,
    A64_MAIR_NORMAL_NC_INDEX, A64_MAIR_WRITEBACK_INDEX, A64_MAIR_WRITETHROUGH_INDEX, A64Flags,
    A64Meta4K48, A64Meta4K52, A64Meta16K48, A64Meta16K52, A64Meta64K48, A64Meta64K52, A64Pte4K48,
    A64Pte4K52, A64Pte16K48, A64Pte16K52, A64Pte64K48, A64Pte64K52, install_a64_mair_el1,
};
pub use tlb::A64Tlb;

/// AArch64 4 KiB granule page table with 48-bit VA/OA.
pub type A64PageTable4K48<Alloc> = crate::PageTableWalker<A64Meta4K48, A64Pte4K48, Alloc>;

/// AArch64 4 KiB granule page table with 52-bit VA/OA.
pub type A64PageTable4K52<Alloc> = crate::PageTableWalker<A64Meta4K52, A64Pte4K52, Alloc>;

/// AArch64 16 KiB granule page table with 48-bit VA/OA.
pub type A64PageTable16K48<Alloc> = crate::PageTableWalker<A64Meta16K48, A64Pte16K48, Alloc>;

/// AArch64 16 KiB granule page table with 52-bit VA/OA.
pub type A64PageTable16K52<Alloc> = crate::PageTableWalker<A64Meta16K52, A64Pte16K52, Alloc>;

/// AArch64 64 KiB granule page table with 48-bit VA/OA.
pub type A64PageTable64K48<Alloc> = crate::PageTableWalker<A64Meta64K48, A64Pte64K48, Alloc>;

/// AArch64 64 KiB granule page table with 52-bit VA/OA.
pub type A64PageTable64K52<Alloc> = crate::PageTableWalker<A64Meta64K52, A64Pte64K52, Alloc>;
