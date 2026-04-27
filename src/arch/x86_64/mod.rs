//! x86_64 PTE + metadata.
//!
//! Uses the architectural x86_64 PTE bit layout directly. We intentionally
//! skip upstream's `Mapper`/`OffsetPageTable` — the crate provides its own
//! walker so IOMMU (VT-d, AMD-Vi) can share the same machinery via a different
//! PTE type.
//!
//! # Cache encoding
//!
//! x86 PAT/PWT/PCD triples encode cache policy per-page. Without a
//! configured PAT MSR, the architectural default maps the `PCD`-only selector
//! to UC-. This module exposes [`X86_PAT_MSR_VALUE`] to install the layout used
//! by [`X86Pte`]:
//!
//! | PWT | PCD | Policy      |
//! |-----|-----|-------------|
//! | 0   | 0   | Writeback   |
//! | 1   | 0   | WriteThrough|
//! | 0   | 1   | WriteCombine|
//! | 1   | 1   | Uncached    |
//!
//! Consumers must install the PAT value on every CPU before activating
//! [`crate::CachePolicy::WriteCombine`] mappings.

mod activation;
mod entry;
mod tlb;

pub use activation::{X86Cr3Mode, X86PagingActivation, X86PagingControls, X86PagingToken};
pub use entry::{
    X86_PAT_MSR, X86_PAT_MSR_VALUE, X86_PAT_TYPE_UC, X86_PAT_TYPE_UC_MINUS, X86_PAT_TYPE_WB,
    X86_PAT_TYPE_WC, X86_PAT_TYPE_WP, X86_PAT_TYPE_WT, X86_PAT_UNCACHED_INDEX,
    X86_PAT_WRITE_COMBINING_INDEX, X86_PAT_WRITE_THROUGH_INDEX, X86_PAT_WRITEBACK_INDEX, X86Flags,
    X86Meta48, X86Meta57, X86Pte, install_x86_pat,
};
pub use tlb::{X86InvlpgbTlb, X86PcidTlb, X86Tlb};

/// x86_64 4-level CPU page table: 48-bit canonical virtual addresses.
pub type X86PageTable48<Alloc> = crate::PageTableWalker<X86Meta48, X86Pte, Alloc>;

/// x86_64 5-level CPU page table: 57-bit canonical virtual addresses.
pub type X86PageTable57<Alloc> = crate::PageTableWalker<X86Meta57, X86Pte, Alloc>;
