//! Per-arch PTE + metadata impls.
//!
//! Selected at compile time via `cfg(target_arch)`. Each arch module
//! exposes only concrete, named metadata/PTE choices:
//!
//! - `<Arch>Pte{Granule}{AddrBits}` — concrete [`crate::PageTableEntry`] impl
//! - `<Arch>Meta{Granule}{AddrBits}` — concrete [`crate::PagingMetaData`] impl
//!
//! The generic internals stay private so consumers choose an explicit paging
//! mode instead of relying on a compatibility alias. Frame allocation bridging
//! to the kraph PT arena lives in the consuming kraph arch module — the crate
//! stays free of allocator/runtime knowledge.
//!
//! The arch root also provides target-gated aliases for shared page-table
//! shapes. These aliases are named by translation shape, not by host/default
//! status, so callers still choose the paging mode explicitly.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "riscv64")]
pub mod riscv64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::{
    X86Flags as Flags4K48, X86Flags as Flags4K57, X86Meta48 as Meta4K48, X86Meta57 as Meta4K57,
    X86PageTable48 as PageTable4K48, X86PageTable57 as PageTable4K57, X86Pte as Pte4K48,
    X86Pte as Pte4K57, X86Tlb as Tlb4K48, X86Tlb as Tlb4K57,
};

#[cfg(target_arch = "x86_64")]
pub const ARCH_NAME: &str = "x86_64";

#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    A64Flags as Flags4K48, A64Flags as Flags4K52, A64Flags as Flags16K48, A64Flags as Flags16K52,
    A64Flags as Flags64K48, A64Flags as Flags64K52, A64Meta4K48 as Meta4K48,
    A64Meta4K52 as Meta4K52, A64Meta16K48 as Meta16K48, A64Meta16K52 as Meta16K52,
    A64Meta64K48 as Meta64K48, A64Meta64K52 as Meta64K52, A64PageTable4K48 as PageTable4K48,
    A64PageTable4K52 as PageTable4K52, A64PageTable16K48 as PageTable16K48,
    A64PageTable16K52 as PageTable16K52, A64PageTable64K48 as PageTable64K48,
    A64PageTable64K52 as PageTable64K52, A64Pte4K48 as Pte4K48, A64Pte4K52 as Pte4K52,
    A64Pte16K48 as Pte16K48, A64Pte16K52 as Pte16K52, A64Pte64K48 as Pte64K48,
    A64Pte64K52 as Pte64K52, A64Tlb as Tlb4K48, A64Tlb as Tlb4K52, A64Tlb as Tlb16K48,
    A64Tlb as Tlb16K52, A64Tlb as Tlb64K48, A64Tlb as Tlb64K52,
};

#[cfg(target_arch = "aarch64")]
pub const ARCH_NAME: &str = "aarch64";

#[cfg(target_arch = "riscv64")]
pub use riscv64::{
    Rv64Flags as Flags4K39, Rv64Flags as Flags4K48, Rv64Flags as Flags4K57, Rv64Meta39 as Meta4K39,
    Rv64Meta48 as Meta4K48, Rv64Meta57 as Meta4K57, Rv64PageTable39 as PageTable4K39,
    Rv64PageTable48 as PageTable4K48, Rv64PageTable57 as PageTable4K57, Rv64Pte as Pte4K39,
    Rv64Pte as Pte4K48, Rv64Pte as Pte4K57, Rv64Tlb as Tlb4K39, Rv64Tlb as Tlb4K48,
    Rv64Tlb as Tlb4K57,
};

#[cfg(target_arch = "riscv64")]
pub const ARCH_NAME: &str = "riscv64";
