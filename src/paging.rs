use core::fmt::Debug;

use memory_addr::{MemoryAddr, PhysAddr, PhysAddrRange, VirtAddr};

use crate::{PageSize, PagingResult};

/// A present page-table entry's decoded role at a particular level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PageTableEntryKind {
    /// Entry points at the next lower-level table.
    Table,
    /// Entry terminates the walk and maps a physical leaf/block.
    Leaf,
    /// Entry encoding is not valid at this level.
    Invalid,
}

impl PageTableEntryKind {
    #[inline]
    pub const fn is_leaf(self) -> bool {
        matches!(self, Self::Leaf)
    }

    #[inline]
    pub const fn is_table(self) -> bool {
        matches!(self, Self::Table)
    }
}

/// A page-table entry.
///
/// # Representation contract
///
/// Every impl must have the same size and alignment as `u64` so the
/// walker can cast a page-table frame to a `PTE` array and manipulate
/// entries in place. The walker asserts this at construction time; impls
/// should still use `#[repr(transparent)]` over a `u64` so the layout is
/// stable across compiler versions.
///
/// # Level parameter on `new_table`
///
/// Intermediate table PTEs at level `L` point at child tables at level
/// `L - 1`. Most CPU arches ignore `level`; AMD-Vi requires it to encode
/// `pm_level_enc(child_level)` at bits 11:9 of each intermediate PDE.
pub trait PageTableEntry: Copy + Debug + Sync + Send + 'static {
    /// Per-impl permission + memory-type bundle. Opaque to the walker.
    /// `Eq` is required so `merge_at` can verify all child entries share
    /// identical flags before coalescing.
    type Flags: Copy + Debug + Default + Eq;

    /// Construct a leaf entry at the target page size.
    fn new_leaf(paddr: PhysAddr, flags: Self::Flags, size: PageSize) -> Self;

    /// Construct an intermediate entry pointing at a child table.
    ///
    /// `level` is the level *of this entry* (i.e., the level of the
    /// containing table, not the child's level).
    fn new_table(paddr: PhysAddr, level: u8) -> Self;

    /// Physical address the entry points at.
    fn paddr(&self) -> PhysAddr;

    /// Current flags on this entry.
    fn flags(&self) -> Self::Flags;

    /// Whether the entry is present (valid mapping or valid table).
    fn is_present(&self) -> bool;

    /// Decode the entry's role at `level`.
    ///
    /// The walker calls this only after [`is_present`](Self::is_present)
    /// returns true. The `level` argument is part of the contract because
    /// several descriptor formats are level-sensitive.
    fn entry_kind(&self, level: u8) -> PageTableEntryKind;

    /// Convenience for callers that only need terminality.
    #[inline]
    fn is_leaf_at(&self, level: u8) -> bool {
        self.entry_kind(level).is_leaf()
    }

    /// Zero out the entry.
    fn clear(&mut self);

    /// Raw 64-bit backing.
    fn bits(&self) -> u64;

    /// Reconstruct from raw 64-bit backing.
    fn from_bits(bits: u64) -> Self;
}

/// Per-arch paging metadata.
///
/// # Configuring table geometry
///
/// The defaults match a 4 KiB page-table frame with 9-bit table indices
/// (512 entries per frame) — the layout used by x86_64 4-level, aarch64
/// 4K granule, and riscv Sv39/Sv48/Sv57.
pub trait PagingMetaData: Sync + Send + 'static {
    /// Number of translation levels (e.g. x86 4-level = 4).
    const LEVELS: usize;

    /// Maximum physical address bits the PTE can encode.
    const PA_MAX_BITS: usize;

    /// Maximum virtual (or IO-virtual) address bits the walker handles.
    const VA_MAX_BITS: usize;

    /// Bits per table index.
    const INDEX_BITS: u32 = 9;

    /// Finest translation granule for this paging format.
    const BASE_PAGE_SIZE: PageSize = PageSize::Size4K;

    /// Physical size/alignment of an intermediate page-table frame.
    const TABLE_FRAME_SIZE: PageSize = Self::BASE_PAGE_SIZE;

    /// Entries per page-table frame.
    const ENTRIES_PER_TABLE: usize = 1usize << Self::INDEX_BITS;

    /// The VA type the walker accepts. CPU paging uses [`VirtAddr`];
    /// IOMMU impls use an IOVA newtype.
    type VirtAddr: MemoryAddr;

    /// Bit-shift for indexing into the table at `level`.
    fn level_shift(level: u8) -> u32;

    /// Number of table-index bits consumed at `level`.
    ///
    /// Most formats use a uniform width at every level. AArch64 16 KiB and
    /// 64 KiB granules have narrower root tables for some VA widths, so the
    /// walker asks per level before masking an index.
    #[inline]
    fn level_index_bits(_level: u8) -> u32 {
        Self::INDEX_BITS
    }

    /// Whether a leaf of `size` may terminate at `level`.
    fn level_supports_leaf(level: u8, size: PageSize) -> bool;

    /// Default: derive leaf size from level via `level_shift`.
    #[inline]
    fn leaf_size_at_level(level: u8) -> Option<PageSize> {
        match Self::level_shift(level) {
            12 => Some(PageSize::Size4K),
            14 => Some(PageSize::Size16K),
            16 => Some(PageSize::Size64K),
            21 => Some(PageSize::Size2M),
            25 => Some(PageSize::Size32M),
            29 => Some(PageSize::Size512M),
            30 => Some(PageSize::Size1G),
            36 => Some(PageSize::Size64G),
            39 => Some(PageSize::Size512G),
            42 => Some(PageSize::Size4T),
            _ => None,
        }
    }

    /// Default: pick the lowest level that natively supports `size`.
    #[inline]
    fn size_to_level(size: PageSize) -> Option<u8> {
        let mut lvl = 1u8;
        while (lvl as usize) <= Self::LEVELS {
            if Self::level_supports_leaf(lvl, size) {
                return Some(lvl);
            }
            lvl += 1;
        }
        None
    }

    /// Whether `vaddr` falls inside the legal input-address range.
    #[inline]
    fn vaddr_is_valid(vaddr: Self::VirtAddr) -> bool {
        let v: usize = vaddr.into();
        if Self::VA_MAX_BITS >= usize::BITS as usize {
            return true;
        }
        v < (1usize << Self::VA_MAX_BITS)
    }

    /// Whether `paddr` is representable in the PTE's physical-address field.
    #[inline]
    fn paddr_is_valid(paddr: PhysAddr) -> bool {
        let p: usize = paddr.as_usize();
        if Self::PA_MAX_BITS >= usize::BITS as usize {
            return true;
        }
        p < (1usize << Self::PA_MAX_BITS)
    }
}

/// Post-mutation invalidation contract.
pub trait TlbInvalidation<V: MemoryAddr>: Sync + Send + 'static {
    /// Invalidate one base-granule TLB entry on the local CPU.
    fn flush_tlb_local(&self, vaddr: V);

    /// Invalidate the entire local TLB.
    fn flush_tlb_all_local(&self);

    /// Hardware-assisted range invalidation. Default: per-page sweep.
    fn flush_tlb_range_local(&self, start: V, page_size: PageSize, count_pages: usize) {
        let stride = page_size.bytes();
        let mut base: usize = start.into();
        for _ in 0..count_pages {
            self.flush_tlb_local(<V as From<usize>>::from(base));
            base = base.saturating_add(stride);
        }
    }

    /// Policy: should an accumulated batch of `pending_count` per-page
    /// invalidations be replaced by a single full-TLB flush?
    fn prefer_full_flush(&self, pending_count: usize) -> bool {
        pending_count > 32
    }
}

/// No-op invalidation.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFlush;

impl<V: MemoryAddr> TlbInvalidation<V> for NoFlush {
    #[inline(always)]
    fn flush_tlb_local(&self, _vaddr: V) {}
    #[inline(always)]
    fn flush_tlb_all_local(&self) {}
    #[inline(always)]
    fn flush_tlb_range_local(&self, _start: V, _page_size: PageSize, _count_pages: usize) {}
    #[inline(always)]
    fn prefer_full_flush(&self, _pending_count: usize) -> bool {
        false
    }
}

/// Installed address-space token metadata shared by every activator.
///
/// Tokens are plain hardware-context descriptors and are safe to move or share
/// across CPUs; per-CPU validity/coherency still belongs to the controls that
/// produced the token.
pub trait AddrSpaceToken: Copy + Debug + Eq + Send + Sync + 'static {
    /// Physical root frame represented by this token.
    fn root(self) -> PhysAddr;
}

/// CPU-local address-space installation and activation policy.
///
/// Page tables produce a root physical address; callers provide the
/// architecture- and CPU-local controls required to turn that root into a live
/// hardware context. On x86_64 the token might include a CR3 frame, PCID, and
/// no-flush policy. On aarch64 it might include TTBR state plus an ASID, and on
/// RISC-V it might encode SATP mode, PPN, and ASID.
///
/// Activators are immutable validators/encoders by default. If an
/// implementation tracks per-CPU state, it should do so with interior
/// mutability so generic callers do not need to hold exclusive access just to
/// encode or activate a token.
pub trait AddrSpaceActivation: Sync + Send + 'static {
    /// Opaque installed address-space handle understood by this activator.
    type Token: AddrSpaceToken;

    /// Caller policy used to encode the hardware token.
    type Controls: Copy + Debug + Eq;

    /// Validate a page-table root and encode it with `controls`, but do not
    /// make it active on the CPU.
    fn install(&self, root: PhysAddr, controls: Self::Controls) -> PagingResult<Self::Token>;

    /// Make a previously installed token active on the current CPU.
    ///
    /// # Safety
    ///
    /// The caller must ensure that switching to `token` is legal for the
    /// current execution context: the target page table must map the code,
    /// stack, and data needed to continue execution; controls that change the
    /// active root's interpretation, such as x86 LA57 or AArch64 SCTLR.M, must
    /// already match the new root shape before they are written; and any
    /// architecture-specific interrupt/preemption constraints must be upheld.
    unsafe fn activate(&self, token: Self::Token) -> PagingResult;

    /// Return the currently active token when the activator can observe it.
    ///
    /// Tokens returned here carry observed hardware controls, not caller
    /// defaults.
    #[inline]
    fn current(&self) -> PagingResult<Option<Self::Token>> {
        Ok(None)
    }

    /// Release activator-local state for a previously installed token.
    #[inline]
    fn uninstall(&self, _token: Self::Token) -> PagingResult {
        Ok(())
    }
}

/// Frame-allocation contract used by the walker.
pub trait FrameAllocator: Sync + Send + 'static {
    /// Allocate a zeroed contiguous physical range.
    fn allocate(size: usize, align: PageSize) -> PagingResult<PhysAddrRange>;

    /// Release a range previously returned by [`allocate`](Self::allocate).
    fn deallocate(range: PhysAddrRange) -> PagingResult;

    /// HHDM-style translation: physical frame → kernel-visible virtual pointer.
    fn phys_to_virt(paddr: PhysAddr) -> VirtAddr;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct MockToken {
        root: PhysAddr,
    }

    impl AddrSpaceToken for MockToken {
        fn root(self) -> PhysAddr {
            self.root
        }
    }

    #[derive(Default)]
    struct MockActivation;

    impl AddrSpaceActivation for MockActivation {
        type Token = MockToken;
        type Controls = ();

        fn install(&self, root: PhysAddr, _controls: Self::Controls) -> PagingResult<Self::Token> {
            Ok(MockToken { root })
        }

        unsafe fn activate(&self, _token: Self::Token) -> PagingResult {
            Ok(())
        }
    }

    #[test]
    fn activation_trait_installs_tokens_with_portable_root() {
        let activation = MockActivation;
        let root = PhysAddr::from(0x2000usize);

        let token = activation.install(root, ()).unwrap();
        assert_eq!(token.root(), root);
        assert_eq!(activation.current().unwrap(), None);

        unsafe {
            activation.activate(token).unwrap();
        }

        activation.uninstall(token).unwrap();
    }
}
