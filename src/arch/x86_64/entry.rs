use memory_addr::{PhysAddr, VirtAddr};

use x86_64::registers::model_specific::Msr;

use crate::{
    AccessFlags, CachePolicy, MemoryAttributes, PageSize, PageTableEntry, PageTableEntryKind,
    PagingMetaData,
};

/// Bit mask for the physical address portion of an x86_64 PTE.
///
/// PTEs pack bits [51:12] as the target address (4 KiB-aligned); the
/// upper 12 bits are NX + software + available; the lower 12 bits are
/// flags. Processors limit effective PA width to 52 bits or less.
const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;
const PRESENT: u64 = 1 << 0;
const WRITABLE: u64 = 1 << 1;
const USER_ACCESSIBLE: u64 = 1 << 2;
const WRITE_THROUGH: u64 = 1 << 3;
const NO_CACHE: u64 = 1 << 4;
const HUGE_PAGE: u64 = 1 << 7;
const GLOBAL: u64 = 1 << 8;
const NO_EXECUTE: u64 = 1 << 63;

const TABLE_FLAGS: u64 = PRESENT | WRITABLE | USER_ACCESSIBLE;
const CACHE_PAT_INDEX: [u64; 4] = [
    X86_PAT_WRITEBACK_INDEX as u64,
    X86_PAT_UNCACHED_INDEX as u64,
    X86_PAT_WRITE_COMBINING_INDEX as u64,
    X86_PAT_WRITE_THROUGH_INDEX as u64,
];

/// IA32_PAT MSR number.
pub const X86_PAT_MSR: u32 = 0x0000_0277;

/// PAT memory type: uncacheable.
pub const X86_PAT_TYPE_UC: u64 = 0x00;
/// PAT memory type: write-combining.
pub const X86_PAT_TYPE_WC: u64 = 0x01;
/// PAT memory type: write-through.
pub const X86_PAT_TYPE_WT: u64 = 0x04;
/// PAT memory type: write-protected.
pub const X86_PAT_TYPE_WP: u64 = 0x05;
/// PAT memory type: write-back.
pub const X86_PAT_TYPE_WB: u64 = 0x06;
/// PAT memory type: uncacheable-minus.
pub const X86_PAT_TYPE_UC_MINUS: u64 = 0x07;

/// PAT index used for [`CachePolicy::Writeback`].
pub const X86_PAT_WRITEBACK_INDEX: u8 = 0;
/// PAT index used for [`CachePolicy::WriteThrough`].
pub const X86_PAT_WRITE_THROUGH_INDEX: u8 = 1;
/// PAT index used for [`CachePolicy::WriteCombine`].
pub const X86_PAT_WRITE_COMBINING_INDEX: u8 = 2;
/// PAT index used for [`CachePolicy::Uncached`].
pub const X86_PAT_UNCACHED_INDEX: u8 = 3;

/// IA32_PAT value expected by the x86_64 PTE cache-policy encoder.
///
/// Slots 0..3 are selected by the PWT/PCD bits alone, which keeps
/// `PageTableEntry::flags()` decodable without a page size or level:
///
/// - index 0: WB  (`PWT=0, PCD=0`)
/// - index 1: WT  (`PWT=1, PCD=0`)
/// - index 2: WC  (`PWT=0, PCD=1`)
/// - index 3: UC  (`PWT=1, PCD=1`)
///
/// Slots 4..7 preserve useful high-bank aliases for mappings created by
/// external code that uses the PAT bit.
pub const X86_PAT_MSR_VALUE: u64 = pat_entry(0, X86_PAT_TYPE_WB)
    | pat_entry(1, X86_PAT_TYPE_WT)
    | pat_entry(2, X86_PAT_TYPE_WC)
    | pat_entry(3, X86_PAT_TYPE_UC)
    | pat_entry(4, X86_PAT_TYPE_WB)
    | pat_entry(5, X86_PAT_TYPE_WT)
    | pat_entry(6, X86_PAT_TYPE_UC_MINUS)
    | pat_entry(7, X86_PAT_TYPE_UC);

const fn pat_entry(index: u8, ty: u64) -> u64 {
    ty << ((index as u64) * 8)
}

/// Install [`X86_PAT_MSR_VALUE`] into IA32_PAT on the current CPU.
///
/// # Safety
///
/// The caller must ensure this runs at CPL0 on an x86_64 CPU that supports
/// PAT, and must synchronize installation across CPUs before creating or
/// activating mappings that use [`CachePolicy::WriteCombine`].
pub unsafe fn install_x86_pat() {
    unsafe {
        Msr::new(X86_PAT_MSR).write(X86_PAT_MSR_VALUE);
    }
}

// ── Flags ───────────────────────────────────────────────────────────────────

/// x86_64 PTE flag bundle — access bits + memory attributes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Flags {
    pub access: AccessFlags,
    pub attrs: MemoryAttributes,
}

impl X86Flags {
    #[inline]
    pub const fn new(access: AccessFlags) -> Self {
        Self {
            access: access.union(AccessFlags::READ),
            attrs: MemoryAttributes::writeback(),
        }
    }

    #[inline]
    pub const fn with_cache(mut self, cache: CachePolicy) -> Self {
        self.attrs = self.attrs.with_cache(cache);
        self
    }

    #[inline]
    pub const fn with_attrs(mut self, attrs: MemoryAttributes) -> Self {
        self.attrs = attrs;
        self
    }

    #[inline]
    pub const fn cache(self) -> CachePolicy {
        self.attrs.cache()
    }

    #[inline]
    pub const fn attrs(self) -> MemoryAttributes {
        self.attrs
    }

    /// Encode x86 PTE bits appropriate for a leaf entry at the given size.
    #[inline(always)]
    fn to_leaf_bits(self, size: PageSize) -> u64 {
        let access = self.access.bits() as u64;
        PRESENT
            | (access & AccessFlags::WRITE.bits() as u64)
            | ((access & AccessFlags::USER.bits() as u64) >> 1)
            | ((access & AccessFlags::GLOBAL.bits() as u64) << 4)
            | (((access & AccessFlags::EXECUTE.bits() as u64 == 0) as u64) << 63)
            | ((matches!(size, PageSize::Size2M | PageSize::Size1G) as u64) << 7)
            | cache_bits(self.attrs.cache())
    }

    /// Decode from raw x86 PTE bits.
    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        // On x86, PRESENT implies readable — every present mapping grants read.
        let access = ((bits & PRESENT) as u8)
            | ((bits & WRITABLE) as u8)
            | (((bits & USER_ACCESSIBLE) as u8) << 1)
            | (((bits & GLOBAL) >> 4) as u8)
            | (((bits & NO_EXECUTE == 0) as u8) << 2);
        let cache = match ((bits & NO_CACHE) != 0, (bits & WRITE_THROUGH) != 0) {
            (false, false) => CachePolicy::Writeback,
            (false, true) => CachePolicy::WriteThrough,
            (true, false) => CachePolicy::WriteCombine,
            (true, true) => CachePolicy::Uncached,
        };
        Self {
            access: AccessFlags::from_bits_truncate(access),
            attrs: MemoryAttributes::writeback().with_cache(cache),
        }
    }
}

#[inline(always)]
fn cache_bits(cache: CachePolicy) -> u64 {
    let index = CACHE_PAT_INDEX[cache as usize];
    ((index & 0b001) << 3) | ((index & 0b010) << 3)
}

// ── PTE ─────────────────────────────────────────────────────────────────────

/// x86_64 page-table entry.
///
/// `#[repr(transparent)]` over `u64` so the walker can treat a 4 KiB
/// frame as `[X86Pte; 512]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct X86Pte(u64);

impl PageTableEntry for X86Pte {
    type Flags = X86Flags;

    #[inline]
    fn new_leaf(paddr: PhysAddr, flags: Self::Flags, size: PageSize) -> Self {
        Self(((paddr.as_usize() as u64) & ADDR_MASK) | flags.to_leaf_bits(size))
    }

    #[inline]
    fn new_table(paddr: PhysAddr, _level: u8) -> Self {
        // x86 intermediate encoding is uniform across all levels — the
        // `level` parameter is ignored here. (AMD-Vi, same physical
        // architecture but IOMMU stage, does need the level — handled
        // in the AMD-Vi PTE impl, not this CPU-paging one.)
        Self(((paddr.as_usize() as u64) & ADDR_MASK) | TABLE_FLAGS)
    }

    #[inline]
    fn paddr(&self) -> PhysAddr {
        PhysAddr::from_usize((self.0 & ADDR_MASK) as usize)
    }

    #[inline]
    fn flags(&self) -> Self::Flags {
        X86Flags::from_bits(self.0)
    }

    #[inline]
    fn is_present(&self) -> bool {
        (self.0 & PRESENT) != 0
    }

    #[inline]
    fn entry_kind(&self, level: u8) -> PageTableEntryKind {
        // On x86, HUGE_PAGE bit at L2/L3 indicates a block leaf. At L1
        // every present entry is a 4K leaf. Table pointers are uniform
        // at the remaining present intermediate levels.
        if level == 1 || (self.0 & HUGE_PAGE) != 0 {
            PageTableEntryKind::Leaf
        } else {
            PageTableEntryKind::Table
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.0 = 0;
    }

    #[inline]
    fn bits(&self) -> u64 {
        self.0
    }

    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

// ── Metadata ────────────────────────────────────────────────────────────────

/// x86_64 4-level paging metadata (48-bit virtual addresses).
#[derive(Clone, Copy, Debug)]
pub struct X86Meta48;

impl PagingMetaData for X86Meta48 {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 48;

    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        // Level 1 (PT) → bit 12; each level up adds 9 bits.
        // Level 4 (PML4) → bit 39.
        12 + ((level as u32) - 1) * 9
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        match (level, size) {
            (1, PageSize::Size4K) => true,
            (2, PageSize::Size2M) => true,
            (3, PageSize::Size1G) => true, // 1 GiB leaves require CPUID[EDX:1GB]; walker is permissive.
            _ => false,
        }
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// x86_64 5-level paging metadata (57-bit virtual addresses).
///
/// LA57 extends the paging tree with a PML5 root but keeps the CPU leaf
/// sizes unchanged: 4 KiB pages, 2 MiB pages, and 1 GiB pages. Level 4 and
/// level 5 entries are table pointers, not 512 GiB or 256 TiB leaves.
#[derive(Clone, Copy, Debug)]
pub struct X86Meta57;

impl PagingMetaData for X86Meta57 {
    const LEVELS: usize = 5;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 57;

    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        12 + ((level as u32) - 1) * 9
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        X86Meta48::level_supports_leaf(level, size)
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

fn canonical_vaddr_is_valid(vaddr: VirtAddr, bits: usize) -> bool {
    let v = vaddr.as_usize() as u64;
    if bits >= u64::BITS as usize {
        return true;
    }

    let upper = v >> bits;
    let sign_set = (v & (1u64 << (bits - 1))) != 0;
    if sign_set {
        upper == ((1u64 << (64 - bits)) - 1)
    } else {
        upper == 0
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_4k_round_trip() {
        let flags = X86Flags::new(AccessFlags::READ | AccessFlags::WRITE);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x1000), flags, PageSize::Size4K);

        assert!(pte.is_present());
        assert!(pte.is_leaf_at(1)); // L1 present entries are 4K leaves.
        assert!(!pte.is_leaf_at(2)); // HUGE bit not set at intermediate levels.
        assert_eq!(pte.paddr().as_usize(), 0x1000);

        let decoded = pte.flags();
        assert!(decoded.access.contains(AccessFlags::READ));
        assert!(decoded.access.contains(AccessFlags::WRITE));
        assert!(!decoded.access.contains(AccessFlags::EXECUTE));
        assert_eq!(decoded.cache(), CachePolicy::Writeback);
    }

    #[test]
    fn flags_new_normalizes_read() {
        let flags = X86Flags::new(AccessFlags::WRITE);
        assert!(flags.access.contains(AccessFlags::READ));
        assert!(flags.access.contains(AccessFlags::WRITE));
    }

    #[test]
    fn leaf_2m_marks_huge() {
        let flags = X86Flags::new(AccessFlags::READ);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x20_0000), flags, PageSize::Size2M);
        assert!(pte.is_leaf_at(2));
        assert!(pte.is_present());
    }

    #[test]
    fn table_entry_is_not_leaf() {
        let pte = X86Pte::new_table(PhysAddr::from_usize(0x4000), 2);
        assert!(pte.is_present());
        assert!(!pte.is_leaf_at(2));
        assert_eq!(pte.paddr().as_usize(), 0x4000);
    }

    #[test]
    fn uncached_sets_nocache_bit() {
        let flags = X86Flags::new(AccessFlags::READ).with_cache(CachePolicy::Uncached);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x1000), flags, PageSize::Size4K);
        let decoded = pte.flags();
        assert_eq!(
            pte.bits() & (NO_CACHE | WRITE_THROUGH),
            NO_CACHE | WRITE_THROUGH
        );
        assert_eq!(decoded.cache(), CachePolicy::Uncached);
    }

    #[test]
    fn writecombine_uses_pat_wc_slot() {
        let flags = X86Flags::new(AccessFlags::READ).with_cache(CachePolicy::WriteCombine);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x1000), flags, PageSize::Size4K);
        let decoded = pte.flags();
        assert_eq!(pte.bits() & (NO_CACHE | WRITE_THROUGH), NO_CACHE);
        assert_eq!(decoded.cache(), CachePolicy::WriteCombine);
    }

    #[test]
    fn writecombine_huge_uses_same_low_pat_slot() {
        let flags = X86Flags::new(AccessFlags::READ).with_cache(CachePolicy::WriteCombine);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x20_0000), flags, PageSize::Size2M);
        let decoded = pte.flags();
        assert_eq!(pte.bits() & HUGE_PAGE, HUGE_PAGE);
        assert_eq!(pte.bits() & (NO_CACHE | WRITE_THROUGH), NO_CACHE);
        assert_eq!(decoded.cache(), CachePolicy::WriteCombine);
    }

    #[test]
    fn pat_msr_value_matches_cache_policy_indices() {
        let entry = |index: u8| (X86_PAT_MSR_VALUE >> (index * 8)) & 0xff;
        assert_eq!(entry(X86_PAT_WRITEBACK_INDEX), X86_PAT_TYPE_WB);
        assert_eq!(entry(X86_PAT_WRITE_THROUGH_INDEX), X86_PAT_TYPE_WT);
        assert_eq!(entry(X86_PAT_WRITE_COMBINING_INDEX), X86_PAT_TYPE_WC);
        assert_eq!(entry(X86_PAT_UNCACHED_INDEX), X86_PAT_TYPE_UC);
    }

    #[test]
    fn attrs_round_trip_cache_dimension() {
        let attrs = MemoryAttributes::writeback()
            .with_cache(CachePolicy::WriteThrough)
            .with_shareability(crate::Shareability::System)
            .with_coherency(crate::Coherency::Coherent);
        let flags = X86Flags::new(AccessFlags::READ).with_attrs(attrs);
        let pte = X86Pte::new_leaf(PhysAddr::from_usize(0x1000), flags, PageSize::Size4K);
        let decoded = pte.flags();

        assert_eq!(decoded.cache(), CachePolicy::WriteThrough);
        // x86 CPU paging has no descriptor bits for these dimensions, so
        // decode falls back to the neutral coherent inner-shareable attrs.
        assert_eq!(decoded.attrs().shareability(), crate::Shareability::Inner);
        assert_eq!(decoded.attrs().coherency(), crate::Coherency::Coherent);
    }

    #[test]
    fn meta_level_shifts() {
        assert_eq!(X86Meta48::level_shift(1), 12);
        assert_eq!(X86Meta48::level_shift(2), 21);
        assert_eq!(X86Meta48::level_shift(3), 30);
        assert_eq!(X86Meta48::level_shift(4), 39);
        assert_eq!(X86Meta57::level_shift(5), 48);
    }

    #[test]
    fn meta_level_supports_leaf() {
        assert!(X86Meta48::level_supports_leaf(1, PageSize::Size4K));
        assert!(X86Meta48::level_supports_leaf(2, PageSize::Size2M));
        assert!(X86Meta48::level_supports_leaf(3, PageSize::Size1G));
        assert!(!X86Meta48::level_supports_leaf(2, PageSize::Size4K));
        assert!(!X86Meta48::level_supports_leaf(4, PageSize::Size1G));
        assert!(!X86Meta57::level_supports_leaf(4, PageSize::Size512G));
        assert!(!X86Meta57::level_supports_leaf(5, PageSize::Size512G));
    }

    #[test]
    fn meta_4l_accepts_canonical_high_half() {
        assert!(X86Meta48::vaddr_is_valid(VirtAddr::from_usize(
            0xffff_8000_0000_0000
        )));
        assert!(X86Meta48::vaddr_is_valid(VirtAddr::from_usize(
            0x0000_7fff_ffff_ffff
        )));
        assert!(!X86Meta48::vaddr_is_valid(VirtAddr::from_usize(
            0x0000_8000_0000_0000
        )));
        assert!(!X86Meta48::vaddr_is_valid(VirtAddr::from_usize(
            0xffff_7fff_ffff_ffff
        )));
    }

    #[test]
    fn meta_5l_accepts_57_bit_canonical_halves() {
        assert_eq!(X86Meta57::LEVELS, 5);
        assert_eq!(X86Meta57::VA_MAX_BITS, 57);
        assert!(X86Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0x00ff_ffff_ffff_ffff
        )));
        assert!(X86Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0xff00_0000_0000_0000
        )));
        assert!(!X86Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0x0100_0000_0000_0000
        )));
        assert!(!X86Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0xfeff_ffff_ffff_ffff
        )));
    }
}
