use core::fmt;
use core::marker::PhantomData;

use aarch64_cpu::{
    asm::barrier::{SY, dsb, isb},
    registers::{MAIR_EL1, Writeable},
};
use aarch64_cpu_ext::structures::tte::{
    AccessPermission, Granule, Granule4KB, Granule16KB, Granule64KB, OA, OA48, OA52,
    Shareability as A64Shareability, TTE64,
};
use memory_addr::{PhysAddr, VirtAddr};

use crate::{
    AccessFlags, CachePolicy, Coherency, MemoryAttributes, PageSize, PageTableEntry,
    PageTableEntryKind, PagingMetaData, Shareability,
};

/// MAIR index used for normal write-back cacheable memory.
pub const A64_MAIR_WRITEBACK_INDEX: u64 = 0;
/// MAIR index used for strongly ordered device memory.
pub const A64_MAIR_DEVICE_INDEX: u64 = 1;
/// MAIR index used for normal non-cacheable memory.
///
/// AArch64 does not have an x86-style per-PTE write-combining memory type.
/// `CachePolicy::WriteCombine` maps to this Normal-NC slot as the portable
/// framebuffer/device-buffer analogue while preserving normal-memory ordering.
pub const A64_MAIR_NORMAL_NC_INDEX: u64 = 2;
/// MAIR index used for normal write-through cacheable memory.
pub const A64_MAIR_WRITETHROUGH_INDEX: u64 = 3;

/// MAIR attribute byte for normal write-back cacheable memory.
pub const A64_MAIR_ATTR_WRITEBACK: u64 = 0xff;
/// MAIR attribute byte for Device-nGnRnE memory.
pub const A64_MAIR_ATTR_DEVICE_NGNRNE: u64 = 0x00;
/// MAIR attribute byte for normal non-cacheable memory.
pub const A64_MAIR_ATTR_NORMAL_NC: u64 = 0x44;
/// MAIR attribute byte for normal write-through cacheable memory.
pub const A64_MAIR_ATTR_WRITETHROUGH: u64 = 0xbb;

/// Conventional MAIR_EL1 value matching the attribute indices above.
///
/// Consumers may choose a different MAIR layout, but then they must use an
/// AArch64 flag/PTE type with matching cache-policy encoding.
pub const A64_MAIR_EL1_VALUE: u64 = A64_MAIR_ATTR_WRITEBACK
    | (A64_MAIR_ATTR_DEVICE_NGNRNE << (A64_MAIR_DEVICE_INDEX * 8))
    | (A64_MAIR_ATTR_NORMAL_NC << (A64_MAIR_NORMAL_NC_INDEX * 8))
    | (A64_MAIR_ATTR_WRITETHROUGH << (A64_MAIR_WRITETHROUGH_INDEX * 8));

/// Install [`A64_MAIR_EL1_VALUE`] into MAIR_EL1 on the current CPU.
///
/// # Safety
///
/// The caller must ensure this runs at EL1 or a privileged context allowed to
/// write MAIR_EL1. The value must be installed, with appropriate global CPU
/// synchronization, before activating page tables whose descriptors were
/// encoded by this module.
pub unsafe fn install_a64_mair_el1() {
    MAIR_EL1.set(A64_MAIR_EL1_VALUE);
    dsb(SY);
    isb(SY);
}

/// AArch64 stage-1 PTE flag bundle: access bits + memory attributes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct A64Flags {
    pub access: AccessFlags,
    pub attrs: MemoryAttributes,
}

impl A64Flags {
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

    fn apply_to_tte<G: Granule, O: OA>(self, tte: &mut TTE64<G, O>) {
        tte.set_attr_index(cache_to_attr_index(self.attrs.cache()));
        tte.set_access_permission(access_to_ap(self.access));
        tte.set_shareability(shareability_to_a64(self.attrs.shareability()));
        tte.set_executable(self.access.contains(AccessFlags::EXECUTE));
        tte.set_privileged_executable(self.access.contains(AccessFlags::EXECUTE));

        if !self.access.contains(AccessFlags::GLOBAL) {
            tte.set_not_global();
        }
    }

    fn from_tte<G: Granule, O: OA>(tte: TTE64<G, O>) -> Self {
        let ap = tte.access_permission();
        let mut access = AccessFlags::READ;

        if ap.allows_privileged_write() {
            access |= AccessFlags::WRITE;
        }
        if ap.allows_unprivileged() {
            access |= AccessFlags::USER;
        }
        if tte.is_executable() || tte.is_privileged_executable() {
            access |= AccessFlags::EXECUTE;
        }
        if tte.is_global() {
            access |= AccessFlags::GLOBAL;
        }

        let shareability = shareability_from_a64(tte.shareability());
        let coherency = if matches!(shareability, Shareability::NonShareable) {
            Coherency::NonCoherent
        } else {
            Coherency::Coherent
        };

        Self {
            access,
            attrs: MemoryAttributes::new(
                cache_from_attr_index(tte.attr_index()),
                shareability,
                coherency,
            ),
        }
    }
}

fn access_to_ap(access: AccessFlags) -> AccessPermission {
    match (
        access.contains(AccessFlags::USER),
        access.contains(AccessFlags::WRITE),
    ) {
        (false, false) => AccessPermission::PrivilegedReadOnly,
        (false, true) => AccessPermission::PrivilegedReadWrite,
        (true, false) => AccessPermission::ReadOnly,
        (true, true) => AccessPermission::ReadWrite,
    }
}

fn shareability_to_a64(shareability: Shareability) -> A64Shareability {
    match shareability {
        Shareability::NonShareable => A64Shareability::NonShareable,
        Shareability::Inner => A64Shareability::InnerShareable,
        Shareability::Outer | Shareability::System => A64Shareability::OuterShareable,
    }
}

fn shareability_from_a64(shareability: A64Shareability) -> Shareability {
    match shareability {
        A64Shareability::NonShareable => Shareability::NonShareable,
        A64Shareability::OuterShareable => Shareability::Outer,
        A64Shareability::InnerShareable => Shareability::Inner,
    }
}

fn cache_to_attr_index(cache: CachePolicy) -> u64 {
    match cache {
        CachePolicy::Writeback => A64_MAIR_WRITEBACK_INDEX,
        CachePolicy::Uncached => A64_MAIR_DEVICE_INDEX,
        CachePolicy::WriteCombine => A64_MAIR_NORMAL_NC_INDEX,
        CachePolicy::WriteThrough => A64_MAIR_WRITETHROUGH_INDEX,
    }
}

fn cache_from_attr_index(index: u64) -> CachePolicy {
    match index {
        A64_MAIR_WRITEBACK_INDEX => CachePolicy::Writeback,
        A64_MAIR_DEVICE_INDEX => CachePolicy::Uncached,
        A64_MAIR_NORMAL_NC_INDEX => CachePolicy::WriteCombine,
        A64_MAIR_WRITETHROUGH_INDEX => CachePolicy::WriteThrough,
        _ => CachePolicy::Writeback,
    }
}

/// AArch64 stage-1 page-table entry implementation shared by the concrete
/// public PTE wrappers.
#[repr(transparent)]
struct A64PteFor<G: Granule, O: OA> {
    bits: u64,
    _p: PhantomData<fn() -> (G, O)>,
}

impl<G: Granule, O: OA> A64PteFor<G, O> {
    #[inline]
    const fn new(bits: u64) -> Self {
        Self {
            bits,
            _p: PhantomData,
        }
    }

    #[inline]
    fn tte(self) -> TTE64<G, O> {
        TTE64::new(self.bits)
    }
}

impl<G: Granule, O: OA> Clone for A64PteFor<G, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Granule, O: OA> Copy for A64PteFor<G, O> {}

impl<G: Granule, O: OA> Default for A64PteFor<G, O> {
    #[inline]
    fn default() -> Self {
        Self::new(0)
    }
}

impl<G: Granule, O: OA> PartialEq for A64PteFor<G, O> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<G: Granule, O: OA> Eq for A64PteFor<G, O> {}

impl<G: Granule, O: OA> fmt::Debug for A64PteFor<G, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("A64Pte")
            .field(&format_args!("{:#018x}", self.bits))
            .finish()
    }
}

impl<G, O> PageTableEntry for A64PteFor<G, O>
where
    G: Granule + Send + Sync + 'static,
    O: OA + Send + Sync + 'static,
{
    type Flags = A64Flags;

    #[inline]
    fn new_leaf(paddr: PhysAddr, flags: Self::Flags, size: PageSize) -> Self {
        let mut tte = if matches!(
            size,
            PageSize::Size4K | PageSize::Size16K | PageSize::Size64K
        ) {
            TTE64::<G, O>::new_table(paddr.as_usize() as u64)
        } else {
            TTE64::<G, O>::new_block(paddr.as_usize() as u64)
        };
        flags.apply_to_tte(&mut tte);
        Self::new(tte.get())
    }

    #[inline]
    fn new_table(paddr: PhysAddr, _level: u8) -> Self {
        Self::new(TTE64::<G, O>::new_table(paddr.as_usize() as u64).get())
    }

    #[inline]
    fn paddr(&self) -> PhysAddr {
        PhysAddr::from_usize(self.tte().address() as usize)
    }

    #[inline]
    fn flags(&self) -> Self::Flags {
        A64Flags::from_tte(self.tte())
    }

    #[inline]
    fn is_present(&self) -> bool {
        self.tte().is_valid()
    }

    #[inline]
    fn entry_kind(&self, level: u8) -> PageTableEntryKind {
        let tte = self.tte();
        if !tte.is_valid() {
            PageTableEntryKind::Invalid
        } else if level == 1 || tte.is_block() {
            PageTableEntryKind::Leaf
        } else if tte.is_table() {
            PageTableEntryKind::Table
        } else {
            PageTableEntryKind::Invalid
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.bits = 0;
    }

    #[inline]
    fn bits(&self) -> u64 {
        self.bits
    }

    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self::new(bits)
    }
}

macro_rules! define_a64_pte {
    ($name:ident, $granule:ty, $oa:ty) => {
        #[doc = concat!("AArch64 stage-1 PTE for ", stringify!($name), ".")]
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name(A64PteFor<$granule, $oa>);

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(&format_args!("{:#018x}", self.0.bits()))
                    .finish()
            }
        }

        impl PageTableEntry for $name {
            type Flags = A64Flags;

            #[inline]
            fn new_leaf(paddr: PhysAddr, flags: Self::Flags, size: PageSize) -> Self {
                Self(<A64PteFor<$granule, $oa> as PageTableEntry>::new_leaf(
                    paddr, flags, size,
                ))
            }

            #[inline]
            fn new_table(paddr: PhysAddr, level: u8) -> Self {
                Self(<A64PteFor<$granule, $oa> as PageTableEntry>::new_table(
                    paddr, level,
                ))
            }

            #[inline]
            fn paddr(&self) -> PhysAddr {
                self.0.paddr()
            }

            #[inline]
            fn flags(&self) -> Self::Flags {
                self.0.flags()
            }

            #[inline]
            fn is_present(&self) -> bool {
                self.0.is_present()
            }

            #[inline]
            fn entry_kind(&self, level: u8) -> PageTableEntryKind {
                self.0.entry_kind(level)
            }

            #[inline]
            fn clear(&mut self) {
                self.0.clear();
            }

            #[inline]
            fn bits(&self) -> u64 {
                self.0.bits()
            }

            #[inline]
            fn from_bits(bits: u64) -> Self {
                Self(<A64PteFor<$granule, $oa> as PageTableEntry>::from_bits(
                    bits,
                ))
            }
        }
    };
}

define_a64_pte!(A64Pte4K48, Granule4KB, OA48);
define_a64_pte!(A64Pte4K52, Granule4KB, OA52);
define_a64_pte!(A64Pte16K48, Granule16KB, OA48);
define_a64_pte!(A64Pte16K52, Granule16KB, OA52);
define_a64_pte!(A64Pte64K48, Granule64KB, OA48);
define_a64_pte!(A64Pte64K52, Granule64KB, OA52);

/// AArch64 4 KiB granule, 48-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta4K48;

impl PagingMetaData for A64Meta4K48 {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 48;
    const VA_MAX_BITS: usize = 48;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        12 + ((level as u32) - 1) * 9
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size4K) | (2, PageSize::Size2M) | (3, PageSize::Size1G)
        )
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// AArch64 4 KiB granule, 52-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta4K52;

impl PagingMetaData for A64Meta4K52 {
    const LEVELS: usize = 5;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 52;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        12 + ((level as u32) - 1) * 9
    }

    #[inline]
    fn level_index_bits(level: u8) -> u32 {
        if level == 5 { 4 } else { 9 }
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size4K) | (2, PageSize::Size2M) | (3, PageSize::Size1G)
        )
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// AArch64 16 KiB granule, 48-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta16K48;

impl PagingMetaData for A64Meta16K48 {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 48;
    const VA_MAX_BITS: usize = 48;
    const INDEX_BITS: u32 = 11;
    const BASE_PAGE_SIZE: PageSize = PageSize::Size16K;
    const TABLE_FRAME_SIZE: PageSize = PageSize::Size16K;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        14 + ((level as u32) - 1) * 11
    }

    #[inline]
    fn level_index_bits(level: u8) -> u32 {
        if level == 4 { 1 } else { 11 }
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size16K) | (2, PageSize::Size32M) | (3, PageSize::Size64G)
        )
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// AArch64 16 KiB granule, 52-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta16K52;

impl PagingMetaData for A64Meta16K52 {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 52;
    const INDEX_BITS: u32 = 11;
    const BASE_PAGE_SIZE: PageSize = PageSize::Size16K;
    const TABLE_FRAME_SIZE: PageSize = PageSize::Size16K;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        14 + ((level as u32) - 1) * 11
    }

    #[inline]
    fn level_index_bits(level: u8) -> u32 {
        if level == 4 { 5 } else { 11 }
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size16K) | (2, PageSize::Size32M) | (3, PageSize::Size64G)
        )
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// AArch64 64 KiB granule, 48-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta64K48;

impl PagingMetaData for A64Meta64K48 {
    const LEVELS: usize = 3;
    const PA_MAX_BITS: usize = 48;
    const VA_MAX_BITS: usize = 48;
    const INDEX_BITS: u32 = 13;
    const BASE_PAGE_SIZE: PageSize = PageSize::Size64K;
    const TABLE_FRAME_SIZE: PageSize = PageSize::Size64K;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        16 + ((level as u32) - 1) * 13
    }

    #[inline]
    fn level_index_bits(level: u8) -> u32 {
        if level == 3 { 6 } else { 13 }
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size64K) | (2, PageSize::Size512M) | (3, PageSize::Size4T)
        )
    }

    #[inline]
    fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
        canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
    }
}

/// AArch64 64 KiB granule, 52-bit VA/OA stage-1 metadata.
#[derive(Clone, Copy, Debug)]
pub struct A64Meta64K52;

impl PagingMetaData for A64Meta64K52 {
    const LEVELS: usize = 3;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 52;
    const INDEX_BITS: u32 = 13;
    const BASE_PAGE_SIZE: PageSize = PageSize::Size64K;
    const TABLE_FRAME_SIZE: PageSize = PageSize::Size64K;
    type VirtAddr = VirtAddr;

    #[inline]
    fn level_shift(level: u8) -> u32 {
        16 + ((level as u32) - 1) * 13
    }

    #[inline]
    fn level_index_bits(level: u8) -> u32 {
        if level == 3 { 10 } else { 13 }
    }

    #[inline]
    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size64K) | (2, PageSize::Size512M) | (3, PageSize::Size4T)
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_4k_round_trip() {
        let flags = A64Flags::new(AccessFlags::READ | AccessFlags::WRITE)
            .with_cache(CachePolicy::WriteThrough);
        let pte = A64Pte4K48::new_leaf(PhysAddr::from_usize(0x4000), flags, PageSize::Size4K);

        assert!(pte.is_present());
        assert_eq!(pte.entry_kind(1), PageTableEntryKind::Leaf);
        assert_eq!(pte.paddr(), PhysAddr::from_usize(0x4000));
        assert!(pte.flags().access.contains(AccessFlags::WRITE));
        assert_eq!(pte.flags().cache(), CachePolicy::WriteThrough);
    }

    #[test]
    fn block_and_table_discriminate_by_level() {
        let block = A64Pte4K48::new_leaf(
            PhysAddr::from_usize(0x20_0000),
            A64Flags::new(AccessFlags::READ),
            PageSize::Size2M,
        );
        let table = A64Pte4K48::new_table(PhysAddr::from_usize(0x8000), 3);

        assert_eq!(block.entry_kind(2), PageTableEntryKind::Leaf);
        assert_eq!(table.entry_kind(3), PageTableEntryKind::Table);
    }

    #[test]
    fn invalid_entry_is_invalid_kind() {
        assert_eq!(
            A64Pte4K48::default().entry_kind(1),
            PageTableEntryKind::Invalid
        );
    }

    #[test]
    fn canonical_va_accepts_configured_halves() {
        assert!(A64Meta4K48::vaddr_is_valid(VirtAddr::from_usize(
            0x0000_7fff_ffff_f000
        )));
        assert!(A64Meta4K48::vaddr_is_valid(VirtAddr::from_usize(
            0xffff_8000_0000_0000
        )));
        assert!(!A64Meta4K48::vaddr_is_valid(VirtAddr::from_usize(
            0x0001_0000_0000_0000
        )));

        assert!(A64Meta4K52::vaddr_is_valid(VirtAddr::from_usize(
            0x0007_ffff_ffff_f000
        )));
        assert!(A64Meta4K52::vaddr_is_valid(VirtAddr::from_usize(
            0xfff8_0000_0000_0000
        )));
        assert!(!A64Meta4K52::vaddr_is_valid(VirtAddr::from_usize(
            0x0010_0000_0000_0000
        )));
    }

    #[test]
    fn metadata_covers_arm_granule_width_matrix() {
        assert_eq!(A64Meta4K48::LEVELS, 4);
        assert_eq!(A64Meta4K52::LEVELS, 5);
        assert_eq!(A64Meta16K48::BASE_PAGE_SIZE, PageSize::Size16K);
        assert_eq!(A64Meta16K48::level_index_bits(4), 1);
        assert_eq!(A64Meta16K52::level_index_bits(4), 5);
        assert_eq!(A64Meta64K48::BASE_PAGE_SIZE, PageSize::Size64K);
        assert_eq!(A64Meta64K48::level_index_bits(3), 6);
        assert_eq!(A64Meta64K52::level_index_bits(3), 10);
    }

    #[test]
    fn huge_sizes_match_non_4k_granule_levels() {
        assert!(A64Meta16K48::level_supports_leaf(2, PageSize::Size32M));
        assert!(A64Meta16K48::level_supports_leaf(3, PageSize::Size64G));
        assert!(A64Meta64K48::level_supports_leaf(2, PageSize::Size512M));
        assert!(A64Meta64K48::level_supports_leaf(3, PageSize::Size4T));
    }

    #[test]
    fn pte_types_cover_output_widths() {
        let _: A64Pte4K52 = A64Pte4K52::default();
        let _: A64Pte16K48 = A64Pte16K48::default();
        let _: A64Pte16K52 = A64Pte16K52::default();
        let _: A64Pte64K48 = A64Pte64K48::default();
        let _: A64Pte64K52 = A64Pte64K52::default();
    }

    #[test]
    fn mair_indices_are_stable() {
        assert_eq!(cache_to_attr_index(CachePolicy::Writeback), 0);
        assert_eq!(cache_to_attr_index(CachePolicy::Uncached), 1);
        assert_eq!(cache_to_attr_index(CachePolicy::WriteCombine), 2);
        assert_eq!(cache_to_attr_index(CachePolicy::WriteThrough), 3);
    }

    #[test]
    fn mair_value_matches_cache_policy_indices() {
        let attr = |index: u64| (A64_MAIR_EL1_VALUE >> (index * 8)) & 0xff;
        assert_eq!(attr(A64_MAIR_WRITEBACK_INDEX), A64_MAIR_ATTR_WRITEBACK);
        assert_eq!(attr(A64_MAIR_DEVICE_INDEX), A64_MAIR_ATTR_DEVICE_NGNRNE);
        assert_eq!(attr(A64_MAIR_NORMAL_NC_INDEX), A64_MAIR_ATTR_NORMAL_NC);
        assert_eq!(
            attr(A64_MAIR_WRITETHROUGH_INDEX),
            A64_MAIR_ATTR_WRITETHROUGH
        );
    }
}
