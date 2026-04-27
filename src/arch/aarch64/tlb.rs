use aarch64_cpu::asm::barrier::{ISH, SY, dsb, isb};
use aarch64_cpu_ext::asm::tlb::{VAAE1IS, VMALLE1IS, tlbi};
use memory_addr::VirtAddr;

use crate::TlbInvalidation;

/// EL1 stage-1 AArch64 TLB invalidation.
///
/// This conservative implementation invalidates by VA for all ASIDs in the
/// inner-shareable domain. ASID-scoped variants can be added once the kernel
/// address-space layer owns an ASID lease type.
#[derive(Clone, Copy, Debug, Default)]
pub struct A64Tlb;

impl TlbInvalidation<VirtAddr> for A64Tlb {
    #[inline]
    fn flush_tlb_local(&self, vaddr: VirtAddr) {
        dsb(ISH);
        tlbi(VAAE1IS::new(vaddr.as_usize()));
        dsb(ISH);
        isb(SY);
    }

    #[inline]
    fn flush_tlb_all_local(&self) {
        dsb(ISH);
        tlbi(VMALLE1IS);
        dsb(ISH);
        isb(SY);
    }
}
