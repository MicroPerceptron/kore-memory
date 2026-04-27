use core::arch::asm;

use memory_addr::VirtAddr;

use crate::TlbInvalidation;

/// Local RISC-V `sfence.vma` invalidation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64Tlb;

impl TlbInvalidation<VirtAddr> for Rv64Tlb {
    #[inline]
    fn flush_tlb_local(&self, vaddr: VirtAddr) {
        unsafe {
            asm!("sfence.vma {}, zero", in(reg) vaddr.as_usize(), options(nostack));
        }
    }

    #[inline]
    fn flush_tlb_all_local(&self) {
        unsafe {
            asm!("sfence.vma zero, zero", options(nostack));
        }
    }
}
