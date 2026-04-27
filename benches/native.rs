use super::*;
use kpte::arch::{ARCH_NAME, Flags4K48, PageTable4K48, Pte4K48};

type BenchPt4K48 = PageTable4K48<BenchAlloc>;

pub(super) fn run() {
    println!();
    bench_ops(
        &format!("{ARCH_NAME}/4k48/map/range/4k-sequential"),
        SAMPLE_COUNT,
        SEQUENTIAL_PAGES,
        bench_range_map_sequential,
    );
    bench_prepared(
        &format!("{ARCH_NAME}/4k48/query/hot/4k"),
        SAMPLE_COUNT,
        QUERY_ITERS,
        setup_query_hot,
        run_query_hot,
    );
    bench_ops(
        &format!("{ARCH_NAME}/4k48/pte/encode-decode"),
        SAMPLE_COUNT,
        QUERY_ITERS,
        bench_pte_encode_decode,
    );
}

fn bench_range_map_sequential() {
    let mut pt = BenchPt4K48::try_new().unwrap();
    pt.map(
        VirtAddrRange::from_start_size(
            VirtAddr::from_usize(0x1000_0000),
            SEQUENTIAL_PAGES * PAGE_4K,
        ),
        PhysAddr::from_usize(0x4000_0000),
        MappingFlags::scattered(
            Flags4K48::new(AccessFlags::READ | AccessFlags::WRITE),
            PageSize::Size4K,
        ),
        &NoFlush,
    )
    .unwrap();
    black_box(pt.root());
}

fn setup_query_hot() -> BenchPt4K48 {
    let mut pt = BenchPt4K48::try_new().unwrap();
    pt.map(
        VirtAddrRange::from_start_size(VirtAddr::from_usize(0x2000_0000), QUERY_PAGES * PAGE_4K),
        PhysAddr::from_usize(0x6000_0000),
        MappingFlags::scattered(Flags4K48::new(AccessFlags::READ), PageSize::Size4K),
        &NoFlush,
    )
    .unwrap();
    pt
}

fn run_query_hot(pt: &mut BenchPt4K48) {
    let mut acc = 0usize;
    for i in 0..QUERY_ITERS {
        let idx = (i.wrapping_mul(131)) & (QUERY_PAGES - 1);
        let mapping = pt
            .query(VirtAddr::from_usize(0x2000_0000 + idx * PAGE_4K))
            .unwrap();
        acc ^= mapping.paddr.as_usize();
        acc ^= mapping.flags.access.bits() as usize;
    }
    black_box(acc);
}

fn bench_pte_encode_decode() {
    let flags = black_box(Flags4K48::new(
        AccessFlags::READ | AccessFlags::WRITE | AccessFlags::GLOBAL,
    ));
    let mut acc = 0u64;
    for i in 0..QUERY_ITERS {
        let offset = black_box(i & 0xffff);
        let paddr = black_box(PhysAddr::from_usize(0x4000_0000 + offset * PAGE_4K));
        let pte = black_box(Pte4K48::new_leaf(
            paddr,
            black_box(flags),
            black_box(PageSize::Size4K),
        ));
        let decoded = black_box(pte.flags());
        acc = black_box(acc ^ pte.bits());
        acc = black_box(acc ^ decoded.access.bits() as u64);
        acc = black_box(acc ^ pte.paddr().as_usize() as u64);
    }
    black_box(acc);
}
