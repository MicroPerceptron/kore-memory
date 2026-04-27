use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use kpte::{
    AccessFlags, CachePolicy, FrameAllocator, MappingFlags, NoFlush, PageSize, PageTableEntry,
    PageTableEntryKind, PageTableWalker, PagingError, PagingMetaData, PagingResult,
    TlbInvalidation,
};
use memory_addr::{PhysAddr, PhysAddrRange, VirtAddr, VirtAddrRange};

mod native;

const PAGE_4K: usize = 0x1000;
const SEQUENTIAL_PAGES: usize = 4096;
const FRAGMENTED_PAGES: usize = 2048;
const QUERY_PAGES: usize = 4096;
const QUERY_ITERS: usize = 262_144;
const PROTECT_ITERS: usize = 16_384;
const SAMPLE_COUNT: usize = 9;

fn main() {
    println!("kpte walker host benchmarks");
    println!("feature: bench");
    println!();

    bench_ops(
        "map/range/4k-sequential",
        SAMPLE_COUNT,
        SEQUENTIAL_PAGES,
        bench_range_map_sequential,
    );
    bench_ops(
        "map/loop/4k-sequential",
        SAMPLE_COUNT,
        SEQUENTIAL_PAGES,
        bench_loop_map_sequential,
    );
    bench_ops(
        "map/range/4k-scattered",
        SAMPLE_COUNT,
        FRAGMENTED_PAGES,
        bench_range_map_fragmented,
    );
    bench_prepared(
        "query/hot/4k",
        SAMPLE_COUNT,
        QUERY_ITERS,
        setup_query_hot,
        run_query_hot,
    );
    bench_prepared(
        "protect/loop/4k-sequential",
        SAMPLE_COUNT,
        PROTECT_ITERS,
        setup_protect_sequential,
        run_protect_sequential,
    );
    bench_prepared(
        "split/2m-to-4k",
        SAMPLE_COUNT,
        1,
        setup_split_2m,
        run_split_2m,
    );
    bench_prepared(
        "merge/4k-to-2m",
        SAMPLE_COUNT,
        1,
        setup_merge_2m,
        run_merge_2m,
    );

    native::run();
}

fn bench_ops(name: &str, samples: usize, ops_per_sample: usize, mut f: impl FnMut()) {
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    let mut total_allocs = 0usize;
    let mut total_alloc_bytes = 0usize;

    for _ in 0..samples {
        BenchAlloc::reset();
        reset_tlb_counts();
        reset_alloc_counts();

        let start = Instant::now();
        f();
        let elapsed = start.elapsed();

        best = best.min(elapsed);
        total += elapsed;
        total_allocs += alloc_calls();
        total_alloc_bytes += alloc_bytes();
        BenchAlloc::reset();
    }

    print_sample(
        name,
        samples,
        ops_per_sample,
        best,
        total,
        total_allocs,
        total_alloc_bytes,
    );
}

fn bench_prepared<S>(
    name: &str,
    samples: usize,
    ops_per_sample: usize,
    mut setup: impl FnMut() -> S,
    mut run: impl FnMut(&mut S),
) {
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    let mut total_allocs = 0usize;
    let mut total_alloc_bytes = 0usize;

    for _ in 0..samples {
        BenchAlloc::reset();
        reset_tlb_counts();

        let mut state = setup();
        reset_tlb_counts();
        reset_alloc_counts();

        let start = Instant::now();
        run(&mut state);
        let elapsed = start.elapsed();

        best = best.min(elapsed);
        total += elapsed;
        total_allocs += alloc_calls();
        total_alloc_bytes += alloc_bytes();
        drop(state);
        BenchAlloc::reset();
    }

    print_sample(
        name,
        samples,
        ops_per_sample,
        best,
        total,
        total_allocs,
        total_alloc_bytes,
    );
}

fn print_sample(
    name: &str,
    samples: usize,
    ops_per_sample: usize,
    best: Duration,
    total: Duration,
    total_allocs: usize,
    total_alloc_bytes: usize,
) {
    let avg = total.as_nanos() / samples as u128;
    let best_ns_per_op = best.as_nanos() as f64 / ops_per_sample as f64;
    let avg_ns_per_op = avg as f64 / ops_per_sample as f64;
    let best_ops_per_sec = 1_000_000_000f64 / best_ns_per_op;
    let allocs_per_sample = total_allocs as f64 / samples as f64;
    let alloc_kib_per_sample = total_alloc_bytes as f64 / samples as f64 / 1024.0;

    println!(
        "{name:<32} best {:>10.2} ns/op  avg {:>10.2} ns/op  {:>12.0} ops/s  allocs {:>6.1}  KiB {:>8.1}",
        best_ns_per_op, avg_ns_per_op, best_ops_per_sec, allocs_per_sample, alloc_kib_per_sample
    );
}

fn bench_range_map_sequential() {
    let mut pt = RecordingPt::try_new().unwrap();
    pt.map(
        VirtAddrRange::from_start_size(
            VirtAddr::from_usize(0x1000_0000),
            SEQUENTIAL_PAGES * PAGE_4K,
        ),
        PhysAddr::from_usize(0x4000_0000),
        MappingFlags::scattered(
            flags(AccessFlags::READ | AccessFlags::WRITE),
            PageSize::Size4K,
        ),
        &RECORDING_TLB,
    )
    .unwrap();
    black_box(tlb_range_flushes());
}

fn bench_loop_map_sequential() {
    let mut pt = RecordingPt::try_new().unwrap();
    for i in 0..SEQUENTIAL_PAGES {
        pt.map(
            vrange(0x1000_0000 + i * PAGE_4K, PageSize::Size4K),
            PhysAddr::from_usize(0x4000_0000 + i * PAGE_4K),
            flags(AccessFlags::READ | AccessFlags::WRITE),
            &RECORDING_TLB,
        )
        .unwrap();
    }
    black_box(tlb_range_flushes());
}

fn bench_range_map_fragmented() {
    let mut pt = RecordingPt::try_new().unwrap();
    let ranges: Vec<PhysAddrRange> = (0..FRAGMENTED_PAGES)
        .map(|i| {
            PhysAddrRange::from_start_size(PhysAddr::from_usize(0x4000_0000 + i * 0x2000), PAGE_4K)
        })
        .collect();
    pt.map(
        VirtAddrRange::from_start_size(
            VirtAddr::from_usize(0x1000_0000),
            FRAGMENTED_PAGES * PAGE_4K,
        ),
        ranges.as_slice(),
        MappingFlags::scattered(flags(AccessFlags::READ), PageSize::Size4K),
        &RECORDING_TLB,
    )
    .unwrap();
    black_box(tlb_range_flushes());
}

fn setup_query_hot() -> BenchPt {
    let mut pt = BenchPt::try_new().unwrap();
    pt.map(
        VirtAddrRange::from_start_size(VirtAddr::from_usize(0x2000_0000), QUERY_PAGES * PAGE_4K),
        PhysAddr::from_usize(0x6000_0000),
        MappingFlags::scattered(flags(AccessFlags::READ), PageSize::Size4K),
        &NO_FLUSH,
    )
    .unwrap();
    pt
}

fn run_query_hot(pt: &mut BenchPt) {
    let mut acc = 0usize;
    for i in 0..QUERY_ITERS {
        let idx = (i.wrapping_mul(131)) & (QUERY_PAGES - 1);
        let mapping = pt
            .query(VirtAddr::from_usize(0x2000_0000 + idx * PAGE_4K))
            .unwrap();
        acc ^= mapping.paddr.as_usize();
    }
    black_box(acc);
}

fn setup_protect_sequential() -> RecordingPt {
    let mut pt = RecordingPt::try_new().unwrap();
    pt.map(
        VirtAddrRange::from_start_size(VirtAddr::from_usize(0x3000_0000), PROTECT_ITERS * PAGE_4K),
        PhysAddr::from_usize(0x7000_0000),
        MappingFlags::scattered(flags(AccessFlags::READ), PageSize::Size4K),
        &RECORDING_TLB,
    )
    .unwrap();
    pt
}

fn run_protect_sequential(pt: &mut RecordingPt) {
    reset_tlb_counts();
    for i in 0..PROTECT_ITERS {
        pt.protect(
            vrange(0x3000_0000 + i * PAGE_4K, PageSize::Size4K),
            flags(AccessFlags::READ | AccessFlags::WRITE),
            &RECORDING_TLB,
        )
        .unwrap();
    }
    black_box(tlb_range_flushes());
}

fn setup_split_2m() -> (BenchPt, VirtAddrRange) {
    let mut pt = BenchPt::try_new().unwrap();
    let range = vrange(0x4000_0000, PageSize::Size2M);
    pt.map(
        range,
        PhysAddr::from_usize(0x8000_0000),
        flags(AccessFlags::READ),
        &NO_FLUSH,
    )
    .unwrap();
    (pt, range)
}

fn run_split_2m((pt, range): &mut (BenchPt, VirtAddrRange)) {
    black_box(pt.split_at(*range, &NO_FLUSH).unwrap());
}

fn setup_merge_2m() -> (BenchPt, VirtAddrRange) {
    let mut pt = BenchPt::try_new().unwrap();
    let range = vrange(0x4000_0000, PageSize::Size2M);
    pt.map(
        range,
        PhysAddr::from_usize(0x8000_0000),
        MappingFlags::scattered(flags(AccessFlags::READ), PageSize::Size4K),
        &NO_FLUSH,
    )
    .unwrap();
    (pt, range)
}

fn run_merge_2m((pt, range): &mut (BenchPt, VirtAddrRange)) {
    black_box(pt.merge_at(*range, &NO_FLUSH).unwrap());
}

type BenchPt = PageTableWalker<BenchMeta, BenchPte, BenchAlloc>;
type RecordingPt = PageTableWalker<BenchMeta, BenchPte, BenchAlloc>;

const NO_FLUSH: NoFlush = NoFlush;
const RECORDING_TLB: RecordingTlb = RecordingTlb;

fn vrange(start: usize, size: PageSize) -> VirtAddrRange {
    VirtAddrRange::from_start_size(VirtAddr::from_usize(start), size.bytes())
}

fn flags(access: AccessFlags) -> BenchFlags {
    BenchFlags {
        access,
        cache: CachePolicy::Writeback,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
struct BenchPte(u64);

const PTE_PRESENT: u64 = 1 << 0;
const PTE_LEAF: u64 = 1 << 1;
const PTE_PADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BenchFlags {
    access: AccessFlags,
    cache: CachePolicy,
}

impl PageTableEntry for BenchPte {
    type Flags = BenchFlags;

    fn new_leaf(paddr: PhysAddr, flags: Self::Flags, _size: PageSize) -> Self {
        let mut bits = (paddr.as_usize() as u64) & PTE_PADDR_MASK;
        bits |= PTE_PRESENT | PTE_LEAF;
        bits |= ((flags.access.bits() as u64) & 0x1f) << 2;
        bits |= ((flags.cache as u64) & 0x3) << 7;
        Self(bits)
    }

    fn new_table(paddr: PhysAddr, _level: u8) -> Self {
        Self(((paddr.as_usize() as u64) & PTE_PADDR_MASK) | PTE_PRESENT)
    }

    fn paddr(&self) -> PhysAddr {
        PhysAddr::from_usize((self.0 & PTE_PADDR_MASK) as usize)
    }

    fn flags(&self) -> Self::Flags {
        let access_bits = ((self.0 >> 2) & 0x1f) as u8;
        let cache_bits = ((self.0 >> 7) & 0x3) as u8;
        let mut access = AccessFlags::empty();
        if access_bits & AccessFlags::READ.bits() != 0 {
            access |= AccessFlags::READ;
        }
        if access_bits & AccessFlags::WRITE.bits() != 0 {
            access |= AccessFlags::WRITE;
        }
        if access_bits & AccessFlags::EXECUTE.bits() != 0 {
            access |= AccessFlags::EXECUTE;
        }
        if access_bits & AccessFlags::USER.bits() != 0 {
            access |= AccessFlags::USER;
        }
        if access_bits & AccessFlags::GLOBAL.bits() != 0 {
            access |= AccessFlags::GLOBAL;
        }
        let cache = match cache_bits {
            0 => CachePolicy::Writeback,
            1 => CachePolicy::Uncached,
            2 => CachePolicy::WriteCombine,
            _ => CachePolicy::WriteThrough,
        };
        BenchFlags { access, cache }
    }

    fn is_present(&self) -> bool {
        self.0 & PTE_PRESENT != 0
    }

    fn entry_kind(&self, level: u8) -> PageTableEntryKind {
        if level == 1 || self.0 & PTE_LEAF != 0 {
            PageTableEntryKind::Leaf
        } else {
            PageTableEntryKind::Table
        }
    }

    fn clear(&mut self) {
        self.0 = 0;
    }

    fn bits(&self) -> u64 {
        self.0
    }

    fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

struct BenchMeta;

impl PagingMetaData for BenchMeta {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 52;
    const VA_MAX_BITS: usize = 48;

    type VirtAddr = VirtAddr;

    fn level_shift(level: u8) -> u32 {
        12 + ((level as u32) - 1) * 9
    }

    fn level_supports_leaf(level: u8, size: PageSize) -> bool {
        matches!(
            (level, size),
            (1, PageSize::Size4K) | (2, PageSize::Size2M) | (3, PageSize::Size1G)
        )
    }
}

struct FrameBlock {
    ptr: *mut u8,
    layout: Layout,
}

type FrameStore = HashMap<usize, FrameBlock>;

struct FrameStoreCell(Mutex<Option<FrameStore>>);

unsafe impl Sync for FrameStoreCell {}

static FRAMES: FrameStoreCell = FrameStoreCell(Mutex::new(None));
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct BenchAlloc;

impl BenchAlloc {
    fn reset() {
        let mut guard = frames_lock();
        if let Some(store) = guard.as_mut() {
            for (_, block) in store.drain() {
                unsafe { dealloc(block.ptr, block.layout) };
            }
        }
    }
}

impl FrameAllocator for BenchAlloc {
    fn allocate(size: usize, align: PageSize) -> PagingResult<PhysAddrRange> {
        if size == 0 || (size & (PAGE_4K - 1)) != 0 {
            return Err(PagingError::NotAligned);
        }
        let layout =
            Layout::from_size_align(size, align.bytes()).map_err(|_| PagingError::NotAligned)?;
        let raw = unsafe { alloc_zeroed(layout) };
        if raw.is_null() {
            return Err(PagingError::OutOfMemory);
        }
        let key = raw as usize;
        frames_lock()
            .as_mut()
            .unwrap()
            .insert(key, FrameBlock { ptr: raw, layout });
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
        Ok(PhysAddrRange::from_start_size(
            PhysAddr::from_usize(key),
            size,
        ))
    }

    fn deallocate(range: PhysAddrRange) -> PagingResult {
        let Some(block) = frames_lock()
            .as_mut()
            .unwrap()
            .remove(&range.start.as_usize())
        else {
            return Err(PagingError::AddressOutOfRange);
        };
        if block.layout.size() != range.size() {
            frames_lock()
                .as_mut()
                .unwrap()
                .insert(range.start.as_usize(), block);
            return Err(PagingError::NotAligned);
        }
        unsafe { dealloc(block.ptr, block.layout) };
        Ok(())
    }

    fn phys_to_virt(paddr: PhysAddr) -> VirtAddr {
        VirtAddr::from_usize(paddr.as_usize())
    }
}

fn frames_lock() -> std::sync::MutexGuard<'static, Option<FrameStore>> {
    let mut guard = FRAMES.0.lock().unwrap_or_else(|err| err.into_inner());
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    guard
}

fn reset_alloc_counts() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn alloc_calls() -> usize {
    ALLOC_CALLS.load(Ordering::Relaxed)
}

fn alloc_bytes() -> usize {
    ALLOC_BYTES.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Default)]
struct RecordingTlb;

static RANGE_FLUSHES: AtomicUsize = AtomicUsize::new(0);
static LOCAL_FLUSHES: AtomicUsize = AtomicUsize::new(0);
static FULL_FLUSHES: AtomicUsize = AtomicUsize::new(0);
static RANGE_FLUSHED_PAGES: AtomicUsize = AtomicUsize::new(0);

impl TlbInvalidation<VirtAddr> for RecordingTlb {
    fn flush_tlb_local(&self, _vaddr: VirtAddr) {
        LOCAL_FLUSHES.fetch_add(1, Ordering::Relaxed);
    }

    fn flush_tlb_all_local(&self) {
        FULL_FLUSHES.fetch_add(1, Ordering::Relaxed);
    }

    fn flush_tlb_range_local(&self, _start: VirtAddr, _page_size: PageSize, count_pages: usize) {
        RANGE_FLUSHES.fetch_add(1, Ordering::Relaxed);
        RANGE_FLUSHED_PAGES.fetch_add(count_pages, Ordering::Relaxed);
    }

    fn prefer_full_flush(&self, _pending_count: usize) -> bool {
        false
    }
}

fn reset_tlb_counts() {
    RANGE_FLUSHES.store(0, Ordering::Relaxed);
    LOCAL_FLUSHES.store(0, Ordering::Relaxed);
    FULL_FLUSHES.store(0, Ordering::Relaxed);
    RANGE_FLUSHED_PAGES.store(0, Ordering::Relaxed);
}

fn tlb_range_flushes() -> usize {
    RANGE_FLUSHES.load(Ordering::Relaxed) ^ RANGE_FLUSHED_PAGES.load(Ordering::Relaxed)
}
