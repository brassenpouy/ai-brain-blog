---
title: "CuTe DSL 4.8.1 API 全量索引"
date: "2026-11-24T21:28:21+08:00"
lastmod: "2026-11-24T21:31:56+08:00"
description: "CUTLASS 4.8.1 中 cutlass 和 cutlass.cute 的模块级公开导出，按模块分组。"
tags:
  - "CUTLASS"
  - "CuTe DSL"
  - "API"
draft: false
slug: "cute-dsl-4-8-1-api-index"
---

## 范围

本页列 CUTLASS 4.8.1 的模块级公开导出。数据取自 `python/CuTeDSL` 源码和 4.8.1 API 文档，保留源码中的大小写与拼写。

覆盖 `cutlass` 基础类型、运行时和测试模块、`cutlass.cute` 及其子模块、`cutlass.pipeline`、`cutlass.utils`、`cutlass.tensor_utils`，以及有明确导出的 experimental 模块。`cutlass.torch` 和 `cutlass.jax` 作为可选适配层单列。

共 59 个模块或命名空间，1329 个导出项。索引按模块和完全限定名组织，不做跨模块语义去重；同名 API 可能指向同一对象，也可能是不同实现。每节内部只合并重复字符串：例如 `cutlass.cute.arch` 的 `__all__` 有 167 项，合并后 162 项。

不包含 `base_dsl`、`_mlir` 内部模块、私有 `_` 符号、类方法和参数签名。`cutlass` 一节只列基础类型；`cutlass` 顶层的控制流、编译配置和 `cutlass_dsl` 入口不在本页展开。Task Scheduling 子模块按 4.8.1 文档拆开列出。`cutlass.cute.experimental` 中未统一导出的内部子模块也不展开。完整签名查 [CuTe DSL 4.8.1 API Reference](https://docs.nvidia.com/cutlass/4.8.1/media/docs/pythonDSL/cute_dsl_api.html)。源码版本：[v4.8.1](https://github.com/NVIDIA/cutlass/tree/v4.8.1)。

## 符号清单

### cutlass 基础类型（31）

```text
BFloat16, Boolean, Float, Float16, Float32, Float4E2M1FN, Float4E2M1FNx2
Float64, Float6E2M3FN, Float6E3M2FN, Float8E4M3, Float8E4M3B11FNUZ
Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Int128, Int16, Int32, Int4
Int64, Int8, Integer, Numeric, Pointer, TFloat32, Uint128, Uint16, Uint32
Uint64, Uint8, Vector
```

### cutlass.block（1）

```text
block_copy
```

### cutlass.runtime（5）

```text
find_runtime_libraries, load_module, make_fake_array
make_fake_compact_array, make_fake_stream
```

### cutlass.testing（12）

```text
add_tensor_init_args, benchmark, CantImplementError, CuptiProfiler
get_workspace_count, JitArguments, sample_pytest, should_use_normal_init
tensor_init_config_from_args, TensorInitConfig, tune
validate_tensor_init_args
```

### cutlass.torch（13）

```text
as_tensor, convert_cute_tensor, create_and_permute_torch_tensor
current_stream, cute_tensor_like, default_stream, GaussianInitConfig
get_leading_dim, matrix, prepare_tensors_for_gemm, RandomInitConfig
ScalarInitConfig, TensorInitType
```

### cutlass.jax（21）

```text
CUTE_DSL_JAX_MIN_SUPPORTED_PYTHON_VERSION
CUTE_DSL_MIN_SUPPORTED_JAX_VERSION, cutlass_call, cutlass_to_jax_dtype
cutlass_to_jax_layout_order, disable_automatic_ffi_registration
from_dlpack, get_cutlass_call_ffi_name, get_cutlass_call_ffi_version
get_export_disabled_safety_checks, is_available, is_ffi_registered
jax_to_cutlass_dtype, jax_to_cutlass_layout_order, JaxArray, register_ffi
release_compile_cache, set_ffi_call_targets, TensorMode, TensorSpec
testing
```

### cutlass.cute（180）

```text
absf, AddressSpace, all_, any_, append, append_ones, arch, ArtifactType
assume, Atom, autovec_copy, basic_copy, basic_copy_if, basis_get
basis_value, BitCode, blocked_product, CacheEvictionPriority, ceil_div
coalesce, compile, compile_to, complement, ComposedLayout, composition
ConstValue, copy, copy_atom_call, CopyAtom, cosize, crd2idx, depth, dice
DisableCuteExtCompile, domain_offset, E, elem_less, empty_like, export
extern, fast_divmod_create_divisor, fast_divmod_create_divisor_v2
FastDivmodDivisor, FastDivmodDivisorV2, ffi, filter, filter_tuple
filter_zeros, find, find_if, flat_divide, flat_product, flatten
flatten_to_tuple, front, full, full_like, gemm, get, get_divisibility
get_leaves, get_nonswizzle_portion, get_swizzle_portion, group_modes
has_scaled_basis, has_underscore, idx2crd, increment_coord, IntValue
is_congruent, is_int_tuple, is_int_tuple_type, is_integer, is_major
is_static, is_valid_leaf, is_weakly_congruent, jit, kernel, Layout
leading_dim, left_inverse, local_partition, local_tile, logical_divide
logical_product, make_atom, make_composed_layout, make_copy_atom
make_cotiled_copy, make_fragment_like, make_identity_layout
make_identity_tensor, make_layout, make_layout_image_mask
make_layout_like, make_layout_tv, make_mma_atom, make_ordered_layout
make_ptr, make_rmem_tensor, make_rmem_tensor_like, make_swizzle
make_tensor, make_tiled_copy, make_tiled_copy_A, make_tiled_copy_B
make_tiled_copy_C, make_tiled_copy_C_atom, make_tiled_copy_D
make_tiled_copy_S, make_tiled_copy_tv, make_tiled_mma, mangle
max_common_layout, max_common_vector, mma_atom_call, MmaAtom, nullspace
Numeric, nvgpu, ones_like, PreCompiledMlirArtifact, prefetch, prepend
prepend_ones, pretty_str, print_tensor, printf, product, product_each
product_like, raked_product, rank, recast_layout, recast_ptr
recast_tensor, ReductionKind, ReductionOp, register_jit_arg_adapter
repeat, repeat_as_tuple, repeat_like, right_inverse, round_up, runtime
ScaledBasis, select, shape, shape_div, size, size_in_bytes, slice_
slice_and_offset, static, struct, Swizzle, SymInt, Tensor, TensorSSA
testing, ThrCopy, ThrMma, tile_to_shape, tiled_divide, tiled_product
TiledCopy, TiledMma, transform_apply, transform_leaf, tuple_cat
unflatten, union, unwrap, viz, where, wrap, zeros_like, zipped_divide
zipped_product
```

### cutlass.cute.core（81）

```text
append, append_ones, assume, blocked_product, ceil_div, coalesce
complement, composition, cosize, crd2idx, depth, dice, E, filter
filter_zeros, flat_divide, flat_product, flatten, front, get
get_divisibility, get_leaves, get_nonswizzle_portion, get_swizzle_portion
group_modes, has_scaled_basis, has_underscore, idx2crd, increment_coord
IntValue, is_congruent, is_major, is_static, is_valid_leaf
is_weakly_congruent, leading_dim, left_inverse, local_partition
local_tile, logical_divide, logical_product, make_composed_layout
make_identity_layout, make_layout, make_layout_image_mask
make_layout_like, make_layout_tv, make_ordered_layout, make_ptr
make_swizzle, max_common_layout, max_common_vector, nullspace, prepend
prepend_ones, pretty_str, printf, raked_product, rank, recast_layout
recast_ptr, repeat, repeat_as_tuple, repeat_like, right_inverse, round_up
select, shape, shape_div, size, size_in_bytes, slice_, slice_and_offset
static, struct, Swizzle, tile_to_shape, tiled_divide, tiled_product
zipped_divide, zipped_product
```

### cutlass.cute.tensor（18）

```text
all_, any_, domain_offset, empty_like, full, full_like
make_fragment_like, make_identity_tensor, make_rmem_tensor
make_rmem_tensor_like, make_tensor, ones_like, print_tensor
recast_tensor, ReductionOp, TensorSSA, where, zeros_like
```

### cutlass.cute.tuple（14）

```text
elem_less, filter_tuple, find, find_if, flatten_to_tuple, product
product_each, product_like, transform_apply, transform_leaf, tuple_cat
unflatten, unwrap, wrap
```

### cutlass.cute.typing（54）

```text
Array, as_numeric, BFloat16, Boolean, ComposedLayout, Coord, Float
Float16, Float32, Float4E2M1FN, Float4E2M1FNx2, Float64, Float6E2M3FN
Float6E2M3FNx4, Float6E3M2FN, Float6E3M2FNx4, Float8E4M3
Float8E4M3B11FNUZ, Float8E4M3FN, Float8E5M2, Float8E8M0FNU
FloatNV8E5M3FNU, Int128, Int16, Int32, Int4, Int64, Int8, Integer
IntTuple, is_int_tuple, is_int_tuple_type, is_integer, Layout, Numeric
NumericMeta, Pointer, ScaledBasis, Shape, Stride, sym_int, sym_int32
sym_int64, SymInt, Tensor, TFloat32, Tile, Tiler, TypedTensor, Uint16
Uint32, Uint64, Uint8, XTuple
```

### cutlass.cute.math（55）

```text
abs, absf, absi, acos, acosh, add, asin, asinh, atan, atan2, atanh, cbrt
ceil, clamp, copysign, cos, cosh, div, erf, erfc, exp, exp2, expm1, floor
fma, fpowi, ipowi, isfinite, isinf, isnan, isnormal, log, log10, log1p
log2, MathOperand, max, min, mul, neg, pow, rcp, rem, round, roundeven
RoundingMode, rsqrt, sin, sincos, sinh, sqrt, sub, tan, tanh, trunc
```

### cutlass.cute.viz（7）

```text
Band, PALETTES, print_latex, print_latex_tv, tikz_color_rainbow
tikz_color_tv, tikz_escape
```

### cutlass.cute.arch（162）

```text
activemask, add_cc, add_packed_bf16x2, add_packed_bf16x2_f32x2_f32x2
add_packed_f16x2, add_packed_f16x2_f32x2_f32x2, add_packed_f32x2
add_packed_f32x2_bf16x2_f32x2, add_packed_f32x2_f16x2_f32x2, add_sat_int
addc, aggr_smem_size, alloc_smem, alloc_tmem, atomic_add, atomic_and
atomic_cas, atomic_exch, atomic_fmax, atomic_fmin, atomic_max
atomic_max_float32, atomic_min, atomic_or, atomic_xor, barrier
barrier_arrive, bfe, bfi, bfind, block_dim, block_idx
block_idx_in_cluster, block_in_cluster_dim, block_in_cluster_idx, brev
clc_response, clock, clock64, cluster_arrive, cluster_arrive_relaxed
cluster_dim, cluster_idx, cluster_size, cluster_wait, clz
cp_async_bulk_commit_group, cp_async_bulk_wait_group
cp_async_commit_group, cp_async_wait_group, cvt_f32_tf32
cvt_f32x2_bf16x2, cvt_f4e2m1_f16_intrinsic, cvt_i4_bf16_intrinsic
cvt_i8_bf16, cvt_i8_bf16_intrinsic, cvt_i8x2_to_bf16x2, cvt_i8x2_to_f32x2
cvt_i8x4_to_bf16x4, cvt_i8x4_to_f32x4, dealloc_tmem, dynamic_smem_size
elect_one, exp2, exp2_packed_bf16x2, exp2_packed_f16x2
fence_acq_rel_cluster, fence_acq_rel_cta, fence_acq_rel_gpu
fence_acq_rel_sys, fence_proxy, fence_view_async_tmem_load
fence_view_async_tmem_store, fma_bf16, fma_f16, fma_packed_bf16x2
fma_packed_f16x2, fma_packed_f32x2, fma_packed_f32x2_bf16x2_f32x2_f32x2
fma_packed_f32x2_f16x2_f32x2_f32x2, fmax, fmin, get_dyn_smem
get_dyn_smem_size, get_max_tmem_alloc_cols, get_min_tmem_alloc_cols
globaltimer, globaltimer_lo, grid_dim, gridid, issue_clc_query, lane_idx
lanemask_eq, lanemask_ge, lanemask_gt, lanemask_le, lanemask_lt, load
lop3, mad24, mad_cc, madc, make_warp_uniform, match_sync, mbarrier_arrive
mbarrier_arrive_and_expect_tx, mbarrier_conditional_try_wait
mbarrier_expect_tx, mbarrier_init, mbarrier_init_fence
mbarrier_test_wait, mbarrier_try_wait, mbarrier_wait, mul24, mul_hi
mul_packed_f32x2, mul_wide, nsmid, nwarpid, physical_warp_id, popc, prmt
rcp_approx, red, relinquish_tmem_alloc_permit, retrieve_tmem_ptr
setmaxregister_decrease, setmaxregister_increase, shf, shuffle_sync
shuffle_sync_bfly, shuffle_sync_down, shuffle_sync_up, smid, store
store_async_dsmem, sub_cc, sub_packed_bf16x2
sub_packed_bf16x2_f32x2_f32x2, sub_packed_f16x2
sub_packed_f16x2_f32x2_f32x2, sub_packed_f32x2
sub_packed_f32x2_bf16x2_f32x2, sub_packed_f32x2_f16x2_f32x2, sub_sat_int
subc, sync_threads, sync_warp, thread_idx, THREADS_PER_WARPGROUP
total_smem_size, vote_all_sync, vote_any_sync, vote_ballot_sync
vote_uni_sync, warp_idx, warp_redux_sync, WARP_SIZE, warpgroup_reg_alloc
warpgroup_reg_dealloc, WARPS_PER_WARPGROUP, warpsize
```

### cutlass.cute.nvgpu（30）

```text
CacheEvictionPriority, CopyG2ROp, CopyG2RTrait, CopyR2GOp, CopyR2GTrait
CopyR2SOp, CopyR2STrait, CopyS2ROp, CopyS2RTrait, CopyUniversalOp
CopyUniversalTrait, cpasync, L2PrefetchSize, LoadCacheMode
make_im2col_tma_atom_A, make_tiled_tma_atom_A, make_tiled_tma_atom_B
make_tiled_tma_atom_C, MemoryOrder, MemoryScope, MmaUniversalOp
MmaUniversalTrait, normalize_field_to_ir_name, OperandMajorMode
OutputMajorMode, SharedSpace, StoreCacheMode, tcgen05, warp, warpgroup
```

### cutlass.cute.nvgpu.common（22）

```text
CacheEvictionPriority, CopyG2ROp, CopyG2RTrait, CopyR2GOp, CopyR2GTrait
CopyR2SOp, CopyR2STrait, CopyS2ROp, CopyS2RTrait, CopyUniversalOp
CopyUniversalTrait, L2PrefetchSize, LoadCacheMode, MemoryOrder
MemoryScope, MmaUniversalOp, MmaUniversalTrait
normalize_field_to_ir_name, OperandMajorMode, OutputMajorMode
SharedSpace, StoreCacheMode
```

### cutlass.cute.nvgpu.helpers（4）

```text
make_im2col_tma_atom_A, make_tiled_tma_atom_A, make_tiled_tma_atom_B
make_tiled_tma_atom_C
```

### cutlass.cute.nvgpu.cpasync（24）

```text
copy_tensormap, CopyBulkG2SMulticastOp, CopyBulkG2SOp
CopyBulkS2GByteMaskOp, CopyBulkS2GOp, CopyBulkS2SOp
CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp
CopyBulkTensorTileS2GOp, CopyDsmemStoreOp, CopyG2SOp
CopyReduceBulkTensorTileS2GOp, cp_fence_tma_desc_release
create_tma_multicast_mask, fence_tma_desc_acquire, fence_tma_desc_release
group_bulk_copy_modes, LoadCacheMode, make_tiled_tma_atom
prefetch_descriptor, tma_partition, TmaCopyOp, TmaInfo
update_tma_descriptor
```

### cutlass.cute.nvgpu.warp（14）

```text
Field, LdMatrix16x16x8bOp, LdMatrix16x8x8bOp, LdMatrix8x8x16bOp
MmaF16BF16Op, MmaFP8Op, MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op
MmaMXF8Op, MmaTF32Op, MXF8F6F4_SUPPORTED_PAIRS, StMatrix16x8x8bOp
StMatrix8x8x16bOp
```

### cutlass.cute.nvgpu.warpgroup（10）

```text
commit_group, fence, Field, make_smem_layout_atom, MmaF16BF16Op, MmaF8Op
OperandMajorMode, OperandSource, SmemLayoutAtomKind, wait_group
```

### cutlass.cute.nvgpu.tcgen05（47）

```text
append_s2t_broadcast_mode, CollectorBBuffer, CollectorOp, commit
CtaGroup, Field, find_tmem_tensor_col_offset, get_s2t_smem_desc_tensor
get_tmem_copy_properties, is_tmem_load, is_tmem_store, Ld16x128bOp
Ld16x256bOp, Ld16x32bx2Op, Ld16x64bOp, Ld32x32bOp, make_s2t_copy
make_smem_layout_atom, make_tmem_copy, make_umma_smem_desc, MmaF16BF16Op
MmaF16BF16SparseOp, MmaF8F6F4Op, MmaFP8Op, MmaFP8SparseOp, MmaI8Op
MmaMXF4NVF4Op, MmaMXF4Op, MmaMXF8F6F4Op, MmaMXF8Op, MmaTF32Op
OperandMajorMode, OperandSource, Pack, Repetition
SM107MmaF16BF16SparseOp, SM107MmaFP8Op, SM107MmaFP8SparseOp
SmemLayoutAtomKind, St16x128bOp, St16x256bOp, St16x32bx2Op, St16x64bOp
St32x32bOp, tile_to_mma_shape, TmemLoadRedOp, Unpack
```

### cutlass.cute.runtime（11）

```text
find_runtime_libraries, from_dlpack, load_module, make_fake_compact_tensor
make_fake_stream, make_fake_tensor, make_ptr, nullptr, Pointer
Tensor, TensorAdapter
```

### cutlass.cute.testing（18）

```text
add_tensor_init_args, assert_, Assertion, AssertionError, autotune_jit
benchmark, CantImplementError, convert, CuptiProfiler
get_workspace_count, JitArguments, RuntimeAssertion, sample_pytest
should_use_normal_init, tensor_init_config_from_args, TensorInitConfig
tune, validate_tensor_init_args
```

### cutlass.cute.export（1）

```text
CuteCHeaderGenerator
```

### cutlass.cute.ffi（5）

```text
BitCode, ConstValue, extern, ffi, mangle
```

### cutlass.pipeline（33）

```text
Agent, agent_sync, arrive, arrive_and_wait, arrive_unaligned
compute_k_tile, CooperativeGroup, make_pipeline_state, MbarrierArray
MbarrierLayout, NamedBarrier, pipeline_init_arrive, pipeline_init_wait
PipelineAsync, PipelineAsyncUmma, PipelineClcFetchAsync, PipelineConsumer
PipelineCpAsync, PipelineOp, PipelineOrder, PipelineProducer
PipelineState, PipelineTmaAsync, PipelineTmaMultiConsumersAsync
PipelineTmaStore, PipelineTmaUmma, PipelineUmmaAsync, PipelineUserType
sync, SyncObject, TmaStoreFence, wait, wait_unaligned
```

### cutlass.memory（8）

```text
compute_tmem_cols_from_layout, get_kernel_smem_size
get_num_tmem_alloc_cols, get_smem_capacity_in_bytes, SmemAllocator
SmemPartition, TmemAllocator, TmemBufferPool
```

### cutlass.tensor_utils（3）

```text
LayoutEnum, TensorMapManager, TensorMapUpdateMode
```

### cutlass.tensor_utils.layout（1）

```text
LayoutEnum
```

### cutlass.tensor_utils.tensormap（2）

```text
TensorMapManager, TensorMapUpdateMode
```

### cutlass.utils（64）

```text
Band, block_copy, ClcDynamicPersistentTileScheduler
ClcDynamicPersistentTileSchedulerParams, cluster_shape_to_tma_atom_A
cluster_shape_to_tma_atom_B, cluster_shape_to_tma_atom_SFB
compute_epilogue_tile_shape, compute_smem_layout
compute_tmem_cols_from_layout, create_cute_tensor_for_fp8
create_initial_search_state, create_tensor_a, distributed, epilogue
epilogue_tma_store, gemm, get_copy_atom_a_transform, get_divisibility
get_gmem_layout_scale, get_kernel_smem_size, get_num_tmem_alloc_cols
get_permutation_mnk, get_smem_capacity_in_bytes, get_smem_layout_atom_ab
get_smem_layout_atom_epi, get_smem_layout_scale, get_smem_store_op
get_tma_atom_kind, get_tmem_load_op, get_transform_a_source
GroupedGemmGroupSearchState, GroupedGemmTileSchedulerHelper
GroupSearchResult, HardwareInfo, is_fp8_dtype, is_valid_scale_granularity
LayoutEnum, make_blockscaled_trivial_tiled_mma, make_smem_layout
make_smem_layout_a, make_smem_layout_b, make_smem_layout_epi
make_trivial_tiled_mma, PALETTES, PersistentTileSchedulerParams
print_latex, print_latex_tv, scale_partition, scale_tma_partition, sm100
sm90, sm90_get_smem_layout_atom, sm90_make_trivial_tiled_mma
SmemAllocator, StaticPersistentRuntimeTileScheduler
StaticPersistentTileScheduler, TensorMapManager, TensorMapUpdateMode
TmemAllocator, TmemBufferPool, transform_partition, TransformMode
WorkTileInfo
```

### cutlass.utils.blackwell_helpers（28）

```text
cluster_shape_to_tma_atom_A, cluster_shape_to_tma_atom_B
cluster_shape_to_tma_atom_SFB, compute_acc_tmem_cols_per_stage
compute_epilogue_tile_shape, compute_epilogue_tile_size, get_layoutSFA_TV
get_layoutSFB_TV, get_num_tmem_alloc_cols, get_permutation_mnk
get_smem_store_op, get_tmem_load_op, make_blockscaled_trivial_tiled_mma
make_smem_layout_a, make_smem_layout_b, make_smem_layout_epi
make_sparse_a_dtype, make_sparse_e_dtype, make_sparse_gmem_layout_a
make_sparse_gmem_layout_e, make_sparse_smem_layout_a
make_sparse_smem_layout_e, make_sparse_tmem_layout_e
make_trivial_tiled_mma, partition_fragment_SFA, partition_fragment_SFB
thrfrg_SFA, thrfrg_SFB
```

### cutlass.utils.block（1）

```text
block_copy
```

### cutlass.utils.distributed（25）

```text
multimem_ld_reduce, multimem_ld_reduce_16xe4m3
multimem_ld_reduce_16xe5m2, multimem_ld_reduce_1xf32
multimem_ld_reduce_2xbf16, multimem_ld_reduce_2xf16
multimem_ld_reduce_2xf32, multimem_ld_reduce_4xbf16
multimem_ld_reduce_4xe4m3, multimem_ld_reduce_4xe5m2
multimem_ld_reduce_4xf16, multimem_ld_reduce_4xf32
multimem_ld_reduce_8xbf16, multimem_ld_reduce_8xe4m3
multimem_ld_reduce_8xe5m2, multimem_ld_reduce_8xf16, multimem_red_add1
multimem_st, multimem_st_1xb32, multimem_st_2xb32, multimem_st_4xb32
red_add1, spin_lock_atom_cas_acquire_wait
spin_lock_atom_cas_relaxed_wait, spin_lock_ld_lt_relaxed_wait
```

### cutlass.utils.gemm（1）

```text
sm100
```

### cutlass.utils.gemm.sm100（2）

```text
epilogue, epilogue_tma_store
```

### cutlass.utils.gemm.tensor_utils（5）

```text
create_gemm_tensor_torch, create_scale_factor_tensor, decode_float4e2m1fn
get_gemm_tensor, get_gemm_tensors
```

### cutlass.utils.hopper_helpers（5）

```text
compute_tile_shape_or_override, get_smem_store_op, make_smem_layout_a
make_smem_layout_b, make_smem_layout_epi
```

### cutlass.utils.layout（1）

```text
LayoutEnum
```

### cutlass.utils.print_latex（9）

```text
Band, PALETTES, print_latex, print_latex_tv, tikz_color_bwx8
tikz_color_rainbow, tikz_color_tv, tikz_color_white, tikz_escape
```

### cutlass.utils.rubin_helpers（10）

```text
canonicalize_rubin_resource_arch, make_blockscaled_trivial_tiled_mma
make_sparse_a_dtype, make_sparse_e_dtype, make_sparse_gmem_layout_e
make_sparse_smem_layout_e, make_sparse_tmem_layout_e
make_sparse_trivial_tiled_mma, make_tensor_e_atom_layout
make_trivial_tiled_mma
```

### cutlass.utils.smem_allocator（5）

```text
get_kernel_smem_size, get_smem_capacity_in_bytes, SMEM_CAPACITY_MAP
SmemAllocator, SmemPartition
```

### cutlass.utils.tensormap_manager（2）

```text
TensorMapManager, TensorMapUpdateMode
```

### cutlass.utils.tmem_allocator（4）

```text
compute_tmem_cols_from_layout, get_num_tmem_alloc_cols, TmemAllocator
TmemBufferPool
```

### cutlass.experimental（3）

```text
cuda, primitives, task_scheduling
```

### cutlass.experimental.cuda（8）

```text
create_tensor_map_tiled, create_tensor_map_tiled_from_view, TensorMap
TensorMapDataFormat, TensorMapFloatOOBFill, TensorMapInterleave
TensorMapL2Promotion, TensorMapSwizzle
```

### cutlass.experimental.cuda.tensor_map（11）

```text
create_tensor_map_tiled, create_tensor_map_tiled_from_view
get_dsl_type_to_tensormap_type, get_tensormap_type_to_dsl_type, TensorMap
TensorMapDataFormat, TensorMapDataType, TensorMapFloatOOBFill
TensorMapInterleave, TensorMapL2Promotion, TensorMapSwizzle
```

### cutlass.experimental.primitives（177）

```text
add_packed_bf16x2_f32x2_f32x2, add_packed_f16x2_f32x2_f32x2
add_packed_f32x2, add_packed_f32x2_bf16x2_f32x2
add_packed_f32x2_f16x2_f32x2, atomicrmw, auto, bar_warp_sync
barrier_cluster_arrive, barrier_cluster_arrive_aligned
barrier_cluster_arrive_relaxed, barrier_cluster_arrive_relaxed_aligned
barrier_cluster_wait, barrier_cluster_wait_aligned, barrier_cta_arrive
barrier_cta_arrive_aligned, barrier_cta_red, barrier_cta_red_aligned
barrier_cta_sync, barrier_cta_sync_aligned, breakpoint, cluster_ctarank
clusterlaunchcontrol_query_cancel, clusterlaunchcontrol_try_cancel
convert, convert_and_pack_integer, convert_bf16x2_to_s2f6x2
convert_f32x2_to_s2f6x2, convert_f8x2_to_bf16x2, convert_float_to_integer
convert_float_to_tf32, convert_s2f6x2_to_bf16x2
cp_async_bulk_commit_group, cp_async_bulk_global_shared_cta
cp_async_bulk_prefetch, cp_async_bulk_shared_cluster_global
cp_async_bulk_shared_cluster_shared_cta
cp_async_bulk_tensor_global_shared_cta, cp_async_bulk_tensor_prefetch
cp_async_bulk_tensor_reduce, cp_async_bulk_tensor_shared_cluster_global
cp_async_bulk_tensor_shared_cta_global, cp_async_bulk_wait_group
cp_async_commit_group, cp_async_mbarrier_arrive, cp_async_shared_global
cp_async_wait_group, cp_reduce_async_bulk_global_shared_cta
cvt_e2m1_word_to_e4m3_words, cvt_e2m1x2_byte_to_bf16x2
cvt_e4m3x2_byte_to_bf16x2, cvt_e4m3x2_to_bf16x2
cvt_e4m3x2_word_to_bf16x2, cvt_f32x2_to_f4x2, cvt_f32x2_to_f8x2
cvt_packfloat, cvt_packfloat_f32, cvt_packfloat_f32_sf, cvt_packfloat_sf
cvta_to, dot_accumulate_2way, dot_accumulate_4way, elect_sync, exit, fabs
fence_acq_rel, fence_mbarrier_init, fence_proxy, fence_proxy_acquire
fence_proxy_async_acquire_sync_restrict
fence_proxy_async_release_sync_restrict, fence_proxy_release
fence_proxy_sync_restrict, fence_sc_cluster, fence_sync_restrict
fma_packed_f32x2, fma_packed_f32x2_bf16x2_f32x2_f32x2
fma_packed_f32x2_f16x2_f32x2_f32x2, fmin, griddepcontrol, inline_ptx
inline_ptx_hl, ldmatrix, load_ext, mapa, match_sync, mbarrier_arrive
mbarrier_arrive_drop, mbarrier_arrive_drop_expect_tx
mbarrier_arrive_drop_nocomplete, mbarrier_arrive_expect_tx
mbarrier_arrive_nocomplete, mbarrier_complete_tx, mbarrier_expect_tx
mbarrier_init, mbarrier_inval, mbarrier_test_wait, mbarrier_try_wait
mbarrier_try_wait_parity, mbarrier_try_wait_timelimit, mbarrier_wait
mbarrier_wait_parity, memory_barrier, mma_block_scale, mma_smem_desc
mma_sp_block_scale, mma_sp_sync, mma_sync, mov_b32, mul, mul_bf16x2
mul_packed_bf16x2_bf16x2_f16x2, mul_packed_bf16x2_f32x2_f32x2
mul_packed_f16x2_f16x2_bf16x2, mul_packed_f16x2_f32x2_f32x2
mul_packed_f32x2, MxScaleFormat, nanosleep, pmevent, prefetch_l1
prefetch_l2, prefetch_tensormap, prefetchu, print_runtime, prmt
ptx_add_comment, read_sreg_hw, red, redux_sync, setmaxregister, shfl_sync
st_bulk, stmatrix, store_ext, sub_packed_bf16x2_f32x2_f32x2
sub_packed_f16x2_f32x2_f32x2, sub_packed_f32x2
sub_packed_f32x2_bf16x2_f32x2, sub_packed_f32x2_f16x2_f32x2
tcgen05_alloc, tcgen05_commit, tcgen05_cp, tcgen05_dealloc, tcgen05_fence
tcgen05_ld, tcgen05_ld_red, tcgen05_mma, tcgen05_mma_block_scale
tcgen05_mma_sp, tcgen05_mma_sp_block_scale, tcgen05_mma_ws
tcgen05_mma_ws_sp, tcgen05_relinquish_alloc_permit, tcgen05_shift
tcgen05_st, tcgen05_wait, Tcgen05InputFormat, Tcgen05InstrDesc
Tcgen05MxInstrDesc, Tcgen05MxOmmaInstrDesc, Tcgen05OutputFormat
Tcgen05SmemDesc, Tcgen05SmemSwizzle, tensormap_cp_fenceproxy
tensormap_replace, TmemAddr, trace_mark, vote_sync
wgmma_commit_group_sync_aligned, wgmma_fence_aligned, wgmma_mma_async
wgmma_wait_group_sync_aligned, WgmmaSmemDesc, WgmmaSwizzle, wmma_load
wmma_mma, wmma_store
```

### cutlass.experimental.task_scheduling（12）

```text
consumer_work, domain_loop, Every, IterationPredicate, OpaqueCondition
pipeline, producer_work, schedule, when_false, when_true, work_tile_loop
WorkAttr
```

### cutlass.experimental.task_scheduling.enums（16）

```text
BlockGuard, Every, guard_fires, is_iteration_predicate, is_loop_guard
is_skippable_guard, IterationPredicate, LoopGuard, OpaqueCondition
PipelineGroupMode, PipelineType, ScheduleStage, ScheduleStageType
SignalingThreads, TileSchedulerType, WorkAttr
```

### cutlass.experimental.task_scheduling.memory（5）

```text
ResourceContext, SmemAllocation, SmemAllocator, TmemAllocation
TmemAllocator
```

### cutlass.experimental.task_scheduling.pipeline（3）

```text
TSPipelineAsyncUmma, TSPipelineTmaAsync, TSPipelineUmmaUmma
```

### cutlass.experimental.task_scheduling.pipeline_group（1）

```text
PipelineGroup
```

### cutlass.experimental.task_scheduling.resources（13）

```text
bind_task_local_variables, consumer_work, get_task_local_variables
MemoryResource, PdlLaunchBarrier, PdlWaitBarrier, PipelineConfig
producer_work, StageInfo, TaskLocalVariable, TileSchedulerConfig
WorkQueue, WorkTileInfo
```

### cutlass.experimental.task_scheduling.schedule_builder（18）

```text
ConditionalBlock, DataFlowToken, domain_loop, DomainLoop, DomainLoopProxy
ResourceProxy, Route, schedule, Schedule, ScheduleBuilder, ScheduleError
ScheduleResult, Step, when_false, when_true, work_tile_loop, WorkTileLoop
WorkTileLoopProxy
```

### cutlass.experimental.task_scheduling.task（2）

```text
normalize_and_validate_schedule_list, Task
```

### cutlass.experimental.task_scheduling.task_manager（5）

```text
DependencyGraphControlEdge, DependencyGraphEdge, expand_loop
print_schedule_list, TaskManager
```

### cutlass.cute.experimental.cluster_mailbox（4）

```text
b32_payload, B32PayloadCodec, ClusterMailbox, CTA_PAIR_PEER_LOCALE
```

### cutlass.cute.experimental.host_runtime（2）

```text
AllocationRequirement, QueryDeviceWorkspaceFunc
```
