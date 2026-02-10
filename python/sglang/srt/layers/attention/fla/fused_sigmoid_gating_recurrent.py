from math import e
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

from sglang.srt.layers.attention.fla.utils import input_guard

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel2(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    b_dt_bias = gl.load(p_dt_bias).to(gl.float32) # f32
    b_A_log = gl.load(p_A_log).to(gl.float32) # f32

    # Gating computation pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv

    b_a = gl.load(p_a).to(gl.float32) # f32
    b_b = gl.load(p_b).to(gl.float32) # f32

    # 4*dwords
    DWORDS_SIZE: gl.constexpr = 4*4 #Bytes
    if q.dtype.element_ty == gl.float16:
        DTYPE_SIZE : gl.constexpr = 2
    elif q.dtype.element_ty == gl.float8e5 or q.dtype.element_ty == gl.float8e4nv:
        DTYPE_SIZE : gl.constexpr = 1
    elif q.dtype.element_ty == gl.float32:
        DTYPE_SIZE : gl.constexpr = 4
    else:
        DTYPE_SIZE : gl.constexpr = 2

    ELE_PER_TILE: gl.constexpr = DWORDS_SIZE // DTYPE_SIZE
    gl.static_assert(BV%ELE_PER_TILE==0)
    gl.static_assert(BK%ELE_PER_TILE==0)

    WARP_PER_CTA: gl.constexpr = 4
    T_PER_WARP_V: gl.constexpr = BV // ELE_PER_TILE // WARP_PER_CTA
    THREAD_PER_WARP: gl.constexpr = 64
    T_PER_WARP_K: gl.constexpr = THREAD_PER_WARP // T_PER_WARP_V

    # BV = 128, warp 4
    #   thread: [8, 8]
    #   thread: [16, 4]
    #   thread: [1, 4]

    # BV = 128, warp 1
    #   thread: [32, 8]
    #   thread: [4, 16]
    #   thread: [1, 1]

    # BV = 64, warp 4
    #   thread: [4, 8]
    #   thread: [32, 2]
    #   thread: [1, 4]

    # BV = 64, warp 1
    #   thread: [16, 8]
    #   thread: [8, 8]
    #   thread: [1, 1]

    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BK//T_PER_WARP_K, ELE_PER_TILE],
        threads_per_warp=[T_PER_WARP_K, T_PER_WARP_V],
        warps_per_cta=[1, WARP_PER_CTA],
        order=[1, 0],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BK//T_PER_WARP_K, 1],
        threads_per_warp=[T_PER_WARP_K, THREAD_PER_WARP//T_PER_WARP_K],
        warps_per_cta=[1, WARP_PER_CTA],
        order=[0, 1],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, ELE_PER_TILE],
        threads_per_warp=[64//T_PER_WARP_V, T_PER_WARP_V],
        warps_per_cta=[1, WARP_PER_CTA],
        order=[1, 0],
    )
    slice1_b2d: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice2_b2d: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )

    slice1_b1d1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked1,
    )
    slice2_b1d1: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked1,
    )

    slice1_b1d2: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2,
    )
    slice2_b1d2: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2,
    )

    o_k_blocked_M = i_k * BK + gl.arange(0, BK, layout=slice1_b1d1)
    o_k_blocked_N = i_k * BK + gl.arange(0, BK, layout=slice2_b1d1)
    o_v_blocked_M = i_v * BV + gl.arange(0, BV, layout=slice1_b1d2)
    o_v_blocked_N = i_v * BV + gl.arange(0, BV, layout=slice2_b1d2)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1_b2d)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice2_b2d)

    p_q = q + (bos * H + i_h) * K + o_k_blocked_M
    p_k = k + (bos * H + i_h) * K + o_k_blocked_M
    p_v = v + (bos * HV + i_hv) * V + o_v_blocked_N
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v_blocked_N

    mask_k_blocked = o_k_blocked_M < K
    mask_v_blocked = o_v_blocked_N < V
    mask_k_slice = o_k_slice < K
    mask_v_slice = o_v_slice < V
    mask_h = mask_k_slice[:, None] & mask_v_slice[None, :]

    b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            b_h += gl.load(p_h0, mask=mask_h, other=0.0).to(gl.float32)  # BKxBVxf32
            
    for _ in range(0, T):
        b_k = gl.load(p_k, mask=mask_k_blocked, other=0.0).to(gl.float32)  # BKxf32
        b_q = gl.load(p_q, mask=mask_k_blocked, other=0.0).to(gl.float32)  # BKxf32
        b_v = gl.load(p_v, mask=mask_v_blocked, other=0.0).to(gl.float32)  # BVxf32
        
        softplus_beta_inv = 1.0 / softplus_beta
        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_A = -gl.exp(b_A_log)
        b_g = b_A * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + gl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            k_norm_rev = 1 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
            tl.assume(k_norm_rev > 0)
            b_k = b_k * k_norm_rev # BKxf32

        # shared_b_v = gl.allocate_shared_memory(b_v0.dtype, [BV], shared_layout, b_v0) # BVxf32
        # # Apply L2 normalization to q
        if USE_QK_L2NORM_IN_KERNEL:
            q_norm_rev = 1 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
            tl.assume(q_norm_rev > 0)
            b_q = b_q * q_norm_rev # BKxf32
        b_q = b_q * scale # BKxf32
        # b_q = gl.convert_layout(b_q, layout=slice1)

        # Apply gating to hidden state: h *= exp(g)
        b_h *= gl.exp(b_g) # BKxBVxf32

        # b_v = shared_b_v.load(slice4)

        # Delta rule: v -= sum(h * k, dim=0)
        # @TODO: place K in the 2nd axis
        b_k_col = gl.convert_layout(b_k[:, None], b_h.type.layout)
        b_q_col = gl.convert_layout(b_q[:, None], b_h.type.layout)

        delta = gl.sum(b_h * b_k_col, 0)       
        delta = gl.convert_layout(delta, b_v.type.layout)
        b_v = b_v - delta        
        b_v = b_v * b_beta

        b_v_row = gl.convert_layout(b_v[None, :], b_h.type.layout)
        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k_col * b_v_row # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

        # Compute output: o = sum(h * q, dim=0)
        # @TODO: place K in 2nd axis
        b_o = gl.sum(b_h * b_q_col, 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
        b_o = gl.convert_layout(b_o, o_v_blocked_N.type.layout)
        gl.store(p_o, b_o.to(p_o.dtype.element_ty) )

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel1(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    b_dt_bias = gl.load(p_dt_bias) # f32
    b_A_log = gl.load(p_A_log).to(gl.float32) # f32

    # Gating computation pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv

    b_a = gl.load(p_a) # f32
    b_b = gl.load(p_b) # f32

    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 2],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blocked_linear: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1]],
        lane_bases=[[2], [4], [8], [16], [32], [64]],
        warp_bases=[[0], [0]],
        block_bases=[],
        shape=[128],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0]
    )

    o_k_blocked = i_k * BK + gl.arange(0, BK, layout=blocked1)
    o_v_blocked = i_v * BV + gl.arange(0, BV, layout=blocked1)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)

    p_q = q + (bos * H + i_h) * K + o_k_blocked
    p_k = k + (bos * H + i_h) * K + o_k_blocked
    p_v = v + (bos * HV + i_hv) * V + o_v_blocked
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v_blocked



    mask_k_blocked = o_k_blocked < K
    mask_v_blocked = o_v_blocked < V
    mask_k_slice = o_k_slice < K
    mask_v_slice = o_v_slice < V
    mask_h = mask_k_slice[:, None] & mask_v_slice[None, :]

    b_k0 = gl.load(p_k, mask=mask_k_blocked, other=0)  # BKxf32
    b_q0 = gl.load(p_q, mask=mask_k_blocked, other=0)  # BKxf32
    b_v0 = gl.load(p_v, mask=mask_v_blocked, other=0)  # BVxf32
    
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k_slice[:, None] * V
                + o_v_slice[None, :]
            )
            b_h = gl.load(p_h0).to(gl.float32)  # BKxBVxf32

            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_b += HV
            p_a += HV

            softplus_beta_inv = 1.0 / softplus_beta
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
            beta_x = softplus_beta * x # f32
            # Apply softplus with numerical stability
            softplus_x = gl.where(
                beta_x <= softplus_threshold,
                softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
                x,
            )
            b_A = -gl.exp(b_A_log)
            b_g = b_A * softplus_x # f32

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

            shared_b_k = gl.allocate_shared_memory(b_k0.dtype, [BK], shared_layout, b_k0)
            shared_b_q = gl.allocate_shared_memory(b_q0.dtype, [BK], shared_layout, b_q0) # BKxf32

            gl.amd.cdna3.sched_barrier(0)

            b_k = shared_b_k.load(slice1).to(gl.float32) # BKxf32
            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                k_norm_rev = 1 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
                tl.assume(k_norm_rev > 0)
                b_k = b_k * k_norm_rev # BKxf32
            # b_k = gl.convert_layout(b_k, layout=slice1)

            shared_b_v = gl.allocate_shared_memory(b_v0.dtype, [BV], shared_layout, b_v0) # BVxf32
            # Apply L2 normalization to q
            b_q = shared_b_q.load(slice1).to(gl.float32) # BKxf32
            if USE_QK_L2NORM_IN_KERNEL:
                q_norm_rev = 1 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
                tl.assume(q_norm_rev > 0)
                b_q = b_q * q_norm_rev # BKxf32
            b_q = b_q * scale # BKxf32
            # b_q = gl.convert_layout(b_q, layout=slice1)


            # Apply gating to hidden state: h *= exp(g)
            b_h *= gl.exp(b_g) # BKxBVxf32

            b_v = shared_b_v.load(slice4)
            b_v = b_v.to(gl.float32)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

            # Apply beta gating: v *= beta
            b_v *= b_beta # BVxf32

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

            # Compute output: o = sum(h * q, dim=0)
            b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
            b_o = gl.convert_layout(b_o, layout=blocked1)
            gl.store(p_o, b_o)

            # Store final state back to h0_source with bounds checking
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

        else:
            b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_b += HV
            p_a += HV

            softplus_beta_inv = 1.0 / softplus_beta
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a.to(gl.float32) + b_dt_bias.to(gl.float32) # f32
            beta_x = softplus_beta * x # f32
            # Apply softplus with numerical stability
            softplus_x = gl.where(
                beta_x <= softplus_threshold,
                softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
                x,
            )
            b_A = -gl.exp(b_A_log)
            b_g = b_A * softplus_x # f32

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + gl.exp(-b_b.to(gl.float32))) # f32

            shared_b_k = gl.allocate_shared_memory(b_k0.dtype, [BK], shared_layout, b_k0)
            shared_b_q = gl.allocate_shared_memory(b_q0.dtype, [BK], shared_layout, b_q0) # BKxf32
            gl.amd.cdna3.sched_barrier(0)

            b_k = shared_b_k.load(slice1).to(gl.float32)
            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                k_norm_rev = 1 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
                tl.assume(k_norm_rev > 0)
                b_k = b_k * k_norm_rev # BKxf32
            # b_k = gl.convert_layout(b_k, layout=slice1)

            shared_b_v = gl.allocate_shared_memory(b_v0.dtype, [BV], shared_layout, b_v0) # BVxf32
            # Apply L2 normalization to q
            b_q = shared_b_q.load(slice1).to(gl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                q_norm_rev = 1 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
                tl.assume(q_norm_rev > 0)
                b_q = b_q * q_norm_rev # BKxf32
            b_q = b_q * scale # BKxf32
            # b_q = gl.convert_layout(b_q, layout=slice1)

            # Apply gating to hidden state: h *= exp(g)
            b_h *= gl.exp(b_g) # BKxBVxf32

            b_v = shared_b_v.load(slice4).to(gl.float32)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= gl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma

            # Apply beta gating: v *= beta
            b_v *= b_beta # BVxf32

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma


            # Compute output: o = sum(h * q, dim=0)
            b_o = gl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
            b_o = gl.convert_layout(b_o, layout=blocked1)
            gl.store(p_o, b_o)

do_bench = lambda kernel, quantiles: triton.testing.do_bench(kernel, quantiles=quantiles, warmup=1, rep=1)
def gdr_get_configs():
    # Only keep configs that make threads_per_warp = [T_PER_WARP_V, T_PER_WARP_K]
    # valid: T_PER_WARP_V = warp_size // T_PER_WARP_K must be >= 1.
    return [
        # T_PER_WARP_V: 32, V_PER_THREAD: 1
        triton.Config({"T_PER_WARP_K": 2}, num_warps=4, num_stages=1),
        # T_PER_WARP_V: 16, V_PER_THREAD: 2
        triton.Config({"T_PER_WARP_K": 4}, num_warps=4, num_stages=1),
        # T_PER_WARP_V: 8, V_PER_THREAD: 4
        triton.Config({"T_PER_WARP_K": 8}, num_warps=4, num_stages=1),

        # T_PER_WARP_V: 64, V_PER_THREAD: 1
        triton.Config({"T_PER_WARP_K": 1}, num_warps=2, num_stages=1),
        # T_PER_WARP_V: 32, V_PER_THREAD: 2
        triton.Config({"T_PER_WARP_K": 2}, num_warps=2, num_stages=1),
        # T_PER_WARP_V: 16, V_PER_THREAD: 4
        triton.Config({"T_PER_WARP_K": 4}, num_warps=2, num_stages=1),
        # T_PER_WARP_V: 8, V_PER_THREAD: 8
        triton.Config({"T_PER_WARP_K": 8}, num_warps=2, num_stages=1),
        # T_PER_WARP_V: 4, V_PER_THREAD: 16
        triton.Config({"T_PER_WARP_K": 16}, num_warps=2, num_stages=1),

        # T_PER_WARP_V: 64, V_PER_THREAD: 2
        triton.Config({"T_PER_WARP_K": 1}, num_warps=1, num_stages=1),
        # T_PER_WARP_V: 32, V_PER_THREAD: 4
        triton.Config({"T_PER_WARP_K": 2}, num_warps=1, num_stages=1),
        # T_PER_WARP_V: 16, V_PER_THREAD: 8
        triton.Config({"T_PER_WARP_K": 4}, num_warps=1, num_stages=1),
        # T_PER_WARP_V: 8, V_PER_THREAD: 16
        triton.Config({"T_PER_WARP_K": 8}, num_warps=1, num_stages=1),
    ]

@triton.autotune(configs=gdr_get_configs(), key=['K'], do_bench=do_bench)
@triton.heuristics(
    {
        "T_PER_WARP_V": lambda nargs: 64 // nargs["T_PER_WARP_K"],
        "WARP_SIZE": lambda nargs: nargs["num_warps"]
    }
)  # test kwargs
@gluon.jit(do_not_specialize=["T"])
def gluon_fused_sigmoid_gating_delta_rule_update_kernel3(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    T_PER_WARP_K: gl.constexpr,
    T_PER_WARP_V: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    
    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    # Gating computation pointers
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv

    # 4*dwords
    DWORDS_SIZE: gl.constexpr = 4*4 #Bytes
    if q.dtype.element_ty == gl.float16:
        DTYPE_SIZE : gl.constexpr = 2
    elif q.dtype.element_ty == gl.float8e5 or q.dtype.element_ty == gl.float8e4nv:
        DTYPE_SIZE : gl.constexpr = 1
    elif q.dtype.element_ty == gl.float32:
        DTYPE_SIZE : gl.constexpr = 4
    else:
        DTYPE_SIZE : gl.constexpr = 2

    ELE_PER_TILE: gl.constexpr = DWORDS_SIZE // DTYPE_SIZE
    # gl.static_assert(BV%ELE_PER_TILE==0)
    # gl.static_assert(BK%ELE_PER_TILE==0)

    # h
    V_PER_THREAD: gl.constexpr=BV//T_PER_WARP_V//WARP_SIZE
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[V_PER_THREAD, BK//T_PER_WARP_K],
        threads_per_warp=[T_PER_WARP_V, T_PER_WARP_K],
        warps_per_cta=[WARP_SIZE, 1],
        order=[1, 0],
    )
    # k
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BK//T_PER_WARP_K],
        threads_per_warp=[T_PER_WARP_V, T_PER_WARP_K],
        warps_per_cta=[WARP_SIZE, 1],
        order=[1, 0],
    )
    # v
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[V_PER_THREAD, 1],
        threads_per_warp=[T_PER_WARP_V, T_PER_WARP_K],
        warps_per_cta=[WARP_SIZE, 1],
        order=[0, 1],
    )
    
    slice1_b2d: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )
    slice2_b2d: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )

    # slice1_b1d1: gl.constexpr = gl.SliceLayout(
    #     dim=1,
    #     parent=blocked1,
    # )
    slice2_b1d1: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked1,
    )

    slice1_b1d2: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2,
    )
    # slice2_b1d2: gl.constexpr = gl.SliceLayout(
    #     dim=0,
    #     parent=blocked2,
    # )

    # slice1_b1d3: gl.constexpr = gl.SliceLayout(
    #     dim=1,
    #     parent=blocked3,
    # )
    # slice2_b1d3: gl.constexpr = gl.SliceLayout(
    #     dim=0,
    #     parent=blocked3,
    # )

    o_k_base = i_k * BK
    o_v_base = i_v * BV
    # o_k_blocked_M = i_k * BK + gl.arange(0, BK, layout=slice1_b1d1)
    o_k_blocked_N = o_k_base + gl.arange(0, BK, layout=slice2_b1d1)
    o_v_blocked_M = o_v_base + gl.arange(0, BV, layout=slice1_b1d2)
    # o_v_blocked_N = i_v * BV + gl.arange(0, BV, layout=slice2_b1d2)
    o_k_slice = o_k_base + gl.arange(0, BK, layout=slice1_b2d)
    o_v_slice = o_v_base + gl.arange(0, BV, layout=slice2_b2d)

    TH_base_k = (bos * H + i_h) * K
    p_q = q + TH_base_k + o_k_blocked_N
    p_k = k + TH_base_k + o_k_blocked_N
    TH_base_v = (bos * HV + i_hv) * V
    p_v = v + TH_base_v + o_v_blocked_M

    T_PER_WARP_O: gl.constexpr = BV // ELE_PER_TILE // WARP_SIZE
    DEN: gl.constexpr = 64 // T_PER_WARP_O
    gl.static_assert((DEN & (DEN - 1)) == 0)  
    expand_bo: gl.constexpr = DEN.bit_length() - 1

    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, ELE_PER_TILE],
        threads_per_warp=[64//T_PER_WARP_O, T_PER_WARP_O],
        warps_per_cta=[1, WARP_SIZE],
        order=[1,0],
    )

    slice1_b1d3: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=out_layout,
    )
    slice2_b1d3: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=out_layout,
    )
    x_axis = gl.arange(0, BV, layout=slice1_b1d3)
    y_axis = gl.arange(0, 64//T_PER_WARP_O, layout=slice2_b1d3)
    offs = x_axis[None,:] + y_axis[:, None]*BV
    p_o = o + ((i_k * all) * HV ) * V + TH_base_v + offs
    maskx = x_axis<V 
    masky = y_axis == 0
    mask_o = maskx[None,:]&masky[:,None]

    b_h = gl.zeros([BV, BK], dtype=gl.float32, layout=blocked2d)
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v_slice[:, None] * K
                + o_k_slice[None, :] 
            )
            # b_h += gl.load(p_h0, mask=mask_h, other=0.0).to(gl.float32)  # BKxBVxf32
            b_h += gl.load(p_h0).to(gl.float32)  # BKxBVxf32
            
    for _ in range(0, T):
        b_dt_bias = gl.load(p_dt_bias).to(gl.float32) # f32
        b_A_log = gl.load(p_A_log).to(gl.float32) # f32
        b_a = gl.load(p_a).to(gl.float32) # f32
        b_b = gl.load(p_b).to(gl.float32) # f32
        # b_k = gl.load(p_k, mask=mask_k_blocked, other=0.0).to(gl.float32)  # BKxf32
        # b_q = gl.load(p_q, mask=mask_k_blocked, other=0.0).to(gl.float32)  # BKxf32
        # b_v = gl.load(p_v, mask=mask_v_blocked, other=0.0).to(gl.float32)  # BVxf32
        b_k = gl.load(p_k).to(gl.float32)  # BKxf32
        b_q = gl.load(p_q).to(gl.float32)  # BKxf32
        b_v = gl.load(p_v).to(gl.float32)  # BVxf32
        
        softplus_beta_inv = 1.0 / softplus_beta
        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            softplus_beta_inv * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_A = -gl.exp(b_A_log)
        b_g = b_A * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + gl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            k_norm_rev = 1.0 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6)
            tl.assume(k_norm_rev > 0)
            b_k = b_k * k_norm_rev # BKxf32
            q_norm_rev = 1.0 / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6)
            tl.assume(q_norm_rev > 0)
            b_q = b_q * q_norm_rev # BKxf32

        b_q = b_q * scale # BKxf32
        # b_q = gl.convert_layout(b_q, layout=slice1)

        # Apply gating to hidden state: h *= exp(g)
        b_h *= gl.exp(b_g) # BKxBVxf32

        # b_v = shared_b_v.load(slice4)

        # Delta rule: v -= sum(h * k, dim=0)
        # @TODO: place K in the 2nd axis
        b_k_col = gl.convert_layout(b_k[None, :], b_h.type.layout)
        b_q_col = gl.convert_layout(b_q[None, :], b_h.type.layout)

        delta = gl.sum(b_h * b_k_col, 1)       
        delta = gl.convert_layout(delta, b_v.type.layout)
        b_v = b_v - delta        
        b_v = b_v * b_beta

        b_v_row = gl.convert_layout(b_v[:, None], b_h.type.layout)
        b_h += b_k_col * b_v_row # BKxBVxf32  --->  BKxf32 * 16xBVxf32 = BKxBVxf32 ---> diag(bk)16... @ b_v  ---> BK/16 x mfma

        # Compute output: o = sum(h * q, dim=0)
        # @TODO: place K in 2nd axis
        b_o = gl.sum(b_h * b_q_col, 1) # BKxBVxf32 x BKxf32 -> BVxf32  ---> 16xBKxf32 @ BKxBVxf32 = 16xBVxf32  ---> BK/16 x mfma
        
        # smem_bo.store(b_o)
        for i in gl.static_range(0,expand_bo):
            b_o = gl.join(b_o, b_o)
            
        b_o = b_o.reshape(BV,DEN).permute(1,0)
        b_o = gl.convert_layout(b_o, out_layout)

        gl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_o)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v_slice[:, None] * K
                + o_k_slice[None, :] 
            )
            gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # BKxBVxf32

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # BVxf32
        b_b = tl.load(p_b).to(tl.float32) # f32

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32) # f32
        b_a = tl.load(p_a).to(tl.float32) # f32
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32) # f32

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6)) # BKxf32
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6)) # BKxf32

        b_q = b_q * scale # BKxf32

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g) # BKxBVxf32

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32

        # Apply beta gating: v *= beta
        b_v *= b_beta # BVxf32

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :] # BKxBVxf32

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0) # BKxBVxf32 x BKxf32 -> BVxf32
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel_VK(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[: ,None]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # BKxBVxf32

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # BKxf32
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # BVxf32
        b_b = tl.load(p_b).to(tl.float32) # f32

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32) # f32
        b_a = tl.load(p_a).to(tl.float32) # f32
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32) # f32

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias # f32
        beta_x = softplus_beta * x # f32
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x # f32

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b)) # f32

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6)) # BKxf32
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6)) # BKxf32

        b_q = b_q * scale # BKxf32

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g) # BKxBVxf32

        # Delta rule: v -= sum(h * k, dim=1)
        b_v -= tl.sum(b_h * b_k[None, :], 1) # BVxBKxf32 x BKxf32 -> BVxf32

        # Apply beta gating: v *= beta
        b_v *= b_beta # BVxf32

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[None, :] * b_v[:, None] # BKxBVxf32

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[None, :], 1) # BVxBKxf32 x BKxf32 -> BVxf32
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@input_guard
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    gluon: bool = True,
    min_hdim: int = 64,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), min_hdim)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)

    if gluon:        
        initial_state_source_test = initial_state_source.clone() if initial_state_source is not None else None
        o_test = o.clone()
        print("run into gluon")
        gluon_fused_sigmoid_gating_delta_rule_update_kernel3[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o, # write
            h0_source=initial_state_source, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            T_PER_WARP_K=16,
            T_PER_WARP_V=4,
            WARP_SIZE=2,
            num_warps=2,
            num_stages=3,
        )
        # print(gluon_fused_sigmoid_gating_delta_rule_update_kernel3.best_config)
        # print(gluon_fused_sigmoid_gating_delta_rule_update_kernel3.configs_timings)

        torch.cuda.synchronize()
        ms = triton.testing.do_bench(lambda: gluon_fused_sigmoid_gating_delta_rule_update_kernel3[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o_test, # write
            h0_source=initial_state_source_test, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            T_PER_WARP_K=16,
            T_PER_WARP_V=4,
            WARP_SIZE=2,
            num_warps=2,
            num_stages=3,
            ),
            warmup=2500, rep=3000,)
        torch.cuda.synchronize()
        print("opt kernel ms", ms)

    else:        
        initial_state_source_test = initial_state_source.clone() if initial_state_source is not None else None
        o_test = o.clone()
        fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o, # write
            h0_source=initial_state_source, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
        )

        torch.cuda.synchronize()
        ms = triton.testing.do_bench(lambda: fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o_test, # write
            h0_source=initial_state_source_test, # update
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
            ),
            warmup=2500, rep=3000,)
        torch.cuda.synchronize()
        print("ori kernel ms", ms)

    o = o.squeeze(0)

    return o