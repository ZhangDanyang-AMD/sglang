"""
Performance benchmark for fused_sigmoid_gating_delta_rule_update kernel.
Tests the speed of the fused Triton kernel with different configurations.
"""

import argparse
import time
from typing import Optional

import torch
import triton

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_opt,
)


def verify_correctness(
    batch_size: int = 4,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim_k: int = 128,
    head_dim_v: int = 128,
    num_heads_v: Optional[int] = None,
    use_varlen: bool = False,
    use_initial_state: bool = True,
    use_qk_l2norm: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Verify that the original and optimized implementations produce the same results.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of query/key heads
        head_dim_k: Dimension of each key/query head
        head_dim_v: Dimension of each value head
        num_heads_v: Number of value heads (defaults to num_heads)
        use_varlen: Whether to use variable length sequences
        use_initial_state: Whether to use initial hidden state
        use_qk_l2norm: Whether to use QK L2 normalization
        dtype: Data type for tensors
        device: Device to run on
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        dict: Dictionary containing verification results
    """
    if num_heads_v is None:
        num_heads_v = num_heads
    
    # print(f"\n{'='*80}")
    # print(f"Verifying Correctness:")
    # print(f"  Batch Size: {batch_size}")
    # print(f"  Sequence Length: {seq_len}")
    # print(f"  Num Heads (Q/K): {num_heads}")
    # print(f"  Num Heads (V): {num_heads_v}")
    # print(f"  Head Dim (K): {head_dim_k}")
    # print(f"  Head Dim (V): {head_dim_v}")
    # print(f"  Use VarLen: {use_varlen}")
    # print(f"  Use Initial State: {use_initial_state}")
    # print(f"  Use QK L2 Norm: {use_qk_l2norm}")
    # print(f"  Data Type: {dtype}")
    # print(f"  Tolerance: rtol={rtol}, atol={atol}")
    # print(f"{'='*80}")
    
    # Create input tensors with fixed seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim_k, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim_k, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, seq_len, num_heads_v, head_dim_v, dtype=dtype, device=device
    )
    b = torch.randn(batch_size, seq_len, num_heads_v, dtype=dtype, device=device)
    
    # Gating parameters
    A_log = torch.randn(num_heads_v, dtype=dtype, device=device)
    a = torch.randn(batch_size, seq_len, num_heads_v, dtype=dtype, device=device)
    dt_bias = torch.randn(num_heads_v, dtype=dtype, device=device)
    softplus_beta = 1.0
    softplus_threshold = 20.0
    
    # Initial state setup
    if use_initial_state:
        initial_state_source_orig = torch.randn(
            batch_size, num_heads_v, head_dim_k, head_dim_v, dtype=dtype, device=device
        )
        initial_state_source_opt = initial_state_source_orig.clone()
        initial_state_indices = (torch.arange(batch_size, device=device, dtype=torch.int32) % batch_size).contiguous()
    else:
        initial_state_source_orig = None
        initial_state_source_opt = None
        initial_state_indices = None
    
    # Variable length setup
    if use_varlen:
        cu_seqlens = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.randint(seq_len // 2, seq_len + 1, (batch_size,), dtype=torch.int32, device=device).cumsum(0)
        ])
    else:
        cu_seqlens = None
    
    # Run original implementation
    print("Running original implementation...")
    output_orig = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source_orig,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
    )
            
    # Run optimized implementation
    print("Running optimized implementation...")
    output_opt = fused_sigmoid_gating_delta_rule_update_opt(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source_opt,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
    )

    # Compare outputs
    print("\nComparing outputs...")
    output_match = torch.allclose(output_orig, output_opt, rtol=rtol, atol=atol)
    
    # Calculate statistics
    abs_diff = torch.abs(output_orig - output_opt)
    rel_diff = abs_diff / (torch.abs(output_orig) + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Compare hidden states if applicable
    state_match = True
    if use_initial_state:
        print("Comparing hidden states...")
        state_match = torch.allclose(
            initial_state_source_orig, initial_state_source_opt, 
            rtol=rtol, atol=atol
        )
        
        state_abs_diff = torch.abs(initial_state_source_orig - initial_state_source_opt)
        state_rel_diff = state_abs_diff / (torch.abs(initial_state_source_orig) + 1e-8)
        
        max_state_abs_diff = state_abs_diff.max().item()
        mean_state_abs_diff = state_abs_diff.mean().item()
        max_state_rel_diff = state_rel_diff.max().item()
        mean_state_rel_diff = state_rel_diff.mean().item()
    
    # Print results
    print(f"\n{'='*80}")
    print("Verification Results:")
    print(f"{'='*80}")
    print(f"Output Match: {'✓ PASS' if output_match else '✗ FAIL'}")
    print(f"\nOutput Differences:")
    print(f"  Max Absolute Diff: {max_abs_diff:.6e}")
    print(f"  Mean Absolute Diff: {mean_abs_diff:.6e}")
    print(f"  Max Relative Diff: {max_rel_diff:.6e}")
    print(f"  Mean Relative Diff: {mean_rel_diff:.6e}")
    
    if use_initial_state:
        print(f"\nHidden State Match: {'✓ PASS' if state_match else '✗ FAIL'}")
        print(f"\nHidden State Differences:")
        print(f"  Max Absolute Diff: {max_state_abs_diff:.6e}")
        print(f"  Mean Absolute Diff: {mean_state_abs_diff:.6e}")
        print(f"  Max Relative Diff: {max_state_rel_diff:.6e}")
        print(f"  Mean Relative Diff: {mean_state_rel_diff:.6e}")
    
    overall_pass = output_match and state_match
    print(f"\n{'='*80}")
    print(f"Overall: {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    return {
        "output_match": output_match,
        "state_match": state_match,
        "overall_pass": overall_pass,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "max_state_abs_diff": max_state_abs_diff if use_initial_state else None,
        "mean_state_abs_diff": mean_state_abs_diff if use_initial_state else None,
        "max_state_rel_diff": max_state_rel_diff if use_initial_state else None,
        "mean_state_rel_diff": mean_state_rel_diff if use_initial_state else None,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused sigmoid gating delta rule update kernel"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1, help="Sequence length")
    parser.add_argument("--num-heads-qk", type=int, default=4, help="Number of qk heads")
    parser.add_argument("--num-heads-v", type=int, default=8, help="Number of v heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Key head dimension")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-varlen", action="store_true", help="Use variable length sequences")
    parser.add_argument("--no-initial-state", action="store_true", help="Disable initial state")
    parser.add_argument("--use-qk-l2norm", action="store_true", help="Enable QK L2 normalization")
    parser.add_argument("--sweep", action="store_true", help="Run a parameter sweep")
    parser.add_argument("--verify", action="store_true", help="Verify correctness by comparing outputs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for verification")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for verification")
    
    args = parser.parse_args()

    torch.manual_seed(1234)

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    # print(f"Using device: {torch.cuda.get_device_name(0)}")
    # print(f"CUDA version: {torch.version.cuda}")
    # print(f"Triton version: {triton.__version__}")
    
    # Run correctness verification
    print("\n" + "="*80)
    print("Running Correctness Verification...")
    print("="*80)
    
    verify_correctness(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads_qk,
        head_dim_k=args.head_dim,
        head_dim_v=args.head_dim,
        num_heads_v=args.num_heads_v,
        use_varlen=args.use_varlen,
        use_initial_state=not args.no_initial_state,
        use_qk_l2norm=args.use_qk_l2norm,
        dtype=dtype,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()
