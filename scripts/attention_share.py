"""Attention share, computed independently in the op view and the kernel view.

A PyTorch profiler export mixes two row populations: framework operators
(`aten::add`, `flash_attn::_flash_attn_forward`) and device kernels
(`void flash::flash_fwd_kernel<...>`, `Memcpy HtoD`). A CUDA op's time appears
in BOTH -- once against the op that launched it, once against the kernel that
ran. Summing them together double-counts, but only for ops that have a
distinguishable kernel row, so the distortion is uneven and cannot be divided
out.

So the share is computed twice, in each population separately. If the two
agree, the number is solid. If they disagree, the honest output is the range,
not whichever end supports the decision we would like to make.
"""
import json, sys
from collections import defaultdict

ATTENTION_MARKERS = (
    "flash_attn", "scaled_dot_product_attention", "sdpa", "efficient_attention",
    "sageattn", "video_sparse_attn", "fmha", "flash::", "flash_fwd",
)

def is_attention(n): 
    low = n.lower()
    return any(m.lower() in low for m in ATTENTION_MARKERS)

def is_kernel_row(n):
    """Device-side row: a CUDA kernel signature or a memcpy/memset."""
    return (n.startswith("void ") or n.startswith("Memcpy") or n.startswith("Memset")
            or ("(" in n and "::" not in n.split("(")[0]) or "cutlass" in n.lower()
            or n.startswith("std::") or "kernel" in n.lower() and "aten::" not in n)

for path in sys.argv[1:]:
    d = json.load(open(path))
    wl = d.get("workload", {})
    ops, kernels = defaultdict(float), defaultdict(float)
    for r in d["rows"]:
        n = r.get("name", "")
        t = float(r.get("self_cuda_time_us") or 0.0)
        (kernels if is_kernel_row(n) else ops)[n] += t

    def summarize(pop, label):
        total = sum(pop.values())
        attn = sum(v for k, v in pop.items() if is_attention(k))
        share = attn / total if total else 0.0
        ceil = 1.0 / (1.0 - share) if share < 1 else float("inf")
        print(f"    {label:12s} total={total/1e6:7.3f}s  attention={attn/1e6:6.3f}s  "
              f"share={share*100:5.2f}%  ceiling={ceil:.4f}x")
        return share, ceil

    print(f"=== {wl.get('workload_id', path)}")
    s_op, c_op = summarize(ops, "OP VIEW")
    s_k, c_k = summarize(kernels, "KERNEL VIEW")
    lo, hi = sorted([c_op, c_k])
    print(f"    ---> attention share {min(s_op,s_k)*100:.2f}%-{max(s_op,s_k)*100:.2f}%, "
          f"Amdahl ceiling {lo:.4f}x-{hi:.4f}x")
    verdict = ("REACHABLE" if lo >= 1.3 else
               "UNREACHABLE" if hi < 1.3 else "AMBIGUOUS (ceiling straddles 1.3x)")
    print(f"    ---> 1.3x gate: {verdict}")
    print(f"    kernel-view attention rows:")
    for k, v in sorted(((k,v) for k,v in kernels.items() if is_attention(k)), key=lambda x:-x[1])[:4]:
        print(f"        {v/1e6:7.4f}s {k[:78]}")
    print(f"    op-view attention rows:")
    for k, v in sorted(((k,v) for k,v in ops.items() if is_attention(k)), key=lambda x:-x[1])[:4]:
        print(f"        {v/1e6:7.4f}s {k[:78]}")
    print()
