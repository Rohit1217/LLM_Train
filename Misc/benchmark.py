"""
Correctness + benchmark harness for varlen FA2 (forward & backward).

Import your kernels and this file drives them:
    from fwd_kernel import triton_intradoc_attention          # (q,k,v,cuseq,max_len,group_size) -> (o, lse_base2_fp32)
    from bwd_kernel import flash_att_backward_split           # (q,k,v,o,do,lse,cuseq,max_len,group_size) -> (dq,dk,dv)

Contract assumed (matches your kernels):
  - q: [T, Hq, d], k/v: [T, Hkv, d], bf16, packed varlen with cuseq (int32, [num_docs+1])
  - No positional encoding fused inside the kernels (plain scaled-dot causal attention)
  - LSE stored/consumed in BASE 2, fp32, shape [Hq, T]
  - group_size = Hq // Hkv (GQA)

Protocol: kernel-verify. Rule zero — no timing is reported for a config that
hasn't passed correctness in this run.
"""

import math
import torch
import torch.nn.functional as F
import triton

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
LOG2E = 1.4426950408889634

torch.manual_seed(0)
torch.cuda.set_device(DEVICE)

# ---------------------------------------------------------------------------
# Import your kernels here
# ---------------------------------------------------------------------------
from intradocatt_fwd_kernels import triton_intradoc_attention          # noqa: E402
from intradocatt_backwd_kernels import flash_att_backward_split           # noqa: E402

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False


# ---------------------------------------------------------------------------
# Reference: fp32 per-doc causal attention, autograd-able
# ---------------------------------------------------------------------------

def reference_fwd_bwd(q, k, v, do, cuseq, group_size):
    """Per-doc causal attention in fp32, gradients via autograd.
    Returns o (bf16), lse_base2 (fp32 [Hq,T]), dq, dk, dv (bf16, KV grads
    reduced over the GQA group). The reference is deliberately eager/naive —
    independent of any tiling algorithm."""
    T, Hq, d = q.shape
    Hkv = k.shape[1]
    assert Hq == Hkv * group_size
    sm_scale = 1.0 / math.sqrt(d)

    qf = q.detach().to(torch.float32).requires_grad_(True)
    kf = k.detach().to(torch.float32).requires_grad_(True)
    vf = v.detach().to(torch.float32).requires_grad_(True)

    # GQA: expand KV heads for the reference contraction (autograd sums the
    # group's contributions back into the shared KV head — exactly the dK/dV
    # semantics your dkv kernel implements with its group loop).
    kf_e = kf.repeat_interleave(group_size, dim=1)            # [T, Hq, d]
    vf_e = vf.repeat_interleave(group_size, dim=1)

    lse2 = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
    outs = []
    for s, e in zip(cuseq[:-1].tolist(), cuseq[1:].tolist()):
        L = e - s
        scores = torch.einsum("qhd,khd->hqk", qf[s:e], kf_e[s:e]) * sm_scale
        mask = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        lse2[:, s:e] = torch.logsumexp(scores, dim=-1) * LOG2E   # base-2, fp32
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vf_e[s:e]))
    o = torch.cat(outs, dim=0)

    o.backward(do.to(torch.float32))
    return (o.detach().to(DTYPE), lse2.detach(),
            qf.grad.to(DTYPE), kf.grad.to(DTYPE), vf.grad.to(DTYPE))


# ---------------------------------------------------------------------------
# Error-structure diagnostics (print the pattern, not just pass/fail)
# ---------------------------------------------------------------------------

def check(name, out, ref, cuseq, atol=2e-2, rtol=2e-2):
    out32, ref32 = out.to(torch.float32), ref.to(torch.float32)
    ok = torch.allclose(out32, ref32, atol=atol, rtol=rtol)
    err = (out32 - ref32).abs()
    print(f"  {name:>4}: {'PASS' if ok else 'FAIL':4}  max={err.max():.4e}  "
          f"mean={err.mean():.4e}")
    if not ok:
        # Spatial pattern along tokens -> diagnoses the bug class
        per_tok = err.flatten(1).amax(dim=1) if err.dim() > 1 else err
        bad = (per_tok > atol).nonzero().flatten()
        first, last = bad[0].item(), bad[-1].item()
        doc_of_first = (torch.searchsorted(cuseq, first, right=True) - 1).item()
        ratio = (out32 / ref32.clamp_min(1e-6)).flatten()
        med_ratio = ratio[ref32.flatten().abs() > 1e-3].median().item()
        print(f"        first bad token {first} (doc {doc_of_first}, "
              f"intra-doc pos {first - cuseq[doc_of_first].item()}), last {last}")
        print(f"        median out/ref ratio {med_ratio:.4f} "
              f"(≈1.4427 or 0.6931 → base-2/scale leak; uniform ≠1 → scale leak; "
              f"clean head then garbage → grid under-launch)")
    return ok


# ---------------------------------------------------------------------------
# Shape suite: adversarial doc layouts (misaligned boundaries are where
# block-skipping logic dies)
# ---------------------------------------------------------------------------

def make_case(lengths, Hq, Hkv, d):
    lengths = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    cuseq = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEVICE)
    cuseq[1:] = torch.cumsum(lengths, 0)
    T = int(cuseq[-1])
    q = torch.randn(T, Hq, d, device=DEVICE, dtype=DTYPE) * 0.5
    k = torch.randn(T, Hkv, d, device=DEVICE, dtype=DTYPE) * 0.5
    v = torch.randn(T, Hkv, d, device=DEVICE, dtype=DTYPE) * 0.5
    do = torch.randn(T, Hq, d, device=DEVICE, dtype=DTYPE) * 0.5
    return q, k, v, do, cuseq, int(lengths.max())


SHAPE_SUITE = [
    ("single-doc (reduces to vanilla)", [1024]),
    ("misaligned boundaries",           [1, 977, 130, 63, 640, 381]),
    ("many tiny docs",                  [1] * 17 + [3, 2, 5, 300]),
    ("uneven realistic",                [2048, 512, 1024, 777, 4096 - 2048 - 512 - 1024 - 777 + 3000]),
]


def run_correctness(Hq=12, Hkv=4, d=128):
    group_size = Hq // Hkv
    all_ok = True
    print(f"\n=== Correctness (Hq={Hq}, Hkv={Hkv}, d={d}, GQA group={group_size}) ===")
    for name, lengths in SHAPE_SUITE:
        q, k, v, do, cuseq, max_len = make_case(lengths, Hq, Hkv, d)
        print(f"\n[{name}]  T={q.shape[0]}, docs={len(lengths)}")

        o_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference_fwd_bwd(
            q, k, v, do, cuseq, group_size)

        # ---- forward ----
        o_tri, lse_tri = triton_intradoc_attention(q, k, v, cuseq, max_len,
                                                   group_size)
        assert lse_tri.dtype == torch.float32, "LSE must stay fp32 end-to-end"
        fwd_ok = check("o", o_tri, o_ref, cuseq)
        fwd_ok &= check("lse", lse_tri.T, lse_ref.T, cuseq, atol=1e-2, rtol=1e-3)

        # ---- backward (fed by the kernel's own o/lse, run TWICE:
        #      a pass-then-fail sequence means stale accumulator state) ----
        bwd_ok = True
        if fwd_ok:
            for trial in (1, 2):
                dq, dk, dv = flash_att_backward_split(
                    q, k, v, o_tri, do, lse_tri, cuseq, max_len,
                    group_size)
                t = f"(run {trial})"
                bwd_ok &= check(f"dq{t}", dq, dq_ref, cuseq)
                bwd_ok &= check(f"dk{t}", dk, dk_ref, cuseq)
                bwd_ok &= check(f"dv{t}", dv, dv_ref, cuseq)
        else:
            print("  bwd:  SKIPPED (forward failed — fix forward first)")
            bwd_ok = False
        all_ok &= fwd_ok and bwd_ok
    return all_ok


# ---------------------------------------------------------------------------
# Flex baseline: doc-causal attention, compiled — apples-to-apples
# ---------------------------------------------------------------------------

def build_flex_baseline(q, k, v, do, cuseq, group_size):
    T, Hq, d = q.shape
    doc_ids = torch.repeat_interleave(
        torch.arange(len(cuseq) - 1, device=DEVICE),
        (cuseq[1:] - cuseq[:-1]).long())

    def mask_mod(b, h, qi, ki):
        return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])

    block_mask = create_block_mask(mask_mod, B=None, H=None,
                                   Q_LEN=T, KV_LEN=T, device=DEVICE)

    # sm_86 has ~101 KB usable SMEM; Flex's default template configs assume
    # Hopper-class budgets and can demand >104 KB. Pin Ampere-sized tiles.
    # (fwd uses BLOCK_M/N; bwd uses M1/N1 for the dQ pass, M2/N2 for dK/dV.)
    kernel_options = {
        "BLOCK_M": 64, "BLOCK_N": 64,
        "BLOCK_M1": 32, "BLOCK_N1": 64,
        "BLOCK_M2": 64, "BLOCK_N2": 32,
    }

    def full(qq, kk, vv):
        qr = qq.transpose(0, 1).unsqueeze(0)                  # [1,Hq,T,d]
        kr = kk.transpose(0, 1).unsqueeze(0)                  # [1,Hkv,T,d]
        vr = vv.transpose(0, 1).unsqueeze(0)
        return flex_attention(qr, kr, vr, block_mask=block_mask,
                              enable_gqa=(group_size > 1),
                              kernel_options=kernel_options)

    full_c = torch.compile(full, dynamic=False)

    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    vg = v.detach().clone().requires_grad_(True)
    do_f = do.transpose(0, 1).unsqueeze(0).contiguous()

    def fwd():
        return full_c(qg, kg, vg)

    out = fwd()  # warm/compile
    out.backward(do_f, retain_graph=True)

    def bwd():
        qg.grad = kg.grad = vg.grad = None
        out.backward(do_f, retain_graph=True)

    return fwd, bwd


# ---------------------------------------------------------------------------
# Benchmark: real shapes, causal FLOP accounting over sum(L_i^2)
# ---------------------------------------------------------------------------

def run_benchmark(Hq=12, Hkv=4, d=128,
                  lengths=(4096, 2048, 2048, 1024, 1024, 777, 3471, 1998),
                  peak_tflops=155.0):
    group_size = Hq // Hkv
    q, k, v, do, cuseq, max_len = make_case(list(lengths), Hq, Hkv, d)
    T = q.shape[0]
    sum_L2 = float(sum(L * L for L in lengths))

    # Causal algorithmic FLOPs (2 FLOP/MAC, triangular halves the L^2):
    #   fwd:  QK^T + PV            -> 2 * (2 * Hq * d * L^2/2) = 2*Hq*d*sum_L2
    #   bwd:  5 matmuls algorithmic -> 2.5x fwd
    # Actual hardware work of the split-recompute backward: 7 matmuls/tile
    # (S and P recomputed in both dq and dkv kernels) -> 3.5x fwd. Report both;
    # they answer different questions (user-experienced speed vs engineering).
    flops_fwd = 2.0 * Hq * d * sum_L2
    flops_bwd_algo = 2.5 * flops_fwd
    flops_bwd_actual = 3.5 * flops_fwd

    print(f"\n=== Benchmark  T={T}, docs={len(lengths)}, Hq={Hq}, Hkv={Hkv}, "
          f"d={d}, bf16 (fp32 acc) ===")

    # warm (also triggers autotune outside the timed region)
    o_tri, lse_tri = triton_intradoc_attention(q, k, v, cuseq, max_len,
                                               group_size)
    _ = flash_att_backward_split(q, k, v, o_tri, do, lse_tri, cuseq, max_len,
                                 group_size)

    ms_fwd = triton.testing.do_bench(
        lambda: triton_intradoc_attention(q, k, v, cuseq, max_len,
                                          group_size))
    ms_bwd = triton.testing.do_bench(
        lambda: flash_att_backward_split(q, k, v, o_tri, do, lse_tri, cuseq,
                                         max_len, group_size))

    rows = [("Triton fwd", ms_fwd, flops_fwd, flops_fwd),
            ("Triton bwd", ms_bwd, flops_bwd_algo, flops_bwd_actual)]

    if HAS_FLEX:
        try:
            flex_fwd, flex_bwd = build_flex_baseline(q, k, v, do, cuseq,
                                                     group_size)
            ms_ffwd = triton.testing.do_bench(flex_fwd)
            ms_fbwd = triton.testing.do_bench(flex_bwd)
            rows += [("Flex fwd (attn)", ms_ffwd, flops_fwd, flops_fwd),
                     ("Flex bwd (attn)", ms_fbwd, flops_bwd_algo,
                      flops_bwd_algo)]
        except Exception as e:
            print(f"[flex baseline unavailable on this GPU/torch build: "
                  f"{type(e).__name__}] reporting Triton-only numbers.")

    print(f"\n{'impl':<22} | {'ms':>9} | {'TFLOP/s algo':>12} | "
          f"{'TFLOP/s actual':>14} | {'MFU%':>5}")
    print("-" * 76)
    for name, ms, fa, fx in rows:
        ta, tx = fa / ms / 1e9, fx / ms / 1e9
        print(f"{name:<22} | {ms:>9.4f} | {ta:>12.1f} | {tx:>14.1f} | "
              f"{100 * ta / peak_tflops:>5.1f}")
    print(f"\n(peak {peak_tflops:.0f} TFLOP/s bf16 dense; MFU uses algorithmic "
          f"count. Flex timings are plain attention, matching the kernels.)")


if __name__ == "__main__":
    ok = run_correctness()
    if ok:
        run_benchmark()
    else:
        print("\nRule zero: correctness failed — no benchmark numbers reported. "
              "Fix the FAILs above, re-run.")
