import triton
import triton.language as tl
import torch

BLOCK_K = 128


@triton.jit
def searchsorted_bk_kernel(
    sorted_ptr,
    q_ptr,
    out_ptr,
    B,
    DIM,
    K,
    s_sorted_b,
    s_sorted_dim,
    s_q_b,
    s_q_k,
    s_out_b,
    s_out_k,
    side: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # pid = batch index
    pid = tl.program_id(0)
    if pid >= B:
        return

    # block index along k dimension
    kid = tl.program_id(1)
    offs = kid * BLOCK + tl.arange(0, BLOCK)
    mask_k = offs < K

    # base pointers
    sorted_row = sorted_ptr + pid * s_sorted_b
    q_row = q_ptr + pid * s_q_b
    out_row = out_ptr + pid * s_out_b

    # load queries
    q_ptrs = q_row + offs * s_q_k
    q = tl.load(q_ptrs, mask=mask_k, other=0.0)

    # initialize binary search bounds
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], DIM, dtype=tl.int32)

    # active mask tracker
    active = tl.zeros([1], dtype=tl.int32)

    # log2(2^32)=32 rounds, safe upper bound
    for _ in range(32):
        cond = lo < hi
        cond_i32 = cond.to(tl.int32)
        # accumulate activity
        active = active | tl.sum(cond_i32, axis=0)

        mid = (lo + hi) >> 1

        # load sorted[mid]
        sorted_ptrs = sorted_row + mid * s_sorted_dim
        sv = tl.load(sorted_ptrs, mask=cond & mask_k, other=0.0)

        if side == 0:  # left
            ge = sv >= q
            hi = tl.where(cond & ge, mid, hi)
            lo = tl.where(cond & (~ge), mid + 1, lo)
        else:  # right
            gt = sv > q
            hi = tl.where(cond & gt, mid, hi)
            lo = tl.where(cond & (~gt), mid + 1, lo)

    # write output
    out_ptrs = out_row + offs * s_out_k
    tl.store(out_ptrs, lo, mask=mask_k)


# ---------------------------- Python 封装 ----------------------------


def triton_searchsorted(sorted_seq: torch.Tensor, q: torch.Tensor, side="left"):
    """
    sorted_seq: [B, dim]
    q:          [B, k]
    return:     [B, k], dtype=int32
    """
    assert sorted_seq.ndim == 2
    assert q.ndim == 2
    B, dim = sorted_seq.shape
    B2, k = q.shape
    assert B == B2

    sorted_seq = sorted_seq.contiguous()
    q = q.contiguous()

    out = torch.empty_like(q, dtype=torch.int32)

    grid = (B, (k + BLOCK_K - 1) // BLOCK_K)

    searchsorted_bk_kernel[grid](
        sorted_seq,
        q,
        out,
        B,
        dim,
        k,
        sorted_seq.stride(0),
        sorted_seq.stride(1),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        0 if side == "left" else 1,
        BLOCK_K,
        num_warps=4,
    )
    return out
