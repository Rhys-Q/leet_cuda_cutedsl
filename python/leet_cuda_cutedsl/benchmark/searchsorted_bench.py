import time
import numpy as np
import torch
import json
from leet_cuda_cutedsl.kernels.searchsorted.triton import triton_searchsorted
from leet_cuda_cutedsl.benchmark.common import (
    KernelBenchmark,
    KernelBenchmarkInfer,
    InputCase,
    benchmark,
)


def create_input_case_searchsorted(B, dim, k, dtype=torch.float32, device="cuda"):
    sorted_seq = torch.sort(
        torch.randn(B, dim, dtype=dtype, device=device), dim=1
    ).values
    q = torch.randn(B, k, dtype=dtype, device=device)
    name = f"{B}x{dim}_k{k}_{dtype}"
    return InputCase(
        name=name,
        inputs=(sorted_seq, q),
    )


test_cases = [
    create_input_case_searchsorted(B=32, dim=1024, k=512),
    create_input_case_searchsorted(B=64, dim=2048, k=1024),
    create_input_case_searchsorted(B=128, dim=4096, k=2048),
    create_input_case_searchsorted(B=256, dim=8192, k=4096),
]


class SearchSortedTorch(KernelBenchmarkInfer):
    def infer(self, inputs: tuple) -> tuple:
        sorted_seq, q = inputs
        out = torch.searchsorted(sorted_seq, q, right=False)
        return out

    def kernel_name(self):
        return "searchsorted"

    def implement_name(self):
        return "torch"


class SearchSortedTriton(KernelBenchmarkInfer):
    def infer(self, inputs: tuple) -> tuple:
        sorted_seq, q = inputs
        out = triton_searchsorted(sorted_seq, q, side="left")
        return out

    def kernel_name(self):
        return "searchsorted"

    def implement_name(self):
        return "triton"


if __name__ == "__main__":
    benchmark(
        kernel_impls=[SearchSortedTorch(), SearchSortedTriton()],
        input_cases=test_cases,
        save_path="./benchmark_results",
    )
