import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
import argparse
import torch

import logging
@dataclass
class InputCase:
    inputs: tuple
    name: str


class KernelBenchmarkInfer(ABC):
    @abstractmethod
    def infer(self, inputs: tuple) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def kernel_name(self):
        """
        return kernel name(such as 'searchsorted_bk_kernel')
        """
        raise NotImplementedError

    @abstractmethod
    def implement_name(self):
        """
        return implement name(such as 'torch', 'triton', 'cutedsl')
        """
        raise NotImplementedError


IMPLEMENT_LIST = ["torch", "triton", "cutedsl"]


class KernelBenchmark:
    @staticmethod
    def get_gpu_name():
        """
        获取当前主 GPU 名称，如 'NVIDIA GeForce RTX 3080 Ti'
        """
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            gpu_names = result.stdout.strip().split("\n")
            return gpu_names[0] if gpu_names else None
        except Exception:
            return None

    def __init__(
        self, kernel_impl: list[KernelBenchmarkInfer], input_cases: list[InputCase]
    ):
        # Automatically use the implementation with implement_name=='torch' as the reference
        for kernel in kernel_impl:
            assert (
                kernel.implement_name() in IMPLEMENT_LIST
            ), f"implement name {kernel.implement_name()} not in {IMPLEMENT_LIST}"
        self.torch_impl = next(
            (impl for impl in kernel_impl if impl.implement_name() == "torch"), None
        )
        if self.torch_impl is None:
            logging.warning(
                "No implementation with implement_name=='torch' found. Accuracy check skip."
            )
        other_impls = [impl for impl in kernel_impl if impl.implement_name() != "torch"]
        self.kernel_impl = [self.torch_impl] + other_impls
        self.input_cases = input_cases

    def _check_accuracy(self):
        """
        Compare the outputs of all implementations for all input_cases and return error statistics.
        The first implementation is used as the reference by default.
        Returns: List[Dict], each dict contains input_case, outputs of each implementation, and errors
        """
        results = []
        for input_case in self.input_cases:
            ref_impl = self.kernel_impl[0]
            ref_out = ref_impl.infer(input_case.inputs)
            case_result = {
                "input_case": input_case.name,
                # "reference": {"name": ref_impl.implement_name(), "output": ref_out},
                "accuracy": [],
            }
            for impl in self.kernel_impl[1:]:
                out = impl.infer(input_case.inputs)
                # Error calculation: supports numpy/tensor, default max_abs
                try:
                    import numpy as np

                    err = np.max(
                        np.abs(
                            np.array(ref_out.detach().cpu())
                            - np.array(out.detach().cpu())
                        )
                    ).item()
                except Exception:
                    err = None
                case_result["accuracy"].append(
                    {"name": impl.implement_name(), "diff": err}
                )
            results.append(case_result)
        return results

    def _benchmark(self, repeat=10, warmup=5):
        """
        Benchmark the performance (mean time, std deviation) of all implementations for all input_cases.
        Returns: List[Dict], each dict contains input_case and performance of each implementation
        """
        import time

        results = []
        for input_case in self.input_cases:
            case_result = {"input_case": input_case.name, "perf": []}
            for impl in self.kernel_impl:
                if impl is None:
                    continue
                for _ in range(warmup):
                    _ = impl.infer(input_case.inputs)
                start = time.time()
                for _ in range(repeat):
                    _ = impl.infer(input_case.inputs)
                torch.cuda.synchronize()
                mean = (time.time() - start) * 1e3 / repeat
                case_result["perf"].append(
                    {
                        "name": impl.implement_name(),
                        "mean": f"{mean} ms",
                    }
                )
            results.append(case_result)
        return results

    def run_all(self, repeat=10, warmup=5):
        """
        Run both accuracy and performance tests for all input_cases and return the results.
        """
        if self.torch_impl is None:
            logging.warning(
                "No reference implementation found. Only performance benchmark will be run."
            )
            accuracy_results = []
        else:
            accuracy_results = self._check_accuracy()
        perf_results = self._benchmark(repeat=repeat, warmup=warmup)
        return {"accuracy": accuracy_results, "performance": perf_results}


def save_as_md(results, save_path):
    """
    Save the benchmark results as a markdown file.
    """
    with open(save_path, "w") as f:
        f.write("``` json\n")
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
        f.write("\n```\n")


def benchmark(
    kernel_impls: list[KernelBenchmarkInfer],
    input_cases: list[InputCase],
    save_path: str = None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--impl', type=str, default="all", help='Number of repetitions for benchmarking')
    args = parser.parse_args()
    if args.impl != "all":
        kernel_impls = [impl for impl in kernel_impls if impl.implement_name() == args.impl]
        assert len(kernel_impls) > 0, f"No implementation found for {args.impl}"
    bench = KernelBenchmark(
        kernel_impl=kernel_impls,
        input_cases=input_cases,
    )
    results = bench.run_all(repeat=10, warmup=5)
    pretty_json_str = json.dumps(results, indent=4, ensure_ascii=False)
    print(pretty_json_str)
    if save_path:
        # get gpu name
        gpu_name = KernelBenchmark.get_gpu_name()
        gpu_name = gpu_name.replace(" ", "_") if gpu_name else "unknown_gpu"
        save_path = os.path.join(save_path, gpu_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        kernel_name = kernel_impls[0].kernel_name()
        save_path = os.path.join(save_path, f"{kernel_name}_benchmark.md")
        save_as_md(results, save_path)
