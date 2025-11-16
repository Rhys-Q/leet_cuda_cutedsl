``` json
{
    "accuracy": [
        {
            "input_case": "32x1024_k512_torch.float32",
            "accuracy": [
                {
                    "name": "triton",
                    "diff": 0
                }
            ]
        },
        {
            "input_case": "64x2048_k1024_torch.float32",
            "accuracy": [
                {
                    "name": "triton",
                    "diff": 0
                }
            ]
        },
        {
            "input_case": "128x4096_k2048_torch.float32",
            "accuracy": [
                {
                    "name": "triton",
                    "diff": 0
                }
            ]
        },
        {
            "input_case": "256x8192_k4096_torch.float32",
            "accuracy": [
                {
                    "name": "triton",
                    "diff": 0
                }
            ]
        }
    ],
    "performance": [
        {
            "input_case": "32x1024_k512_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.03662109375 ms"
                },
                {
                    "name": "triton",
                    "mean": "0.06072521209716797 ms"
                }
            ]
        },
        {
            "input_case": "64x2048_k1024_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.016999244689941406 ms"
                },
                {
                    "name": "triton",
                    "mean": "0.05230903625488281 ms"
                }
            ]
        },
        {
            "input_case": "128x4096_k2048_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.023674964904785156 ms"
                },
                {
                    "name": "triton",
                    "mean": "0.05536079406738281 ms"
                }
            ]
        },
        {
            "input_case": "256x8192_k4096_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.116729736328125 ms"
                },
                {
                    "name": "triton",
                    "mean": "0.1703023910522461 ms"
                }
            ]
        }
    ]
}
```
