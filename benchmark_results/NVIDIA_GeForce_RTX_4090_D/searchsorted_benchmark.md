``` json
{
    "accuracy": [
        {
            "input_case": "32x1024_k512_torch.float32",
            "accuracy": []
        },
        {
            "input_case": "64x2048_k1024_torch.float32",
            "accuracy": []
        },
        {
            "input_case": "128x4096_k2048_torch.float32",
            "accuracy": []
        },
        {
            "input_case": "256x8192_k4096_torch.float32",
            "accuracy": []
        }
    ],
    "performance": [
        {
            "input_case": "32x1024_k512_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.008106231689453125 ms"
                }
            ]
        },
        {
            "input_case": "64x2048_k1024_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.004792213439941406 ms"
                }
            ]
        },
        {
            "input_case": "128x4096_k2048_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.01430511474609375 ms"
                }
            ]
        },
        {
            "input_case": "256x8192_k4096_torch.float32",
            "perf": [
                {
                    "name": "torch",
                    "mean": "0.04820823669433594 ms"
                }
            ]
        }
    ]
}
```
