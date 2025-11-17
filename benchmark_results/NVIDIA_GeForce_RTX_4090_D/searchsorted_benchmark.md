``` json
{
    "accuracy": [],
    "performance": [
        {
            "input_case": "32x1024_k512_torch.float32",
            "perf": [
                {
                    "name": "triton",
                    "mean": "882.8835487365723 ms"
                }
            ]
        },
        {
            "input_case": "64x2048_k1024_torch.float32",
            "perf": [
                {
                    "name": "triton",
                    "mean": "865.2434587478638 ms"
                }
            ]
        },
        {
            "input_case": "128x4096_k2048_torch.float32",
            "perf": [
                {
                    "name": "triton",
                    "mean": "910.8263731002808 ms"
                }
            ]
        },
        {
            "input_case": "256x8192_k4096_torch.float32",
            "perf": [
                {
                    "name": "triton",
                    "mean": "920.7778453826904 ms"
                }
            ]
        }
    ]
}
```
