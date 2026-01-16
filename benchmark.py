import torch
import numpy as np
import flash_attention_ext as ext

def benchmark(fn, Q, K, V, runs=100, warmup=10):
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        out = fn(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return out, np.array(times)

def main():
    B, H, N, D = 1, 8, 512, 64
    Q = torch.randn(B, H, N, D, device="cuda")
    K = torch.randn(B, H, N, D, device="cuda")
    V = torch.randn(B, H, N, D, device="cuda")

    fns = {
        "PyTorch SDPA": torch.nn.functional.scaled_dot_product_attention,
        "Naive CUDA": ext.naive_forward,
        "Optimized CUDA": ext.optimized_forward,
        "Flash CUDA": ext.flash_forward,
    }

    outputs = {}
    timings = {}

    for name, fn in fns.items():
        out, t = benchmark(fn, Q, K, V)
        outputs[name] = out
        timings[name] = t
        print(f"{name:15s} mean: {t.mean():.3f} ms")

    # Correctness check vs PyTorch
    ref = outputs["PyTorch SDPA"]
    for name, out in outputs.items():
        diff = (ref - out).abs().max().item()
        print(f"Max diff vs PyTorch ({name}): {diff:.6f}")

if __name__ == "__main__":
    main()
