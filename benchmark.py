"""
Benchmark all five attention kernel stages against PyTorch SDPA.

Usage:
    python benchmark.py              # Quick comparison at N=512
    python benchmark.py --scale      # Scaling test across N=256..16384
"""
import argparse
import torch
import numpy as np
import flash_attention_ext as ext


def benchmark(fn, Q, K, V, runs=100, warmup=10):
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        out = fn(Q, K, V)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return out, np.array(times)


def measure_memory(fn, Q, K, V):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = fn(Q, K, V)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return out, (peak - before) / (1024 ** 2)


def run_quick():
    """Compare all 5 kernels at N=512."""
    B, H, N, D = 1, 8, 512, 64
    Q = torch.randn(B, H, N, D, device="cuda")
    K = torch.randn(B, H, N, D, device="cuda")
    V = torch.randn(B, H, N, D, device="cuda")

    fns = {
        "PyTorch SDPA":     torch.nn.functional.scaled_dot_product_attention,
        "Naive":            ext.naive_forward,
        "Optimized":        ext.optimized_forward,
        "Flash v1":         ext.flash_forward,
        "Flash v2 (1-warp)": ext.flash_v2_forward,
        "Flash v3 (4-warp)": ext.flash_v3_forward,
    }

    print(f"Config: B={B}, H={H}, N={N}, D={D}  (100 runs, 10 warmup)")
    print()
    print(f"{'Kernel':>20}  {'mean':>9}  {'median':>9}  {'min':>9}  {'max err':>9}")
    print("-" * 65)

    outputs = {}
    for name, fn in fns.items():
        out, t = benchmark(fn, Q, K, V)
        outputs[name] = out
        print(f"{name:>20}  {t.mean():>8.3f}ms  {np.median(t):>8.3f}ms  {t.min():>8.3f}ms", end="")
        if name == "PyTorch SDPA":
            print(f"  {'--':>9}")
        else:
            diff = (outputs["PyTorch SDPA"] - out).abs().max().item()
            print(f"  {diff:>9.1e}")


def run_scaling():
    """Scaling test across sequence lengths."""
    B, H, D = 1, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 95)
    print("Scaling Benchmark")
    print(f"Config: B={B}, H={H}, D={D}")
    print("=" * 95)
    print()
    print(f"{'N':>6} | {'Kernel':>20} | {'Median ms':>10} | {'Min ms':>8} | {'Peak MB':>8} | {'Correct':>7}")
    print("-" * 95)

    for N in seq_lengths:
        if N <= 1024:
            runs, warmup = 100, 20
        elif N <= 4096:
            runs, warmup = 50, 10
        else:
            runs, warmup = 20, 5

        Q = torch.randn(B, H, N, D, device="cuda")
        K = torch.randn(B, H, N, D, device="cuda")
        V = torch.randn(B, H, N, D, device="cuda")

        # PyTorch SDPA (reference)
        try:
            ref, t = benchmark(torch.nn.functional.scaled_dot_product_attention, Q, K, V, runs, warmup)
            _, mem = measure_memory(torch.nn.functional.scaled_dot_product_attention, Q, K, V)
            print(f"{N:>6} | {'PyTorch SDPA':>20} | {np.median(t):>10.3f} | {t.min():>8.3f} | {mem:>8.1f} | {'ref':>7}")
        except RuntimeError:
            print(f"{N:>6} | {'PyTorch SDPA':>20} | {'OOM':>10} |")
            ref = None

        # Naive (hard limit: N <= 1024 threads/block)
        if N <= 1024:
            try:
                out, t = benchmark(ext.naive_forward, Q, K, V, runs, warmup)
                _, mem = measure_memory(ext.naive_forward, Q, K, V)
                diff = (ref - out).abs().max().item() if ref is not None else -1
                ok = "yes" if diff < 1e-3 else f"NO {diff:.0e}"
                print(f"{'':>6} | {'Naive':>20} | {np.median(t):>10.3f} | {t.min():>8.3f} | {mem:>8.1f} | {ok:>7}")
            except RuntimeError:
                print(f"{'':>6} | {'Naive':>20} | {'OOM':>10} |")
        else:
            smb = B * H * N * N * 4 / (1024 ** 2)
            print(f"{'':>6} | {'Naive':>20} | {'--':>10} | {'--':>8} | {smb:>7.0f}* | {'skip':>7}")

        # Flash v2
        try:
            out, t = benchmark(ext.flash_v2_forward, Q, K, V, runs, warmup)
            _, mem = measure_memory(ext.flash_v2_forward, Q, K, V)
            diff = (ref - out).abs().max().item() if ref is not None else -1
            ok = "yes" if diff < 1e-3 else f"NO {diff:.0e}"
            print(f"{'':>6} | {'Flash v2 (1-warp)':>20} | {np.median(t):>10.3f} | {t.min():>8.3f} | {mem:>8.1f} | {ok:>7}")
        except RuntimeError:
            print(f"{'':>6} | {'Flash v2 (1-warp)':>20} | {'OOM':>10} |")

        # Flash v3
        try:
            out, t = benchmark(ext.flash_v3_forward, Q, K, V, runs, warmup)
            _, mem = measure_memory(ext.flash_v3_forward, Q, K, V)
            diff = (ref - out).abs().max().item() if ref is not None else -1
            ok = "yes" if diff < 1e-3 else f"NO {diff:.0e}"
            print(f"{'':>6} | {'Flash v3 (4-warp)':>20} | {np.median(t):>10.3f} | {t.min():>8.3f} | {mem:>8.1f} | {ok:>7}")
        except RuntimeError:
            print(f"{'':>6} | {'Flash v3 (4-warp)':>20} | {'OOM':>10} |")

        print()
        del Q, K, V
        torch.cuda.empty_cache()

    print("=" * 95)
    print("* Naive launches N threads/block (CUDA limit = 1024), so it cannot run for seq len N > 1024.")
    print("  Memory marked * is theoretical score matrix size (B*H*N*N*4 bytes).")
    print("  Flash v2/v3 never allocate the N*N matrix -- memory stays O(N*D), not O(N^2).")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", action="store_true", help="Run scaling benchmark across N=256..16384")
    args = parser.parse_args()

    if args.scale:
        run_scaling()
    else:
        run_quick()
