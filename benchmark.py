import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from flashattention_triton import TritonAttention, triton_decode


def naive_decode_eager(Q, K_cache, V_cache, cache_len, softmax_scale=None):
    """
    Naive PyTorch eager decode baseline (Static Preallocated Cache):
    Slices cache up to cache_len, expands GQA, computes scaled dot-product attention manually.
    Materializes intermediate expanded KV, QK^T scores, and Softmax tensors.
    """
    BATCH_SIZE, NUM_HEADS, _, HEAD_DIM = Q.shape
    NUM_KV_HEADS = K_cache.shape[1]
    group_size = NUM_HEADS // NUM_KV_HEADS

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    # Slice valid cache
    K_valid = K_cache[:, :, :cache_len, :]  # (B, H_kv, S, D)
    V_valid = V_cache[:, :, :cache_len, :]  # (B, H_kv, S, D)

    # Repeat KV heads for GQA alignment
    if group_size > 1:
        K_valid = K_valid.repeat_interleave(group_size, dim=1)
        V_valid = V_valid.repeat_interleave(group_size, dim=1)

    # Q @ K^T -> (B, H, 1, S)
    scores = torch.matmul(Q, K_valid.transpose(-1, -2)) * softmax_scale
    probs = F.softmax(scores, dim=-1)

    # Probs @ V -> (B, H, 1, D)
    O = torch.matmul(probs, V_valid)
    return O


def naive_decode_cat(Q, K_history, V_history, K_new, V_new, softmax_scale=None):
    """
    Naive PyTorch eager decode baseline (Dynamic Reallocation via torch.cat):
    Simulates dynamic tensor growth without preallocated memory buffers.
    """
    BATCH_SIZE, NUM_HEADS, _, HEAD_DIM = Q.shape
    NUM_KV_HEADS = K_history.shape[1]
    group_size = NUM_HEADS // NUM_KV_HEADS

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    # Dynamically grow cache tensors
    K_full = torch.cat([K_history, K_new], dim=2)
    V_full = torch.cat([V_history, V_new], dim=2)

    # Repeat KV heads for GQA alignment
    if group_size > 1:
        K_full = K_full.repeat_interleave(group_size, dim=1)
        V_full = V_full.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(Q, K_full.transpose(-1, -2)) * softmax_scale
    probs = F.softmax(scores, dim=-1)
    O = torch.matmul(probs, V_full)
    return O


def run_benchmark_latency_memory(fn, warmup=10, rep=30):
    """
    Measures median wall-clock latency (ms) and peak INCREMENTAL CUDA memory (MB) over N runs.
    Ensures memory measurements represent only the temporary allocation delta of the function.
    """
    # 1. Warmup runs (autotune / compilation overhead discarded)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 2. Incremental Peak Memory Measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    incremental_mem_bytes = max(0, peak_mem_bytes - mem_before)
    peak_mem_mb = incremental_mem_bytes / (1024 * 1024)

    # 3. Latency Measurement (Median of N repetitions)
    timings = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # ms

    timings.sort()
    median_latency = timings[len(timings) // 2]
    return median_latency, peak_mem_mb


# =====================================================================
# Benchmark 1: Prefill Regime (_attn_fwd vs F.scaled_dot_product_attention)
# =====================================================================
def benchmark_prefill():
    print("\n--- Running Prefill Benchmark ---")
    torch.manual_seed(42)

    BATCH_SIZE = 2
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    group_size = NUM_HEADS // NUM_KV_HEADS

    seq_lengths = [2048, 4096, 8192]
    results = {
        "seq_len": seq_lengths,
        "triton_lat": [], "sdpa_lat": [],
        "triton_mem": [], "sdpa_mem": []
    }

    for seq_len in seq_lengths:
        print(f"Benchmarking Prefill SEQ_LEN={seq_len}...")
        dtype = torch.float16
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

        Q = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM), device="cuda", dtype=dtype)
        K = torch.randn((BATCH_SIZE, NUM_KV_HEADS, seq_len, HEAD_DIM), device="cuda", dtype=dtype)
        V = torch.randn((BATCH_SIZE, NUM_KV_HEADS, seq_len, HEAD_DIM), device="cuda", dtype=dtype)

        # PyTorch SDPA setup (expand GQA heads)
        K_sdpa = K.repeat_interleave(group_size, dim=1)
        V_sdpa = V.repeat_interleave(group_size, dim=1)

        # Functions to benchmark
        fn_triton = lambda: TritonAttention.apply(Q, K, V, True, softmax_scale)
        fn_sdpa = lambda: F.scaled_dot_product_attention(Q, K_sdpa, V_sdpa, is_causal=True, scale=softmax_scale)

        lat_triton, mem_triton = run_benchmark_latency_memory(fn_triton)
        lat_sdpa, mem_sdpa = run_benchmark_latency_memory(fn_sdpa)

        results["triton_lat"].append(lat_triton)
        results["sdpa_lat"].append(lat_sdpa)
        results["triton_mem"].append(mem_triton)
        results["sdpa_mem"].append(mem_sdpa)

        print(f"  [Triton] Latency: {lat_triton:.2f} ms | Peak Incremental Mem: {mem_triton:.2f} MB")
        print(f"  [SDPA]   Latency: {lat_sdpa:.2f} ms | Peak Incremental Mem: {mem_sdpa:.2f} MB")

    # Plotting Prefill Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(seq_lengths, results["triton_lat"], label="Triton _attn_fwd", marker="o")
    ax1.plot(seq_lengths, results["sdpa_lat"], label="PyTorch SDPA", marker="s", linestyle="--")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Median Latency (ms)")
    ax1.set_title("Prefill Latency Comparison (Causal)")
    ax1.set_xticks(seq_lengths)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(seq_lengths, results["triton_mem"], label="Triton _attn_fwd", marker="o")
    ax2.plot(seq_lengths, results["sdpa_mem"], label="PyTorch SDPA", marker="s", linestyle="--")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Incremental Memory (MB)")
    ax2.set_title("Prefill Intermediate Memory Footprint")
    ax2.set_xticks(seq_lengths)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("prefill_benchmark.png", dpi=300)
    plt.close()
    print("Prefill benchmark complete. Saved plot to 'prefill_benchmark.png'.")


# =====================================================================
# Benchmark 2: Decode Regime (_attn_decode vs PyTorch Eager Loop)
# =====================================================================
def benchmark_decode():
    print("\n--- Running Decode Benchmark ---")
    torch.manual_seed(42)

    BATCH_SIZE = 2
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    MAX_CACHE_LEN = 8192
    dtype = torch.float16
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    cache_lengths = [128, 512, 2048, 8192]
    results = {
        "cache_len": cache_lengths,
        "triton_lat": [], "eager_slice_lat": [], "eager_cat_lat": [],
        "triton_mem": [], "eager_slice_mem": [], "eager_cat_mem": []
    }

    # Preallocated Static Buffer for Triton and Eager Slicing
    K_cache = torch.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), device="cuda", dtype=dtype)
    V_cache = torch.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), device="cuda", dtype=dtype)
    K_cache.normal_(0.0, 0.5)
    V_cache.normal_(0.0, 0.5)

    for cache_len in cache_lengths:
        print(f"Benchmarking Decode CACHE_LEN={cache_len}...")
        Q = torch.randn((BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), device="cuda", dtype=dtype)

        # 1. Benchmark Triton Decode (Static Cache)
        fn_triton = lambda: triton_decode(Q, K_cache, V_cache, cache_len=cache_len, softmax_scale=softmax_scale)
        lat_triton, mem_triton = run_benchmark_latency_memory(fn_triton)

        # 2. Benchmark PyTorch Eager Slicing (Static Cache)
        fn_eager_slice = lambda: naive_decode_eager(Q, K_cache, V_cache, cache_len=cache_len, softmax_scale=softmax_scale)
        lat_slice, mem_slice = run_benchmark_latency_memory(fn_eager_slice)

        # 3. Benchmark PyTorch Eager Dynamic Concatenation
        # Scope variables locally to prevent memory pollution during Triton and Slice benchmark runs
        K_history = K_cache[:, :, :cache_len - 1, :].clone()
        V_history = V_cache[:, :, :cache_len - 1, :].clone()
        K_new = K_cache[:, :, cache_len - 1:cache_len, :].clone()
        V_new = V_cache[:, :, cache_len - 1:cache_len, :].clone()

        fn_eager_cat = lambda: naive_decode_cat(Q, K_history, V_history, K_new, V_new, softmax_scale=softmax_scale)
        lat_cat, mem_cat = run_benchmark_latency_memory(fn_eager_cat)

        # Clean up dynamic allocations immediately after execution
        del K_history, V_history, K_new, V_new
        torch.cuda.empty_cache()

        results["triton_lat"].append(lat_triton)
        results["eager_slice_lat"].append(lat_slice)
        results["eager_cat_lat"].append(lat_cat)

        results["triton_mem"].append(mem_triton)
        results["eager_slice_mem"].append(mem_slice)
        results["eager_cat_mem"].append(mem_cat)

        print(f"  [Triton Decode] Latency: {lat_triton:.3f} ms | Incremental Peak Mem: {mem_triton:.2f} MB")
        print(f"  [Eager Slice]   Latency: {lat_slice:.3f} ms | Incremental Peak Mem: {mem_slice:.2f} MB")
        print(f"  [Eager Dynamic] Latency: {lat_cat:.3f} ms | Incremental Peak Mem: {mem_cat:.2f} MB")

    # Plotting Decode Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency Plot (Log-Scale X-Axis)
    ax1.plot(cache_lengths, results["triton_lat"], label="Triton _attn_decode (Fused Streaming)", marker="o", color="green")
    ax1.plot(cache_lengths, results["eager_slice_lat"], label="PyTorch Eager (Static Slice)", marker="s", linestyle="--", color="orange")
    ax1.plot(cache_lengths, results["eager_cat_lat"], label="PyTorch Eager (torch.cat Reallocation)", marker="^", linestyle=":", color="red")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Cache Length (tokens, log2 scale)")
    ax1.set_ylabel("Latency per Token (ms)")
    ax1.set_title("Decode Latency vs. Cache Length")
    ax1.set_xticks(cache_lengths)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Peak Memory Plot (Log-Scale X-Axis)
    ax2.plot(cache_lengths, results["triton_mem"], label="Triton Static Cache (O(1) Intermediate)", marker="o", color="green")
    ax2.plot(cache_lengths, results["eager_slice_mem"], label="PyTorch Eager (Materializes Expanded KV)", marker="s", linestyle="--", color="orange")
    ax2.plot(cache_lengths, results["eager_cat_mem"], label="PyTorch Eager (Dynamic Reallocation)", marker="^", linestyle=":", color="red")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Cache Length (tokens, log2 scale)")
    ax2.set_ylabel("Peak Incremental Memory (MB)")
    ax2.set_title("Decode Peak Intermediate Memory Allocation")
    ax2.set_xticks(cache_lengths)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("decode_benchmark.png", dpi=300)
    plt.close()
    print("Decode benchmark complete. Saved plot to 'decode_benchmark.png'.")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA environment required for benchmarking."
    print(f"Running benchmarks on GPU: {torch.cuda.get_device_name(0)}")
    
    benchmark_prefill()
    benchmark_decode()