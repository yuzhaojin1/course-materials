import cupy as cp
from cupyx.profiler import benchmark
import argparse

def simulate_terminal_stats(N: int):
    """
    Original logic:
      - 100 steps per walk
      - first step is manually set to 0
      - full path is built with cumsum
      - only final value is used

    Since the first step is overwritten with 0, the terminal value is:
        100 + sum of the remaining 99 shocks

    So we only generate the 99 useful shocks and reduce them on the GPU.
    This minimizes memory use and avoids building the full random-walk path.
    """

    # Use float32 to reduce GPU memory pressure and improve throughput.
    # For large N, memory bandwidth is often a major bottleneck.
    steps = cp.random.normal(
        loc=0.0,
        scale=1.0,
        size=(N, 99),
        dtype=cp.float32
    )

    # Compute only terminal values, not the full path matrix.
    finish = cp.float32(100.0) + steps.sum(axis=1, dtype=cp.float32)

    # Keep the reductions on the GPU; only move final scalars back to CPU.
    average_finish = finish.mean(dtype=cp.float64)
    std_finish = finish.std(dtype=cp.float64)

    return average_finish, std_finish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10**6)
    args = parser.parse_args()

    N = args.n

    # Warm up GPU context so first-use overhead is not included in timing.
    _ = cp.random.normal(size=(1,), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    average_finish, std_finish = simulate_terminal_stats(N)
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)

    # Transfer only the two output scalars back to host memory.
    average_finish = float(average_finish.item())
    std_finish = float(std_finish.item())

    print(f"N = {N}")
    print(f"Average Finish: {average_finish}")
    print(f"Standard Deviation: {std_finish}")
    print(f"GPU Time (s): {elapsed_ms / 1000:.6f}")


if __name__ == "__main__":
    main()