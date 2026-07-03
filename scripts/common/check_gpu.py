# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import sys

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA devices visible", file=sys.stderr)
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    failed = []

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        try:
            device = torch.device(f"cuda:{i}")
            a = torch.randn(4096, 4096, device=device)
            b = torch.randn(4096, 4096, device=device)
            c = a @ b
            torch.cuda.synchronize(device)

            if not torch.isfinite(c).all():
                raise RuntimeError("matmul produced non-finite values")

            mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            print(f"GPU {i} ({name}): OK, {mem_gb:.2f} GiB allocated")
        except Exception as e:
            failed.append(i)
            print(f"GPU {i} ({name}): FAILED, {e}", file=sys.stderr)

    if failed:
        print(f"{len(failed)}/{num_gpus} GPU(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)

    print(f"all {num_gpus} GPU(s) OK")


if __name__ == "__main__":
    main()
