import torch

print("=" * 50)
print("PyTorch GPU Diagnostic")
print("=" * 50)

# PyTorch version
print("PyTorch version:", torch.__version__)

# CUDA availability
print("CUDA available:", torch.cuda.is_available())

# CUDA version compiled into PyTorch
print("PyTorch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():

    print("\nGPU detected!")

    print("Number of GPUs:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print("Name:", torch.cuda.get_device_name(i))
        print("Capability:", torch.cuda.get_device_capability(i))

        props = torch.cuda.get_device_properties(i)

        print(
            "VRAM:",
            round(props.total_memory / (1024 ** 3), 2),
            "GB"
        )

    # Actually perform a calculation on GPU
    device = torch.device("cuda")

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    z = x @ y

    torch.cuda.synchronize()

    print("\nGPU computation test: SUCCESS")
    print("Tensor device:", z.device)

    # Memory information
    print(
        "Allocated GPU memory:",
        round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "MB"
    )

    print(
        "Reserved GPU memory:",
        round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "MB"
    )

else:

    print("\nNO CUDA GPU AVAILABLE TO PYTORCH")
    print("\nPossible reasons:")
    print("1. NVIDIA driver is not installed correctly")
    print("2. You installed the CPU-only version of PyTorch")
    print("3. CUDA/PyTorch installation is incorrect")
    print("4. Your Python environment is different from the one you expect")

print("\n" + "=" * 50)