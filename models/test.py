import torch

# 1. 检查版本
print(f"PyTorch Version: {torch.__version__}")

# 2. 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 3. 检查 PyTorch 内部认为的 CUDA 版本 (应该是 11.6)
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    # 4. 检查您的 GPU
    print(f"Device Name: {torch.cuda.get_device_name(0)}")