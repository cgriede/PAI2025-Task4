import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Quick tensor test (should be on GPU)
    x = torch.rand(3, 3).cuda()
    print(f"Sample tensor on GPU: {x.device}")
else:
    print("Fallback to CPU—check drivers!")