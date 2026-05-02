# Build flash-attn Wheel for Granite

Run this **on the host** (not in LXC container), after installing ROCm 7.1.1:

```bash
# Install host ROCm environment (if not already done)
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
sudo apt update
sudo amdgpu-install -y --usecase=rocm --no-dkms

# Install torch with ROCm support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install setuptools wheel psutil

# Build wheel in this directory
cd /path/to/granite
HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIP_PLATFORM=amd MAX_JOBS=1 \
  python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation -w ./

# Verify wheel created
ls -la flash_attn*.whl
```

This wheel will be COPY'd during Docker build.
