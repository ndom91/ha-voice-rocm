# Wyoming ROCm Setup for Ubuntu LXC

## Prerequisites

You're running this in Ubuntu 24.04 LXC on Proxmox with ROCm 7.1.1 target.

## 1. LXC Container GPU Passthrough

Add to your LXC container config (`/etc/pve/lxc/<VMID>.conf`):

```ini
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 242:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=dir
```

Then restart the container.

## 2. Install ROCm in Container

```bash
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
sudo apt update
sudo amdgpu-install -y --usecase=rocm --no-dkms
```

## 3. Configure User Permissions

```bash
sudo usermod -a -G render,video $USER
# Log out and back in for group changes to take effect
```

Verify:

```bash
rocminfo
```

## 4. Build flash-attn Wheel (on Proxmox Host)

```bash
# On Proxmox host (not in LXC)
cd /path/to/granite

HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIP_PLATFORM=amd MAX_JOBS=1 \
  python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation -w ./

# Copy wheel into LXC or mount directory
```

See [`granite/BUILD_WHEEL.md`](./granite/BUILD_WHEEL.md).

## 5. Set GPU Architecture

In `.env`:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1  # For Radeon 8060S (RDNA 3.5)
```

Verify gfx version:

```bash
rocminfo | grep gfx
```

## 6. Build & Run Services

```bash
# Build granite (requires flash-attn wheel)
docker compose build wyoming-granite

# Run
docker compose up -d wyoming-granite

# Verify
docker compose logs -f wyoming-granite
rocminfo  # Should show GPU in use
```

## Troubleshooting

**GPU not visible in container:**
- Check LXC config has `/dev/dri` and `/dev/kfd` mounted
- Verify user is in `render,video` groups
- Run `rocminfo` in container

**flash-attn build fails:**
- Build wheel on **host**, not in LXC
- Use `MAX_JOBS=1` to avoid memory issues
- Check `/opt/rocm` path exists

**Granite segfault (Exit 139):**
- Ensure `attn_implementation="flash_attention_2"` is set (check `granite_handler.py`)
- Use `torch.bfloat16` dtype for RDNA 3.5
