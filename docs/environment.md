# Environment

## Hardware

- Device: NVIDIA Jetson Orin NX 16GB
- Device tree name: `NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super`
- Unified memory: 16GB（Linux 可见约 15GiB）
- Storage: 256GB NVMe
- NVMe model: `LOBOROBOT`
- Root filesystem: `/dev/nvme0n1p1`
- Cooling: active cooling + aluminum enclosure

## Software

- Ubuntu: 22.04.5 LTS
- Jetson Linux / L4T: R36.4.3
- Kernel: 5.15.148-tegra
- Architecture: aarch64
- CUDA: 12.6
- cuDNN: 9.3
- TensorRT: 10.3
- GCC / G++: 11.4
- CMake: 3.22.1
- `trtexec`: `/usr/src/tensorrt/bin/trtexec`

## Power

- Current power mode: `MAXN_SUPER`

Formal benchmark later should also record:

- power mode
- `jetson_clocks` status
- fan policy
- temperature
- CPU/GPU frequency
- whether thermal throttling occurred

## Storage Status

At the time of environment inspection:

```text
Root filesystem: /dev/nvme0n1p1
Filesystem size: ~233G
Used: ~31G
Available: ~190G
```

Useful commands:

```bash
lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS
findmnt /
df -h /
```

## Network

Direct Ethernet:

```text
Laptop: 192.168.50.1/24
Jetson: 192.168.50.2/24
```

SSH is enabled and active.

Useful checks:

```bash
ip -br address
systemctl is-active ssh
ss -lntp | grep ':22'
```

The direct Ethernet subnet should not become the default Internet route.

## Internet Access

Intended path:

```text
Jetson TCP/DNS
    ↓
sshuttle
    ↓
SSH
    ↓
Laptop
    ↓
FlClash / VPN
    ↓
Internet
```

Current sshuttle setup observed:

```bash
sshuttle --dns \
  -r paopao@192.168.50.1 \
  -x 192.168.50.0/24 \
  0.0.0.0/0
```

`192.168.50.0/24` is excluded so the Jetson ↔ laptop management link is not routed into its own tunnel.

Useful checks:

```bash
pgrep -af sshuttle
systemctl status laptop-vpn-tunnel.service
curl -I https://github.com
curl -I https://huggingface.co
```

## CUDA / TensorRT Checks

CUDA compiler location:

```text
/usr/local/cuda-12.6/bin/nvcc
```

Useful commands:

```bash
which nvcc
nvcc --version
dpkg -l | grep -E "cuda-toolkit|cuda-compiler"
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

## Environment Snapshot Command

```bash
{
    echo "===== MODEL ====="
    tr -d '\0' < /proc/device-tree/model
    echo

    echo "===== L4T ====="
    cat /etc/nv_tegra_release

    echo "===== OS ====="
    lsb_release -a
    uname -a

    echo "===== MEMORY ====="
    free -h

    echo "===== STORAGE ====="
    lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINTS
    findmnt /
    df -h /

    echo "===== CUDA ====="
    nvcc --version

    echo "===== TENSORRT ====="
    python3 -c "import tensorrt as trt; print(trt.__version__)"

    echo "===== COMPILERS ====="
    gcc --version | head -n 1
    cmake --version | head -n 1

    echo "===== POWER MODE ====="
    sudo nvpmodel -q

    echo "===== NETWORK ====="
    ip -br address

    echo "===== SSH ====="
    systemctl is-active ssh
}
```
