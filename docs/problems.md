# Problems & Troubleshooting

Keep entries short. Use this template:

```text
## Date - Problem title

**Cause:** ...

**Investigation:**
commands / observations

**Fix:**
...

**Learned:**
- ...
```

---

## 2026-08-08 - `nvcc: command not found`

**Cause:** CUDA Toolkit was already installed, but `/usr/local/cuda/bin` was not in `PATH`.

**Investigation:**

```bash
find /usr/local -maxdepth 4 -type f -name nvcc 2>/dev/null
dpkg -l | grep -E "cuda-toolkit|cuda-compiler"
echo "$PATH"
which nvcc
```

Found:

```text
/usr/local/cuda-12.6/bin/nvcc
```

**Fix:**

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Persist in `~/.bashrc` if needed.

**Learned:**

- `command not found` does not necessarily mean the package is missing.
- Check executable location, package state and `PATH` before reinstalling.
- `PATH` controls executable lookup.
- `LD_LIBRARY_PATH` is used for runtime shared-library lookup.

---

## 2026-08-08 - `trtexec --version` reports FAILED

**Cause:** This `trtexec` invocation did not behave as a pure version query. It entered the normal execution path and failed because no model/engine was supplied.

**Observation:**

```text
Model missing or format not recognized
&&&& FAILED TensorRT.trtexec
```

The same output still showed:

```text
TensorRT v100300
```

meaning TensorRT 10.3.

**Verification:**

A real FP16 engine was later loaded and executed successfully:

```text
&&&& PASSED TensorRT.trtexec
```

**Learned:**

- Do not infer that a tool is broken from a generic final `FAILED` line.
- Identify the exact stage that failed.

---

## 2026-08-08 - Direct Ethernet SSH setup

**Configuration:**

```text
Laptop: 192.168.50.1/24
Jetson: 192.168.50.2/24
```

The Ethernet connection is intended only for laptop ↔ Jetson management.

Important option:

```text
ipv4.never-default yes
```

**Useful commands:**

```bash
nmcli device status
ip -br address
ip route
systemctl status ssh
ss -lntp | grep ':22'
```

**Learned:**

A dedicated management subnet is useful for stable SSH/Remote SSH while Wi-Fi/VPN handles Internet access.

---

## 2026-08-08 - SOCKS tunnel works with curl but not `hf`

**Symptom:**

Explicit curl proxy worked:

```bash
curl --proxy socks5h://127.0.0.1:1080 ...
```

but:

```bash
hf download ...
```

reported:

```text
Network is unreachable
```

**Cause:**

A SOCKS listener existing on localhost does not mean every application automatically uses it.

**Learned:**

"Proxy exists" and "application is using the proxy" are different statements.

---

## 2026-08-08 - HF SOCKS proxy requires `socksio`

**Symptom:**

```text
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed
```

**Cause:**

HF CLI uses Python/HTTPX. SOCKS support is optional in that stack.

APT could use SOCKS because APT has its own proxy implementation.

**Learned:**

Different network clients can support the same proxy type differently.

---

## 2026-08-08 - Invalid `ALL_PROXY=socks://...` breaks HF CLI

**Symptom:**

```text
Unknown scheme for proxy URL
URL('socks://127.0.0.1:7890/')
```

Environment included:

```text
HTTP_PROXY=http://127.0.0.1:7890/
HTTPS_PROXY=http://127.0.0.1:7890/
ALL_PROXY=socks://127.0.0.1:7890/
```

**Temporary Fix:**

```bash
env -u ALL_PROXY -u all_proxy hf download ...
```

**Useful Check:**

```bash
env | grep -i proxy
```

**Learned:**

Proxy environment variables can override each other and can break tools even when another valid proxy setting exists.

---

## 2026-08-08 - Per-application proxy became difficult to maintain

**Previous approach:**

Configure proxy separately for:

- curl;
- apt;
- git;
- hf;
- pip.

**Replacement:**

Use transparent routing:

```text
Jetson
 ↓
sshuttle
 ↓
SSH
 ↓
Laptop
 ↓
VPN
 ↓
Internet
```

**Learned:**

When many applications need the same egress path, a network-level transparent solution is easier to maintain than per-application proxy configuration.

---

## 2026-08-08 - HF CLI installer reused a broken venv

**Symptom:**

Initial install failed because `python3-venv` was missing.

After installing `python3-venv`, rerunning the installer still reported:

```text
Virtual environment already exists; reusing
No module named pip
```

**Cause:**

The first failed install left a partially-created `~/.hf-cli/venv`, and the installer reused it.

**Fix:**

```bash
rm -rf ~/.hf-cli
```

Then rerun the installer.

**Learned:**

Failed bootstrap/install operations can leave stale partial state. Re-running the same installer may reproduce the failure until that stale state is removed.
