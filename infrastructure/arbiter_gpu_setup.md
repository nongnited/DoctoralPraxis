# Arbiter Server — GPU Setup Guide
Dell PowerEdge R740 | Dual NVIDIA Tesla P100-PCIE-16GB | 256 GB RAM

## Critical: Rocky Linux Kernel Lock

Rocky Linux 9.8 introduced a change to the Direct Rendering Manager (DRM) API
that breaks NVIDIA driver versions 550, 565, and 575.
The arbiter kernel must be locked to Rocky Linux 9.7 (el9_7).

### Step 1 — Install DNF version lock plugin
```bash
sudo dnf install python3-dnf-plugin-versionlock -y
```

### Step 2 — Lock kernel version
```bash
# List available kernels
rpm -qa kernel

# Lock to el9_7
sudo dnf versionlock add kernel-5.14.0-611.54.1.el9_7.x86_64
sudo dnf versionlock add kernel-core-5.14.0-611.54.1.el9_7.x86_64
sudo dnf versionlock add kernel-modules-5.14.0-611.54.1.el9_7.x86_64

# Verify
sudo dnf versionlock list
```

### Step 3 — Set default kernel in GRUB
```bash
# Find the el9_7 entry index
sudo grubby --info=ALL | grep "title\|index"

# Set it as default (replace N with the el9_7 index)
sudo grubby --set-default-index=N
sudo reboot
```

### Step 4 — Verify kernel after reboot
```bash
uname -r
# Expected: 5.14.0-611.54.1.el9_7.x86_64
```

---

## NVIDIA Driver Installation

### Step 5 — Disable Nouveau
```bash
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
sudo dracut --force
sudo reboot
```

### Step 6 — Install prerequisites
```bash
sudo dnf install -y kernel-devel kernel-headers gcc make dkms
sudo dnf groupinstall -y "Development Tools"
```

### Step 7 — Install NVIDIA driver 550.163.01 with CUDA 12.4
```bash
# Download from NVIDIA
wget https://us.download.nvidia.com/tesla/550.163.01/NVIDIA-Linux-x86_64-550.163.01.run

# Run installer
sudo sh NVIDIA-Linux-x86_64-550.163.01.run --silent --dkms
```

### Step 8 — Verify GPU detection
```bash
nvidia-smi
# Expected output: Two P100 GPUs, driver 550.163.01, CUDA 12.4
```

---

## Ollama Installation

### Step 9 — Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 10 — Configure Ollama for GPU
```bash
# Edit systemd service to expose on all interfaces
sudo systemctl edit ollama --force

# Add:
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="CUDA_VISIBLE_DEVICES=0,1"
```

### Step 11 — Pull models
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Step 12 — Start and verify
```bash
sudo systemctl start ollama
sudo systemctl enable ollama

# Test generation
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"Say hello","stream":false}'

# Test embedding
curl http://localhost:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}'
```

---

## Performance Validation
```bash
# Benchmark CPU vs GPU generation latency
time curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"What is 2+2?","stream":false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Total: {d['total_duration']/1e9:.1f}s\")"
```

Expected with dual P100: ~3-4 seconds for a short response (vs 800-1200 seconds CPU).
