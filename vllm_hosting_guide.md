# Hosting Your Fine-tuned Astrologer Model on a VPS with vLLM

> **Model**: Qwen2.5-7B-Instruct (fine-tuned) | **Serving**: vLLM OpenAI-compatible API

---

## Table of Contents
1. [Choosing and Sizing the VPS](#1-choosing-and-sizing-the-vps)
2. [Initial Server Setup](#2-initial-server-setup)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Uploading / Downloading the Model](#4-uploading--downloading-the-model)
5. [Launching the vLLM Server](#5-launching-the-vllm-server)
6. [Firewall & Port Configuration](#6-firewall--port-configuration)
7. [Testing the Endpoint](#7-testing-the-endpoint)
8. [Keeping It Running (systemd + tmux)](#8-keeping-it-running-systemd--tmux)
9. [Logging & Monitoring](#9-logging--monitoring)
10. [Restart on Failure & Auto-recovery](#10-restart-on-failure--auto-recovery)
11. [Security Best Practices](#11-security-best-practices)
12. [Scaling Tips](#12-scaling-tips)
13. [Cost Comparison Table](#13-cost-comparison-table)

---

## 1. Choosing and Sizing the VPS

### Why GPU matters
The Qwen2.5-7B-Instruct model (merged, full precision) requires:
- **FP16**: ~14 GB VRAM for the model weights alone
- **vLLM KV cache** (for serving concurrent requests): +2-6 GB
- **Recommended minimum**: **16 GB VRAM**

### Recommended VPS configurations

| Provider | Instance | GPU | VRAM | RAM | vCPUs | Approx. Cost |
|----------|----------|-----|------|-----|-------|--------------|
| **Lambda Labs** | `gpu_1x_a10` | A10 | 24 GB | 30 GB | 30 | ~$0.60/hr |
| **RunPod** | A10 Pod | A10 | 24 GB | 15 GB | 8 | ~$0.39/hr |
| **Vast.ai** | RTX 4090 | RTX 4090 | 24 GB | 32 GB | 8 | ~$0.35/hr |
| **AWS** | `g5.xlarge` | A10G | 24 GB | 16 GB | 4 | ~$1.006/hr |
| **GCP** | `a2-highgpu-1g` | A100 40G | 40 GB | 85 GB | 12 | ~$3.67/hr |
| **Azure** | `Standard_NC6s_v3` | V100 | 16 GB | 112 GB | 6 | ~$0.90/hr |

> **Best value for this model**: **RunPod A10 or Vast.ai RTX 4090** — both have 24 GB VRAM and are cost-effective.
> For **production traffic**, use **Lambda Labs A10** or **AWS g5.xlarge** for better reliability SLAs.

### Storage requirements
- Model weights (FP16): ~14 GB
- OS + dependencies: ~10 GB
- Logs + temp: ~5 GB
- **Recommended disk**: >= 50 GB SSD

### Operating System
Use **Ubuntu 22.04 LTS** — best driver support and most documented.

---

## 2. Initial Server Setup

SSH into your VPS as root or a sudo user:

```bash
ssh ubuntu@YOUR_SERVER_IP
```

### System updates and essentials

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install essentials
sudo apt-get install -y git curl wget unzip htop nvtop screen tmux build-essential

# Verify GPU is visible
nvidia-smi
```

Expected output from `nvidia-smi`:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.x    Driver Version: 535.x   CUDA Version: 12.2             |
+-------------------------------+----------------------+----------------------+
| GPU 0  NVIDIA A10 24G         |   MiB /  24576 MiB  |    0%                |
+-----------------------------------------------------------------------------+
```

---

## 3. Installing Dependencies

### Step 3a: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create dedicated environment
conda create -n vllm-serve python=3.10 -y
conda activate vllm-serve
```

### Step 3b: Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.3.0 torchvision torchaudio --index-url https://dl.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA A10
```

### Step 3c: Install vLLM

```bash
pip install vllm==0.6.3.post1

# Verify
python -c "import vllm; print(vllm.__version__)"
```

> **Note**: vLLM 0.6.x supports Qwen2.5 natively. Check [vllm releases](https://github.com/vllm-project/vllm/releases) for the latest stable version.

### Step 3d: Install HuggingFace Hub tools

```bash
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1  # Faster downloads
```

---

## 4. Uploading / Downloading the Model

### Option A: Upload from local machine (SCP)

```bash
# From your LOCAL machine — compress and upload
tar -czf qwen25_astro_merged.tar.gz ./output/qwen25_astro_merged/
scp qwen25_astro_merged.tar.gz ubuntu@YOUR_SERVER_IP:/home/ubuntu/models/

# On SERVER — extract
mkdir -p /home/ubuntu/models
tar -xzf /home/ubuntu/models/qwen25_astro_merged.tar.gz -C /home/ubuntu/models/
```

### Option B: Push to HuggingFace Hub, then pull on server

```bash
# LOCAL: push to Hub
huggingface-cli login
python train_qlora.py --merge_only \
    --adapter_path ./output/qwen25_astro_qlora \
    --push_to_hub \
    --hub_repo_id YOUR_HF_USERNAME/qwen25-vedaz-astrologer

# SERVER: download from Hub
huggingface-cli download YOUR_HF_USERNAME/qwen25-vedaz-astrologer \
    --local-dir /home/ubuntu/models/qwen25_astro_merged \
    --local-dir-use-symlinks False
```

### Option C: rsync (fastest for large models)

```bash
rsync -avz --progress \
    ./output/qwen25_astro_merged/ \
    ubuntu@YOUR_SERVER_IP:/home/ubuntu/models/qwen25_astro_merged/
```

### Verify model files

```bash
ls -lh /home/ubuntu/models/qwen25_astro_merged/
# Should see config.json, tokenizer.json, model-XXXX.safetensors files

du -sh /home/ubuntu/models/qwen25_astro_merged/
# Expected: ~14-15 GB total
```

---

## 5. Launching the vLLM Server

### Basic launch command

```bash
conda activate vllm-serve

python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/qwen25_astro_merged \
    --served-model-name "vedaz-astrologer" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --api-key "your-secret-api-key-here"
```

### Parameter explanations

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--model` | Model directory | Path to merged model |
| `--served-model-name` | `vedaz-astrologer` | Name exposed via API |
| `--host 0.0.0.0` | All interfaces | Allows external access |
| `--port 8000` | 8000 | Standard API port |
| `--tensor-parallel-size 1` | 1 | Single GPU; use 2+ for multi-GPU |
| `--dtype bfloat16` | bfloat16 | Best precision/speed balance on A10 |
| `--max-model-len 4096` | 4096 tokens | Max context window |
| `--gpu-memory-utilization` | 0.90 | Use 90% of VRAM for KV cache |
| `--api-key` | your key | Simple authentication |

### Expected startup output

```
INFO - Initializing an LLM engine ...
INFO - Loading model weights took 13.47 GB
INFO - KV cache: 4096 tokens, 24 layers
INFO - Available routes: GET /health, GET /v1/models, POST /v1/chat/completions
Uvicorn running on http://0.0.0.0:8000
```

Startup takes ~60-90 seconds on first launch.

---

## 6. Firewall & Port Configuration

### Ubuntu UFW

```bash
# Enable UFW (ALWAYS allow SSH first!)
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp
sudo ufw enable

# Check status
sudo ufw status verbose
```

### Cloud provider security groups

If using AWS/GCP/Azure, also open port 8000 in the cloud console security group:

```bash
# AWS CLI example (restrict to your IP in production!)
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp \
    --port 8000 \
    --cidr YOUR_IP/32
```

> **Security tip**: In production, put Nginx with HTTPS in front and close port 8000 externally. See Section 11.

---

## 7. Testing the Endpoint

### Health check

```bash
curl http://YOUR_SERVER_IP:8000/health
# Expected: {"status":"ok"}
```

### List models

```bash
curl http://YOUR_SERVER_IP:8000/v1/models \
    -H "Authorization: Bearer your-secret-api-key-here"
```

### Test chat completions (curl)

```bash
curl http://YOUR_SERVER_IP:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-secret-api-key-here" \
    -d '{
        "model": "vedaz-astrologer",
        "messages": [
            {
                "role": "system",
                "content": "You are Vedaz, an expert Vedic astrologer."
            },
            {
                "role": "user",
                "content": "My DOB is March 15, 1992, 6:30 AM, Delhi. When will my career improve?"
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

### Test with Python (OpenAI client)

```python
# test_endpoint.py
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_SERVER_IP:8000/v1",
    api_key="your-secret-api-key-here",
)

response = client.chat.completions.create(
    model="vedaz-astrologer",
    messages=[
        {
            "role": "system",
            "content": "You are Vedaz, an expert Vedic astrologer with deep knowledge of Jyotish shastra.",
        },
        {
            "role": "user",
            "content": "Namaste ji! My DOB is July 22, 1995, at 11:45 PM, Mumbai. When will I get married?",
        },
    ],
    max_tokens=512,
    temperature=0.7,
)

print("Astrologer:", response.choices[0].message.content)
```

### Streaming example

```python
stream = client.chat.completions.create(
    model="vedaz-astrologer",
    messages=[{"role": "user", "content": "Tell me about Venus in the 7th house"}],
    stream=True,
    max_tokens=300,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 8. Keeping It Running (systemd + tmux)

### Method A: systemd service (Recommended for production)

```bash
sudo nano /etc/systemd/system/vllm-astrologer.service
```

```ini
[Unit]
Description=vLLM Astrologer API Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/miniconda3/envs/vllm-serve/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES=0"

ExecStart=/home/ubuntu/miniconda3/envs/vllm-serve/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/qwen25_astro_merged \
    --served-model-name vedaz-astrologer \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --api-key your-secret-api-key-here

Restart=on-failure
RestartSec=15s
StandardOutput=append:/var/log/vllm-astrologer/stdout.log
StandardError=append:/var/log/vllm-astrologer/stderr.log
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
# Create log dir and enable service
sudo mkdir -p /var/log/vllm-astrologer
sudo chown ubuntu:ubuntu /var/log/vllm-astrologer
sudo systemctl daemon-reload
sudo systemctl enable vllm-astrologer
sudo systemctl start vllm-astrologer
sudo systemctl status vllm-astrologer
```

Useful commands:
```bash
sudo systemctl restart vllm-astrologer    # Restart
sudo journalctl -u vllm-astrologer -f     # Live logs
```

### Method B: tmux (Simple, good for testing)

```bash
# Start in background tmux session
tmux new-session -d -s vllm \
    "conda activate vllm-serve && python -m vllm.entrypoints.openai.api_server \
     --model /home/ubuntu/models/qwen25_astro_merged \
     --served-model-name vedaz-astrologer \
     --host 0.0.0.0 --port 8000 \
     --dtype bfloat16 --max-model-len 4096 \
     --api-key your-secret-api-key-here"

tmux attach-session -t vllm    # Attach to view
# Detach: Ctrl+B then D
```

---

## 9. Logging & Monitoring

```bash
# Live logs (systemd)
sudo journalctl -u vllm-astrologer -f

# Application stdout
tail -f /var/log/vllm-astrologer/stdout.log

# GPU monitoring
watch -n 2 nvidia-smi     # Simple
nvtop                      # Interactive dashboard

# vLLM Prometheus metrics
curl http://localhost:8000/metrics | grep vllm
```

### Log rotation

```bash
sudo nano /etc/logrotate.d/vllm-astrologer
```
```
/var/log/vllm-astrologer/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## 10. Restart on Failure & Auto-recovery

systemd's `Restart=on-failure` handles automatic restarts. Additionally, add a watchdog:

```bash
# /home/ubuntu/watchdog.sh
#!/bin/bash
ENDPOINT="http://localhost:8000/health"
LOG="/var/log/vllm-astrologer/watchdog.log"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$ENDPOINT")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ "$STATUS" != "200" ]; then
    echo "[$TIMESTAMP] Health check failed (HTTP $STATUS). Restarting..." >> "$LOG"
    sudo systemctl restart vllm-astrologer
else
    echo "[$TIMESTAMP] Healthy (HTTP $STATUS)" >> "$LOG"
fi
```

```bash
chmod +x /home/ubuntu/watchdog.sh

# Add to crontab (every 5 minutes)
crontab -e
# Add: */5 * * * * /home/ubuntu/watchdog.sh
```

---

## 11. Security Best Practices

### Nginx HTTPS reverse proxy (strongly recommended)

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo nano /etc/nginx/sites-available/vllm-astrologer
```

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Rate limiting: 10 requests/minute per IP
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
    limit_req zone=api burst=5 nodelay;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
```

```bash
sudo certbot --nginx -d api.yourdomain.com
sudo nginx -t && sudo systemctl reload nginx
# Now close port 8000 externally:
sudo ufw delete allow 8000/tcp
```

---

## 12. Scaling Tips

| Scenario | Solution |
|----------|----------|
| More concurrent users | `--max-num-seqs 512` |
| Larger context | `--max-model-len 8192` (needs more VRAM) |
| Two GPUs | `--tensor-parallel-size 2` |
| Very low VRAM | Use `--quantization awq` with an AWQ-quantized model |
| High throughput | `--enable-chunked-prefill` |

---

## 13. Cost Comparison Table

| Provider | GPU | $/hr | $/month | Notes |
|----------|-----|------|---------|-------|
| Vast.ai | RTX 4090 24GB | ~$0.35 | ~$252 | Cheapest, community hardware |
| RunPod | A10 24GB | ~$0.39 | ~$281 | Good reliability |
| Lambda Labs | A10 24GB | ~$0.60 | ~$432 | Enterprise-grade |
| AWS g5.xlarge | A10G 24GB | ~$1.01 | ~$727 | Most reliable |
| GCP A100 40GB | A100 | ~$3.67 | ~$2,642 | Heavy traffic |

> For a lightweight astrologer chatbot (100-1000 req/day), **RunPod or Lambda Labs** offer the best cost/reliability balance.

---

## Quick Start Cheatsheet

```bash
# 1. SSH in
ssh ubuntu@YOUR_SERVER_IP

# 2. Activate env
conda activate vllm-serve

# 3. Launch vLLM
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/qwen25_astro_merged \
    --served-model-name vedaz-astrologer \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --max-model-len 4096 \
    --api-key YOUR_SECRET_KEY &

# 4. Test
curl http://localhost:8000/health

# 5. Monitor GPU
nvidia-smi
```

---

*Guide version: 1.0 | July 2026*
