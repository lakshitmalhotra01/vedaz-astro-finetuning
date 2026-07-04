
---

## 🧠 My Approach

### Part 1 — Fine-tuning

#### Step 1: Understanding the Data
The chat dataset from the provided Google Drive folder was analyzed using the `--inspect` flag in `data_converter.py`. The converter auto-detects 7 formats:

| Format | Pattern |
|--------|---------|
| JSON/JSONL | `{"messages": [{role, content}]}` |
| Flat JSON | `{"prompt": "...", "response": "..."}` |
| CSV | columns: `user/assistant`, `prompt/response` |
| TXT | `Human: ... Assistant: ...` blocks |
| ShareGPT | `{"human": ..., "gpt": ...}` |
| Alpaca | `{"instruction", "input", "output"}` |
| Excel | Same as CSV |

All formats are normalized into **ChatML-style messages** with a consistent system prompt defining the astrologer persona.

#### Step 2: Model Selection — Qwen2.5-7B-Instruct
I chose **Qwen2.5-7B-Instruct** over other options because:
- ✅ **7B scale** fits in ~4.5 GB VRAM with 4-bit QLoRA — practical for most GPUs
- ✅ **Excellent multilingual support** — handles Hindi/Sanskrit terms (rashi, nakshatra, dasha) naturally
- ✅ **Native ChatML format** (`<|im_start|>` / `<|im_end|>`) — no custom template needed
- ✅ **Strong instruction following** — RLHF-trained baseline makes fine-tuning faster
- ✅ Better than Qwen3-8B for this task due to more stable instruction format

#### Step 3: QLoRA Configuration
Instead of full fine-tuning (which needs ~60 GB VRAM for a 7B model), I used **QLoRA** — 4-bit quantization of the base model + low-rank adapter training:

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Quantization | 4-bit NF4 + double quant | ~75% VRAM savings vs FP16 |
| LoRA rank (r) | 16 | Good balance of expressiveness and regularization |
| LoRA alpha | 32 | Standard 2×r ratio for stable gradient scaling |
| LoRA dropout | 0.05 | Prevents overfitting on small datasets |
| Target modules | All linear layers | Q/K/V/O projections + MLP gate/up/down |
| Learning rate | 2e-4 | Standard for QLoRA (10× higher than full fine-tuning) |
| Effective batch size | 16 | (2 per device × 8 gradient accumulation steps) |
| LR scheduler | Cosine | Smooth decay, prevents forgetting at end of training |
| Warmup ratio | 5% | Prevents unstable gradients in early steps |
| Epochs | 3 | Sufficient for domain adaptation without overfitting |
| Max seq length | 2048 | Covers most astrology conversations |
| Optimizer | paged_adamw_8bit | bitsandbytes paged optimizer — saves VRAM |

#### Step 4: Training with TRL SFTTrainer
Used `SFTTrainer` from the `trl` library because it handles:
- Sequence packing (combines short convos to fill context window → faster training)
- Chat template application automatically
- Gradient checkpointing for memory efficiency
- Evaluation loop and checkpointing built-in

#### Step 5: Adapter Merging
After training, the LoRA adapter is automatically merged back into the base model weights using `merge_and_unload()`. This produces a single standalone model ready for deployment — no adapter loading overhead at inference time.

#### Step 6: Evaluation
`evaluate_model.py` runs 7 astrology-themed benchmark prompts covering:
- Career query
- Marriage query
- Health concern
- Financial loss
- Foreign travel/visa
- General astrology knowledge
- Multi-turn conversation

Each response is scored on keyword presence (rashi, dasha, house, planet names), response length, and generation speed (tokens/sec). Results saved as both JSON and Markdown report.

---

### Part 2a — vLLM Hosting Guide

See [`vllm_hosting_guide.md`](./vllm_hosting_guide.md) for the full step-by-step guide covering:

1. **VPS Sizing** — GPU/RAM recommendations with cost comparison (Vast.ai ~$0.35/hr to A100 ~$3.67/hr). Recommended: RunPod A10 24GB (~$0.39/hr) for best cost/reliability balance.
2. **Server Setup** — Ubuntu 22.04, NVIDIA drivers, CUDA verification
3. **Dependency Installation** — Miniconda + PyTorch (CUDA 12.1) + vLLM 0.6.x
4. **Model Upload** — SCP, rsync, or HuggingFace Hub options
5. **vLLM Launch** — Full command with every flag explained
6. **Firewall Config** — UFW rules + cloud security group setup
7. **Testing** — curl command + Python OpenAI-compatible client code
8. **systemd Service** — Full service file for auto-start on reboot
9. **tmux** — Simpler alternative for dev/testing
10. **Logging** — journalctl, log rotation, GPU monitoring with nvtop
11. **Watchdog Script** — Cron-based health check with auto-restart
12. **Security** — Nginx HTTPS reverse proxy + Let's Encrypt SSL + rate limiting
13. **Scaling Tips** — Multi-GPU, context length, concurrent request tuning

---

### Part 2b — 5 Sample Conversations

See [`sample_conversations.json`](./sample_conversations.json) — manually written, realistic multi-turn conversations in ChatML format, directly usable as additional fine-tuning data.

| # | Topic | Key Astrology Used | Predicted Date Given |
|---|-------|-------------------|---------------------|
| 1 | Career & Job Change | Kanya Lagna, Moola Nakshatra, Rahu in 10th, Rahu Mahadasha, D-10 chart | Jan 15 – Apr 30, 2025 |
| 2 | Marriage Delay | Mesha Lagna, Jyeshtha Nakshatra, Manglik dosha, Venus Mahadasha, Saturn transit | Feb–May 2025 (proposal); Oct 2025–Feb 2026 (wedding) |
| 3 | Mystery Illness | Vrishabha Lagna, Ashlesha Nakshatra, Saturn in 6th, Ketu in 12th, D-6 chart | Jun 2025 (improvement); Sep–Dec 2025 (full recovery) |
| 4 | Financial Losses | Vrishabha Lagna, Hasta Nakshatra, Rahu Mahadasha, Rahu in 2nd house, D-2 chart | Aug 2024–Mar 2025 (first positive); Dec 2025 (stabilized) |
| 5 | Canada Visa Delays | Mithuna Lagna, Bharani Nakshatra, Rahu in 12th, Guru-Rahu transit yoga | Jan–Apr 2025 (visa success); Jul–Sep 2027 (job offer) |

Each conversation includes:
- ✅ Real kundli elements (rashi, nakshatra, dasha periods, divisional charts)
- ✅ Astrologer pausing to "prepare the kundli" before analysis
- ✅ Empathy toward the user's specific concern
- ✅ Specific date or time window prediction at the end
- ✅ Natural multi-turn follow-up handled

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data from Google Drive
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/18p1FLNAC_cbeFXl3SjJ4h-C88ZmSo1gC" -O ./raw_data

# 3. Inspect your data format
python data_converter.py --inspect --input_dir ./raw_data

# 4. Convert to ChatML format
python data_converter.py --input_dir ./raw_data --output ./dataset_chatml.jsonl

# 5. Train
python train_qlora.py --dataset_path ./dataset_chatml.jsonl

# 6. Evaluate
python evaluate_model.py --model_path ./output/qwen25_astro_merged

# OR — run everything at once
chmod +x run_training.sh && ./run_training.sh
