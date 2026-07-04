#!/bin/bash
# =============================================================================
# run_training.sh — End-to-end fine-tuning pipeline
# =============================================================================
#
# USAGE:
#   chmod +x run_training.sh
#   ./run_training.sh
#
# Or with custom data directory:
#   DATA_DIR=/path/to/your/data ./run_training.sh
#
# REQUIREMENTS:
#   - Python 3.10+
#   - CUDA-enabled GPU (16GB+ VRAM recommended)
#   - pip install -r requirements.txt
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — modify as needed
# ---------------------------------------------------------------------------

DATA_DIR="${DATA_DIR:-./raw_data}"          # Directory with downloaded Google Drive data
OUTPUT_JSONL="./dataset_chatml.jsonl"       # Converted dataset
ADAPTER_DIR="./output/qwen25_astro_qlora"  # LoRA adapter output
MERGED_DIR="./output/qwen25_astro_merged"  # Merged model output
EVAL_DIR="./evaluation_output"             # Evaluation results

NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-4
LORA_R=16
LORA_ALPHA=32
MAX_SEQ_LEN=2048

# ---------------------------------------------------------------------------
# Colors for pretty output
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
error() { echo -e "${RED}❌ $*${NC}"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log "Checking prerequisites..."

# Check Python
python --version >/dev/null 2>&1 || error "Python not found. Install Python 3.10+"

# Check GPU
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')" \
    || warn "Could not detect PyTorch/CUDA. Proceeding anyway..."

# Check GPU memory
GPU_MEM=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))" 2>/dev/null || echo "N/A")
log "GPU Memory: ${GPU_MEM}GB"
if [[ "$GPU_MEM" != "N/A" ]] && (( $(echo "$GPU_MEM < 12" | bc -l 2>/dev/null || echo 0) )); then
    warn "GPU has less than 12GB VRAM. Reduce batch size or use more gradient accumulation."
fi

# ---------------------------------------------------------------------------
# Step 0: Install dependencies
# ---------------------------------------------------------------------------

log "Step 0: Installing dependencies..."
pip install -r requirements.txt --quiet
success "Dependencies installed"

# ---------------------------------------------------------------------------
# Step 1: Check data exists
# ---------------------------------------------------------------------------

log "Step 1: Checking data directory: ${DATA_DIR}"

if [ ! -d "$DATA_DIR" ]; then
    warn "Data directory '${DATA_DIR}' not found."
    echo ""
    echo "  Please download the Google Drive folder and place it at: ${DATA_DIR}"
    echo "  Or set DATA_DIR environment variable:"
    echo "    DATA_DIR=/your/data/path ./run_training.sh"
    echo ""
    echo "  To download with gdown (requires: pip install gdown):"
    echo "    pip install gdown"
    echo "    gdown --folder 'https://drive.google.com/drive/folders/18p1FLNAC_cbeFXl3SjJ4h-C88ZmSo1gC' -O ./raw_data"
    echo ""
    read -p "Press Enter to exit, or create the directory and re-run..." || true
    exit 1
fi

# Count files
FILE_COUNT=$(find "$DATA_DIR" -type f | wc -l)
log "Found ${FILE_COUNT} file(s) in ${DATA_DIR}"

# ---------------------------------------------------------------------------
# Step 2: Inspect data format
# ---------------------------------------------------------------------------

log "Step 2: Inspecting data format..."
python data_converter.py --inspect --input_dir "$DATA_DIR"
success "Data inspection complete"

# ---------------------------------------------------------------------------
# Step 3: Convert data to ChatML format
# ---------------------------------------------------------------------------

log "Step 3: Converting data to ChatML format..."
python data_converter.py \
    --input_dir "$DATA_DIR" \
    --output "$OUTPUT_JSONL"

if [ ! -f "$OUTPUT_JSONL" ]; then
    error "Conversion failed — output file not created"
fi

LINE_COUNT=$(wc -l < "$OUTPUT_JSONL")
success "Converted ${LINE_COUNT} conversations → ${OUTPUT_JSONL}"

# ---------------------------------------------------------------------------
# Step 4: Train with QLoRA
# ---------------------------------------------------------------------------

log "Step 4: Starting QLoRA fine-tuning..."
log "  This will take approximately:"
log "    ~2-4 hours on A100 (100-500 conversations)"
log "    ~6-12 hours on RTX 4090"
log "    ~12-24 hours on RTX 3080"

python train_qlora.py \
    --dataset_path "$OUTPUT_JSONL" \
    --output_dir "$ADAPTER_DIR" \
    --merged_output_dir "$MERGED_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --max_seq_length "$MAX_SEQ_LEN" \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100

success "Training complete! Adapter: ${ADAPTER_DIR}"

# ---------------------------------------------------------------------------
# Step 5: Evaluate the merged model
# ---------------------------------------------------------------------------

log "Step 5: Running evaluation..."
python evaluate_model.py \
    --model_path "$MERGED_DIR" \
    --output_dir "$EVAL_DIR" \
    --max_new_tokens 512 \
    --temperature 0.7

success "Evaluation complete! Results: ${EVAL_DIR}"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🎉 TRAINING PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "  📁 Raw data:          ${DATA_DIR}"
echo "  📄 Converted dataset: ${OUTPUT_JSONL} (${LINE_COUNT} conversations)"
echo "  🔧 LoRA adapter:      ${ADAPTER_DIR}"
echo "  🤖 Merged model:      ${MERGED_DIR}"
echo "  📊 Evaluation:        ${EVAL_DIR}/evaluate_results.md"
echo ""
echo "  Next steps:"
echo "  1. Review evaluation: cat ${EVAL_DIR}/evaluate_results.md"
echo "  2. Interactive test:  python evaluate_model.py --model_path ${MERGED_DIR} --interactive"
echo "  3. Deploy with vLLM:  See vllm_hosting_guide.md"
echo "════════════════════════════════════════════════════════════"
