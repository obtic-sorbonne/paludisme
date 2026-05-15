#!/usr/bin/env bash
# =============================================================================
# setup_models.sh — Auto-download required AI models if not present
# Called automatically by run_pipeline.sh on first run
# =============================================================================

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODELFILE="$HOME/Modelfile_limited"

echo "🔍 Checking required models..."

check_model() {
    OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "^$1"
}

# glm-ocr
if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "glm-ocr"; then
    echo "⬇️  Downloading glm-ocr (~2.2 GB)..."
    OLLAMA_HOST="$OLLAMA_HOST" ollama pull glm-ocr
else
    echo "✅ glm-ocr already present"
fi

# qwen2.5vl:7b
if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "qwen2.5vl:7b"; then
    echo "⬇️  Downloading qwen2.5vl:7b (~6 GB)..."
    OLLAMA_HOST="$OLLAMA_HOST" ollama pull qwen2.5vl:7b
else
    echo "✅ qwen2.5vl:7b already present"
fi

# qwen3:30b
if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "qwen3:30b"; then
    echo "⬇️  Downloading qwen3:30b (~18 GB)..."
    OLLAMA_HOST="$OLLAMA_HOST" ollama pull qwen3:30b
else
    echo "✅ qwen3:30b already present"
fi

# qwen2.5vl:72b (needed for qwen72b-limited)
if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "qwen2.5vl:72b"; then
    echo "⬇️  Downloading qwen2.5vl:72b (~48 GB — this will take several hours)..."
    OLLAMA_HOST="$OLLAMA_HOST" ollama pull qwen2.5vl:72b
else
    echo "✅ qwen2.5vl:72b already present"
fi

# qwen72b-limited (created from qwen2.5vl:72b)
if ! OLLAMA_HOST="$OLLAMA_HOST" ollama list 2>/dev/null | grep -q "qwen72b-limited"; then
    echo "🔧 Creating qwen72b-limited from qwen2.5vl:72b..."
    cat > "$MODELFILE" << 'MODELEOF'
FROM qwen2.5vl:72b
PARAMETER num_ctx 8192
PARAMETER num_gpu 20
MODELEOF
    OLLAMA_HOST="$OLLAMA_HOST" ollama create qwen72b-limited -f "$MODELFILE"
    echo "✅ qwen72b-limited created (22 GB VRAM instead of 37 GB)"
else
    echo "✅ qwen72b-limited already present"
fi

echo ""
echo "✅ All models ready!"
