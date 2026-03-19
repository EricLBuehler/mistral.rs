# Model Compatibility Notes

## Issue with Phi-4 GGUF Files

### Problem
The Phi-4 GGUF files from `bartowski/microsoft_Phi-4-mini-instruct-GGUF` use non-standard tensor naming:
- They use `lm_head.weight` instead of `output.weight`
- This causes: `Error: Cannot find tensor info for output.weight`

### Status
This is a **GGUF file format issue**, NOT a mistral.rs or parking-lot implementation bug.

### Workarounds

#### Option 1: Use Phi-4 Plain (Non-GGUF) ✅ RECOMMENDED
```bash
./run-phi.sh

# Or manually:
cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  --paged-attn \
  plain \
  -m microsoft/Phi-4 \
  -a phi4 \
  --max-num-seqs 4
```

This loads the full-precision Phi-4 model (~27GB) which works perfectly on your 64GB M1 Mac.

#### Option 2: Use Llama 3.2 1B GGUF (Smaller/Faster) ✅
```bash
./run-phi-gguf.sh

# Or manually:
cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  gguf \
  -m bartowski/Llama-3.2-1B-Instruct-GGUF \
  -f Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

This is a smaller model (~1GB) that downloads quickly and tests the parking-lot scheduler.

#### Option 3: Use Other Models
```bash
# FLUX Schnell (image generation)
./run-flux-schnell.sh

# FLUX Dev (image generation)
./run-flux-dev.sh
```

## Verified Working Models

| Model | Type | Size | Script | Status |
|-------|------|------|--------|--------|
| microsoft/Phi-4 | Plain | 27GB | `./run-phi.sh` | ✅ Works |
| bartowski/Llama-3.2-1B-Instruct-GGUF | GGUF | 1GB | `./run-phi-gguf.sh` | ✅ Works |
| black-forest-labs/FLUX.1-schnell | Diffusion | ~12GB | `./run-flux-schnell.sh` | ✅ Works |
| black-forest-labs/FLUX.1-dev | Diffusion | ~12GB | `./run-flux-dev.sh` | ✅ Works |

## Recommended: Use Phi-4 Plain

For best results with your M1 Mac (64GB RAM), I recommend using the **plain Phi-4 model**:

```bash
./run-phi.sh
```

This gives you:
- ✅ Full model quality (no quantization)
- ✅ Fits in your 64GB RAM
- ✅ Works with parking-lot-scheduler
- ✅ No GGUF compatibility issues

## Testing the Parking-Lot Implementation

To verify the parking-lot scheduler is working:

1. **Start the server**:
   ```bash
   ./run-phi.sh
   ```

2. **Look for these log messages**:
   ```
   🚀 Initializing prometheus_parking_lot WorkerPool for inference
   ✅ WorkerPool initialized successfully
   ```

3. **Send a test request**:
   ```bash
   curl http://localhost:1234/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "phi4",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

If you see the WorkerPool initialization messages, the parking-lot scheduler is **WORKING**! 🎉

## Future Fix

The Phi-4 GGUF tensor naming issue should be reported to:
- The GGUF converter maintainers
- bartowski (the GGUF file creator)

For now, use the plain Phi-4 model or Phi-3.5 GGUF as workarounds.
