# Running Models - Quick Reference

## ✅ Fixed Scripts

All scripts have been corrected with proper flag ordering.

## Command Structure

**IMPORTANT**: Flags must be in this exact order:

```
mistralrs-server [GLOBAL_FLAGS] <SUBCOMMAND> [MODEL_FLAGS]
```

### Global Flags (Before Subcommand)
- `--port` - Server port
- `--isq` - Quantization type
- `--paged-attn` - Enable paged attention (Metal)
- `--no-paged-attn` - Disable paged attention (CUDA)
- `--pa-gpu-mem` - PagedAttention GPU memory
- `--pa-ctxt-len` - PagedAttention context length

### Subcommands
- `plain` - Text models
- `vision-plain` - Multimodal/vision models
- `diffusion` - Image generation models
- `speech` - Speech models

### Model Flags (After Subcommand)
- `-m` - Model ID
- `-a` - Architecture
- `--max-num-seqs` - Max concurrent sequences

## Running the Scripts

### Phi-4 Multimodal
```bash
./run-phi.sh
```

**What it does**:
- Runs Phi-4 with vision capabilities
- 4-bit quantization (Q4K) - uses ~3-4GB
- Paged attention enabled
- Max 8 concurrent requests

### FLUX.1 Schnell (Fast)
```bash
./run-flux-schnell.sh
```

**What it does**:
- Fast image generation (1-4 steps)
- 8-bit quantization - uses ~12GB
- Good quality

### FLUX.1 Dev (Best Quality)
```bash
./run-flux-dev.sh
```

**What it does**:
- Best quality image generation (20-50 steps)
- Full precision - uses ~24GB
- Slower but better results

## Testing the Server

### For Phi-4 (Chat)
```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

### For FLUX (Image Generation)
```bash
curl -X POST http://localhost:1234/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux",
    "prompt": "A serene mountain landscape at sunset",
    "n": 1,
    "size": "1024x1024"
  }'
```

## Common Issues & Solutions

### "No response received from model"

**Cause**: Flags in wrong order (ISQ/paged-attn after subcommand)

**Solution**: Use the corrected scripts above - global flags MUST come before the subcommand

### Out of Memory

**For Phi-4**:
- Reduce `--max-num-seqs` from 8 to 4 or 2
- Use stronger quantization: `--isq Q4_0` instead of `Q4K`

**For FLUX**:
- Add quantization: `--isq Q8_0` or `--isq Q4K`
- Reduce batch size

### Model Download Slow

First run downloads the model from HuggingFace. This is normal and only happens once.

Cache location: `~/.cache/huggingface/`

## ISQ Quantization Guide

| ISQ Value | Bits | Memory | Quality | Use Case |
|-----------|------|--------|---------|----------|
| `Q4_0` | 4-bit | ~3GB | Good | Maximum savings |
| `Q4K` | 4-bit | ~3-4GB | Better | Recommended for 64GB |
| `Q5K` | 5-bit | ~4-5GB | Very Good | Balance |
| `Q8_0` | 8-bit | ~7-8GB | Excellent | Diffusion models |
| None | 16-bit | ~14GB | Best | If you have RAM |

## Performance Notes

**M1 Mac with 64GB RAM**:
- Phi-4 with Q4K: ~3-4GB, fast inference
- FLUX Schnell with Q8_0: ~12GB, ~2-5s per image
- FLUX Dev full precision: ~24GB, ~10-30s per image

**Concurrent Requests**:
- `--max-num-seqs 2`: Safe, low memory
- `--max-num-seqs 4`: Good balance
- `--max-num-seqs 8`: Max throughput (needs more RAM)

## Monitoring

Check server logs for:
- Model loading progress
- Memory usage
- Request processing times
- Errors

Server is ready when you see:
```
Serving on http://0.0.0.0:1234
```

## API Endpoints

Once running on port 1234:

- `http://localhost:1234/health` - Health check
- `http://localhost:1234/v1/models` - List models
- `http://localhost:1234/v1/chat/completions` - Chat (Phi-4)
- `http://localhost:1234/v1/images/generations` - Images (FLUX)
- `http://localhost:1234/v1/metrics` - Metrics (with parking-lot-scheduler feature)

---

**All scripts are now corrected and ready to use!** 🚀
