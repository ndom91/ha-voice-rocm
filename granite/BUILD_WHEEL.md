# Granite Flash-Attention Notes

flash-attn 2.8.3 has gfx1151 (RDNA 3.5) assembler incompatibilities in hipcc, specifically with inline assembly bfloat16 instructions.

**Solution**: Use PyTorch's native SDPA (Scaled Dot-Product Attention) instead, with eager fallback.

The `granite_handler.py` now loads with:
1. `attn_implementation="sdpa"` (PyTorch native, no external deps)
2. Falls back to `eager` if needed
3. dtype: `torch.bfloat16` for RDNA 3.5 optimization

No flash-attn wheel needed.
