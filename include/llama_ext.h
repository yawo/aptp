/**
 * llama_ext.h — Extension API for APTP: hidden state + KV-cache extraction.
 *
 * Stock llama.cpp only exposes final embeddings and logits. This header
 * defines hooks that must be compiled INTO llama.cpp itself (via a patch)
 * to gain access to per-layer hidden states and raw KV-cache tensor data.
 *
 * ── How to patch llama.cpp ─────────────────────────────────────────────────
 *
 * 1. Add `#include "llama_ext.h"` at the top of `llama.cpp`.
 * 2. In `llama_decode_internal()` (or wherever the forward pass finishes),
 *    capture hidden states after each layer:
 *
 *    ```c
 *    // After each transformer layer's output is written to `buf_output`:
 *    if (g_aptp_capture_hidden) {
 *        aptp_on_layer_output(lctx, layer_idx, buf_output, n_tokens, n_embd);
 *    }
 *    ```
 *
 * 3. Expose KV-cache raw data by adding the `aptp_get_kv_data()` function:
 *
 *    ```c
 *    float* aptp_get_kv_data(llama_context* ctx, int layer, int is_key, int* out_count) {
 *        // Access ctx->kv_cache.k_l[layer] and .v_l[layer]
 *    }
 *    ```
 *
 * ── Why this is needed ────────────────────────────────────────────────────
 *
 * llama.cpp's public C API (llama.h) intentionally hides intermediate states
 * for performance reasons. APTP needs them for agent-to-agent primitive transfer.
 * This header bridges that gap without forking llama.cpp — you just add a few
 * hook calls to your local build.
 *
 * ── Status ────────────────────────────────────────────────────────────────
 *
 * This is a REFERENCE DESIGN. The exact integration points depend on the
 * llama.cpp version. To use this, add these hooks into your llama.cpp build:
 *   - After each transformer layer forward pass, call `aptp_on_layer_output()`
 *   - In the kv_cache struct, expose `k_l` and `v_l` via the key/value accessors
 */
#ifndef LLAMA_EXT_H
#define LLAMA_EXT_H

#include "llama.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Global capture toggle ──────────────────────────────────────────────── */
/* Set to non-zero to enable hidden state capture during the next decode. */
extern int  g_aptp_capture_hidden;

/* ── Callbacks (called from within patched llama.cpp) ──────────────────── */

/* Called after each transformer layer's forward pass completes.
   `buf` points to the layer output (n_tokens * n_embd floats).       */
void aptp_on_layer_output(
    struct llama_context* ctx,
    int32_t               layer_idx,
    const float*          buf,
    int32_t               n_tokens,
    int32_t               n_embd
);

/* ── Extended data access (direct KV-cache read) ────────────────────────── */

/* Get KV-cache key tensor for layer `layer_id`.
   Returns pointer to float array of size `n_embd_head * n_kv_cells`.
   Sets `*out_count` to the number of floats, or 0 on error.          */
float* aptp_get_kv_cache_keys(struct llama_context* ctx, int32_t layer_id, int32_t* out_count);

/* Get KV-cache value tensor for layer `layer_id`.                      */
float* aptp_get_kv_cache_values(struct llama_context* ctx, int32_t layer_id, int32_t* out_count);

/* Clear stored hidden state snapshots.                                */
void aptp_clear_snapshots(struct llama_context* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_EXT_H */
