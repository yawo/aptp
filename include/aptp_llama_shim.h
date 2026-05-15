#ifndef APTP_LLAMA_SHIM_H
#define APTP_LLAMA_SHIM_H

#include <stdint.h>

#ifdef _WIN32
#define APTP_EXPORT   __declspec(dllexport)
#define APTP_WEAK
#else
#define APTP_EXPORT   __attribute__((visibility("default")))
#define APTP_WEAK     __attribute__((weak))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

/* Load a llama.cpp model. Returns opaque handle, or NULL on failure. */
APTP_EXPORT void* aptp_load_model(
    const char* model_path,
    int         n_gpu_layers,
    int         n_ctx,
    int         enable_embeddings
);

/* Free all resources associated with a loaded model. */
APTP_EXPORT void aptp_free_model(void* handle);

/* ── Model Properties ───────────────────────────────────────────────────── */

APTP_EXPORT int32_t aptp_n_vocab(const void* handle);
APTP_EXPORT int32_t aptp_n_embd(const void* handle);
APTP_EXPORT int32_t aptp_n_layer(const void* handle);
APTP_EXPORT int32_t aptp_n_head(const void* handle);
APTP_EXPORT int32_t aptp_n_ctx(const void* handle);

/* Return the model family string (e.g. "llama", "mistral"). Static storage. */
APTP_EXPORT const char* aptp_model_family(const void* handle);

/* ── Inference ──────────────────────────────────────────────────────────── */

/* Run a forward pass over `n_tokens` tokens starting at position `n_past`.
   Returns 0 on success, non-zero on failure. */
APTP_EXPORT int aptp_decode(
    void*        handle,
    const int*   tokens,
    int32_t      n_tokens,
    int32_t      n_past
);

/* ── Output Extraction (stock llama.cpp API) ──────────────────────────── */

/* Get logits for the most recent token. Shape: [n_vocab]. */
APTP_EXPORT float* aptp_get_logits(void* handle);

/* Get the final-layer embedding for the token at `pos`.
   Only valid if `enable_embeddings` was true at load time.
   Returns NULL if unavailable. */
APTP_EXPORT float* aptp_get_embedding(void* handle, int32_t pos);

/* ── Extended Extraction (requires llama_ext.h patch) ────────────────── */
/* These functions return NULL if the extended API is not linked (weak stubs).
   Patch llama.cpp via `llama_ext.h` and recompile for full primitive extraction. */

/* Get hidden state at layer `layer_id` (0-indexed) for token at position `pos`.
   On success: writes `*out_n = n_embd`, returns pointer to float array.
   On failure: returns NULL. */
APTP_EXPORT float* aptp_get_hidden_state(void* handle, int32_t layer_id, int32_t pos, int32_t* out_n);

/* Get KV-cache keys for layer `layer_id`.
   On success: writes `*out_n = n_embd_head * n_cells`, returns pointer to float array.
   On failure: returns NULL. */
APTP_EXPORT float* aptp_get_kv_cache_keys(void* handle, int32_t layer_id, int32_t* out_n);

/* Get KV-cache values for layer `layer_id`. Returns NULL if unavailable. */
APTP_EXPORT float* aptp_get_kv_cache_values(void* handle, int32_t layer_id, int32_t* out_n);

#ifdef __cplusplus
}
#endif

#endif /* APTP_LLAMA_SHIM_H */
