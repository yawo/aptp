/**
 * aptp_llama_shim.c — C bridge between APTP and llama.cpp.
 *
 * Compile alongside your llama.cpp build and link against libllama:
 *   cc -I/path/to/llama.cpp -fPIC -shared -o libaptp_shim.so aptp_llama_shim.c -lllama
 *
 * Then set LLAMA_CPP_DIR or use pkg-config for the Rust build.
 */
#include "aptp_llama_shim.h"
#include "llama.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Internal struct holding all state ─────────────────────────────────── */

typedef struct {
    struct llama_model*   model;
    struct llama_context* ctx;
    int                   embeddings_on;
    /* Cached model info (retrieved once at load time) */
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_ctx_sz;
    char    family[64];
} aptp_model_t;

/* ── Helpers ───────────────────────────────────────────────────────────── */

static void detect_family(aptp_model_t* m) {
    /* Read the model's architecture name via the llama API.
       llama_model_desc() returns something like "llama 8B Q4_0". */
    const char* desc = llama_model_desc(m->model);
    if (!desc) { snprintf(m->family, sizeof(m->family), "unknown"); return; }

    /* Extract the base architecture (first word before space or slash) */
    const char* p = desc;
    while (*p && *p != ' ' && *p != '/') p++;
    size_t len = (size_t)(p - desc);
    if (len >= sizeof(m->family)) len = sizeof(m->family) - 1;
    memcpy(m->family, desc, len);
    m->family[len] = '\0';
}

/* ── Helpers ───────────────────────────────────────────────────────────── */

/* Guard to call llama_backend_init() at most once (thread-safe on first call). */
static atomic_int g_aptp_backend_initialized = 0;

static void ensure_backend_init(void) {
    if (atomic_exchange(&g_aptp_backend_initialized, 1) == 0) {
        llama_backend_init();
    }
}

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

APTP_EXPORT void* aptp_load_model(
    const char* model_path,
    int         n_gpu_layers,
    int         n_ctx,
    int         enable_embeddings
) {
    ensure_backend_init();

    /* Model params */
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap     = 1;

    struct llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) return NULL;

    /* Context params */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = n_ctx > 0 ? (uint32_t)n_ctx : 2048;
    cparams.n_batch     = 512;
    cparams.n_ubatch    = 512;
    cparams.embeddings  = enable_embeddings ? 1 : 0;

    struct llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        return NULL;
    }

    /* Allocate shim state */
    aptp_model_t* m = (aptp_model_t*)calloc(1, sizeof(aptp_model_t));
    if (!m) {
        llama_free(ctx);
        llama_free_model(model);
        return NULL;
    }

    m->model         = model;
    m->ctx           = ctx;
    m->embeddings_on = enable_embeddings;
    m->n_vocab       = llama_n_vocab(model);
    m->n_embd        = llama_n_embd(ctx);
    m->n_layer       = llama_n_layer(ctx);
    m->n_head        = llama_n_head(ctx);
    m->n_ctx_sz      = (int32_t)llama_n_ctx(ctx);
    detect_family(m);

    return (void*)m;
}

APTP_EXPORT void aptp_free_model(void* handle) {
    if (!handle) return;
    aptp_model_t* m = (aptp_model_t*)handle;
    llama_free(m->ctx);
    llama_free_model(m->model);
    free(m);
}

/* ── Model Properties ──────────────────────────────────────────────────── */

APTP_EXPORT int32_t aptp_n_vocab(const void* handle) {
    return ((const aptp_model_t*)handle)->n_vocab;
}
APTP_EXPORT int32_t aptp_n_embd(const void* handle) {
    return ((const aptp_model_t*)handle)->n_embd;
}
APTP_EXPORT int32_t aptp_n_layer(const void* handle) {
    return ((const aptp_model_t*)handle)->n_layer;
}
APTP_EXPORT int32_t aptp_n_head(const void* handle) {
    return ((const aptp_model_t*)handle)->n_head;
}
APTP_EXPORT int32_t aptp_n_ctx(const void* handle) {
    return ((const aptp_model_t*)handle)->n_ctx_sz;
}
APTP_EXPORT const char* aptp_model_family(const void* handle) {
    return ((const aptp_model_t*)handle)->family;
}

/* ── Inference ──────────────────────────────────────────────────────────── */

APTP_EXPORT int aptp_decode(
    void*        handle,
    const int*   tokens,
    int32_t      n_tokens,
    int32_t      n_past
) {
    aptp_model_t* m = (aptp_model_t*)handle;
    struct llama_batch batch = llama_batch_get_one(
        (int32_t*)tokens,
        n_tokens,
        n_past,
        0,   /* n_seq_id */
        0    /* pos_shift */
    );
    return llama_decode(m->ctx, batch);
}

/* ── Output Extraction ──────────────────────────────────────────────────── */

APTP_EXPORT float* aptp_get_logits(void* handle) {
    return llama_get_logits(((aptp_model_t*)handle)->ctx);
}

APTP_EXPORT float* aptp_get_embedding(void* handle, int32_t pos) {
    aptp_model_t* m = (aptp_model_t*)handle;
    if (!m->embeddings_on) return NULL;
    return llama_get_embeddings_ith(m->ctx, pos);
}

/* ── Extended Extraction (weak symbols — return NULL if not patched) ────
 *
 * These are weak so that a patched llama.cpp (via llama_ext.h) can provide
 * strong overrides. When the user patches llama.cpp and links it, the
 * linker picks the patched (strong) definitions over these stubs.
 */

APTP_WEAK
APTP_EXPORT float* aptp_get_hidden_state(void* handle, int32_t layer_id, int32_t pos, int32_t* out_n) {
    (void)handle; (void)layer_id; (void)pos;
    if (out_n) *out_n = 0;
    return NULL;
}

APTP_WEAK
APTP_EXPORT float* aptp_get_kv_cache_keys(void* handle, int32_t layer_id, int32_t* out_n) {
    (void)handle; (void)layer_id;
    if (out_n) *out_n = 0;
    return NULL;
}

APTP_WEAK
APTP_EXPORT float* aptp_get_kv_cache_values(void* handle, int32_t layer_id, int32_t* out_n) {
    (void)handle; (void)layer_id;
    if (out_n) *out_n = 0;
    return NULL;
}
