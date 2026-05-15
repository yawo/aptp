use std::ffi::{CStr, CString};
use libc::{c_char, c_int, c_float};
use sha2::{Digest, Sha256};

use crate::backend::{InferenceSnapshot, PrimitiveExtractor};
use crate::error::{AptpError, Result};
use crate::primitives::KvCache;
use crate::config::BackendConfig;

/* ── FFI declarations: C shim (aptp_llama_shim.c) ─────────────────────── */

extern "C" {
    fn aptp_load_model(
        path: *const c_char,
        n_gpu_layers: c_int,
        n_ctx: c_int,
        enable_embeddings: c_int,
    ) -> *mut std::ffi::c_void;

    fn aptp_free_model(handle: *mut std::ffi::c_void);

    fn aptp_n_vocab(handle: *const std::ffi::c_void) -> c_int;
    fn aptp_n_embd(handle: *const std::ffi::c_void) -> c_int;
    fn aptp_n_layer(handle: *const std::ffi::c_void) -> c_int;
    fn aptp_n_head(handle: *const std::ffi::c_void) -> c_int;
    fn aptp_n_ctx(handle: *const std::ffi::c_void) -> c_int;
    fn aptp_model_family(handle: *const std::ffi::c_void) -> *const c_char;

    fn aptp_decode(
        handle: *mut std::ffi::c_void,
        tokens: *const c_int,
        n_tokens: c_int,
        n_past: c_int,
    ) -> c_int;

    fn aptp_get_logits(handle: *mut std::ffi::c_void) -> *mut c_float;
    fn aptp_get_embedding(handle: *mut std::ffi::c_void, pos: c_int) -> *mut c_float;

    fn aptp_get_hidden_state(
        handle: *mut std::ffi::c_void,
        layer_id: c_int,
        pos: c_int,
        out_n: *mut c_int,
    ) -> *mut c_float;

    fn aptp_get_kv_cache_keys(
        handle: *mut std::ffi::c_void,
        layer_id: c_int,
        out_n: *mut c_int,
    ) -> *mut c_float;

    fn aptp_get_kv_cache_values(
        handle: *mut std::ffi::c_void,
        layer_id: c_int,
        out_n: *mut c_int,
    ) -> *mut c_float;
}

/* ── LlamaCppBackend ───────────────────────────────────────────────────── */

pub struct LlamaCppBackend {
    handle: *mut std::ffi::c_void,
    n_vocab: u32,
    n_embd: u32,
    n_layer: u32,
    n_head: u32,
    n_ctx: u32,
    family: String,
    fingerprint: Vec<u8>,
    n_past: u32,
    embeddings_on: bool,
}

impl LlamaCppBackend {
    /// Load a model via the C shim.
    ///
    /// Requires `libaptp_shim` and `libllama` to be linkable at build time
    /// (enabled via the `backend-llama` feature).
    pub fn new(config: &BackendConfig) -> Result<Self> {
        let model_path = CString::new(config.model_path.as_str())
            .map_err(|e| AptpError::Config(format!("model path contains null byte: {e}")))?;

        let handle = unsafe {
            aptp_load_model(
                model_path.as_ptr(),
                config.n_gpu_layers as c_int,
                config.n_ctx as c_int,
                config.enable_embeddings as c_int,
            )
        };

        if handle.is_null() {
            return Err(AptpError::Config(format!(
                "failed to load model from '{}'",
                config.model_path
            )));
        }

        let n_vocab = unsafe { aptp_n_vocab(handle) as u32 };
        let n_embd = unsafe { aptp_n_embd(handle) as u32 };
        let n_layer = unsafe { aptp_n_layer(handle) as u32 };
        let n_head = unsafe { aptp_n_head(handle) as u32 };
        let n_ctx_sz = unsafe { aptp_n_ctx(handle) as u32 };

        let family = unsafe {
            let ptr = aptp_model_family(handle);
            if ptr.is_null() {
                "unknown".to_owned()
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("unknown").to_owned()
            }
        };

        let fingerprint = compute_model_fingerprint(&config.model_path, n_vocab, n_embd, n_layer, n_head);

        Ok(Self {
            handle,
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_ctx: n_ctx_sz,
            family,
            fingerprint,
            n_past: 0,
            embeddings_on: config.enable_embeddings,
        })
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { aptp_free_model(self.handle) };
        }
    }
}

impl PrimitiveExtractor for LlamaCppBackend {
    fn hidden_size(&self) -> u32 { self.n_embd }
    fn num_layers(&self) -> u32 { self.n_layer }
    fn num_heads(&self) -> u32 { self.n_head }
    fn vocab_size(&self) -> u32 { self.n_vocab }
    fn model_family(&self) -> &str { &self.family }
    fn model_fingerprint(&self) -> Vec<u8> { self.fingerprint.clone() }

    fn infer(&mut self, tokens: &[i32]) -> Result<InferenceSnapshot> {
        let n_past = self.n_past as c_int;
        let n_toks = tokens.len();

        let ret = unsafe {
            aptp_decode(
                self.handle,
                tokens.as_ptr(),
                n_toks as c_int,
                n_past,
            )
        };
        if ret != 0 {
            return Err(AptpError::Internal(format!(
                "llama_decode returned {}",
                ret
            )));
        }

        self.n_past += n_toks as u32;

        let logits = unsafe {
            let ptr = aptp_get_logits(self.handle);
            if ptr.is_null() {
                return Err(AptpError::Internal("aptp_get_logits returned NULL".into()));
            }
            std::slice::from_raw_parts(ptr, self.n_vocab as usize).to_vec()
        };

        // Capture hidden states for the LAST token in the batch
        let last_pos = (n_toks.saturating_sub(1)) as c_int;
        let hidden_states = self.capture_hidden_states(last_pos);
        let kv_cache_layer = self.capture_kv_cache();

        Ok(InferenceSnapshot {
            n_past: self.n_past,
            logits,
            hidden_states,
            kv_cache_layer,
        })
    }
}

/* ── Internal extraction helpers ───────────────────────────────────────── */

impl LlamaCppBackend {
    /// Extract per-layer hidden states for the token at `token_pos` in the batch.
    ///
    /// With stock llama.cpp (via `enable_embeddings=true`):
    ///   returns a single vector = the final layer embedding.
    ///
    /// With patched llama.cpp (via `llama_ext.h`):
    ///   returns one vector per transformer layer.
    fn capture_hidden_states(&self, token_pos: c_int) -> Vec<Vec<f32>> {
        let n_embd = self.n_embd as usize;

        if self.embeddings_on {
            let emb_ptr = unsafe { aptp_get_embedding(self.handle, token_pos) };
            if !emb_ptr.is_null() {
                return vec![unsafe { std::slice::from_raw_parts(emb_ptr, n_embd) }.to_vec()];
            }
        }

        let mut states: Vec<Vec<f32>> = vec![];
        for layer in 0..self.n_layer as c_int {
            let mut out_n: c_int = 0;
            let ptr = unsafe { aptp_get_hidden_state(self.handle, layer, token_pos, &mut out_n) };
            if ptr.is_null() || out_n <= 0 {
                continue;
            }
            states.push(unsafe { std::slice::from_raw_parts(ptr, out_n as usize) }.to_vec());
        }

        states
    }

    /// Extract KV-cache data for the last layer (most recent).
    ///
    /// Only works with patched llama.cpp (`llama_ext.h`).
    /// Returns `None` if the extended API is unavailable.
    fn capture_kv_cache(&self) -> Option<KvCache> {
        let last_layer = (self.n_layer.saturating_sub(1)) as c_int;
        let mut n_keys: c_int = 0;

        let key_ptr = unsafe { aptp_get_kv_cache_keys(self.handle, last_layer, &mut n_keys) };
        if key_ptr.is_null() || n_keys <= 0 {
            return None;
        }

        let mut n_vals: c_int = 0;
        let val_ptr = unsafe { aptp_get_kv_cache_values(self.handle, last_layer, &mut n_vals) };
        if val_ptr.is_null() || n_vals <= 0 {
            return None;
        }

        let keys = unsafe { std::slice::from_raw_parts(key_ptr, n_keys as usize) }.to_vec();
        let values = unsafe { std::slice::from_raw_parts(val_ptr, n_vals as usize) }.to_vec();

        Some(KvCache {
            layer_idx: self.n_layer.saturating_sub(1) as u16,
            keys,
            values,
            shape: crate::primitives::Shape {
                dims: vec![1, self.n_head, self.n_past, self.n_embd / self.n_head],
            },
        })
    }
}

/* ── Model fingerprint ─────────────────────────────────────────────────── */

/// Compute a deterministic fingerprint from model config parameters.
///
/// This avoids hashing the full weights file (which can be 100s of GB).
/// Two models with identical architecture produce the same fingerprint;
/// for weight-level uniqueness, hash a sample of the model file instead.
fn compute_model_fingerprint(
    path: &str,
    n_vocab: u32,
    n_embd: u32,
    n_layer: u32,
    n_head: u32,
) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update(&n_vocab.to_le_bytes());
    h.update(&n_embd.to_le_bytes());
    h.update(&n_layer.to_le_bytes());
    h.update(&n_head.to_le_bytes());
    h.finalize().to_vec()
}

/* ── Tokenization helper ───────────────────────────────────────────────── */

/// Build a `PrimitivePacket` from an inference snapshot, ready to send via APTP.
///
/// ```ignore
/// let packet = backend::llama::snapshot_to_packet(&snapshot, &backend, &session_id, seq);
/// client.send_primitive(packet).await?;
/// ```
pub fn snapshot_to_packet(
    snapshot: &InferenceSnapshot,
    backend: &dyn PrimitiveExtractor,
    sender_id: &str,
    session_id: &str,
    sequence_id: u64,
    layer_index: u16,
) -> crate::primitives::PrimitivePacket {
    use crate::primitives::{Metadata, PrimitivePacket, PrimitivePayload, Shape, APTP_VERSION};

    let hidden_dim = backend.hidden_size() as u32;

        let payload = if let Some(ref kv) = snapshot.kv_cache_layer {
        PrimitivePayload::KvCache(kv.clone())
    } else if !snapshot.hidden_states.is_empty() {
        let idx = (layer_index as usize).min(snapshot.hidden_states.len() - 1);
        let state = snapshot.hidden_states[idx].clone();
        PrimitivePayload::HiddenState(state)
    } else {
        PrimitivePayload::LatentThought(snapshot.logits.clone())
    };

    let dims = match &payload {
        PrimitivePayload::HiddenState(_) => vec![1u32, 1, hidden_dim],
        PrimitivePayload::LatentThought(_) => vec![1, hidden_dim],
        PrimitivePayload::KvCache(kv) => kv.shape.dims.clone(),
    };

    PrimitivePacket {
        version: APTP_VERSION,
        sender_id: sender_id.to_owned(),
        model_fingerprint: backend.model_fingerprint(),
        layer_index,
        payload,
        shape: Shape { dims },
        metadata: Metadata::new_now(session_id, sequence_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fingerprint_is_deterministic() {
        let a = compute_model_fingerprint("/models/llama.gguf", 32000, 4096, 32, 32);
        let b = compute_model_fingerprint("/models/llama.gguf", 32000, 4096, 32, 32);
        assert_eq!(a, b);
    }

    #[test]
    fn test_compute_fingerprint_changes_with_params() {
        let a = compute_model_fingerprint("/models/llama.gguf", 32000, 4096, 32, 32);
        let b = compute_model_fingerprint("/models/llama.gguf", 32001, 4096, 32, 32);
        assert_ne!(a, b);
    }
}
