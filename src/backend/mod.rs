use crate::error::Result;
use crate::primitives::KvCache;

/// Snapshot of model internals captured after a single inference step.
/// Fields are populated based on what the backend can extract —
/// stock llama.cpp gives only logits/embeddings;
/// a patched llama.cpp (via `llama_ext.h`) gives full per-layer states.
pub struct InferenceSnapshot {
    /// Number of tokens processed so far (position in KV-cache).
    pub n_past: u32,
    /// Output logits over the vocabulary — always available. Shape: `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Per-layer hidden states, indexed by layer.
    /// With stock llama.cpp: `vec![]` unless embeddings are enabled (then `vec![final_layer]`).
    /// With patched llama.cpp: one `[hidden_size]` vector per layer.
    pub hidden_states: Vec<Vec<f32>>,
    /// KV-cache snapshot for one selected layer.
    /// Only available with patched llama.cpp (`llama_ext.h`).
    pub kv_cache_layer: Option<KvCache>,
}

impl InferenceSnapshot {
    pub fn new_no_hidden(n_past: u32, logits: Vec<f32>) -> Self {
        Self { n_past, logits, hidden_states: vec![], kv_cache_layer: None }
    }
}

/// Backend-agnostic primitive extractor.
///
/// Implementations connect APTP to a local inference engine (`llama.cpp`,
/// `candle`, etc.) and extract model-native primitives during forward passes.
///
/// # Contract
/// - `infer()` must not panic. Errors must be returned via `Result`.
/// - All returned values must be **immutable views** — the caller may
///   clone them for APTP transmission but must not hold references into
///   the backend's internal state across `infer()` calls.
pub trait PrimitiveExtractor: Send {
    fn hidden_size(&self) -> u32;
    fn num_layers(&self) -> u32;
    fn num_heads(&self) -> u32;
    fn vocab_size(&self) -> u32;
    fn model_family(&self) -> &str;
    /// Deterministic fingerprint of the loaded model (SHA-256 of config params).
    fn model_fingerprint(&self) -> Vec<u8>;
    /// Run the forward pass on the given token sequence and capture primitives.
    fn infer(&mut self, tokens: &[i32]) -> Result<InferenceSnapshot>;
}

#[cfg(feature = "backend-llama")]
pub mod llama;

#[cfg(feature = "backend-llama")]
pub use llama::LlamaCppBackend;
