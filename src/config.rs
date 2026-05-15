use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AptpConfig {
    pub transport: TransportConfig,
    pub validation: ValidationConfig,
    pub adapter: AdapterConfig,
    pub agent: AgentConfig,
    #[serde(default)]
    pub backend: Option<BackendConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub bind_addr: String,
    pub connect_timeout_ms: u64,
    pub max_frame_bytes: usize,
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_pem_path: String,
    pub key_pem_path: String,
    pub ca_pem_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub l2_norm_max: f32,
    pub cosine_sim_min: f32,
    pub fairness_threshold: f32,
    pub probe_vectors_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub allow_dimension_mismatch_passthrough: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_id: String,
    pub model_family: String,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub aptp_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Path to a GGUF model file.
    pub model_path: String,
    /// Number of layers to offload to GPU (0 = CPU only).
    #[serde(default = "default_gpu_layers")]
    pub n_gpu_layers: u32,
    /// Context size (max tokens).
    #[serde(default = "default_n_ctx")]
    pub n_ctx: u32,
    /// Enable embedding output (required for hidden state extraction via stock API).
    #[serde(default)]
    pub enable_embeddings: bool,
}

fn default_gpu_layers() -> u32 { 0 }
fn default_n_ctx() -> u32 { 2048 }

impl AptpConfig {
    pub fn from_toml_file(path: &str) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))?;
        toml::from_str(&raw)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))
    }
}
