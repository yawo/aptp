use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AptpConfig {
    pub transport: TransportConfig,
    pub validation: ValidationConfig,
    pub adapter: AdapterConfig,
    pub agent: AgentConfig,
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

impl AptpConfig {
    pub fn from_toml_file(path: &str) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))?;
        toml::from_str(&raw)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))
    }
}
