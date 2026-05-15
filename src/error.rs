use thiserror::Error;

#[derive(Debug, Error)]
pub enum AptpError {
    #[error("TCP bind/connect failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cap'n Proto RPC error: {0}")]
    Rpc(#[from] ::capnp::Error),

    #[error("Not in schema: {0}")]
    NotInSchema(#[from] ::capnp::NotInSchema),

    #[error("Handshake rejected by peer: {reason}")]
    HandshakeRejected { reason: String },

    #[error("Protocol version mismatch: local={local}, remote={remote}")]
    VersionMismatch { local: u32, remote: u32 },

    #[error("NeuralFirewall: off-manifold injection detected (l2={l2:.4}, cosine={cosine:.4})")]
    OffManifold { l2: f32, cosine: f32 },

    #[error("NeuralFirewall: provenance hash mismatch")]
    ProvenanceMismatch,

    #[error("NeuralFirewall: fairness score below threshold ({score:.3} < {threshold:.3})")]
    FairnessBelowThreshold { score: f32, threshold: f32 },

    #[error("ManifoldAdapter: incompatible dimensions (src={src}, dst={dst})")]
    IncompatibleDimensions { src: u32, dst: u32 },

    #[error("ManifoldAdapter: adapter weights not yet negotiated")]
    AdapterNotReady,

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, AptpError>;
