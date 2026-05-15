# AGENTS.md — Agent-Primitive Transfer Protocol (APTP) v0.1
> Authoritative implementation spec for Claude Opus.  
> Generate **complete, compilable, `cargo fmt`-clean files**. No stubs. No `todo!()`. No `unwrap()` in protocol paths.

---

## 1. Mission & Scope

APTP is a **low-level neural transport library** that lets AI agents collaborate by exchanging model-native primitives (Hidden States, KV-Cache slices, Latent Thought Vectors) instead of serialized text. It plugs into any Rust agent runtime as a `tokio`-compatible async crate.

**What APTP is not:** a high-level agent orchestration protocol (that is A2A). APTP sits below A2A — it is the zero-copy data plane that A2A-level agents call when they need to share internal representations.

**Target environments:** CPU-only inference servers (llama.cpp, candle) and GPU-attached inference nodes. TLS is mandatory for cross-host; Unix sockets for intra-host.

---

## 2. Canonical Directory Layout

Generate **every file below**. Do not add extras; do not skip any.

```
aptp/
├── AGENTS.md                        # This file (do not regenerate)
├── Cargo.toml
├── build.rs
├── schemas/
│   └── aptp.capnp
└── src/
    ├── lib.rs                       # Re-exports; feature gates
    ├── config.rs                    # AptpConfig, TlsConfig
    ├── error.rs                     # AptpError (thiserror)
    ├── schema_capnp.rs              # GENERATED — include! macro only
    ├── primitives.rs                # Rust mirror structs for Cap'n Proto types
    ├── transport/
    │   ├── mod.rs
    │   ├── server.rs                # tokio TcpListener + capnp-rpc dispatch
    │   └── client.rs                # AptpClient, connection pool
    ├── rpc/
    │   ├── mod.rs
    │   ├── handshake.rs             # AgentPrimitiveTransfer::handshake impl
    │   ├── stream.rs                # AgentPrimitiveTransfer::streamPrimitives impl
    │   └── negotiate.rs             # AgentPrimitiveTransfer::negotiateAlignment impl
    ├── validation/
    │   ├── mod.rs
    │   ├── gate.rs                  # ValidationGate trait + NeuralFirewall impl
    │   └── probe.rs                 # SafetyProbe (reference vector store)
    ├── adapter/
    │   ├── mod.rs
    │   ├── manifold.rs              # ManifoldAdapter trait
    │   ├── linear.rs                # LinearProjectionAdapter
    │   └── norm_match.rs            # NormMatchAdapter
    └── bin/
        ├── aptp-server.rs
        └── aptp-client.rs
```

---

## 3. Cargo.toml (exact — do not alter versions without cause)

```toml
[package]
name    = "aptp"
version = "0.1.0"
edition = "2021"
description = "Agent-Primitive Transfer Protocol — zero-copy neural primitive transport"

[[bin]]
name = "aptp-server"
path = "src/bin/aptp-server.rs"

[[bin]]
name = "aptp-client"
path = "src/bin/aptp-client.rs"

[lib]
name = "aptp"
path = "src/lib.rs"

[dependencies]
capnp        = "0.19"
capnp-rpc    = "0.19"
tokio        = { version = "1", features = ["full"] }
tokio-util   = { version = "0.7", features = ["compat"] }
futures      = "0.3"
thiserror    = "1"
tracing      = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
bytes        = "1"
uuid         = { version = "1", features = ["v4"] }
sha2         = "0.10"
digest       = "0.10"
serde        = { version = "1", features = ["derive"] }
toml         = "0.8"
rustls       = { version = "0.23", optional = true }
tokio-rustls = { version = "0.26", optional = true }

[build-dependencies]
capnpc = "0.19"

[dev-dependencies]
criterion  = { version = "0.5", features = ["html_reports"] }
proptest   = "1"
tokio-test = "0.4"

[features]
default = ["tls"]
tls     = ["dep:rustls", "dep:tokio-rustls"]

[[bench]]
name    = "serialization"
harness = false
```

---

## 4. build.rs

```rust
fn main() {
    capnpc::CompilerCommand::new()
        .file("schemas/aptp.capnp")
        .run()
        .expect("capnp schema compilation failed");
}
```

---

## 5. Cap'n Proto Schema (`schemas/aptp.capnp`)

> Run `capnp id` once and replace the placeholder ID below before committing.

```capnp
@0xa1b2c3d4e5f60718;  # Replace: run `capnp id` to generate a fresh unique ID

# ─── Enumerations ────────────────────────────────────────────────────────────

enum AdapterType {
  identity        @0;  # Same architecture; no transform needed
  linearProjection @1; # Dimension mismatch → learned linear map
  normMatch       @2;  # Same dim, disjoint manifold → norm-alignment
}

enum HandshakeStatus {
  accepted      @0;
  rejected      @1;  # Incompatible architecture invariants
  needsAdapter  @2;  # Compatible after ManifoldAdapter negotiation
}

enum PrimitiveKind {
  hiddenState    @0;
  kvCache        @1;
  latentThought  @2;
}

# ─── Core Data Structs ───────────────────────────────────────────────────────

struct Shape {
  # Tensor dimensions, row-major. e.g. [batch, seq, hidden]
  dims @0 :List(UInt32);
}

struct KVCache {
  layerIdx @0 :UInt16;
  keys     @1 :List(Float32);  # Flattened; recover shape via `shape`
  values   @2 :List(Float32);
  shape    @3 :Shape;           # [batch, heads, seq, head_dim]
}

struct Metadata {
  timestampNs    @0 :UInt64;   # Unix nanoseconds (wall clock)
  sequenceId     @1 :UInt64;   # Monotonic per-stream counter
  sessionId      @2 :Text;     # UUIDv4 string
  compressionAlg @3 :Text;     # "none" | "lz4" | "zstd" (future)
}

struct TaggedPrimitive {
  # Every stored primitive MUST be wrapped in this. Never store bare tensors.
  provenanceHash @0 :Data;    # SHA-256(senderId || modelFingerprint || payload bytes)
  fairnessScore  @1 :Float32; # [0.0, 1.0]; 1.0 = maximally fair/neutral
  payload        @2 :PrimitivePacket;
}

struct PrimitivePacket {
  version          @0 :UInt32;  # APTP wire protocol version; current = 1
  senderId         @1 :Text;    # UUIDv4 of the sending agent instance
  modelFingerprint @2 :Data;    # SHA-256 of the model weight file (or config hash)
  layerIndex       @3 :UInt16;  # Source layer from which primitive was extracted

  primitive :union {
    hiddenState   @4 :List(Float32);  # Flattened [batch, seq, hidden_dim]
    kvCache       @5 :KVCache;
    latentThought @6 :List(Float32);  # Flattened [batch, hidden_dim]
  }

  shape    @7 :Shape;
  metadata @8 :Metadata;
}

struct ArchitectureInvariants {
  # The minimal facts two agents must agree on before streaming
  hiddenSize  @0 :UInt32;   # Model hidden dimension
  numLayers   @1 :UInt32;   # Total transformer layers
  numHeads    @2 :UInt32;   # Attention heads
  vocabSize   @3 :UInt32;   # Vocabulary size (used for compat checks)
  modelFamily @4 :Text;     # "llama" | "mistral" | "qwen" | "gemma" | ...
}

struct AgentCard {
  agentId    @0 :Text;                  # UUIDv4
  aptpVersion @1 :UInt32;              # Protocol version this agent supports
  invariants  @2 :ArchitectureInvariants;
  capabilities @3 :List(Text);         # e.g. ["hidden_state", "kv_cache"]
  publicKey   @4 :Data;                # For future mTLS / message signing
}

struct HandshakeResult {
  status         @0 :HandshakeStatus;
  assignedSession @1 :Text;            # UUIDv4; empty if rejected
  serverCard     @2 :AgentCard;
  adapterRequired @3 :AdapterType;    # Meaningful only when status = needsAdapter
  rejectionReason @4 :Text;           # Human-readable; empty if accepted
}

struct AlignmentSpec {
  # Returned by negotiateAlignment; describes the adapter the client must apply
  adapterType   @0 :AdapterType;
  sourceDim     @1 :UInt32;
  targetDim     @2 :UInt32;
  # Serialised adapter weights (e.g. a flattened [sourceDim × targetDim] matrix)
  # Empty for identity or normMatch
  weights       @3 :List(Float32);
}

# ─── RPC Interface ───────────────────────────────────────────────────────────

interface AgentPrimitiveTransfer {
  # Phase 1: Capability exchange
  handshake @0 (card :AgentCard) -> (result :HandshakeResult);

  # Phase 2: Primitive stream
  # Returns false and closes the stream if the NeuralFirewall rejects the packet.
  streamPrimitive @1 (packet :PrimitivePacket) -> (ack :Bool);

  # Phase 3: Optional adapter negotiation (call after handshake returns needsAdapter)
  negotiateAlignment @2 (sourceDim :UInt32, targetDim :UInt32, sourceFamily :Text, targetFamily :Text)
                     -> (spec :AlignmentSpec);

  # Graceful half-close: sender signals it will send no more primitives this session
  finalize @3 (sessionId :Text) -> (receivedCount :UInt64);
}
```

---

## 6. Error Hierarchy (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AptpError {
    // Transport
    #[error("TCP bind/connect failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cap'n Proto RPC error: {0}")]
    Rpc(#[from] ::capnp::Error),

    #[error("Not in schema: {0}")]
    NotInSchema(#[from] ::capnp::NotInSchema),

    // Handshake
    #[error("Handshake rejected by peer: {reason}")]
    HandshakeRejected { reason: String },

    #[error("Protocol version mismatch: local={local}, remote={remote}")]
    VersionMismatch { local: u32, remote: u32 },

    // Validation / Safety
    #[error("NeuralFirewall: off-manifold injection detected (l2={l2:.4}, cosine={cosine:.4})")]
    OffManifold { l2: f32, cosine: f32 },

    #[error("NeuralFirewall: provenance hash mismatch")]
    ProvenanceMismatch,

    #[error("NeuralFirewall: fairness score below threshold ({score:.3} < {threshold:.3})")]
    FairnessBelowThreshold { score: f32, threshold: f32 },

    // Adapter
    #[error("ManifoldAdapter: incompatible dimensions (src={src}, dst={dst})")]
    IncompatibleDimensions { src: u32, dst: u32 },

    #[error("ManifoldAdapter: adapter weights not yet negotiated")]
    AdapterNotReady,

    // Config
    #[error("Configuration error: {0}")]
    Config(String),

    // Internal invariant violations (bugs, not user errors)
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, AptpError>;
```

---

## 7. Configuration (`src/config.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Top-level configuration; load from TOML or build programmatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AptpConfig {
    pub transport: TransportConfig,
    pub validation: ValidationConfig,
    pub adapter: AdapterConfig,
    pub agent: AgentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub bind_addr: String,          // e.g. "0.0.0.0:7878"
    pub connect_timeout_ms: u64,    // default: 5000
    pub max_frame_bytes: usize,     // default: 64 * 1024 * 1024 (64 MiB)
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_pem_path: String,
    pub key_pem_path: String,
    pub ca_pem_path: Option<String>, // None = system roots
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// L2-norm upper bound for incoming vectors. Reject if exceeded.
    pub l2_norm_max: f32,           // default: 1000.0
    /// Cosine similarity minimum vs. SafetyProbe reference. Reject if below.
    pub cosine_sim_min: f32,        // default: -0.95 (block near-antipodal)
    /// Minimum acceptable fairness score in TaggedPrimitive.
    pub fairness_threshold: f32,    // default: 0.1
    /// Path to a NDJSON file of reference "safe" vectors (one JSON array per line).
    /// If None, cosine check is skipped (dev mode only).
    pub probe_vectors_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// When true, allow identity pass-through even if dimensions differ (dev mode).
    pub allow_dimension_mismatch_passthrough: bool, // default: false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_id: String,           // UUIDv4; generated at startup if empty
    pub model_family: String,       // "llama" | "mistral" | ...
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub aptp_version: u32,          // Must be 1 for this release
}

impl AptpConfig {
    pub fn from_toml_file(path: &str) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))?;
        toml::from_str(&raw)
            .map_err(|e| crate::error::AptpError::Config(e.to_string()))
    }
}
```

---

## 8. Primitive Mirror Structs (`src/primitives.rs`)

These are owned Rust types that correspond 1-to-1 with Cap'n Proto structs. Use these inside business logic; only cross into capnp builders at the transport boundary.

```rust
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const APTP_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Shape {
    pub dims: Vec<u32>,
}

impl Shape {
    pub fn element_count(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layer_idx: u16,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub shape: Shape, // [batch, heads, seq, head_dim]
}

#[derive(Debug, Clone)]
pub enum PrimitivePayload {
    HiddenState(Vec<f32>),
    KvCache(KvCache),
    LatentThought(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct Metadata {
    pub timestamp_ns: u64,
    pub sequence_id: u64,
    pub session_id: String,
    pub compression_alg: String,
}

impl Metadata {
    pub fn new_now(session_id: &str, sequence_id: u64) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            timestamp_ns,
            sequence_id,
            session_id: session_id.to_owned(),
            compression_alg: "none".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrimitivePacket {
    pub version: u32,
    pub sender_id: String,
    pub model_fingerprint: Vec<u8>, // 32-byte SHA-256
    pub layer_index: u16,
    pub payload: PrimitivePayload,
    pub shape: Shape,
    pub metadata: Metadata,
}

/// Compute provenance hash: SHA-256(sender_id || model_fingerprint || first 256 payload bytes)
pub fn provenance_hash(packet: &PrimitivePacket) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(packet.sender_id.as_bytes());
    h.update(&packet.model_fingerprint);
    let payload_bytes: &[u8] = match &packet.payload {
        PrimitivePayload::HiddenState(v) | PrimitivePayload::LatentThought(v) => {
            let byte_slice = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len().min(256) * 4)
            };
            byte_slice
        }
        PrimitivePayload::KvCache(kv) => unsafe {
            std::slice::from_raw_parts(kv.keys.as_ptr() as *const u8, kv.keys.len().min(64) * 4)
        },
    };
    h.update(payload_bytes);
    h.finalize().to_vec()
}

#[derive(Debug, Clone)]
pub struct TaggedPrimitive {
    pub provenance_hash: Vec<u8>,
    pub fairness_score: f32,
    pub payload: PrimitivePacket,
}

impl TaggedPrimitive {
    pub fn new(packet: PrimitivePacket, fairness_score: f32) -> Self {
        let provenance_hash = provenance_hash(&packet);
        Self { provenance_hash, fairness_score, payload: packet }
    }

    pub fn verify_provenance(&self) -> bool {
        let expected = provenance_hash(&self.payload);
        expected == self.provenance_hash
    }
}

#[derive(Debug, Clone)]
pub struct AgentCard {
    pub agent_id: String,
    pub aptp_version: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub model_family: String,
    pub capabilities: Vec<String>,
}

impl AgentCard {
    pub fn new_from_config(cfg: &crate::config::AgentConfig) -> Self {
        let agent_id = if cfg.agent_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            cfg.agent_id.clone()
        };
        Self {
            agent_id,
            aptp_version: cfg.aptp_version,
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_layers,
            num_heads: cfg.num_heads,
            vocab_size: cfg.vocab_size,
            model_family: cfg.model_family.clone(),
            capabilities: vec![
                "hidden_state".into(),
                "kv_cache".into(),
                "latent_thought".into(),
            ],
        }
    }
}
```

---

## 9. Validation Layer

### 9.1 `ValidationGate` Trait (`src/validation/gate.rs`)

```rust
use crate::error::Result;
use crate::primitives::PrimitivePayload;

/// All incoming primitives MUST pass through a ValidationGate before use.
pub trait ValidationGate: Send + Sync {
    fn validate(&self, payload: &PrimitivePayload) -> Result<()>;
}
```

### 9.2 `NeuralFirewall` Implementation (`src/validation/gate.rs`, continued)

```rust
use crate::config::ValidationConfig;
use crate::error::AptpError;
use crate::validation::probe::SafetyProbe;

pub struct NeuralFirewall {
    cfg: ValidationConfig,
    probe: Option<SafetyProbe>,
}

impl NeuralFirewall {
    pub fn new(cfg: ValidationConfig) -> Result<Self> {
        let probe = if let Some(ref path) = cfg.probe_vectors_path {
            Some(SafetyProbe::load(path)?)
        } else {
            None
        };
        Ok(Self { cfg, probe })
    }

    fn check_vector(&self, v: &[f32]) -> Result<()> {
        // 1. L2-norm gate
        let l2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if l2 > self.cfg.l2_norm_max {
            return Err(AptpError::OffManifold { l2, cosine: 0.0 });
        }

        // 2. Cosine-similarity gate vs. probe centroid (if probe loaded)
        if let Some(ref probe) = self.probe {
            let cosine = probe.cosine_similarity(v);
            if cosine < self.cfg.cosine_sim_min {
                return Err(AptpError::OffManifold { l2, cosine });
            }
        }
        Ok(())
    }
}

impl ValidationGate for NeuralFirewall {
    fn validate(&self, payload: &PrimitivePayload) -> Result<()> {
        match payload {
            PrimitivePayload::HiddenState(v) | PrimitivePayload::LatentThought(v) => {
                self.check_vector(v)
            }
            PrimitivePayload::KvCache(kv) => {
                self.check_vector(&kv.keys)?;
                self.check_vector(&kv.values)
            }
        }
    }
}
```

### 9.3 `SafetyProbe` (`src/validation/probe.rs`)

```rust
use crate::error::{AptpError, Result};

/// Holds reference "safe" vectors loaded from NDJSON.
/// Each line is a JSON array of f32 values.
/// The probe stores the mean (centroid) for efficient cosine checks.
pub struct SafetyProbe {
    centroid: Vec<f32>, // Normalised mean of all reference vectors
    dim: usize,
}

impl SafetyProbe {
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| AptpError::Config(format!("probe load: {e}")))?;

        let mut sum: Vec<f64> = Vec::new();
        let mut count = 0usize;

        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let vec: Vec<f32> = serde_json::from_str(line)
                .map_err(|e| AptpError::Config(format!("probe parse: {e}")))?;
            if sum.is_empty() {
                sum = vec![0.0f64; vec.len()];
            }
            for (s, &v) in sum.iter_mut().zip(vec.iter()) {
                *s += v as f64;
            }
            count += 1;
        }

        if count == 0 {
            return Err(AptpError::Config("probe file is empty".into()));
        }

        let dim = sum.len();
        let mut centroid: Vec<f32> = sum.iter().map(|&s| (s / count as f64) as f32).collect();
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            centroid.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(Self { centroid, dim })
    }

    /// Returns cosine similarity in [-1, 1] between the incoming vector and the centroid.
    pub fn cosine_similarity(&self, v: &[f32]) -> f32 {
        if v.len() != self.dim {
            return 0.0; // Dimension mismatch → treat as neutral; caller may reject separately
        }
        let dot: f32 = v.iter().zip(self.centroid.iter()).map(|(a, b)| a * b).sum();
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_v < 1e-9 { 0.0 } else { dot / norm_v }
    }
}
```

---

## 10. Manifold Adapter Layer

### 10.1 Trait (`src/adapter/manifold.rs`)

```rust
use crate::error::Result;

/// Transform a primitive vector from one model's representation space to another's.
pub trait ManifoldAdapter: Send + Sync {
    fn source_dim(&self) -> usize;
    fn target_dim(&self) -> usize;
    /// Project `input` into the target space. Output length == target_dim().
    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>>;
}
```

### 10.2 `LinearProjectionAdapter` (`src/adapter/linear.rs`)

For dimension mismatch: apply a pre-negotiated weight matrix `W ∈ ℝ^{target × source}`.  
Weights arrive from `negotiateAlignment` as a flattened row-major matrix.

```rust
use crate::adapter::manifold::ManifoldAdapter;
use crate::error::{AptpError, Result};

pub struct LinearProjectionAdapter {
    pub source_dim: usize,
    pub target_dim: usize,
    /// Row-major [target_dim × source_dim] weight matrix
    pub weights: Vec<f32>,
}

impl LinearProjectionAdapter {
    pub fn new(source_dim: usize, target_dim: usize, weights: Vec<f32>) -> Result<Self> {
        let expected = source_dim * target_dim;
        if weights.len() != expected {
            return Err(AptpError::IncompatibleDimensions {
                src: source_dim as u32,
                dst: target_dim as u32,
            });
        }
        Ok(Self { source_dim, target_dim, weights })
    }
}

impl ManifoldAdapter for LinearProjectionAdapter {
    fn source_dim(&self) -> usize { self.source_dim }
    fn target_dim(&self) -> usize { self.target_dim }

    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.source_dim {
            return Err(AptpError::IncompatibleDimensions {
                src: input.len() as u32,
                dst: self.source_dim as u32,
            });
        }
        let mut output = vec![0.0f32; self.target_dim];
        // output[i] = W[i, :] · input
        for i in 0..self.target_dim {
            let row_start = i * self.source_dim;
            output[i] = self.weights[row_start..row_start + self.source_dim]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum();
        }
        Ok(output)
    }
}
```

### 10.3 `NormMatchAdapter` (`src/adapter/norm_match.rs`)

For same dimension, disjoint manifold: scale the incoming vector so its L2-norm matches the target distribution's expected norm, then apply layer normalisation.

```rust
use crate::adapter::manifold::ManifoldAdapter;
use crate::error::Result;

pub struct NormMatchAdapter {
    pub dim: usize,
    /// Expected L2 norm in the target manifold (computed from probe stats)
    pub target_norm: f32,
}

impl ManifoldAdapter for NormMatchAdapter {
    fn source_dim(&self) -> usize { self.dim }
    fn target_dim(&self) -> usize { self.dim }

    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>> {
        let current_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if current_norm > 1e-9 { self.target_norm / current_norm } else { 1.0 };
        Ok(input.iter().map(|x| x * scale).collect())
    }
}
```

---

## 11. Transport Layer

### 11.1 Server (`src/transport/server.rs`)

```rust
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::AsyncReadExt;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

use crate::config::AptpConfig;
use crate::rpc::AptpRpcServer;

pub async fn run_server(cfg: Arc<AptpConfig>) -> crate::error::Result<()> {
    let addr: SocketAddr = cfg.transport.bind_addr.parse()
        .map_err(|e| crate::error::AptpError::Config(format!("invalid bind addr: {e}")))?;

    let listener = TcpListener::bind(addr).await?;
    info!("APTP server listening on {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!("Accepted connection from {}", peer_addr);

        stream.set_nodelay(true)?;
        let cfg = cfg.clone();

        tokio::task::spawn_local(async move {
            if let Err(e) = handle_connection(stream, cfg).await {
                error!("Connection error from {}: {}", peer_addr, e);
            }
        });
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    cfg: Arc<AptpConfig>,
) -> crate::error::Result<()> {
    use tokio_util::compat::TokioAsyncReadCompatExt;

    let (reader, writer) = stream.into_split();
    let reader = reader.compat();
    let writer = writer.compat_write();

    let network = twoparty::VatNetwork::new(
        futures::io::BufReader::new(reader),
        futures::io::BufWriter::new(writer),
        rpc_twoparty_capnp::Side::Server,
        Default::default(),
    );

    let rpc_server = AptpRpcServer::new(cfg);
    // The Cap'n Proto generated client type goes here after `capnp compile`
    // let client = crate::schema_capnp::agent_primitive_transfer::ToClient::new(rpc_server)
    //     .into_client::<capnp_rpc::Server>();
    // let rpc_system = RpcSystem::new(Box::new(network), Some(client.clone().client));
    // rpc_system.await?;
    //
    // Implementation note for Opus: fill the above using the generated capnp types.
    // The pattern is identical to the capnp-rpc calculator example.
    Ok(())
}
```

> **Implementation note for Opus:** The `handle_connection` body must be completed using the generated `schema_capnp` types after `cargo build` compiles the schema. Follow the capnp-rpc `calculator` sample pattern exactly. The `AptpRpcServer` struct in `src/rpc/mod.rs` must implement the generated `Server` trait for `AgentPrimitiveTransfer`.

### 11.2 Client (`src/transport/client.rs`)

```rust
use std::sync::Arc;
use crate::config::AptpConfig;
use crate::primitives::{AgentCard, PrimitivePacket, TaggedPrimitive};
use crate::error::Result;

/// A connected APTP client. Owns one TCP connection + RPC system.
pub struct AptpClient {
    cfg: Arc<AptpConfig>,
    session_id: Option<String>,
    sequence_counter: u64,
    // capnp client handle (concrete type injected post codegen)
}

impl AptpClient {
    pub async fn connect(cfg: Arc<AptpConfig>) -> Result<Self> {
        // 1. TcpStream::connect to cfg.transport.bind_addr
        // 2. Wrap in twoparty VatNetwork (Client side)
        // 3. Spawn RpcSystem on current_local
        // 4. Return Self with stored capnp client handle
        todo!("complete after capnp codegen; follow capnp-rpc client example")
    }

    /// Execute 3-way handshake. Must be called before streamPrimitive.
    pub async fn handshake(&mut self, card: AgentCard) -> Result<String> {
        // Returns the assigned session_id
        todo!()
    }

    /// Send a single primitive. Applies ValidationGate locally before sending.
    pub async fn send_primitive(&mut self, packet: PrimitivePacket) -> Result<bool> {
        let session_id = self.session_id.as_ref()
            .ok_or_else(|| crate::error::AptpError::Internal("handshake not completed".into()))?;
        let _tagged = TaggedPrimitive::new(packet.clone(), 1.0); // fairness scoring is caller's responsibility
        self.sequence_counter += 1;
        todo!()
    }

    pub async fn finalize(self) -> Result<u64> {
        todo!()
    }
}
```

---

## 12. RPC Handlers (`src/rpc/`)

### `src/rpc/mod.rs`
```rust
pub mod handshake;
pub mod negotiate;
pub mod stream;

use std::sync::Arc;
use crate::config::AptpConfig;
use crate::validation::gate::{NeuralFirewall, ValidationGate};

pub struct AptpRpcServer {
    pub cfg: Arc<AptpConfig>,
    pub firewall: Arc<NeuralFirewall>,
    // Session state: HashMap<session_id, SessionState> behind Arc<Mutex<>>
}

impl AptpRpcServer {
    pub fn new(cfg: Arc<AptpConfig>) -> Self {
        let firewall = Arc::new(
            NeuralFirewall::new(cfg.validation.clone())
                .expect("failed to init NeuralFirewall")
        );
        Self { cfg, firewall }
    }
}
// Implement generated capnp Server trait here using handshake::, stream::, negotiate:: modules.
```

### `src/rpc/handshake.rs`

Rules:
- Validate `card.aptp_version == APTP_VERSION`; else return `HandshakeStatus::Rejected` with `rejectionReason = "version mismatch"`
- Compare `card.invariants.hidden_size` to local config
- If dimensions differ AND config allows adapter: return `HandshakeStatus::NeedsAdapter` + `adapterRequired = AdapterType::LinearProjection`
- If same: return `HandshakeStatus::Accepted` + fresh UUIDv4 `assignedSession`
- Log every handshake result at `info!` level

### `src/rpc/stream.rs`

Rules:
1. Deserialise `PrimitivePacket` from capnp into `crate::primitives::PrimitivePacket`
2. Call `self.firewall.validate(&packet.payload)` → if Err, log at `warn!` and return `ack = false`
3. Wrap in `TaggedPrimitive::new()` — verify provenance hash matches
4. Store or forward according to session routing table
5. Return `ack = true`
6. **Never block** the tokio task; use `tokio::sync::mpsc` channel to hand off to a processing task

### `src/rpc/negotiate.rs`

Rules:
- If `source_dim == target_dim`: return `AdapterType::NormMatch`, empty weights
- If `source_dim != target_dim`: generate a random initialised (Kaiming uniform) `LinearProjectionAdapter` weight matrix, serialise to `AlignmentSpec.weights`, return `AdapterType::LinearProjection`
- Both sides store the spec; subsequent `streamPrimitive` calls must apply it before sending

---

## 13. `src/lib.rs`

```rust
pub mod config;
pub mod error;
pub mod primitives;
pub mod transport;
pub mod rpc;
pub mod validation;
pub mod adapter;

// The generated capnp module — populated after `cargo build`
#[allow(dead_code)]
pub mod schema_capnp {
    include!(concat!(env!("OUT_DIR"), "/aptp_capnp.rs"));
}
```

---

## 14. Binaries

### `src/bin/aptp-server.rs`
```rust
use std::sync::Arc;
use tokio::task::LocalSet;
use tracing_subscriber::EnvFilter;
use aptp::config::AptpConfig;
use aptp::transport::server::run_server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cfg_path = std::env::args().nth(1).unwrap_or_else(|| "aptp.toml".into());
    let cfg = Arc::new(AptpConfig::from_toml_file(&cfg_path)?);

    let local = LocalSet::new();
    local.run_until(run_server(cfg)).await?;
    Ok(())
}
```

### `src/bin/aptp-client.rs`
```rust
// Demo: connect → handshake → send 100 random hidden states → finalize
use std::sync::Arc;
use aptp::config::AptpConfig;
use aptp::primitives::{AgentCard, Metadata, PrimitivePacket, PrimitivePayload, Shape, APTP_VERSION};
use aptp::transport::client::AptpClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg_path = std::env::args().nth(1).unwrap_or_else(|| "aptp.toml".into());
    let cfg = Arc::new(AptpConfig::from_toml_file(&cfg_path)?);

    let mut client = AptpClient::connect(cfg.clone()).await?;
    let card = AgentCard::new_from_config(&cfg.agent);
    let session_id = client.handshake(card).await?;
    println!("Handshake OK — session {}", session_id);

    let hidden_dim = cfg.agent.hidden_size as usize;
    for i in 0..100u64 {
        let vec: Vec<f32> = (0..hidden_dim).map(|j| (j as f32 * 0.001) + (i as f32 * 0.0001)).collect();
        let packet = PrimitivePacket {
            version: APTP_VERSION,
            sender_id: cfg.agent.agent_id.clone(),
            model_fingerprint: vec![0u8; 32],
            layer_index: 16,
            payload: PrimitivePayload::HiddenState(vec),
            shape: Shape { dims: vec![1, 1, hidden_dim as u32] },
            metadata: Metadata::new_now(&session_id, i),
        };
        let ack = client.send_primitive(packet).await?;
        assert!(ack, "server rejected packet {i}");
    }

    let total = client.finalize().await?;
    println!("Done — server confirmed {} packets", total);
    Ok(())
}
```

---

## 15. Tests

### Unit Tests — one `#[cfg(test)]` block per module

**`src/validation/gate.rs` tests:**
```rust
#[tokio::test]
async fn test_firewall_rejects_high_norm() {
    // Build a ValidationConfig with l2_norm_max = 10.0, no probe
    // Create a vector with L2 norm = 50.0
    // Assert NeuralFirewall::validate returns Err(AptpError::OffManifold)
}

#[tokio::test]
async fn test_firewall_accepts_normal_vector() {
    // Build a vector with L2 norm = 1.0
    // Assert validate returns Ok(())
}
```

**`src/adapter/linear.rs` tests:**
```rust
#[test]
fn test_linear_projection_identity_2d() {
    // 2×2 identity matrix, input [1.0, 2.0] → output [1.0, 2.0]
}

#[test]
fn test_linear_projection_upscale() {
    // source_dim=2, target_dim=4, weights = [[1,0],[0,1],[1,1],[0,0]]
    // input [3.0, 5.0] → output [3.0, 5.0, 8.0, 0.0]
}
```

**`src/primitives.rs` tests:**
```rust
#[test]
fn test_tagged_primitive_provenance_roundtrip() {
    // Create PrimitivePacket, wrap in TaggedPrimitive::new
    // Assert verify_provenance() == true
    // Mutate sender_id, assert verify_provenance() == false
}
```

**Integration test (`tests/integration_handshake.rs`):**
```rust
// Spin up run_server in a LocalSet task on a random port
// Connect AptpClient
// Perform handshake with matching AgentCard
// Assert HandshakeStatus::Accepted
// Call finalize and assert count == 0
```

### Proptest (`src/adapter/linear.rs`)
```rust
proptest! {
    #[test]
    fn linear_adapter_output_length(dim_src in 1usize..512, dim_dst in 1usize..512) {
        // Generate random weights Vec<f32> of len dim_src * dim_dst
        // Assert adapt(input).len() == dim_dst for any input of len dim_src
    }
}
```

---

## 16. Benchmarks (`benches/serialization.rs`)

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

// Benchmark 1: PrimitivePacket capnp serialization round-trip for dims [512, 2048, 4096, 8192]
// Target: < 1ms for 8192-dim vector (p99)

// Benchmark 2: NeuralFirewall::validate at dims [512, 4096, 8192]
// Target: < 50µs

// Benchmark 3: LinearProjectionAdapter::adapt 4096→4096
// Target: < 500µs

criterion_group!(benches, bench_serialization, bench_firewall, bench_adapter);
criterion_main!(benches);
```

---

## 17. Observability

All async boundaries and RPC entry points must emit `tracing` spans:

```rust
// Span naming convention:
//   aptp::handshake      (per connection)
//   aptp::stream_receive (per packet)
//   aptp::validate       (per packet)
//   aptp::adapt          (per packet requiring ManifoldAdapter)
//   aptp::finalize       (per session)

// Required fields on aptp::stream_receive:
//   session_id, sequence_id, payload_kind ("hidden_state"|"kv_cache"|"latent_thought"), vector_dim
```

Log levels:
- `error!` — connection drops, firewall panics
- `warn!`  — firewall rejections, provenance mismatches
- `info!`  — handshake accept/reject, session open/close
- `debug!` — per-packet ack, adapter invocation
- `trace!` — raw capnp buffer sizes

---

## 18. Definition of Done

Every box must be checkable by running `cargo test --all && cargo bench`:

- [ ] `cargo build` succeeds with no warnings
- [ ] `cargo clippy -- -D warnings` passes
- [ ] All unit tests pass
- [ ] `test_firewall_rejects_high_norm` and `test_tagged_primitive_provenance_roundtrip` explicitly included
- [ ] Integration handshake test passes (server + client on loopback)
- [ ] Bench `bench_serialization` shows < 1ms for 8192-dim round-trip on a modern laptop
- [ ] No `unwrap()` in `src/transport/`, `src/rpc/`, or `src/validation/`
- [ ] `schema_capnp.rs` is generated (not hand-written)
- [ ] Every `TaggedPrimitive` write site calls `verify_provenance()` before trusting the payload
- [ ] `ManifoldAdapter` is invoked whenever `HandshakeResult.adapterRequired != identity`
- [ ] Server handles connection close gracefully (no panic on EOF)

---

## 19. What Opus Must NOT Do

- Do not use `grpc` or `tonic`; the transport is `capnp-rpc` over raw TCP
- Do not use `unwrap()` or `expect()` in `src/rpc/`, `src/transport/`, `src/validation/`
- Do not invent crates not listed in §3
- Do not skip `TaggedPrimitive` wrapping for any stored primitive
- Do not generate placeholder / stub implementations — every file must be complete and compilable
- Do not reference "Gemini 3.1 Pro" or any specific external model — use model family strings only
- Do not write the `schema_capnp.rs` file by hand; it is always `include!(concat!(env!("OUT_DIR"), ...))` only

---

## 20. Sample `aptp.toml` (generate this file at project root)

```toml
[transport]
bind_addr          = "0.0.0.0:7878"
connect_timeout_ms = 5000
max_frame_bytes    = 67108864   # 64 MiB

# Omit [transport.tls] entirely to disable TLS (loopback / dev only)
[transport.tls]
cert_pem_path = "certs/server.crt"
key_pem_path  = "certs/server.key"
# ca_pem_path  = "certs/ca.crt"   # Uncomment for mTLS

[validation]
l2_norm_max          = 1000.0
cosine_sim_min       = -0.95
fairness_threshold   = 0.10
# probe_vectors_path = "probes/safe_vectors.ndjson"  # Omit to skip cosine check

[adapter]
allow_dimension_mismatch_passthrough = false

[agent]
agent_id     = ""            # Empty → generate UUIDv4 at startup
model_family = "llama"
hidden_size  = 4096
num_layers   = 32
num_heads    = 32
vocab_size   = 32000
aptp_version = 1
```

---

## 21. Missing `serde_json` Dependency (add to Cargo.toml)

The `SafetyProbe` loader in `src/validation/probe.rs` uses `serde_json`. Add this line to `[dependencies]` in `Cargo.toml`:

```toml
serde_json = "1"
```

---

## 22. Module `mod.rs` Files (generate all four)

These files are **not** optional — `cargo build` will fail without them.

### `src/transport/mod.rs`
```rust
pub mod client;
pub mod server;
```

### `src/rpc/mod.rs`
```rust
pub mod handshake;
pub mod negotiate;
pub mod stream;

use std::sync::Arc;
use crate::config::AptpConfig;
use crate::error::{AptpError, Result};
use crate::validation::gate::NeuralFirewall;

pub struct AptpRpcServer {
    pub cfg: Arc<AptpConfig>,
    pub firewall: Arc<NeuralFirewall>,
    pub sessions: Arc<tokio::sync::Mutex<SessionRegistry>>,
}

impl AptpRpcServer {
    pub fn new(cfg: Arc<AptpConfig>) -> Result<Self> {
        let firewall = Arc::new(NeuralFirewall::new(cfg.validation.clone())?);
        Ok(Self {
            cfg,
            firewall,
            sessions: Arc::new(tokio::sync::Mutex::new(SessionRegistry::default())),
        })
    }
}

// Implement the capnp-generated `agent_primitive_transfer::Server` trait on AptpRpcServer.
// Each method delegates to the corresponding sub-module:
//   handshake()            → handshake::handle(...)
//   stream_primitive()     → stream::handle(...)
//   negotiate_alignment()  → negotiate::handle(...)
//   finalize()             → session registry lookup + counter drain
//
// Trait impl goes here after `cargo build` generates schema_capnp.rs.
```

### `src/validation/mod.rs`
```rust
pub mod gate;
pub mod probe;

pub use gate::{NeuralFirewall, ValidationGate};
pub use probe::SafetyProbe;
```

### `src/adapter/mod.rs`
```rust
pub mod linear;
pub mod manifold;
pub mod norm_match;

pub use linear::LinearProjectionAdapter;
pub use manifold::ManifoldAdapter;
pub use norm_match::NormMatchAdapter;

use crate::error::{AptpError, Result};

/// Factory: select the correct adapter from an `AlignmentSpec`.
pub fn from_spec(
    adapter_type: AdapterVariant,
    source_dim: usize,
    target_dim: usize,
    weights: Vec<f32>,
) -> Result<Box<dyn ManifoldAdapter>> {
    match adapter_type {
        AdapterVariant::Identity => {
            // Pass-through: return NormMatchAdapter with target_norm = 1.0 (no-op at unit norm)
            Ok(Box::new(NormMatchAdapter { dim: source_dim, target_norm: 1.0 }))
        }
        AdapterVariant::LinearProjection => {
            Ok(Box::new(LinearProjectionAdapter::new(source_dim, target_dim, weights)?))
        }
        AdapterVariant::NormMatch => {
            let target_norm = if weights.len() == 1 { weights[0] } else { 1.0 };
            Ok(Box::new(NormMatchAdapter { dim: source_dim, target_norm }))
        }
    }
}

/// Rust-side enum mirroring the capnp `AdapterType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterVariant {
    Identity,
    LinearProjection,
    NormMatch,
}
```

---

## 23. Session Registry (`src/rpc/mod.rs`, continued)

The session registry tracks live sessions and per-session packet counters. Generate it in the same `src/rpc/mod.rs` file, below `AptpRpcServer`:

```rust
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: HashMap<String, SessionState>,
}

#[derive(Debug)]
pub struct SessionState {
    pub session_id: String,
    pub peer_agent_id: String,
    pub received_count: u64,
    pub adapter: Option<crate::adapter::AdapterVariant>,
    pub opened_at_ns: u64,
}

impl SessionRegistry {
    pub fn open(&mut self, peer_agent_id: String, adapter: Option<crate::adapter::AdapterVariant>) -> String {
        let session_id = Uuid::new_v4().to_string();
        let opened_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.sessions.insert(session_id.clone(), SessionState {
            session_id: session_id.clone(),
            peer_agent_id,
            received_count: 0,
            adapter,
            opened_at_ns,
        });
        session_id
    }

    pub fn increment(&mut self, session_id: &str) -> Result<()> {
        self.sessions
            .get_mut(session_id)
            .ok_or_else(|| AptpError::Internal(format!("unknown session: {session_id}")))?
            .received_count += 1;
        Ok(())
    }

    pub fn drain(&mut self, session_id: &str) -> Result<u64> {
        self.sessions
            .remove(session_id)
            .map(|s| s.received_count)
            .ok_or_else(|| AptpError::Internal(format!("unknown session: {session_id}")))
    }
}
```

---

## 24. Cap'n Proto ↔ Rust Conversion Bridge (`src/primitives.rs`, continued)

Add these conversion functions at the bottom of `src/primitives.rs`. They form the only crossing point between capnp types and Rust types — all other code uses the mirror structs.

```rust
use crate::error::{AptpError, Result};

/// Deserialise a capnp `PrimitivePacket` reader into an owned `PrimitivePacket`.
/// Call this at the top of every RPC handler that receives a packet.
///
/// Implementation note for Opus:
///   Replace every `todo!()` with the actual capnp field accessor for the
///   generated reader type `crate::schema_capnp::primitive_packet::Reader<'_>`.
pub fn packet_from_capnp<'a>(
    reader: crate::schema_capnp::primitive_packet::Reader<'a>,
) -> Result<PrimitivePacket> {
    let version = reader.get_version();
    let sender_id = reader.get_sender_id()?.to_str()
        .map_err(|e| AptpError::Internal(e.to_string()))?.to_owned();
    let model_fingerprint = reader.get_model_fingerprint()?.to_vec();
    let layer_index = reader.get_layer_index();

    let payload = match reader.get_primitive().which()? {
        crate::schema_capnp::primitive_packet::primitive::HiddenState(r) => {
            let floats: Vec<f32> = r?.iter().collect();
            PrimitivePayload::HiddenState(floats)
        }
        crate::schema_capnp::primitive_packet::primitive::LatentThought(r) => {
            let floats: Vec<f32> = r?.iter().collect();
            PrimitivePayload::LatentThought(floats)
        }
        crate::schema_capnp::primitive_packet::primitive::KvCache(r) => {
            let kv = r?;
            PrimitivePayload::KvCache(KvCache {
                layer_idx: kv.get_layer_idx(),
                keys: kv.get_keys()?.iter().collect(),
                values: kv.get_values()?.iter().collect(),
                shape: shape_from_capnp(kv.get_shape()?)?,
            })
        }
    };

    let shape_reader = reader.get_shape()?;
    let shape = shape_from_capnp(shape_reader)?;

    let meta_reader = reader.get_metadata()?;
    let metadata = Metadata {
        timestamp_ns: meta_reader.get_timestamp_ns(),
        sequence_id: meta_reader.get_sequence_id(),
        session_id: meta_reader.get_session_id()?.to_str()
            .map_err(|e| AptpError::Internal(e.to_string()))?.to_owned(),
        compression_alg: meta_reader.get_compression_alg()?.to_str()
            .map_err(|e| AptpError::Internal(e.to_string()))?.to_owned(),
    };

    Ok(PrimitivePacket { version, sender_id, model_fingerprint, layer_index, payload, shape, metadata })
}

fn shape_from_capnp(
    reader: crate::schema_capnp::shape::Reader<'_>,
) -> Result<Shape> {
    Ok(Shape { dims: reader.get_dims()?.iter().collect() })
}

/// Serialise an owned `PrimitivePacket` into a capnp `Builder`.
/// Returns a `Vec<u8>` (packed serialization) ready to send over the wire.
pub fn packet_to_bytes(packet: &PrimitivePacket) -> Result<Vec<u8>> {
    use capnp::message::Builder;
    use capnp::serialize_packed;

    let mut message = Builder::new_default();
    {
        let mut root = message.init_root::<crate::schema_capnp::primitive_packet::Builder<'_>>();
        root.set_version(packet.version);
        root.set_sender_id(&packet.sender_id);
        root.set_model_fingerprint(&packet.model_fingerprint);
        root.set_layer_index(packet.layer_index);

        match &packet.payload {
            PrimitivePayload::HiddenState(v) => {
                let mut list = root.reborrow().init_primitive().init_hidden_state(v.len() as u32);
                for (i, &val) in v.iter().enumerate() { list.set(i as u32, val); }
            }
            PrimitivePayload::LatentThought(v) => {
                let mut list = root.reborrow().init_primitive().init_latent_thought(v.len() as u32);
                for (i, &val) in v.iter().enumerate() { list.set(i as u32, val); }
            }
            PrimitivePayload::KvCache(kv) => {
                let mut kv_builder = root.reborrow().init_primitive().init_kv_cache();
                kv_builder.set_layer_idx(kv.layer_idx);
                let mut keys = kv_builder.reborrow().init_keys(kv.keys.len() as u32);
                for (i, &v) in kv.keys.iter().enumerate() { keys.set(i as u32, v); }
                let mut vals = kv_builder.reborrow().init_values(kv.values.len() as u32);
                for (i, &v) in kv.values.iter().enumerate() { vals.set(i as u32, v); }
                let mut shape = kv_builder.init_shape();
                let mut dims = shape.init_dims(kv.shape.dims.len() as u32);
                for (i, &d) in kv.shape.dims.iter().enumerate() { dims.set(i as u32, d); }
            }
        }

        // Shape
        let mut shape_builder = root.reborrow().init_shape();
        let mut dims = shape_builder.init_dims(packet.shape.dims.len() as u32);
        for (i, &d) in packet.shape.dims.iter().enumerate() { dims.set(i as u32, d); }

        // Metadata
        let mut meta = root.reborrow().init_metadata();
        meta.set_timestamp_ns(packet.metadata.timestamp_ns);
        meta.set_sequence_id(packet.metadata.sequence_id);
        meta.set_session_id(&packet.metadata.session_id);
        meta.set_compression_alg(&packet.metadata.compression_alg);
    }

    let mut buf = Vec::new();
    serialize_packed::write_message(&mut buf, &message)
        .map_err(AptpError::Rpc)?;
    Ok(buf)
}
```

---

## 25. Kaiming Uniform Weight Initialisation (`src/rpc/negotiate.rs`)

The `negotiateAlignment` handler must return a properly initialised weight matrix — not zeros.  
Generate this helper in `src/rpc/negotiate.rs`:

```rust
/// Kaiming uniform initialisation for a [out_dim × in_dim] weight matrix.
/// Gain = 1.0 (linear activation). Matches PyTorch `torch.nn.init.kaiming_uniform_`.
pub fn kaiming_uniform_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let bound = (3.0f32 / in_dim as f32).sqrt();
    let n = in_dim * out_dim;

    // Deterministic pseudo-random (LCG) — no external rand crate required.
    // Seed from current time to avoid identical matrices across sessions.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();

    let mut state = seed as u64 ^ 0xdeadbeef_cafebabe;
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to [-bound, bound]
        let t = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        weights.push(t * bound);
    }
    weights
}

/// Full `negotiateAlignment` handler. Call from the capnp Server trait impl.
pub fn handle(
    source_dim: u32,
    target_dim: u32,
    _source_family: &str,
    _target_family: &str,
) -> crate::error::Result<(crate::adapter::AdapterVariant, Vec<f32>)> {
    if source_dim == 0 || target_dim == 0 {
        return Err(crate::error::AptpError::IncompatibleDimensions {
            src: source_dim,
            dst: target_dim,
        });
    }
    if source_dim == target_dim {
        // Same dimension: NormMatch, target norm = 1.0 (encoded as single-element weight)
        Ok((crate::adapter::AdapterVariant::NormMatch, vec![1.0f32]))
    } else {
        let weights = kaiming_uniform_weights(source_dim as usize, target_dim as usize);
        Ok((crate::adapter::AdapterVariant::LinearProjection, weights))
    }
}
```

---

## 26. Handshake Handler (`src/rpc/handshake.rs`)

```rust
use crate::config::AptpConfig;
use crate::error::{AptpError, Result};
use crate::primitives::{AgentCard, APTP_VERSION};
use crate::rpc::SessionRegistry;
use crate::adapter::AdapterVariant;
use tracing::{info, warn};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

pub async fn handle(
    remote_card: AgentCard,
    cfg: Arc<AptpConfig>,
    sessions: Arc<Mutex<SessionRegistry>>,
) -> Result<(crate::schema_capnp::handshake_status::Which, String, AdapterVariant, String)> {
    // Returns: (status, session_id_or_empty, adapter_variant, rejection_reason)

    // 1. Version check
    if remote_card.aptp_version != APTP_VERSION {
        let reason = format!(
            "version mismatch: remote={}, local={}",
            remote_card.aptp_version, APTP_VERSION
        );
        warn!(
            agent_id = %remote_card.agent_id,
            remote_version = remote_card.aptp_version,
            local_version = APTP_VERSION,
            "Handshake rejected: {}", reason
        );
        return Ok((
            crate::schema_capnp::handshake_status::Which::Rejected(()),
            String::new(),
            AdapterVariant::Identity,
            reason,
        ));
    }

    // 2. Dimension compatibility check
    let local_hidden = cfg.agent.hidden_size;
    let remote_hidden = remote_card.hidden_size;

    let (status, adapter) = if local_hidden == remote_hidden {
        (crate::schema_capnp::handshake_status::Which::Accepted(()), AdapterVariant::Identity)
    } else if !cfg.adapter.allow_dimension_mismatch_passthrough {
        (crate::schema_capnp::handshake_status::Which::NeedsAdapter(()), AdapterVariant::LinearProjection)
    } else {
        warn!(
            local_hidden,
            remote_hidden,
            "Dimension mismatch passthrough enabled (dev mode)"
        );
        (crate::schema_capnp::handshake_status::Which::Accepted(()), AdapterVariant::Identity)
    };

    // 3. Open session if accepted or needs-adapter (not rejected)
    let session_id = {
        let mut reg = sessions.lock().await;
        reg.open(remote_card.agent_id.clone(), Some(adapter))
    };

    info!(
        agent_id = %remote_card.agent_id,
        model_family = %remote_card.model_family,
        remote_hidden,
        local_hidden,
        session_id = %session_id,
        ?adapter,
        "Handshake complete"
    );

    Ok((status, session_id, adapter, String::new()))
}
```

---

## 27. Stream Handler (`src/rpc/stream.rs`)

```rust
use crate::error::{AptpError, Result};
use crate::primitives::{packet_from_capnp, TaggedPrimitive};
use crate::rpc::SessionRegistry;
use crate::validation::gate::ValidationGate;
use tracing::{debug, warn};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Per-session processing channel. Spin one up per accepted session.
pub struct PacketSink {
    tx: mpsc::Sender<TaggedPrimitive>,
}

impl PacketSink {
    pub fn new(buffer: usize) -> (Self, mpsc::Receiver<TaggedPrimitive>) {
        let (tx, rx) = mpsc::channel(buffer);
        (Self { tx }, rx)
    }

    pub async fn send(&self, tagged: TaggedPrimitive) -> Result<()> {
        self.tx.send(tagged).await
            .map_err(|_| AptpError::Internal("PacketSink channel closed".into()))
    }
}

/// Called by the capnp Server trait impl for each `streamPrimitive` RPC.
/// Returns `true` (ack) on success, `false` on firewall rejection.
/// Never panics.
pub async fn handle<'a>(
    reader: crate::schema_capnp::primitive_packet::Reader<'a>,
    firewall: Arc<dyn ValidationGate>,
    sessions: Arc<Mutex<SessionRegistry>>,
    sink: Arc<PacketSink>,
) -> Result<bool> {
    // 1. Deserialise
    let packet = match packet_from_capnp(reader) {
        Ok(p) => p,
        Err(e) => {
            warn!(error = %e, "Failed to deserialise PrimitivePacket");
            return Ok(false);
        }
    };

    let session_id = packet.metadata.session_id.clone();
    let seq = packet.metadata.sequence_id;
    let kind = match &packet.payload {
        crate::primitives::PrimitivePayload::HiddenState(_) => "hidden_state",
        crate::primitives::PrimitivePayload::KvCache(_)     => "kv_cache",
        crate::primitives::PrimitivePayload::LatentThought(_) => "latent_thought",
    };

    let span = tracing::info_span!(
        "aptp::stream_receive",
        session_id = %session_id,
        sequence_id = seq,
        payload_kind = kind,
    );
    let _enter = span.enter();

    // 2. Firewall validation
    if let Err(e) = firewall.validate(&packet.payload) {
        warn!(error = %e, "NeuralFirewall rejected packet");
        return Ok(false);
    }

    // 3. Wrap in TaggedPrimitive and verify provenance
    let tagged = TaggedPrimitive::new(packet, 1.0);
    if !tagged.verify_provenance() {
        warn!("Provenance hash mismatch — rejecting packet");
        return Err(AptpError::ProvenanceMismatch);
    }

    // 4. Increment session counter (non-blocking — fire and forget if lock is contended)
    {
        let mut reg = sessions.lock().await;
        reg.increment(&session_id)?;
    }

    // 5. Hand off to processing channel — never block the RPC dispatcher
    sink.send(tagged).await?;

    debug!(sequence_id = seq, "Packet accepted and queued");
    Ok(true)
}
```

---

## 28. Benchmark Completions (`benches/serialization.rs`)

Generate this file in full — `criterion_group!` and `criterion_main!` must be present or the bench target will not compile.

```rust
use aptp::primitives::{
    packet_to_bytes, KvCache, Metadata, PrimitivePacket, PrimitivePayload, Shape, APTP_VERSION,
};
use aptp::config::ValidationConfig;
use aptp::validation::gate::{NeuralFirewall, ValidationGate};
use aptp::adapter::linear::LinearProjectionAdapter;
use aptp::adapter::manifold::ManifoldAdapter;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_hidden_state_packet(dim: usize) -> PrimitivePacket {
    PrimitivePacket {
        version: APTP_VERSION,
        sender_id: "bench-agent".to_owned(),
        model_fingerprint: vec![0u8; 32],
        layer_index: 16,
        payload: PrimitivePayload::HiddenState(vec![0.1f32; dim]),
        shape: Shape { dims: vec![1, 1, dim as u32] },
        metadata: Metadata::new_now("bench-session", 0),
    }
}

// ── Benchmark 1: Serialization round-trip ────────────────────────────────────

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_serialization");
    for dim in [512usize, 2048, 4096, 8192] {
        group.bench_with_input(BenchmarkId::new("hidden_state_dim", dim), &dim, |b, &dim| {
            let packet = make_hidden_state_packet(dim);
            b.iter(|| {
                let bytes = packet_to_bytes(&packet).expect("serialization failed");
                criterion::black_box(bytes);
            });
        });
    }
    group.finish();
}

// ── Benchmark 2: NeuralFirewall validation ────────────────────────────────────

fn bench_firewall(c: &mut Criterion) {
    let cfg = ValidationConfig {
        l2_norm_max: 1000.0,
        cosine_sim_min: -0.95,
        fairness_threshold: 0.1,
        probe_vectors_path: None,
    };
    let fw = NeuralFirewall::new(cfg).expect("firewall init");

    let mut group = c.benchmark_group("firewall_validate");
    for dim in [512usize, 4096, 8192] {
        let payload = PrimitivePayload::HiddenState(vec![0.01f32; dim]);
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                fw.validate(criterion::black_box(&payload)).ok();
            });
        });
    }
    group.finish();
}

// ── Benchmark 3: LinearProjectionAdapter ─────────────────────────────────────

fn bench_adapter(c: &mut Criterion) {
    let dim = 4096usize;
    let weights = vec![0.001f32; dim * dim];
    let adapter = LinearProjectionAdapter::new(dim, dim, weights).expect("adapter init");
    let input = vec![0.1f32; dim];

    c.bench_function("linear_adapter_4096x4096", |b| {
        b.iter(|| {
            adapter.adapt(criterion::black_box(&input)).ok();
        });
    });
}

criterion_group!(benches, bench_serialization, bench_firewall, bench_adapter);
criterion_main!(benches);
```

---

## 29. Final Dependency Reconciliation (complete `[dependencies]` block)

Replace the `[dependencies]` section in `Cargo.toml` with this reconciled version that adds `serde_json` and `anyhow` (used in binaries):

```toml
[dependencies]
capnp          = "0.19"
capnp-rpc      = "0.19"
tokio          = { version = "1",    features = ["full"] }
tokio-util     = { version = "0.7",  features = ["compat"] }
futures        = "0.3"
thiserror      = "1"
tracing        = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
bytes          = "1"
uuid           = { version = "1",    features = ["v4"] }
sha2           = "0.10"
digest         = "0.10"
serde          = { version = "1",    features = ["derive"] }
serde_json     = "1"
toml           = "0.8"
anyhow         = "1"
rustls         = { version = "0.23", optional = true }
tokio-rustls   = { version = "0.26", optional = true }
```

---

## 30. Opus Execution Order

Opus MUST generate files in this exact order to avoid forward-reference errors:

1. `Cargo.toml` (§3 + §29 reconciliation)
2. `build.rs` (§4)
3. `schemas/aptp.capnp` (§5)
4. `src/error.rs` (§6)
5. `src/config.rs` (§7)
6. `src/primitives.rs` (§8 + §24 conversion bridge)
7. `src/validation/mod.rs` → `src/validation/probe.rs` → `src/validation/gate.rs` (§22 + §9)
8. `src/adapter/mod.rs` → `src/adapter/manifold.rs` → `src/adapter/linear.rs` → `src/adapter/norm_match.rs` (§22 + §10)
9. `src/rpc/mod.rs` (§22 + §23 session registry)
10. `src/rpc/handshake.rs` (§26)
11. `src/rpc/stream.rs` (§27)
12. `src/rpc/negotiate.rs` (§25)
13. `src/transport/mod.rs` → `src/transport/server.rs` → `src/transport/client.rs` (§22 + §11)
14. `src/lib.rs` (§13)
15. `src/bin/aptp-server.rs` (§14)
16. `src/bin/aptp-client.rs` (§14)
17. `benches/serialization.rs` (§28)
18. `aptp.toml` (§20)
19. `tests/integration_handshake.rs` (§15)

After all files exist: run `cargo build 2>&1`, fix any compilation errors iteratively, then run `cargo test --all`.
