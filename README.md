# APTP — Agent-Primitive Transfer Protocol

**Low-level neural transport library** for AI agents to exchange model-native
primitives (hidden states, KV-cache slices, latent thought vectors) over
the wire. Built on **Cap'n Proto RPC** + **tokio**.

> APTP sits below A2A. It is the zero-copy data plane that high-level agent
> orchestration protocols call when they need to share internal representations.

---

## Features

- **Three primitive kinds** — `HiddenState`, `KvCache`, `LatentThought`
- **Provenance-hashed packets** — every primitive is SHA-256 tagged
- **Neural firewall** — L2-norm gate + probe-based cosine similarity check
- **Manifold adapters** — `LinearProjection`, `NormMatch`, `Identity`
- **Dimension negotiation** — agents auto-detect and negotiate alignment
- **Cap'n Proto RPC** — compact binary wire format, zero-copy deserialization
- **TLS optional** — enable via `tls` feature (default on)

---

## Quick Start

### Dependencies

```bash
# Cap'n Proto schema compiler (required for build)
apt install capnproto
```

### Build

```bash
cargo build --release
```

### Run

```bash
# Server
./target/release/aptp-server aptp.toml

# Client (separate terminal)
./target/release/aptp-client aptp.toml
```

### Test

```bash
cargo test --all
cargo bench
```

---

## Architecture

```
src/
├── bin/
│   ├── aptp-server.rs     # TCP server binary
│   └── aptp-client.rs     # Demo client binary
├── lib.rs                 # Crate root, module exports
├── config.rs              # AptpConfig (TOML deserialization)
├── error.rs               # AptpError enum (thiserror)
├── primitives.rs          # Mirror structs + capnp conversion
├── adapter/
│   ├── mod.rs             # AdapterVariant enum + from_spec factory
│   ├── manifold.rs        # ManifoldAdapter trait
│   ├── linear.rs          # LinearProjectionAdapter
│   └── norm_match.rs      # NormMatchAdapter
├── rpc/
│   ├── mod.rs             # AptpRpcServer, SessionRegistry, PacketSink
│   ├── handshake.rs       # 3-way handshake handler
│   ├── stream.rs          # Primitive stream handler + firewall
│   └── negotiate.rs       # Alignment negotiation + Kaiming init
├── transport/
│   ├── mod.rs
│   ├── server.rs          # tokio TcpListener + capnp-rpc dispatch
│   └── client.rs          # AptpClient with connection pool
└── validation/
    ├── mod.rs
    ├── gate.rs            # NeuralFirewall (ValidationGate impl)
    └── probe.rs           # SafetyProbe (NDJSON reference vectors)
```

### Wire Protocol (4 RPCs)

| Method | Purpose |
|--------|---------|
| `handshake` | Capability exchange, version check, dimension compat |
| `streamPrimitive` | Send a provenance-tagged primitive |
| `negotiateAlignment` | Request adapter weights for dim mismatch |
| `finalize` | Graceful half-close, returns received count |

### Data Flow

```
Client                  Server
  │                       │
  ├── handshake() ──────► │  (version + dimension check)
  │◄──── HandshakeResult ─┤
  │                       │
  ├── streamPrimitive() ► │  (firewall validate + provenance check)
  │◄──── ack ─────────────┤
  │      ...              │
  ├── finalize() ────────►│
  │◄──── receivedCount ───┤
```

---

## Configuration

See [`aptp.toml`](./aptp.toml) for a full example:

```toml
[transport]
bind_addr          = "0.0.0.0:7878"
connect_timeout_ms = 5000
max_frame_bytes    = 67108864

[validation]
l2_norm_max          = 1000.0
cosine_sim_min       = -0.95
fairness_threshold   = 0.10
# probe_vectors_path = "probes/safe_vectors.ndjson"

[adapter]
allow_dimension_mismatch_passthrough = false

[agent]
agent_id     = ""            # Empty → auto-generate UUIDv4
model_family = "llama"
hidden_size  = 4096
num_layers   = 32
num_heads    = 32
vocab_size   = 32000
aptp_version = 1
```

---

## Adapters

When two agents have different hidden dimensions, APTP negotiates an adapter:

| Adapter | When | What it does |
|---------|------|--------------|
| `Identity` | Same architecture | No-op passthrough |
| `LinearProjection` | Dim mismatch | Learns `W ∈ ℝ^{target × source}` via Kaiming init |
| `NormMatch` | Same dim, disjoint manifold | Scales L2 norm to match target distribution |

---

## Neural Firewall

Every incoming primitive passes through `NeuralFirewall` before acceptance:

1. **L2-norm gate** — reject vectors exceeding `l2_norm_max`
2. **Cosine-similarity gate** — optional, requires probe vectors file
3. **Provenance verification** — SHA-256 hash must match

---

## Benchmarks

```bash
cargo bench
```

Target (modern laptop):
- 8192-dim packet serialization: **< 1ms** (p99)
- Firewall validation @ 8192 dim: **< 50µs**
- Linear adapter 4096→4096: **< 500µs**

---

## Crate Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tls`   | yes     | TLS support via `rustls` + `tokio-rustls` |

---

## Project Status

v0.1 — Proof-of-concept. Not yet production-ready.

---
