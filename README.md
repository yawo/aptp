# APTP — Agent-Primitive Transfer Protocol

**Low-level neural transport library** for AI agents to exchange model-native
primitives (hidden states, KV-cache slices, latent thought vectors) over
the wire. Built on **Cap'n Proto RPC** + **tokio**.

> APTP sits below A2A. It is the zero-copy data plane that high-level agent
> orchestration protocols call when they need to share internal representations.

---

## The Problem

Today's AI agents communicate the same way humans do: **serialized text**. One
agent generates a string, pipes it to another agent's context window, and that
agent re-encodes it into its own latent space. This works, but it is wildly
inefficient and lossy:

| Problem | Consequence |
|---------|-------------|
| **Text is a bottleneck** | Agents must decompress full thoughts into language tokens, losing sub-word-level nuance. The receiving agent then re-encodes from scratch. |
| **KV-cache is discarded** | Every text-based handoff discards the sender's attention state. The receiver cannot pick up where the sender left off — it must re-derive all context. |
| **Latent information is lost** | A model's hidden states encode intent, uncertainty, and representational geometry that text cannot express. By the time a thought is serialized to tokens, this signal is gone. |
| **Latency overhead** | Serialize → transmit → decode → re-encode adds 100ms+ per hop for a 4096-dim vector vs. ~50µs for raw tensor transport. |
| **No safety gates at the primitive level** | Text protocols inspect *strings* for safety. But a maliciously crafted hidden state can carry adversarial perturbations invisible to string-based filters. |

**The result:** multi-agent systems today are chatrooms, not swarms. Each agent
operates in isolation, sharing only what can be crammed into natural language.
They cannot share internal representations, cannot collaborate on reasoning at
the representation level, and cannot inspect each other's primitives for safety.

APTP solves this by defining a **wire protocol for neural primitives** — the
actual data structures that LLMs natively operate on, serialized without loss
through Cap'n Proto's zero-copy binary format.

### Who needs this

- **Multi-agent orchestrators** (e.g., A2A implementations) that want agents to
  share KV-cache state instead of re-prompting from scratch.
- **Model routing layers** that dispatch inference subtasks across heterogeneous
  models (different architectures, dimensions, or families).
- **Agent safety infrastructure** that needs to inspect and validate
  representations, not just text, before they reach a downstream model.
- **Collaborative reasoning systems** where agents build on each other's latent
  thought vectors — speculative decoding chains, ensemble verifiers,
  multi-perspective reasoners.

## Features

- **Three primitive kinds** — `HiddenState`, `KvCache`, `LatentThought`
- **Provenance-hashed packets** — every primitive is SHA-256 tagged
- **Neural firewall** — L2-norm gate + probe-based cosine similarity check
- **Manifold adapters** — `LinearProjection`, `NormMatch`, `Identity`
- **Dimension negotiation** — agents auto-detect and negotiate alignment
- **Cap'n Proto RPC** — compact binary wire format, zero-copy deserialization
- **TLS optional** — enable via `tls` feature (default on)

---

## Multi-Agent Topologies

APTP's wire protocol is **language-agnostic** — any agent that speaks Cap'n Proto RPC can
connect, regardless of its host language (Rust, Python, C++, Node.js, etc.).
The [`schemas/aptp.capnp`](./schemas/aptp.capnp) file **is** the protocol contract.
Non-Rust agents need only implement the 4 RPC methods defined there.

```
Primitive kinds: HiddenState, KvCache, LatentThought
RPC methods:     handshake → streamPrimitive* → negotiateAlignment? → finalize
```

### 2-Agent Peer-to-Peer

Each agent runs both a server (to receive) and a client (to send). This lets two
agents exchange primitives bidirectionally.

```
┌─────────────────────────┐          ┌─────────────────────────┐
│  Agent A (Pi / Gemini)  │          │  Agent B (Pi / Gemini)  │
│                         │          │                         │
│  Server → port 7878     │◄────────►│  Server → port 7879     │
│  Client ───────────────►│          │  Client ───────────────►│
│  (connects to Agent B)  │          │  (connects to Agent A)  │
└─────────────────────────┘          └─────────────────────────┘
```

**Rust setup for Agent A:**
```rust
use std::sync::Arc;
use tokio::task::LocalSet;
use aptp::config::AptpConfig;
use aptp::transport::client::AptpClient;
use aptp::transport::server::run_server;

let cfg_a = Arc::new(AptpConfig::from_toml_file("agent_a.toml")?);

// Agent A listens on port 7878
let server = run_server(cfg_a.clone());

// Agent A connects TO Agent B on port 7879
let mut to_b = AptpClient::connect_to(cfg_a.clone(), "127.0.0.1:7879").await?;
let session = to_b.handshake(agent_card).await?;

// Both run concurrently
tokio::join!(server, async {
    to_b.send_primitive(packet).await?;
    to_b.finalize().await?;
});
```

### 3-Agent Master + Workers

One orchestrator (Master) distributes work to two Workers. Each worker runs an
APTP server; the Master creates one client per worker.

```
┌────────────────────────┐
│        Master          │
│  (Gemini / Opencode)   │
│                        │
│  Client ───────────────►  Worker 1  ←── Server on port 7879
│  Client ───────────────►  Worker 2  ←── Server on port 7880
└────────────────────────┘
```

**Rust setup for Master:**
```rust
use std::sync::Arc;
use tokio::task::LocalSet;
use aptp::config::AptpConfig;
use aptp::primitives::AgentCard;
use aptp::transport::client::AptpClient;

let cfg = Arc::new(AptpConfig::from_toml_file("master.toml")?);
let card = AgentCard::new_from_config(&cfg.agent);

LocalSet::new().run_until(async {
    // Connect to both workers
    let mut w1 = AptpClient::connect_to(cfg.clone(), "192.168.1.10:7879").await?;
    let mut w2 = AptpClient::connect_to(cfg.clone(), "192.168.1.11:7880").await?;

    let s1 = w1.handshake(card.clone()).await?;
    let s2 = w2.handshake(card).await?;

    // Send primitives to each worker
    w1.send_primitive(hidden_state_packet).await?;
    w2.send_primitive(kv_cache_packet).await?;

    let count1 = w1.finalize().await?;
    let count2 = w2.finalize().await?;
}).await;
```

**Rust setup for each Worker (e.g. `worker.toml` on port 7879):**
```rust
use std::sync::Arc;
use tokio::task::LocalSet;
use aptp::config::AptpConfig;
use aptp::transport::server::run_server;

let cfg = Arc::new(AptpConfig::from_toml_file("worker.toml")?);
let local = LocalSet::new();
local.run_until(run_server(cfg)).await?;
```

### From Non-Rust Agents

Any language with a Cap'n Proto implementation can connect to APTP:

1. Copy [`schemas/aptp.capnp`](./schemas/aptp.capnp) into your project
2. Generate bindings: `capnp compile -o <lang> schemas/aptp.capnp`
3. Implement the `AgentPrimitiveTransfer` RPC client interface
4. Open a TCP connection, bootstrap Cap'n Proto RPC, call `handshake`

**Python** (using `pycapnp`):
```python
import capnp
import aptp_capnp

client = await capnp.TwoPartyClient.create("host:port")
aptp = client.bootstrap().cast_as(aptp_capnp.AgentPrimitiveTransfer)
result = await aptp.handshake(card)
print(f"Session: {result.assignedSession}")
```

**JavaScript/Node.js** (using `capnp-ts`):
```typescript
import { connect } from 'capnp-ts/rpc';
import { AgentPrimitiveTransfer } from './aptp.capnp';

const conn = connect({ host, port });
const client = conn.bootstrap<AgentPrimitiveTransfer>();
const result = await client.handshake({ card });
console.log(`Session: ${result.assignedSession}`);
```

This means Pi agents, Gemini agents, Opencode, or any other agent runtime can
participate in an APTP mesh — they just need a Cap'n Proto RPC implementation
for their language.

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
