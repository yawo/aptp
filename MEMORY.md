# APTP — Session Memory

## Project Identity

- **APTP** = Agent-Primitive Transfer Protocol v0.1
- Low-level neural transport for AI agents to share model-native primitives (hidden states, KV-cache, latent thought vectors)
- Sits *below* A2A — zero-copy data plane, not orchestration
- Rust crate, Cap'n Proto RPC + tokio

## Key Design Decisions

- **Transport:** `capnp-rpc` over raw TCP, NOT gRPC/tonic
- **No `unwrap()`** in transport/, rpc/, or validation/ paths
- **Every primitive** wrapped in `TaggedPrimitive` (provenance-hashed) before storage/forwarding
- **schema_capnp.rs** is auto-generated via `include!` — never hand-written
- **TLS** optional via `tls` feature (default on), uses `rustls` + `tokio-rustls`

## Architecture

| Layer | Responsibility |
|-------|---------------|
| `validation/` | `NeuralFirewall` (L2-norm + cosine gate) + `SafetyProbe` |
| `adapter/` | `ManifoldAdapter` trait, `LinearProjection`, `NormMatch` |
| `rpc/` | 4 RPC handlers: `handshake`, `streamPrimitive`, `negotiateAlignment`, `finalize` |
| `transport/` | TCP listener/client, capnp-rpc dispatch |

## Protocol Phases

1. **Handshake** — version check, dimension compat, session creation
2. **Stream** — send primitives through firewall + provenance check
3. **Negotiate** — if dim mismatch, server returns Kaiming-initialized linear weights
4. **Finalize** — graceful half-close, server returns received count

## Safety Contract

- `NeuralFirewall.validate()` called on every incoming `PrimitivePayload`
- `TaggedPrimitive.verify_provenance()` checks SHA-256 before trusting payload
- `ManifoldAdapter` invoked when `HandshakeResult.adapterRequired != Identity`

## Dependencies (notable)

- `capnp` + `capnp-rpc` v0.19, `capnpc` v0.19 (build-dep)
- `tokio` with `full` features, `tokio-util` compat
- `sha2` + `digest` for provenance hashing
- `uuid` v1 (v4), `serde` + `toml` for config
- `serde_json` for probe vector loading
- `anyhow` for binary error handling

## Conventions

- `AptpError` enum with typed variants, `Result<T>` = `std::result::Result<T, AptpError>`
- `config.rs` uses `serde::Deserialize` from TOML
- `primitives.rs` has mirror structs + `packet_from_capnp`/`packet_to_bytes` bridge
- Tracing spans: `aptp::handshake`, `aptp::stream_receive`, `aptp::validate`, `aptp::adapt`, `aptp::finalize`
- Log levels: error (drops/panics), warn (rejections), info (handshake/session), debug (per-packet ack), trace (buffer sizes)

## Build Commands

```sh
cargo build
cargo test --all
cargo bench
cargo clippy -- -D warnings
```

Prerequisite: `apt install capnproto`
