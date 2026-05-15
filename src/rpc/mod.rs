pub mod handshake;
pub mod negotiate;
pub mod stream;

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::adapter::AdapterVariant;
use crate::config::AptpConfig;
use crate::error::{AptpError, Result};
use crate::primitives::TaggedPrimitive;
use crate::validation::gate::NeuralFirewall;

pub struct AptpRpcServer {
    pub cfg: Arc<AptpConfig>,
    pub firewall: Arc<NeuralFirewall>,
    pub sessions: Arc<tokio::sync::Mutex<SessionRegistry>>,
    pub packet_sink: Arc<PacketSink>,
}

impl AptpRpcServer {
    pub fn new(cfg: Arc<AptpConfig>, sink_buffer: usize) -> Result<Self> {
        let firewall = Arc::new(NeuralFirewall::new(cfg.validation.clone())?);
        let (sink, _rx) = PacketSink::new(sink_buffer);
        Ok(Self {
            cfg,
            firewall,
            sessions: Arc::new(tokio::sync::Mutex::new(SessionRegistry::default())),
            packet_sink: Arc::new(sink),
        })
    }
}

pub struct PacketSink {
    tx: tokio::sync::mpsc::Sender<TaggedPrimitive>,
}

impl PacketSink {
    pub fn new(buffer: usize) -> (Self, tokio::sync::mpsc::Receiver<TaggedPrimitive>) {
        let (tx, rx) = tokio::sync::mpsc::channel(buffer);
        (Self { tx }, rx)
    }

    pub async fn send(&self, tagged: TaggedPrimitive) -> Result<()> {
        self.tx.send(tagged).await
            .map_err(|_| AptpError::Internal("PacketSink channel closed".into()))
    }
}

#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: HashMap<String, SessionState>,
}

#[derive(Debug)]
pub struct SessionState {
    pub session_id: String,
    pub peer_agent_id: String,
    pub received_count: u64,
    pub adapter: Option<AdapterVariant>,
    pub opened_at_ns: u64,
}

impl SessionRegistry {
    pub fn open(&mut self, peer_agent_id: String, adapter: Option<AdapterVariant>) -> String {
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

impl crate::aptp_capnp::agent_primitive_transfer::Server for AptpRpcServer {
    fn handshake(
        &mut self,
        params: crate::aptp_capnp::agent_primitive_transfer::HandshakeParams,
        mut results: crate::aptp_capnp::agent_primitive_transfer::HandshakeResults,
    ) -> ::capnp::capability::Promise<(), ::capnp::Error> {
        let cfg = self.cfg.clone();
        let sessions = self.sessions.clone();
        let _span = tracing::info_span!("aptp::handshake").entered();
        ::capnp::capability::Promise::from_future(async move {
            let params_reader = params.get().map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            let card_reader = params_reader.get_card()
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            let card = crate::primitives::agent_card_from_capnp(card_reader)
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;

            let (status, session_id, adapter, reason) = handshake::handle(card, cfg.clone(), sessions).await
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;

            let mut h_result = results.get().init_result();
            h_result.set_assigned_session(&session_id);
            h_result.set_rejection_reason(&reason);
            h_result.set_status(status);

            {
                let mut server_card = h_result.reborrow().init_server_card();
                server_card.set_agent_id(&cfg.agent.agent_id);
                server_card.set_aptp_version(crate::primitives::APTP_VERSION);
                server_card.set_public_key(&[]);
                {
                    let mut inv = server_card.reborrow().init_invariants();
                    inv.set_hidden_size(cfg.agent.hidden_size);
                    inv.set_num_layers(cfg.agent.num_layers);
                    inv.set_num_heads(cfg.agent.num_heads);
                    inv.set_vocab_size(cfg.agent.vocab_size);
                    inv.set_model_family(&cfg.agent.model_family);
                }
                {
                    let mut caps = server_card.reborrow().init_capabilities(3);
                    caps.set(0, "hidden_state");
                    caps.set(1, "kv_cache");
                    caps.set(2, "latent_thought");
                }
            }

            let adapter_type = match adapter {
                AdapterVariant::Identity => crate::aptp_capnp::AdapterType::Identity,
                AdapterVariant::LinearProjection => crate::aptp_capnp::AdapterType::LinearProjection,
                AdapterVariant::NormMatch => crate::aptp_capnp::AdapterType::NormMatch,
            };
            h_result.set_adapter_required(adapter_type);

            Ok(())
        })
    }

    fn stream_primitive(
        &mut self,
        params: crate::aptp_capnp::agent_primitive_transfer::StreamPrimitiveParams,
        mut results: crate::aptp_capnp::agent_primitive_transfer::StreamPrimitiveResults,
    ) -> ::capnp::capability::Promise<(), ::capnp::Error> {
        let firewall = self.firewall.clone();
        let sessions = self.sessions.clone();
        let sink = self.packet_sink.clone();
        ::capnp::capability::Promise::from_future(async move {
            let packet_reader = params.get().map_err(|e| ::capnp::Error::failed(e.to_string()))?.get_packet()
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            match stream::handle(packet_reader, firewall, sessions, sink).await {
                Ok(ack) => {
                    results.get().set_ack(ack);
                    Ok(())
                }
                Err(e) => Err(::capnp::Error::failed(e.to_string())),
            }
        })
    }

    fn negotiate_alignment(
        &mut self,
        params: crate::aptp_capnp::agent_primitive_transfer::NegotiateAlignmentParams,
        mut results: crate::aptp_capnp::agent_primitive_transfer::NegotiateAlignmentResults,
    ) -> ::capnp::capability::Promise<(), ::capnp::Error> {
        ::capnp::capability::Promise::from_future(async move {
            let p = params.get().map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            let source_dim = p.get_source_dim();
            let target_dim = p.get_target_dim();
            let source_family = p.get_source_family()
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?
                .to_str().unwrap_or("unknown").to_owned();
            let target_family = p.get_target_family()
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?
                .to_str().unwrap_or("unknown").to_owned();

            let _span = tracing::info_span!("aptp::adapt", source_dim, target_dim).entered();

            let (adapter_variant, weights) = negotiate::handle(source_dim, target_dim, &source_family, &target_family)
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;

            let mut spec = results.get().init_spec();
            let adapter_type = match adapter_variant {
                AdapterVariant::Identity => crate::aptp_capnp::AdapterType::Identity,
                AdapterVariant::LinearProjection => crate::aptp_capnp::AdapterType::LinearProjection,
                AdapterVariant::NormMatch => crate::aptp_capnp::AdapterType::NormMatch,
            };
            spec.set_adapter_type(adapter_type);
            spec.set_source_dim(source_dim);
            spec.set_target_dim(target_dim);
            {
                let mut w = spec.init_weights(weights.len() as u32);
                for (i, &v) in weights.iter().enumerate() {
                    w.set(i as u32, v);
                }
            }
            Ok(())
        })
    }

    fn finalize(
        &mut self,
        params: crate::aptp_capnp::agent_primitive_transfer::FinalizeParams,
        mut results: crate::aptp_capnp::agent_primitive_transfer::FinalizeResults,
    ) -> ::capnp::capability::Promise<(), ::capnp::Error> {
        let sessions = self.sessions.clone();
        ::capnp::capability::Promise::from_future(async move {
            let session_info = params.get().map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            let sid = session_info.get_session_id()
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?
                .to_str().map_err(|e| ::capnp::Error::failed(e.to_string()))?.to_owned();

            let _span = tracing::info_span!("aptp::finalize", session_id = %sid).entered();

            let mut reg = sessions.lock().await;
            let count = reg.drain(&sid)
                .map_err(|e| ::capnp::Error::failed(e.to_string()))?;
            results.get().set_received_count(count);
            Ok(())
        })
    }
}
