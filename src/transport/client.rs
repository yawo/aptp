use std::sync::Arc;
use tokio::net::TcpStream;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::AptpConfig;
use crate::error::Result;
use crate::primitives::{AgentCard, PrimitivePacket};

pub struct AptpClient {
    #[allow(dead_code)]
    cfg: Arc<AptpConfig>,
    session_id: Option<String>,
    sequence_counter: u64,
    client: crate::aptp_capnp::agent_primitive_transfer::Client,
}

impl AptpClient {
    /// Connect to the address specified in `cfg.transport.bind_addr`.
    pub async fn connect(cfg: Arc<AptpConfig>) -> Result<Self> {
        Self::connect_to(cfg.clone(), &cfg.transport.bind_addr).await
    }

    /// Connect to a specific target address (bypasses `cfg.transport.bind_addr`).
    ///
    /// Useful when one agent orchestrates multiple workers on different ports:
    ///
    /// ```ignore
    /// let cfg = Arc::new(AptpConfig::from_toml_file("aptp.toml")?);
    /// let mut w1 = AptpClient::connect_to(cfg.clone(), "127.0.0.1:7879").await?;
    /// let mut w2 = AptpClient::connect_to(cfg.clone(), "127.0.0.1:7880").await?;
    /// ```
    pub async fn connect_to(cfg: Arc<AptpConfig>, addr: &str) -> Result<Self> {
        use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
        use std::net::ToSocketAddrs;

        let addr = addr.to_socket_addrs()
            .map_err(|e| crate::error::AptpError::Config(format!("invalid addr: {e}")))?
            .next()
            .ok_or_else(|| crate::error::AptpError::Config("empty addr".into()))?;
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;

        let (reader, writer) = stream.into_split();
        let reader = reader.compat();
        let writer = writer.compat_write();

        let network = twoparty::VatNetwork::new(
            futures::io::BufReader::new(reader),
            futures::io::BufWriter::new(writer),
            rpc_twoparty_capnp::Side::Client,
            Default::default(),
        );

        let mut rpc_system = RpcSystem::new(Box::new(network), None);
        let client = rpc_system.bootstrap::<crate::aptp_capnp::agent_primitive_transfer::Client>(
            rpc_twoparty_capnp::Side::Server,
        );
        tokio::task::spawn_local(rpc_system);

        Ok(Self {
            cfg,
            session_id: None,
            sequence_counter: 0,
            client,
        })
    }

    pub async fn handshake(&mut self, card: AgentCard) -> Result<String> {
        let mut request = self.client.handshake_request();
        {
            let req = request.get();
            let mut agent_card = req.init_card();
            agent_card.set_agent_id(&card.agent_id);
            agent_card.set_aptp_version(card.aptp_version);
            agent_card.set_public_key(&[]);
            {
                let mut inv = agent_card.reborrow().init_invariants();
                inv.set_hidden_size(card.hidden_size);
                inv.set_num_layers(card.num_layers);
                inv.set_num_heads(card.num_heads);
                inv.set_vocab_size(card.vocab_size);
                inv.set_model_family(&card.model_family);
            }
            {
                let mut caps = agent_card.reborrow().init_capabilities(card.capabilities.len() as u32);
                for (i, c) in card.capabilities.iter().enumerate() {
                    caps.set(i as u32, c);
                }
            }
        }

        let response = request.send().promise.await?;
        let result = response.get()?.get_result()?;

        let status = result.get_status()
            .map_err(|e| crate::error::AptpError::Internal(e.to_string()))?;
        let session_id = result.get_assigned_session()?
            .to_str()
            .map_err(|e| crate::error::AptpError::Internal(e.to_string()))?
            .to_owned();

        match status {
            crate::aptp_capnp::HandshakeStatus::Accepted | crate::aptp_capnp::HandshakeStatus::NeedsAdapter => {
                self.session_id = Some(session_id.clone());
                Ok(session_id)
            }
            crate::aptp_capnp::HandshakeStatus::Rejected => {
                let reason = result.get_rejection_reason()?
                    .to_str().unwrap_or("unknown");
                Err(crate::error::AptpError::HandshakeRejected { reason: reason.to_owned() })
            }
        }
    }

    pub async fn send_primitive(&mut self, packet: PrimitivePacket) -> Result<bool> {
        self.session_id.as_ref()
            .ok_or_else(|| crate::error::AptpError::Internal("handshake not completed".into()))?;
        self.sequence_counter += 1;

        let mut request = self.client.stream_primitive_request();
        {
            let req = request.get();
            let mut p = req.init_packet();
            p.set_version(packet.version);
            p.set_sender_id(&packet.sender_id);
            p.set_model_fingerprint(&packet.model_fingerprint);
            p.set_layer_index(packet.layer_index);
            match &packet.payload {
                crate::primitives::PrimitivePayload::HiddenState(v) => {
                    let mut list = p.reborrow().init_primitive().init_hidden_state(v.len() as u32);
                    for (i, &val) in v.iter().enumerate() { list.set(i as u32, val); }
                }
                crate::primitives::PrimitivePayload::LatentThought(v) => {
                    let mut list = p.reborrow().init_primitive().init_latent_thought(v.len() as u32);
                    for (i, &val) in v.iter().enumerate() { list.set(i as u32, val); }
                }
                crate::primitives::PrimitivePayload::KvCache(kv) => {
                    let mut kv_b = p.reborrow().init_primitive().init_kv_cache();
                    kv_b.set_layer_idx(kv.layer_idx);
                    let mut keys = kv_b.reborrow().init_keys(kv.keys.len() as u32);
                    for (i, &v) in kv.keys.iter().enumerate() { keys.set(i as u32, v); }
                    let mut vals = kv_b.reborrow().init_values(kv.values.len() as u32);
                    for (i, &v) in kv.values.iter().enumerate() { vals.set(i as u32, v); }
                    let shape = kv_b.init_shape();
                    let mut dims = shape.init_dims(kv.shape.dims.len() as u32);
                    for (i, &d) in kv.shape.dims.iter().enumerate() { dims.set(i as u32, d); }
                }
            }
            {
                let shape = p.reborrow().init_shape();
                let mut dims = shape.init_dims(packet.shape.dims.len() as u32);
                for (i, &d) in packet.shape.dims.iter().enumerate() { dims.set(i as u32, d); }
            }
            {
                let mut meta = p.reborrow().init_metadata();
                meta.set_timestamp_ns(packet.metadata.timestamp_ns);
                meta.set_sequence_id(packet.metadata.sequence_id);
                meta.set_session_id(&packet.metadata.session_id);
                meta.set_compression_alg(&packet.metadata.compression_alg);
            }
        }
        let response = request.send().promise.await?;
        Ok(response.get()?.get_ack())
    }

    pub async fn finalize(mut self) -> Result<u64> {
        let session_id = self.session_id.take()
            .ok_or_else(|| crate::error::AptpError::Internal("no active session".into()))?;
        let mut request = self.client.finalize_request();
        request.get().set_session_id(&session_id);
        let response = request.send().promise.await?;
        Ok(response.get()?.get_received_count())
    }
}
