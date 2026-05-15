use crate::adapter::AdapterVariant;
use crate::config::AptpConfig;
use crate::error::Result;
use crate::primitives::{AgentCard, APTP_VERSION};
use crate::rpc::SessionRegistry;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub async fn handle(
    remote_card: AgentCard,
    cfg: Arc<AptpConfig>,
    sessions: Arc<Mutex<SessionRegistry>>,
) -> Result<(crate::aptp_capnp::HandshakeStatus, String, AdapterVariant, String)> {
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
            crate::aptp_capnp::HandshakeStatus::Rejected,
            String::new(),
            AdapterVariant::Identity,
            reason,
        ));
    }

    let local_hidden = cfg.agent.hidden_size;
    let remote_hidden = remote_card.hidden_size;

    let (status, adapter) = if local_hidden == remote_hidden {
        (crate::aptp_capnp::HandshakeStatus::Accepted, AdapterVariant::Identity)
    } else if !cfg.adapter.allow_dimension_mismatch_passthrough {
        (crate::aptp_capnp::HandshakeStatus::NeedsAdapter, AdapterVariant::LinearProjection)
    } else {
        warn!(
            local_hidden,
            remote_hidden,
            "Dimension mismatch passthrough enabled (dev mode)"
        );
        (crate::aptp_capnp::HandshakeStatus::Accepted, AdapterVariant::Identity)
    };

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
