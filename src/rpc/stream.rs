use crate::error::{AptpError, Result};
use crate::primitives::{packet_from_capnp, TaggedPrimitive};
use crate::rpc::{PacketSink, SessionRegistry};
use crate::validation::gate::ValidationGate;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, warn};

pub async fn handle<'a>(
    reader: crate::aptp_capnp::primitive_packet::Reader<'a>,
    firewall: Arc<dyn ValidationGate>,
    sessions: Arc<Mutex<SessionRegistry>>,
    sink: Arc<PacketSink>,
) -> Result<bool> {
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

    let _span = tracing::info_span!(
        "aptp::stream_receive",
        session_id = %session_id,
        sequence_id = seq,
        payload_kind = kind,
    )
    .entered();

    if let Err(e) = firewall.validate(&packet.payload) {
        warn!(error = %e, "NeuralFirewall rejected packet");
        return Ok(false);
    }

    let tagged = TaggedPrimitive::new(packet, 1.0);
    if !tagged.verify_provenance() {
        warn!("Provenance hash mismatch — rejecting packet");
        return Err(AptpError::ProvenanceMismatch);
    }

    sink.send(tagged).await?;

    {
        let mut reg = sessions.lock().await;
        reg.increment(&session_id)?;
    }

    debug!(sequence_id = seq, "Packet accepted and queued");
    Ok(true)
}
