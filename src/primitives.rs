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
    pub shape: Shape,
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
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
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
    pub model_fingerprint: Vec<u8>,
    pub layer_index: u16,
    pub payload: PrimitivePayload,
    pub shape: Shape,
    pub metadata: Metadata,
}

    pub fn provenance_hash(packet: &PrimitivePacket) -> Vec<u8> {
    // SAFETY: f32 has no invalid bit patterns and is trivially transmutable to [u8; 4].
    // The slice length is bounded by min(256, len) * 4 bytes, staying within the Vec allocation.
    // For KvCache the bound is min(64, len) * 4 bytes, similarly safe.
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

use crate::error::{AptpError, Result};

pub fn packet_from_capnp<'a>(
    reader: crate::aptp_capnp::primitive_packet::Reader<'a>,
) -> Result<PrimitivePacket> {
    let version = reader.get_version();
    let sender_id = reader.get_sender_id()?
        .to_str()
        .map_err(|e| AptpError::Internal(e.to_string()))?
        .to_owned();
    let model_fingerprint = reader.get_model_fingerprint()?.to_vec();
    let layer_index = reader.get_layer_index();

    let payload = match reader.get_primitive().which().map_err(AptpError::NotInSchema)? {
        crate::aptp_capnp::primitive_packet::primitive::HiddenState(r) => {
            let r: ::capnp::primitive_list::Reader<f32> = r?;
            let floats: Vec<f32> = r.iter().collect();
            PrimitivePayload::HiddenState(floats)
        }
        crate::aptp_capnp::primitive_packet::primitive::LatentThought(r) => {
            let r: ::capnp::primitive_list::Reader<f32> = r?;
            let floats: Vec<f32> = r.iter().collect();
            PrimitivePayload::LatentThought(floats)
        }
        crate::aptp_capnp::primitive_packet::primitive::KvCache(r) => {
            let kv: crate::aptp_capnp::k_v_cache::Reader<'_> = r?;
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
    let session_id = meta_reader.get_session_id()?
        .to_str()
        .map_err(|e| AptpError::Internal(e.to_string()))?
        .to_owned();
    let compression_alg = meta_reader.get_compression_alg()?
        .to_str()
        .map_err(|e| AptpError::Internal(e.to_string()))?
        .to_owned();
    let metadata = Metadata {
        timestamp_ns: meta_reader.get_timestamp_ns(),
        sequence_id: meta_reader.get_sequence_id(),
        session_id,
        compression_alg,
    };

    Ok(PrimitivePacket { version, sender_id, model_fingerprint, layer_index, payload, shape, metadata })
}

fn shape_from_capnp(
    reader: crate::aptp_capnp::shape::Reader<'_>,
) -> Result<Shape> {
    Ok(Shape { dims: reader.get_dims()?.iter().collect() })
}

pub fn packet_to_bytes(packet: &PrimitivePacket) -> Result<Vec<u8>> {
    use capnp::message::Builder;
    use capnp::serialize_packed;

    let mut message = Builder::new_default();
    {
        let mut root = message.init_root::<crate::aptp_capnp::primitive_packet::Builder<'_>>();
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
                let shape = kv_builder.init_shape();
                let mut dims = shape.init_dims(kv.shape.dims.len() as u32);
                for (i, &d) in kv.shape.dims.iter().enumerate() { dims.set(i as u32, d); }
            }
        }

        let shape_builder = root.reborrow().init_shape();
        let mut dims = shape_builder.init_dims(packet.shape.dims.len() as u32);
        for (i, &d) in packet.shape.dims.iter().enumerate() { dims.set(i as u32, d); }

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

pub fn agent_card_from_capnp(
    reader: crate::aptp_capnp::agent_card::Reader<'_>,
) -> Result<AgentCard> {
    let invariants = reader.get_invariants()?;
    Ok(AgentCard {
        agent_id: reader.get_agent_id()?
            .to_str()
            .map_err(|e| AptpError::Internal(e.to_string()))?
            .to_owned(),
        aptp_version: reader.get_aptp_version(),
        hidden_size: invariants.get_hidden_size(),
        num_layers: invariants.get_num_layers(),
        num_heads: invariants.get_num_heads(),
        vocab_size: invariants.get_vocab_size(),
        model_family: invariants.get_model_family()?
            .to_str()
            .map_err(|e| AptpError::Internal(e.to_string()))?
            .to_owned(),
        capabilities: {
            let cap_list = reader.get_capabilities()?;
            let mut caps = Vec::with_capacity(cap_list.len() as usize);
            for i in 0..cap_list.len() {
                caps.push(cap_list.get(i)?.to_str()
                    .map_err(|e| AptpError::Internal(e.to_string()))?.to_owned());
            }
            caps
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tagged_primitive_provenance_roundtrip() {
        let packet = PrimitivePacket {
            version: APTP_VERSION,
            sender_id: "test-agent".to_owned(),
            model_fingerprint: vec![0xAB; 32],
            layer_index: 8,
            payload: PrimitivePayload::HiddenState(vec![0.5f32; 128]),
            shape: Shape { dims: vec![1, 1, 128] },
            metadata: Metadata::new_now("test-session", 0),
        };
        let tagged = TaggedPrimitive::new(packet, 0.9);
        assert!(tagged.verify_provenance());

        let mut forged_tagged = tagged.clone();
        match &mut forged_tagged.payload.payload {
            PrimitivePayload::HiddenState(v) => v[0] = -99.0,
            _ => unreachable!(),
        }
        assert!(!forged_tagged.verify_provenance());
    }
}
