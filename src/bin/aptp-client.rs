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
