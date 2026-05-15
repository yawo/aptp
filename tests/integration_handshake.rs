/// Integration test: server on one thread, client on another, verify handshake.
use std::sync::Arc;
use aptp::config::{AgentConfig, AptpConfig, TransportConfig, ValidationConfig, AdapterConfig};
use aptp::primitives::AgentCard;
use aptp::transport::client::AptpClient;

fn test_config(bind_addr: &str, agent_id: &str) -> AptpConfig {
    AptpConfig {
        transport: TransportConfig {
            bind_addr: bind_addr.into(),
            connect_timeout_ms: 5000,
            max_frame_bytes: 67108864,
            tls: None,
        },
        validation: ValidationConfig {
            l2_norm_max: 1000.0,
            cosine_sim_min: -0.95,
            fairness_threshold: 0.1,
            probe_vectors_path: None,
        },
        adapter: AdapterConfig {
            allow_dimension_mismatch_passthrough: false,
        },
        backend: None,
        agent: AgentConfig {
            agent_id: agent_id.into(),
            model_family: "llama".into(),
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 32000,
            aptp_version: 1,
        },
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn integration_handshake() {
    let (port_tx, port_rx) = tokio::sync::oneshot::channel::<u16>();

    let server_thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            port_tx.send(port).ok();

            loop {
                let (stream, _) = listener.accept().await.unwrap();
                stream.set_nodelay(true).unwrap();
                let (reader, writer) = stream.into_split();
                let reader = tokio_util::compat::TokioAsyncReadCompatExt::compat(reader);
                let writer = tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(writer);

                let network = capnp_rpc::twoparty::VatNetwork::new(
                    futures::io::BufReader::new(reader),
                    futures::io::BufWriter::new(writer),
                    capnp_rpc::rpc_twoparty_capnp::Side::Server,
                    Default::default(),
                );

                let rpc_server = aptp::rpc::AptpRpcServer::new(
                    Arc::new(test_config("127.0.0.1:0", "server")),
                    8,
                )
                .unwrap();
                let typed_client = capnp_rpc::new_client::<
                    aptp::aptp_capnp::agent_primitive_transfer::Client,
                    _,
                >(rpc_server);
                let rpc_system =
                    capnp_rpc::RpcSystem::new(Box::new(network), Some(typed_client.client));
                tokio::task::spawn_local(rpc_system);
            }
        });
    });

    let port = port_rx.await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client_cfg = Arc::new(test_config(&format!("127.0.0.1:{}", port), "test-client"));
    let card = AgentCard::new_from_config(&client_cfg.agent);
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut client = AptpClient::connect(client_cfg).await.unwrap();
            let session_id = client.handshake(card).await.unwrap();
            assert!(!session_id.is_empty());
            let total = client.finalize().await.unwrap();
            assert_eq!(total, 0);
        })
        .await;

    drop(server_thread);
}
