use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing::{error, info};

use crate::config::AptpConfig;
use crate::rpc::AptpRpcServer;

pub async fn run_server(cfg: Arc<AptpConfig>) -> crate::error::Result<()> {
    let addr: SocketAddr = cfg.transport.bind_addr.parse()
        .map_err(|e| crate::error::AptpError::Config(format!("invalid bind addr: {e}")))?;

    let listener = TcpListener::bind(addr).await?;
    info!("APTP server listening on {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!("Accepted connection from {}", peer_addr);

        stream.set_nodelay(true)?;
        let cfg = cfg.clone();

        tokio::task::spawn_local(async move {
            if let Err(e) = handle_connection(stream, cfg).await {
                error!("Connection error from {}: {}", peer_addr, e);
            }
        });
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    cfg: Arc<AptpConfig>,
) -> crate::error::Result<()> {
    let (reader, writer) = stream.into_split();
    let reader = reader.compat();
    let writer = writer.compat_write();

    let network = twoparty::VatNetwork::new(
        futures::io::BufReader::new(reader),
        futures::io::BufWriter::new(writer),
        rpc_twoparty_capnp::Side::Server,
        Default::default(),
    );

    let rpc_server = AptpRpcServer::new(cfg, 1024)?;
    let typed_client = capnp_rpc::new_client::<
        crate::aptp_capnp::agent_primitive_transfer::Client,
        _,
    >(rpc_server);

    let rpc_system = RpcSystem::new(Box::new(network), Some(typed_client.client));
    rpc_system.await?;

    Ok(())
}
