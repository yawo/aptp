use std::sync::Arc;
use tokio::task::LocalSet;
use tracing_subscriber::EnvFilter;
use aptp::config::AptpConfig;
use aptp::transport::server::run_server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cfg_path = std::env::args().nth(1).unwrap_or_else(|| "aptp.toml".into());
    let cfg = Arc::new(AptpConfig::from_toml_file(&cfg_path)?);

    let local = LocalSet::new();
    local.run_until(run_server(cfg)).await?;
    Ok(())
}
