use crate::adapter::AdapterVariant;
use std::sync::atomic::{AtomicU64, Ordering};

static KAIMING_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn kaiming_uniform_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
    let bound = (3.0f32 / in_dim as f32).sqrt();
    let n = in_dim * out_dim;

    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let counter = KAIMING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let seed = time_seed ^ counter;

    let mut state = seed ^ 0xdeadbeef_cafebabe;
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let t = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        weights.push(t * bound);
    }
    weights
}

pub fn handle(
    source_dim: u32,
    target_dim: u32,
    _source_family: &str,
    _target_family: &str,
) -> crate::error::Result<(AdapterVariant, Vec<f32>)> {
    if source_dim == 0 || target_dim == 0 {
        return Err(crate::error::AptpError::IncompatibleDimensions {
            src: source_dim,
            dst: target_dim,
        });
    }
    if source_dim == target_dim {
        Ok((AdapterVariant::NormMatch, vec![1.0f32]))
    } else {
        let weights = kaiming_uniform_weights(source_dim as usize, target_dim as usize);
        Ok((AdapterVariant::LinearProjection, weights))
    }
}
