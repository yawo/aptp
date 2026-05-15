pub mod linear;
pub mod manifold;
pub mod norm_match;

pub use linear::LinearProjectionAdapter;
pub use manifold::ManifoldAdapter;
pub use norm_match::NormMatchAdapter;

use crate::error::Result;

pub fn from_spec(
    adapter_type: AdapterVariant,
    source_dim: usize,
    target_dim: usize,
    weights: Vec<f32>,
) -> Result<Box<dyn ManifoldAdapter>> {
    match adapter_type {
        AdapterVariant::Identity => {
            Ok(Box::new(NormMatchAdapter { dim: source_dim, target_norm: 1.0 }))
        }
        AdapterVariant::LinearProjection => {
            Ok(Box::new(LinearProjectionAdapter::new(source_dim, target_dim, weights)?))
        }
        AdapterVariant::NormMatch => {
            let target_norm = if weights.len() == 1 { weights[0] } else { 1.0 };
            Ok(Box::new(NormMatchAdapter { dim: source_dim, target_norm }))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterVariant {
    Identity,
    LinearProjection,
    NormMatch,
}
