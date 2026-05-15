use crate::adapter::manifold::ManifoldAdapter;
use crate::error::Result;

pub struct NormMatchAdapter {
    pub dim: usize,
    pub target_norm: f32,
}

impl ManifoldAdapter for NormMatchAdapter {
    fn source_dim(&self) -> usize { self.dim }
    fn target_dim(&self) -> usize { self.dim }

    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>> {
        let current_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if current_norm > 1e-9 { self.target_norm / current_norm } else { 1.0 };
        Ok(input.iter().map(|x| x * scale).collect())
    }
}
