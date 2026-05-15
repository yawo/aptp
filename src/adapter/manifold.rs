use crate::error::Result;

pub trait ManifoldAdapter: Send + Sync {
    fn source_dim(&self) -> usize;
    fn target_dim(&self) -> usize;
    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>>;
}
