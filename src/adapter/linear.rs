use crate::adapter::manifold::ManifoldAdapter;
use crate::error::{AptpError, Result};

pub struct LinearProjectionAdapter {
    pub source_dim: usize,
    pub target_dim: usize,
    pub weights: Vec<f32>,
}

impl LinearProjectionAdapter {
    pub fn new(source_dim: usize, target_dim: usize, weights: Vec<f32>) -> Result<Self> {
        let expected = source_dim * target_dim;
        if weights.len() != expected {
            return Err(AptpError::IncompatibleDimensions {
                src: source_dim as u32,
                dst: target_dim as u32,
            });
        }
        Ok(Self { source_dim, target_dim, weights })
    }
}

impl ManifoldAdapter for LinearProjectionAdapter {
    fn source_dim(&self) -> usize { self.source_dim }
    fn target_dim(&self) -> usize { self.target_dim }

    fn adapt(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.source_dim {
            return Err(AptpError::IncompatibleDimensions {
                src: input.len() as u32,
                dst: self.source_dim as u32,
            });
        }
        let mut output = vec![0.0f32; self.target_dim];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.target_dim {
            let row_start = i * self.source_dim;
            output[i] = self.weights[row_start..row_start + self.source_dim]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum();
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_projection_identity_2d() {
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let adapter = LinearProjectionAdapter::new(2, 2, weights).unwrap();
        let input = vec![1.0, 2.0];
        let output = adapter.adapt(&input).unwrap();
        assert_eq!(output, vec![1.0, 2.0]);
    }

    #[test]
    fn test_linear_projection_upscale() {
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let adapter = LinearProjectionAdapter::new(2, 4, weights).unwrap();
        let input = vec![3.0, 5.0];
        let output = adapter.adapt(&input).unwrap();
        assert_eq!(output, vec![3.0, 5.0, 8.0, 0.0]);
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use crate::adapter::manifold::ManifoldAdapter;
    use crate::adapter::LinearProjectionAdapter;

    proptest! {
        #[test]
        fn linear_adapter_output_length(dim_src in 1usize..64, dim_dst in 1usize..64) {
            let n = dim_src * dim_dst;
            let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let adapter = LinearProjectionAdapter::new(dim_src, dim_dst, weights).unwrap();
            let input: Vec<f32> = (0..dim_src).map(|i| (i as f32) * 0.1).collect();
            let output = adapter.adapt(&input).unwrap();
            assert_eq!(output.len(), dim_dst);
        }
    }
}
