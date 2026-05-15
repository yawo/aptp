use crate::error::{AptpError, Result};

pub struct SafetyProbe {
    centroid: Vec<f32>,
    dim: usize,
}

impl SafetyProbe {
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| AptpError::Config(format!("probe load: {e}")))?;

        let mut sum: Vec<f64> = Vec::new();
        let mut count = 0usize;

        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let vec: Vec<f32> = serde_json::from_str(line)
                .map_err(|e| AptpError::Config(format!("probe parse: {e}")))?;
            if sum.is_empty() {
                sum = vec![0.0f64; vec.len()];
            }
            for (s, &v) in sum.iter_mut().zip(vec.iter()) {
                *s += v as f64;
            }
            count += 1;
        }

        if count == 0 {
            return Err(AptpError::Config("probe file is empty".into()));
        }

        let dim = sum.len();
        let mut centroid: Vec<f32> = sum.iter().map(|&s| (s / count as f64) as f32).collect();
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            centroid.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(Self { centroid, dim })
    }

    pub fn cosine_similarity(&self, v: &[f32]) -> f32 {
        if v.len() != self.dim {
            return 0.0;
        }
        let dot: f32 = v.iter().zip(self.centroid.iter()).map(|(a, b)| a * b).sum();
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_v < 1e-9 { 0.0 } else { dot / norm_v }
    }
}
