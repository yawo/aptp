use crate::config::ValidationConfig;
use crate::error::{AptpError, Result};
use crate::primitives::PrimitivePayload;
use crate::validation::probe::SafetyProbe;

pub trait ValidationGate: Send + Sync {
    fn validate(&self, payload: &PrimitivePayload) -> Result<()>;
}

pub struct NeuralFirewall {
    cfg: ValidationConfig,
    probe: Option<SafetyProbe>,
}

impl NeuralFirewall {
    pub fn new(cfg: ValidationConfig) -> Result<Self> {
        let probe = if let Some(ref path) = cfg.probe_vectors_path {
            Some(SafetyProbe::load(path)?)
        } else {
            None
        };
        Ok(Self { cfg, probe })
    }

    fn check_vector(&self, v: &[f32]) -> Result<()> {
        let l2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if l2 > self.cfg.l2_norm_max {
            return Err(AptpError::OffManifold { l2, cosine: 0.0 });
        }

        if let Some(ref probe) = self.probe {
            let cosine = probe.cosine_similarity(v);
            if cosine < self.cfg.cosine_sim_min {
                return Err(AptpError::OffManifold { l2, cosine });
            }
        }
        Ok(())
    }
}

impl ValidationGate for NeuralFirewall {
    fn validate(&self, payload: &PrimitivePayload) -> Result<()> {
        match payload {
            PrimitivePayload::HiddenState(v) | PrimitivePayload::LatentThought(v) => {
                self.check_vector(v)
            }
            PrimitivePayload::KvCache(kv) => {
                self.check_vector(&kv.keys)?;
                self.check_vector(&kv.values)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ValidationConfig;

    #[tokio::test]
    async fn test_firewall_rejects_high_norm() {
        let cfg = ValidationConfig {
            l2_norm_max: 10.0,
            cosine_sim_min: -0.95,
            fairness_threshold: 0.1,
            probe_vectors_path: None,
        };
        let fw = NeuralFirewall::new(cfg).unwrap();
        let v: Vec<f32> = vec![50.0; 16];
        let err = fw.check_vector(&v).unwrap_err();
        match err {
            AptpError::OffManifold { l2, cosine: _ } => assert!(l2 > 10.0),
            _ => panic!("expected OffManifold"),
        }
    }

    #[tokio::test]
    async fn test_firewall_accepts_normal_vector() {
        let cfg = ValidationConfig {
            l2_norm_max: 1000.0,
            cosine_sim_min: -0.95,
            fairness_threshold: 0.1,
            probe_vectors_path: None,
        };
        let fw = NeuralFirewall::new(cfg).unwrap();
        let v: Vec<f32> = vec![0.1; 16];
        assert!(fw.check_vector(&v).is_ok());
    }
}
