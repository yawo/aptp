use aptp::adapter::linear::LinearProjectionAdapter;
use aptp::adapter::manifold::ManifoldAdapter;
use aptp::config::ValidationConfig;
use aptp::primitives::{
    packet_to_bytes, KvCache, Metadata, PrimitivePacket, PrimitivePayload, Shape, APTP_VERSION,
};
use aptp::validation::gate::{NeuralFirewall, ValidationGate};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_hidden_state_packet(dim: usize) -> PrimitivePacket {
    PrimitivePacket {
        version: APTP_VERSION,
        sender_id: "bench-agent".to_owned(),
        model_fingerprint: vec![0u8; 32],
        layer_index: 16,
        payload: PrimitivePayload::HiddenState(vec![0.1f32; dim]),
        shape: Shape { dims: vec![1, 1, dim as u32] },
        metadata: Metadata::new_now("bench-session", 0),
    }
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_serialization");
    for dim in [512usize, 2048, 4096, 8192] {
        group.bench_with_input(BenchmarkId::new("hidden_state_dim", dim), &dim, |b, &dim| {
            let packet = make_hidden_state_packet(dim);
            b.iter(|| {
                let bytes = packet_to_bytes(&packet).expect("serialization failed");
                criterion::black_box(bytes);
            });
        });
    }
    group.finish();
}

fn bench_firewall(c: &mut Criterion) {
    let cfg = ValidationConfig {
        l2_norm_max: 1000.0,
        cosine_sim_min: -0.95,
        fairness_threshold: 0.1,
        probe_vectors_path: None,
    };
    let fw = NeuralFirewall::new(cfg).expect("firewall init");

    let mut group = c.benchmark_group("firewall_validate");
    for dim in [512usize, 4096, 8192] {
        let payload = PrimitivePayload::HiddenState(vec![0.01f32; dim]);
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                fw.validate(criterion::black_box(&payload)).ok();
            });
        });
    }
    group.finish();
}

fn bench_adapter(c: &mut Criterion) {
    let dim = 4096usize;
    let weights = vec![0.001f32; dim * dim];
    let adapter = LinearProjectionAdapter::new(dim, dim, weights).expect("adapter init");
    let input = vec![0.1f32; dim];

    c.bench_function("linear_adapter_4096x4096", |b| {
        b.iter(|| {
            adapter.adapt(criterion::black_box(&input)).ok();
        });
    });
}

criterion_group!(benches, bench_serialization, bench_firewall, bench_adapter);
criterion_main!(benches);
