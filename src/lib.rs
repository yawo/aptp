pub mod config;
pub mod error;
pub mod primitives;
pub mod transport;
pub mod rpc;
pub mod validation;
pub mod adapter;

#[cfg(feature = "backend-llama")]
pub mod backend;

#[allow(dead_code)]
pub mod aptp_capnp {
    include!(concat!(env!("OUT_DIR"), "/schemas/aptp_capnp.rs"));
}

pub use aptp_capnp as schema_capnp;
