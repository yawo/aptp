fn main() {
    capnpc::CompilerCommand::new()
        .file("schemas/aptp.capnp")
        .run()
        .expect("capnp schema compilation failed");

    // Compile the C shim when the llama.cpp backend is enabled.
    // The user must have llama.cpp headers available.
    if std::env::var("CARGO_FEATURE_BACKEND_LLAMA").is_ok() {
        let mut build = cc::Build::new();
        build.file("include/aptp_llama_shim.c");
        build.flag_if_supported("-Wno-unused-parameter");

        // Allow the user to point to a custom llama.cpp checkout:
        //   LLAMA_CPP_DIR=/path/to/llama.cpp cargo build --features backend-llama
        if let Ok(dir) = std::env::var("LLAMA_CPP_DIR") {
            build.include(&dir);
        }

        build.compile("aptp_shim");
        println!("cargo:rustc-link-lib=llama");
        println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    }
}
