fn main() {
    capnpc::CompilerCommand::new()
        .file("schemas/aptp.capnp")
        .run()
        .expect("capnp schema compilation failed");
}
