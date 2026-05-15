@0xfcb00019726628dd;

# ─── Enumerations ────────────────────────────────────────────────────────────

enum AdapterType {
  identity        @0;
  linearProjection @1;
  normMatch       @2;
}

enum HandshakeStatus {
  accepted      @0;
  rejected      @1;
  needsAdapter  @2;
}

enum PrimitiveKind {
  hiddenState    @0;
  kvCache        @1;
  latentThought  @2;
}

# ─── Core Data Structs ───────────────────────────────────────────────────────

struct Shape {
  dims @0 :List(UInt32);
}

struct KVCache {
  layerIdx @0 :UInt16;
  keys     @1 :List(Float32);
  values   @2 :List(Float32);
  shape    @3 :Shape;
}

struct Metadata {
  timestampNs    @0 :UInt64;
  sequenceId     @1 :UInt64;
  sessionId      @2 :Text;
  compressionAlg @3 :Text;
}

struct TaggedPrimitive {
  provenanceHash @0 :Data;
  fairnessScore  @1 :Float32;
  payload        @2 :PrimitivePacket;
}

struct PrimitivePacket {
  version          @0 :UInt32;
  senderId         @1 :Text;
  modelFingerprint @2 :Data;
  layerIndex       @3 :UInt16;

  primitive :union {
    hiddenState   @4 :List(Float32);
    kvCache       @5 :KVCache;
    latentThought @6 :List(Float32);
  }

  shape    @7 :Shape;
  metadata @8 :Metadata;
}

struct ArchitectureInvariants {
  hiddenSize  @0 :UInt32;
  numLayers   @1 :UInt32;
  numHeads    @2 :UInt32;
  vocabSize   @3 :UInt32;
  modelFamily @4 :Text;
}

struct AgentCard {
  agentId    @0 :Text;
  aptpVersion @1 :UInt32;
  invariants  @2 :ArchitectureInvariants;
  capabilities @3 :List(Text);
  publicKey   @4 :Data;
}

struct HandshakeResult {
  status         @0 :HandshakeStatus;
  assignedSession @1 :Text;
  serverCard     @2 :AgentCard;
  adapterRequired @3 :AdapterType;
  rejectionReason @4 :Text;
}

struct AlignmentSpec {
  adapterType   @0 :AdapterType;
  sourceDim     @1 :UInt32;
  targetDim     @2 :UInt32;
  weights       @3 :List(Float32);
}

# ─── RPC Interface ───────────────────────────────────────────────────────────

interface AgentPrimitiveTransfer {
  handshake @0 (card :AgentCard) -> (result :HandshakeResult);

  streamPrimitive @1 (packet :PrimitivePacket) -> (ack :Bool);

  negotiateAlignment @2 (sourceDim :UInt32, targetDim :UInt32, sourceFamily :Text, targetFamily :Text)
                     -> (spec :AlignmentSpec);

  finalize @3 (sessionId :Text) -> (receivedCount :UInt64);
}
