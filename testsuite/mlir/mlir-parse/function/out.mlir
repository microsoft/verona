

module {
  func @foo(%arg0: !type.imm, %arg1: !type<"U64&imm">) -> !type<"U64&imm"> {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.add"(%arg0, %arg1) : (!type.imm, !type<"U64&imm">) -> !type.unk
    %2 = "verona.store"(%1, %0) : (!type.unk, !type.alloca) -> !type.unk
    %3 = "verona.alloca"() : () -> !type.alloca
    %4 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %5 = "verona.store"(%4, %3) : (!type.unk, !type.alloca) -> !type.unk
    %6 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %7 = "verona.cast"(%6) : (!type.unk) -> !type<"U64&imm">
    return %7 : !type<"U64&imm">
  }
  func @apply() -> none {
    %0 = "verona.none"() : () -> none
    return %0 : none
  }
}