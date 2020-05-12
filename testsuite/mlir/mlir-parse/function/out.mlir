

module {
  func @foo(%arg0: !type.imm, %arg1: !type<"U64&imm">) -> !type<"U64&imm"> {
    %0 = "verona.add"(%arg0, %arg1) : (!type.imm, !type<"U64&imm">) -> !type.ret
    %1 = "verona.cast"(%0) : (!type.ret) -> !type<"U64&imm">
    return %1 : !type<"U64&imm">
  }
  func @apply() -> none {
    %0 = "verona.none"() : () -> none
    return %0 : none
  }
}