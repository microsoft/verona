

module {
  func @bar(%arg0: !verona.U64) -> !type<"U64 & imm"> {
    %0 = verona.foo %arg0 : !verona.U64
    %1 = "verona.cast"(%0) : (!verona.U64) -> !verona.S64
    %2 = "verona.test"(%1) : (!verona.S64) -> !type<"U64 & imm">
    return %2 : !type<"U64 & imm">
  }
}