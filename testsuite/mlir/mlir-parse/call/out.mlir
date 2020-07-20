

module {
  func @foo(%arg0: !type.imm, %arg1: !type<"U64&imm">) -> !type<"U64&imm"> {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (!type.imm, !type.alloca) -> !type.unk
    %2 = "verona.alloca"() : () -> !type.alloca
    %3 = "verona.store"(%arg1, %2) : (!type<"U64&imm">, !type.alloca) -> !type.unk
    %4 = "verona.alloca"() : () -> !type.alloca
    %5 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %6 = "verona.load"(%2) : (!type.alloca) -> !type.unk
    %7 = "verona.add"(%5, %6) : (!type.unk, !type.unk) -> !type.unk
    %8 = "verona.store"(%7, %4) : (!type.unk, !type.alloca) -> !type.unk
    %9 = "verona.alloca"() : () -> !type.alloca
    %10 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %11 = "verona.load"(%2) : (!type.alloca) -> !type.unk
    %12 = "verona.sub"(%10, %11) : (!type.unk, !type.unk) -> !type.unk
    %13 = "verona.store"(%12, %9) : (!type.unk, !type.alloca) -> !type.unk
    %14 = "verona.load"(%4) : (!type.alloca) -> !type.unk
    %15 = "verona.constant(100)"() : () -> !type.int
    %16 = "verona.lt"(%14, %15) : (!type.unk, !type.int) -> i1
    cond_br %16, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %17 = "verona.load"(%4) : (!type.alloca) -> !type.unk
    %18 = "verona.cast"(%17) : (!type.unk) -> !type.imm
    %19 = "verona.load"(%9) : (!type.alloca) -> !type.unk
    %20 = "verona.cast"(%19) : (!type.unk) -> !type<"U64&imm">
    %21 = call @foo(%18, %20) : (!type.imm, !type<"U64&imm">) -> !type<"U64&imm">
    return %21 : !type<"U64&imm">
  ^bb2:  // pred: ^bb0
    %22 = "verona.load"(%4) : (!type.alloca) -> !type.unk
    %23 = "verona.load"(%9) : (!type.alloca) -> !type.unk
    %24 = "verona.add"(%22, %23) : (!type.unk, !type.unk) -> !type.unk
    %25 = "verona.cast"(%24) : (!type.unk) -> !type<"U64&imm">
    return %25 : !type<"U64&imm">
  }
  func @apply() -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.constant(10)"() : () -> !type.int
    %2 = "verona.store"(%1, %0) : (!type.int, !type.alloca) -> !type.unk
    %3 = "verona.alloca"() : () -> !type.alloca
    %4 = "verona.constant(20)"() : () -> !type.int
    %5 = "verona.store"(%4, %3) : (!type.int, !type.alloca) -> !type.unk
    %6 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %7 = "verona.cast"(%6) : (!type.unk) -> !type.imm
    %8 = "verona.load"(%3) : (!type.alloca) -> !type.unk
    %9 = "verona.cast"(%8) : (!type.unk) -> !type<"U64&imm">
    %10 = call @foo(%7, %9) : (!type.imm, !type<"U64&imm">) -> !type<"U64&imm">
    %11 = "verona.cast"(%10) : (!type<"U64&imm">) -> none
    return %11 : none
  }
}