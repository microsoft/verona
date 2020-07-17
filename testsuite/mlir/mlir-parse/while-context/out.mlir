

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk
    %2 = "verona.alloca"() : () -> !type.alloca
    %3 = "verona.constant(1)"() : () -> !type.int
    %4 = "verona.store"(%3, %2) : (!type.int, !type.alloca) -> !type.unk
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %5 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %6 = "verona.constant(5)"() : () -> !type.int
    %7 = "verona.lt"(%5, %6) : (!type.unk, !type.int) -> i1
    cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = "verona.alloca"() : () -> !type.alloca
    %9 = "verona.constant(2)"() : () -> !type.int
    %10 = "verona.store"(%9, %8) : (!type.int, !type.alloca) -> !type.unk
    %11 = "verona.alloca"() : () -> !type.alloca
    %12 = "verona.load"(%8) : (!type.alloca) -> !type.unk
    %13 = "verona.constant(3)"() : () -> !type.int
    %14 = "verona.add"(%12, %13) : (!type.unk, !type.int) -> !type.unk
    %15 = "verona.store"(%14, %11) : (!type.unk, !type.alloca) -> !type.unk
    %16 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %17 = "verona.load"(%11) : (!type.alloca) -> !type.unk
    %18 = "verona.add"(%16, %17) : (!type.unk, !type.unk) -> !type.unk
    %19 = "verona.store"(%18, %0) : (!type.unk, !type.alloca) -> !type.unk
    br ^bb1
  ^bb3:  // pred: ^bb1
    %20 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %21 = "verona.cast"(%20) : (!type.unk) -> none
    return %21 : none
  }
}