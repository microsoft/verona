

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb6
    %2 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %3 = "verona.constant(50)"() : () -> !type.int
    %4 = "verona.lt"(%2, %3) : (!type.unk, !type.int) -> i1
    cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = "verona.alloca"() : () -> !type.alloca
    %6 = "verona.constant(1)"() : () -> !type.int
    %7 = "verona.store"(%6, %5) : (!type.int, !type.alloca) -> !type.unk
    br ^bb4
  ^bb3:  // pred: ^bb1
    %8 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %9 = "verona.cast"(%8) : (!type.unk) -> none
    return %9 : none
  ^bb4:  // 2 preds: ^bb2, ^bb5
    %10 = "verona.load"(%5) : (!type.alloca) -> !type.unk
    %11 = "verona.constant(10)"() : () -> !type.int
    %12 = "verona.lt"(%10, %11) : (!type.unk, !type.int) -> i1
    cond_br %12, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %13 = "verona.load"(%5) : (!type.alloca) -> !type.unk
    %14 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %15 = "verona.add"(%13, %14) : (!type.unk, !type.unk) -> !type.unk
    %16 = "verona.store"(%15, %5) : (!type.unk, !type.alloca) -> !type.unk
    br ^bb4
  ^bb6:  // pred: ^bb4
    %17 = "verona.load"(%5) : (!type.alloca) -> !type.unk
    %18 = "verona.store"(%17, %0) : (!type.unk, !type.alloca) -> !type.unk
    br ^bb1
  }
}