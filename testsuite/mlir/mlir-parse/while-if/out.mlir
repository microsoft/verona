

module {
  func @f(%arg0: none, %arg1: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk
    %2 = "verona.alloca"() : () -> !type.alloca
    %3 = "verona.store"(%arg1, %2) : (none, !type.alloca) -> !type.unk
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    %4 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %5 = "verona.constant(5)"() : () -> !type.int
    %6 = "verona.lt"(%4, %5) : (!type.unk, !type.int) -> i1
    cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %7 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %8 = "verona.load"(%2) : (!type.alloca) -> !type.unk
    %9 = "verona.ne"(%7, %8) : (!type.unk, !type.unk) -> i1
    cond_br %9, ^bb4, ^bb5
  ^bb3:  // pred: ^bb1
    %10 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %11 = "verona.cast"(%10) : (!type.unk) -> none
    return %11 : none
  ^bb4:  // pred: ^bb2
    %12 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %13 = "verona.constant(1)"() : () -> !type.int
    %14 = "verona.add"(%12, %13) : (!type.unk, !type.int) -> !type.unk
    %15 = "verona.store"(%14, %0) : (!type.unk, !type.alloca) -> !type.unk
    br ^bb5
  ^bb5:  // 2 preds: ^bb2, ^bb4
    br ^bb1
  }
}