

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk
    verona.while  {
      %4 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %5 = "verona.constant(5)"() : () -> !type.int
      %6 = "verona.lt"(%4, %5) : (!type.unk, !type.int) -> i1
      verona.loop_exit %6 : i1
      %7 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %8 = "verona.constant(1)"() : () -> !type.int
      %9 = "verona.add"(%7, %8) : (!type.unk, !type.int) -> !type.unk
      %10 = "verona.store"(%9, %0) : (!type.unk, !type.alloca) -> !type.unk
      %11 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %12 = "verona.constant(5)"() : () -> !type.int
      %13 = "verona.lt"(%11, %12) : (!type.unk, !type.int) -> i1
    ^bb1:  // pred: ^bb0
      verona.break
    ^bb2:  // pred: ^bb0
      verona.continue
    }
    %2 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %3 = "verona.cast"(%2) : (!type.unk) -> none
    return %3 : none
  }
}