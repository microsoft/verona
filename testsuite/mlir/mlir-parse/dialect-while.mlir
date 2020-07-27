

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk

    verona.while {
      // Condition
      %2 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %3 = "verona.constant(5)"() : () -> !type.int
      %4 = "verona.lt"(%2, %3) : (!type.unk, !type.int) -> i1
      verona.loop_exit %4 : i1

      // Body
      %5 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %6 = "verona.constant(1)"() : () -> !type.int
      %7 = "verona.add"(%5, %6) : (!type.unk, !type.int) -> !type.unk
      %8 = "verona.store"(%7, %0) : (!type.unk, !type.alloca) -> !type.unk

      // Test control flow inside while
      %9 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %10 = "verona.constant(5)"() : () -> !type.int
      %11 = "verona.lt"(%9, %10) : (!type.unk, !type.int) -> i1
      cond_br %11, ^bb2, ^bb3

    ^bb2:
      // Break the loop
      verona.break

    ^bb3:
      // Default terminator, continue the loop to the condition
      verona.continue
    }

    %12 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %13 = "verona.cast"(%12) : (!type.unk) -> none
    return %13 : none
  }
}
