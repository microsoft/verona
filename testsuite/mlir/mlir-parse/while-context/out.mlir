

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.store"(%arg0, %0) : (none, !type.alloca) -> !type.unk
    %2 = "verona.alloca"() : () -> !type.alloca
    %3 = "verona.constant(1)"() : () -> !type.int
    %4 = "verona.store"(%3, %2) : (!type.int, !type.alloca) -> !type.unk
    verona.while  {
      %7 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %8 = "verona.constant(5)"() : () -> !type.int
      %9 = "verona.lt"(%7, %8) : (!type.unk, !type.int) -> i1
      verona.loop_exit %9 : i1
      %10 = "verona.alloca"() : () -> !type.alloca
      %11 = "verona.constant(2)"() : () -> !type.int
      %12 = "verona.store"(%11, %10) : (!type.int, !type.alloca) -> !type.unk
      %13 = "verona.alloca"() : () -> !type.alloca
      %14 = "verona.load"(%10) : (!type.alloca) -> !type.unk
      %15 = "verona.constant(3)"() : () -> !type.int
      %16 = "verona.add"(%14, %15) : (!type.unk, !type.int) -> !type.unk
      %17 = "verona.store"(%16, %13) : (!type.unk, !type.alloca) -> !type.unk
      %18 = "verona.load"(%0) : (!type.alloca) -> !type.unk
      %19 = "verona.load"(%13) : (!type.alloca) -> !type.unk
      %20 = "verona.add"(%18, %19) : (!type.unk, !type.unk) -> !type.unk
      %21 = "verona.store"(%20, %0) : (!type.unk, !type.alloca) -> !type.unk
      verona.continue
    }
    %5 = "verona.load"(%0) : (!type.alloca) -> !type.unk
    %6 = "verona.cast"(%5) : (!type.unk) -> none
    return %6 : none
  }
}