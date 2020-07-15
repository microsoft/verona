

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.alloca"() : () -> !type.alloca
    %1 = "verona.constant(42)"() : () -> !type.int
    %2 = "verona.store"(%1, %0) : (!type.int, !type.alloca) -> !type.unk
    %3 = "verona.alloca"() : () -> !type.alloca
    %4 = "verona.constant(3.1415)"() : () -> !type.float
    %5 = "verona.store"(%4, %3) : (!type.float, !type.alloca) -> !type.unk
    %6 = "verona.cast"(%3) : (!type.alloca) -> none
    return %6 : none
  }
}