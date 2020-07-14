

module {
  func @f(%arg0: none) -> none {
    %0 = "verona.constant(42)"() : () -> !type.int
    %1 = "verona.constant(3.1415)"() : () -> !type.float
    %2 = "verona.cast"(%1) : (!type.float) -> none
    return %2 : none
  }
}