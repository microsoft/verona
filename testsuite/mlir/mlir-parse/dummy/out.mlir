

module {
  verona.class @C {
  }
  verona.class @D {
    verona.field "f" : !verona.U64
    verona.field "g" : !verona.U64
  }
  func @bar() -> !verona.U64 {
    %0 = verona.new_region @C [] : !verona.U64
    %1 = verona.new_object @D ["f", "g"](%0, %0 : !verona.U64, !verona.U64) in(%0 : !verona.U64) : !verona.U64
    %2 = verona.view %0 : !verona.U64 -> !verona.U64
    %3 = verona.field_read %1["f"] : !verona.U64 -> !verona.U64
    %4 = verona.field_write %1["g"], %2 : !verona.U64 -> !verona.U64 -> !verona.U64
    verona.tidy %0 : !verona.U64
    verona.drop %0 : !verona.U64
    return %0 : !verona.U64
  }
}