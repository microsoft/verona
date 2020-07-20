// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

module {
  verona.class @C {
  }

  verona.class @D {
    verona.field "f" : !verona.U64
    verona.field "g" : !verona.U64
  }

  func @bar() -> !verona.U64 {
    %a = verona.new_region @C [ ] : !verona.U64
    %b = verona.new_object @D [ "f", "g" ] (%a, %a : !verona.U64, !verona.U64) in (%a : !verona.U64) : !verona.U64

    %c = verona.view %a : !verona.U64 -> !verona.U64
    %d = verona.field_read %b["f"] : !verona.U64 -> !verona.U64
    verona.field_write %b["g"], %c : !verona.U64 -> !verona.U64 -> !verona.U64

    verona.tidy %a : !verona.U64
    verona.drop %a : !verona.U64

    return %a : !verona.U64
  }
}
