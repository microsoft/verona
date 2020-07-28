// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is a simple example to demonstrate the Verona MLIR dialect. The types
// used are just a placeholder and not correct yet.
//
// ```verona
// class C {  }
// class D {
//   f: U64;
//   g: S32;
// }
//
// bar() {
//     let a = new C;
//     let b = view a;
//     let c = new D { f: b, g: b } in a;
//     c.g = c.f;
//
//     tidy(a);
//     drop(a);
//
//     return new C;
// }
// ```

module {
  verona.class @C {
  }

  verona.class @D {
    verona.field "f" : !verona.U64
    verona.field "g" : !verona.S32
  }

  func @bar() -> !verona.U64 {
    %a = verona.new_region @C [ ] : !verona.U64
    %b = verona.view %a : !verona.U64 -> !verona.U64

    %c = verona.new_object @D [ "f", "g" ] (%b, %b : !verona.U64, !verona.U64) in (%a : !verona.U64) : !verona.S64

    %d = verona.field_read %c["f"] : !verona.S64 -> !verona.U64
    verona.field_write %c["f"], %d : !verona.S64 -> !verona.U64 -> !verona.U64

    %e = verona.field_read %c["g"] : !verona.S64 -> !verona.S32

    verona.tidy %a : !verona.U64
    verona.drop %a : !verona.U64

    %f = verona.new_region @C [ ] : !verona.U64
    return %f : !verona.U64
  }
}
