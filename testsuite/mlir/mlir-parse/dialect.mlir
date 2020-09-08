// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is a simple example to demonstrate the Verona MLIR dialect.
//
// ```verona
// class C {  }
// class D {
//   f: U64 & imm;
//   g: S32 & mut;
//   h: F32;
//   i: F64;
//   j: bool;
// }
//
// bar(x: U64 & imm, y: U64 & imm) {
//     let a : C & iso = new C;
//     let b : C & mut = view a;
//     let c : D & mut = new D { f: x, g: b } in a;
//     let d : U64 & imm = c.f;
//     let e : D & mut = c.g;
//     let f : U64 & imm = (c.g = y);
//
//     tidy(a);
//     drop(a);
// }
// ```

!C = type !verona.class<"C">
!D = type !verona.class<"D", "f" : meet<U64, imm>, "g" : meet<!C, mut>>
module {
  verona.class @C {
  }

  verona.class @D {
    verona.field "f" : !verona.meet<U64, imm>
    verona.field "g" : !verona.meet<!C, mut>
    verona.field "h" : !verona.F32
    verona.field "i" : !verona.F64
    verona.field "j" : !verona.bool
  }

  func @bar(%x: !verona.meet<U64, imm>, %y: !verona.meet<U64, imm>) {
    %a = verona.new_region @C [ ] : !verona.meet<!C, iso>
    %b = verona.view %a : !verona.meet<!C, iso> -> !verona.meet<!C, mut>

    %c = verona.new_object @D [ "f", "g" ] (%x, %b : !verona.meet<U64, imm>, !verona.meet<!C, mut>)
      in (%a : !verona.meet<!C, iso>)
      : !verona.meet<!D, mut>

    %d = verona.field_read %c["f"]
       : !verona.meet<!D, mut>
      -> !verona.meet<U64, imm>

    %e = verona.field_read %c["g"]
       : !verona.meet<!D, mut>
      -> !verona.meet<!C, mut>

    %f = verona.field_write %c["f"], %y
       : !verona.meet<!D, mut>
      -> !verona.meet<U64, imm>
      -> !verona.meet<U64, imm>

     verona.tidy %a : !verona.meet<!C, iso>
     verona.drop %a : !verona.meet<!C, iso>

    return
  }
}
