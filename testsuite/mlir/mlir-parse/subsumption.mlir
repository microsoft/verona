// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test checks that Verona operations allow subsumption in their arguments.
// For instance, we can copy a value from type A & B to a value of type A.

module {
  func @test1(%a: !verona.meet<U64, imm>) -> !verona.imm {
    // We can drop components of an intersection.
    %b = verona.copy %a : !verona.meet<U64, imm> -> !verona.U64

    // We can extend a type to a disjunction.
    %c = verona.copy %b : !verona.U64 -> !verona.join<U64, S64>

    // We can widen the disjunction to more types.
    %d = verona.copy %c : !verona.join<U64, S64> -> !verona.join<U64, S64, U32>

    // We can re-order types of a join.
    %e = verona.copy %d : !verona.join<U64, S64, U32> -> !verona.join<U32, U64, S64>

    // We can copy anything into top (ie. an empty meet).
    %f = verona.copy %e : !verona.join<U32, U64, S64> -> !verona.top

    // Join and meet can nest
    %g = verona.copy %a : !verona.meet<U64, imm> -> !verona.join<meet<U64, imm>, S64>

    // Return also supports subsumption: the function's return value is imm.
    verona.return %a : !verona.meet<U64, imm>
  }

  func @test2(%a: !verona.bottom) -> !verona.U64 {
    // Bottom (ie. an empty join) is a subtype of anything.
    verona.return %a: !verona.bottom
  }

  func @test3(%a: !verona.meet<U64, imm>) -> (!verona.U64, !verona.imm) {
    // Return supports multiple operands, which are subtyped individually
    verona.return %a, %a : !verona.meet<U64, imm>, !verona.meet<U64, imm>
  }
}
