// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test checks that Verona operations allow subsumption in their arguments.
// For instance, we can copy a value from type A & B to a value of type A.

module {
  func @test1(%a: !verona.meet<U64, imm>) {
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

    return
  }

  func @test2(%a: !verona.bottom) {
    // Bottom (ie. an empty join) is a subtype of anything.
    %b = verona.copy %a: !verona.bottom -> !verona.U64

    return
  }

  func @test_distributivity(%a: !verona.meet<U64, join<iso, mut>>) {
    // We allow distributivity of join over meets.
    %b = verona.copy %a: !verona.meet<U64, join<iso, mut>> -> !verona.join<meet<U64, iso>, meet<U64, mut>> 

    return
  }
}
