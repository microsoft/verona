// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

// This test checks that Verona operations allow subsumption in their arguments.
// For instance, we can copy a value from type A & B to a value of type A.

module {
  func @test1(%a: !verona.meet<class<"U64">, imm>) {
    // We can drop components of an intersection.
    %b = verona.copy %a : !verona.meet<class<"U64">, imm> -> !verona.class<"U64">

    // We can extend a type to a disjunction.
    %c = verona.copy %b : !verona.class<"U64"> -> !verona.join<class<"U64">, class<"S64">>

    // We can widen the disjunction to more types.
    %d = verona.copy %c : !verona.join<class<"U64">, class<"S64">> -> !verona.join<class<"U64">, class<"S64">, class<"U32">>

    // We can re-order types of a join.
    %e = verona.copy %d : !verona.join<class<"U64">, class<"S64">, class<"U32">> -> !verona.join<class<"U32">, class<"U64">, class<"S64">>

    // We can copy anything into top (ie. an empty meet).
    %f = verona.copy %e : !verona.join<class<"U32">, class<"U64">, class<"S64">> -> !verona.top

    // Join and meet can nest
    %g = verona.copy %a : !verona.meet<class<"U64">, imm> -> !verona.join<meet<class<"U64">, imm>, class<"S64">>

    return
  }

  func @test2(%a: !verona.bottom) {
    // Bottom (ie. an empty join) is a subtype of anything.
    %b = verona.copy %a: !verona.bottom -> !verona.class<"U64">

    return
  }

  func @test_distributivity(%a: !verona.meet<class<"U64">, join<iso, mut>>) {
    // We allow distributivity of join over meets.
    %b = verona.copy %a: !verona.meet<class<"U64">, join<iso, mut>> -> !verona.join<meet<class<"U64">, iso>, meet<class<"U64">, mut>> 

    return
  }
}
