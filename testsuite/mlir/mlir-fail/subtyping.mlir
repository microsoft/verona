// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

module {
  func @test(%x: !verona.meet<U64, iso>) {
    // expected-error@+1 {{'!verona.meet<U64, iso>' is not a subtype of '!verona.meet<U64, mut>'}}
    %y = verona.copy %x : !verona.meet<U64, iso> -> !verona.meet<U64, mut>
    return
  }
}

// -----

module {
  func @test(%x: !verona.meet<class<"C">, mut>) {
    // expected-error@+1 {{'!verona.meet<class<"C">, mut>' is not a subtype of '!verona.meet<class<"D">, mut>'}}
    %y = verona.copy %x : !verona.meet<class<"C">, mut> -> !verona.meet<class<"D">, mut>
    return
  }
}
