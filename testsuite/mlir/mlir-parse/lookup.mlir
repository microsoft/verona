// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

module
{
  // Lookup a field in (C1 & mut), where the field in C1 has type (D1 & mut)
  func @test_class_mut(
    %x: !verona.meet<class<"C1", "f": meet<class<"D1">, mut>>, mut>)
    -> !verona.meet<class<"D1">, mut>
  {
    %y = verona.field_read %x["f"]
      : !verona.meet<class<"C1", "f": meet<class<"D1">, mut>>, mut>
      -> !verona.meet<class<"D1">, mut>

    return %y : !verona.meet<class<"D1">, mut>
  }

  // Lookup a field in (C1 | C2) & mut, where the field in C1 and C2
  // respectively have type (D1 & mut) and (D2 & mut)
  func @test_join(
    %x: !verona.meet<join<class<"C1", "f": meet<class<"D1">, mut>>, class<"C2", "f": meet<class<"D2">, mut>>>, mut>)
    -> !verona.meet<join<class<"D1">, class<"D2">>, mut>
  {
    %y = verona.field_read %x["f"] : !verona.meet<join<class<"C1", "f": meet<class<"D1">, mut>>, class<"C2", "f": meet<class<"D2">, mut>>>, mut>
    -> !verona.meet<join<class<"D1">, class<"D2">>, mut>

    return %y : !verona.meet<join<class<"D1">, class<"D2">>, mut>
  }

  // Lookup a field in (C1 & C2 & mut), where the field in C1 and C2
  // respectively have type (D1 & mut) and (D2 & mut)
  func @test_meet(
    %x: !verona.meet<class<"C1", "f": meet<class<"D1">, mut>>, class<"C2", "f": meet<class<"D2">, mut>>, mut>)
    -> !verona.meet<class<"D1">, class<"D2">, mut>
  {
    %y = verona.field_read %x["f"] : !verona.meet<class<"C1", "f": meet<class<"D1">, mut>>, class<"C2", "f": meet<class<"D2">, mut>>, mut>
    -> !verona.meet<class<"D1">, class<"D2">, mut>

    return %y : !verona.meet<class<"D1">, class<"D2">, mut>
  }

  // We can read anything out of a bottom type
  func @test_bottom(%x: !verona.bottom) {
    %y = verona.field_read %x["f"] : !verona.bottom -> !verona.bottom
    %z = verona.field_read %x["g"] : !verona.bottom -> !verona.meet<class<"D1">, class<"D2">, iso>
    return
  }
}
