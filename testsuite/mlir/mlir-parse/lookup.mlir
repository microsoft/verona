// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

!D1 = type !verona.class<"D1">
!D2 = type !verona.class<"D2">
!C1 = type !verona.class<"C1", "f": meet<!D1, mut>>
!C2 = type !verona.class<"C2", "f": meet<!D2, mut>>

// R1 is recursive, with a field of type `R1 & mut`.
!R1 = type !verona.class<"R1", "f": meet<class<"R1">, mut>>

module
{
  // Lookup a field in (C1 & mut), where the field in C1 has type (D1 & mut)
  func @test_class_mut(%x: !verona.meet<!C1, mut>) -> !verona.meet<!D1, mut>
  {
    %y = verona.field_read %x["f"]
      : !verona.meet<!C1, mut>
     -> !verona.meet<!D1, mut>

    return %y : !verona.meet<!D1, mut>
  }

  // Lookup a field in (C1 | C2) & mut, where the field in C1 and C2
  // respectively have type (D1 & mut) and (D2 & mut)
  func @test_join(%x: !verona.meet<join<!C1, !C2>, mut>) -> !verona.meet<join<!D1, !D2>, mut>
  {
    %y = verona.field_read %x["f"]
       : !verona.meet<join<!C1, !C2>, mut>
      -> !verona.meet<join<!D1, !D2>, mut>

    return %y : !verona.meet<join<!D1, !D2>, mut>
  }

  // Lookup a field in (C1 & C2 & mut), where the field in C1 and C2
  // respectively have type (D1 & mut) and (D2 & mut)
  func @test_meet(%x: !verona.meet<!C1, !C2, mut>) -> !verona.meet<!D1, !D2, mut>
  {
    %y = verona.field_read %x["f"]
       : !verona.meet<!C1, !C2, mut>
      -> !verona.meet<!D1, !D2, mut>

    return %y : !verona.meet<!D1, !D2, mut>
  }

  // We can read anything out of a bottom type
  func @test_bottom(%x: !verona.bottom) {
    %y = verona.field_read %x["f"] : !verona.bottom -> !verona.bottom
    %z = verona.field_read %x["g"] : !verona.bottom -> !verona.meet<!D1, !D2, iso>
    return
  }

  func @test_recursive(%x: !verona.meet<!R1, mut>) -> !verona.meet<!R1, mut>
  {
    %y = verona.field_read %x["f"] : !verona.meet<!R1, mut> -> !verona.meet<!R1, mut>
    %z = verona.field_read %y["f"] : !verona.meet<!R1, mut> -> !verona.meet<!R1, mut>
    return %z : !verona.meet<!R1, mut>
  }

  func @test_recursive_loop(%x: !verona.meet<!R1, mut>) -> !verona.meet<!R1, mut>
  {
    br ^loop(%x: !verona.meet<!R1, mut>)

  ^loop(%a1: !verona.meet<!R1, mut>):
    %a2 = verona.field_read %a1["f"] : !verona.meet<!R1, mut> -> !verona.meet<!R1, mut>
    %b = constant 1 : i1
    cond_br %b, ^loop(%a2: !verona.meet<!R1, mut>), ^end

  ^end:
    return %a2 : !verona.meet<!R1, mut>

  }
}
