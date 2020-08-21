

module {
  func @test1(%arg0: !verona.meet<U64, imm>) {
    %0 = verona.copy %arg0 : !verona.meet<U64, imm> -> !verona.U64
    %1 = verona.copy %0 : !verona.U64 -> !verona.join<U64, S64>
    %2 = verona.copy %1 : !verona.join<U64, S64> -> !verona.join<U64, S64, U32>
    %3 = verona.copy %2 : !verona.join<U64, S64, U32> -> !verona.join<U32, U64, S64>
    %4 = verona.copy %3 : !verona.join<U32, U64, S64> -> !verona.top
    %5 = verona.copy %arg0 : !verona.meet<U64, imm> -> !verona.join<meet<U64, imm>, S64>
    return
  }
  func @test2(%arg0: !verona.bottom) {
    %0 = verona.copy %arg0 : !verona.bottom -> !verona.U64
    return
  }
  func @test_distributivity(%arg0: !verona.meet<U64, join<iso, mut>>) {
    %0 = verona.copy %arg0 : !verona.meet<U64, join<iso, mut>> -> !verona.join<meet<U64, iso>, meet<U64, mut>>
    return
  }
}