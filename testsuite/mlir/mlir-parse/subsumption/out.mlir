

module {
  func @test1(%arg0: !verona.meet<U64, imm>) -> !verona.imm {
    %0 = verona.copy %arg0 : !verona.meet<U64, imm> -> !verona.U64
    %1 = verona.copy %0 : !verona.U64 -> !verona.join<U64, S64>
    %2 = verona.copy %1 : !verona.join<U64, S64> -> !verona.join<U64, S64, U32>
    %3 = verona.copy %2 : !verona.join<U64, S64, U32> -> !verona.join<U32, U64, S64>
    %4 = verona.copy %3 : !verona.join<U32, U64, S64> -> !verona.top
    %5 = verona.copy %arg0 : !verona.meet<U64, imm> -> !verona.join<meet<U64, imm>, S64>
    verona.return %arg0 : !verona.meet<U64, imm>
  }
  func @test2(%arg0: !verona.bottom) -> !verona.U64 {
    verona.return %arg0 : !verona.bottom
  }
}