module {
  func @test_alias_reflexive(%x: !verona.mut) {
    return
  }

  func @test_alias_copy(%x: !verona.mut) {
    %y = verona.copy %x : !verona.mut -> !verona.mut
    return
  }

  func @test_alias_view(%x: !verona.mut) {
    %y = verona.view %x : !verona.mut -> !verona.mut
    return
  }

  func @test_alias_transitive(%x: !verona.mut) {
    %y = verona.view %x : !verona.mut -> !verona.mut
    %z = verona.view %y : !verona.mut -> !verona.mut
    return
  }

  func @test_alias_block_argument(%x: !verona.mut) {
    br ^bb1(%x, %x: !verona.mut, !verona.mut)

  ^bb1(%y: !verona.mut, %z: !verona.mut):
    return
  }

  func @test_alias_intersect(%b : i1, %x: !verona.mut) {
     cond_br %b, ^bb1, ^bb2
 
   ^bb1:
     %y0 = verona.copy %x : !verona.mut -> !verona.mut
     br ^end(%y0 : !verona.mut)
 
   ^bb2:
     %y1 = verona.view %x : !verona.mut -> !verona.mut
     br ^end(%y1 : !verona.mut)
 
   ^end(%y: !verona.mut):
     return
  }
}
