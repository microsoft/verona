// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// RUN: verona-mlir %s -o - | verona-mlir | FileCheck %s

module {
  // CHECK: func @bar(%arg0: !verona.U64) -> !type<"U64 & imm"> {
  func @bar(%arg0: !verona.U64) -> !type<"U64 & imm"> {
    // CHECK: %[[res:[0-9]+]] = verona.foo %arg0 : !verona.U64
    %res = verona.foo %arg0 : !verona.U64
    // CHECK: %[[cast:[0-9]+]] = "verona.cast"(%[[res]]) : (!verona.U64) -> !verona.S64
    %cast = "verona.cast"(%res) : (!verona.U64) -> !verona.S64
    // CHECK: %[[foo:[0-9]+]] = "verona.test"(%[[cast]]) : (!verona.S64) -> !type<"U64 & imm">
    %foo = "verona.test"(%cast) : (!verona.S64) -> (!type<"U64 & imm">)
    // CHECK: return %[[foo]] : !type<"U64 & imm">
    return %foo : !type<"U64 & imm">
  }
}
