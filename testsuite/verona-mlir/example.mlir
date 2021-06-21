// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

module @__  {
  func @"____$module-0__Math__getTruth"() -> i64 {
    %c42_i64 = constant 42 : i64
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %1 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %2 = llvm.load %1 : !llvm.ptr<i64>
    llvm.store %c42_i64, %1 : !llvm.ptr<i64>
    %3 = llvm.load %1 : !llvm.ptr<i64>
    return %3 : i64
  }
  func @"____$module-0__Math__getRandom"() -> i64 {
    %c1_i64 = constant 1 : i64
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %1 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %2 = llvm.load %1 : !llvm.ptr<i64>
    llvm.store %c1_i64, %1 : !llvm.ptr<i64>
    %3 = llvm.load %1 : !llvm.ptr<i64>
    return %3 : i64
  }
  func @"____$module-0__bar"() -> i64 {
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %1 = call @"____$module-0__Math__getRandom"() : () -> i64
    %2 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %3 = llvm.load %2 : !llvm.ptr<i64>
    llvm.store %1, %2 : !llvm.ptr<i64>
    %4 = llvm.load %2 : !llvm.ptr<i64>
    return %4 : i64
  }
  func @"____$module-0__foo"(%arg0: i64) -> i64 {
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %1 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    llvm.store %arg0, %1 : !llvm.ptr<i64>
    %2 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %3 = call @"____$module-0__bar"() : () -> i64
    %4 = llvm.getelementptr %2[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %5 = llvm.load %4 : !llvm.ptr<i64>
    llvm.store %3, %4 : !llvm.ptr<i64>
    %6 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %7 = llvm.load %4 : !llvm.ptr<i64>
    %8 = llvm.load %1 : !llvm.ptr<i64>
    %9 = addi %7, %8 : i64
    %10 = llvm.getelementptr %6[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %11 = llvm.load %10 : !llvm.ptr<i64>
    llvm.store %9, %10 : !llvm.ptr<i64>
    %12 = llvm.load %10 : !llvm.ptr<i64>
    return %12 : i64
  }
  func @main() -> i64 {
    %c21_i64 = constant 21 : i64
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = llvm.alloca %c1_i32 x !llvm.struct<"Math", (i64, i64)> : (i32) -> !llvm.ptr<struct<"Math", (i64, i64)>>
    %1 = call @"____$module-0__Math__getTruth"() : () -> i64
    %2 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %3 = llvm.getelementptr %2[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %4 = llvm.load %3 : !llvm.ptr<i64>
    llvm.store %1, %3 : !llvm.ptr<i64>
    %5 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<"Math", (i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
    %6 = llvm.load %3 : !llvm.ptr<i64>
    %7 = llvm.load %5 : !llvm.ptr<i64>
    llvm.store %6, %5 : !llvm.ptr<i64>
    %8 = llvm.getelementptr %0[%c0_i32, %c1_i32] : (!llvm.ptr<struct<"Math", (i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
    %9 = llvm.load %3 : !llvm.ptr<i64>
    %10 = llvm.load %3 : !llvm.ptr<i64>
    %11 = addi %9, %10 : i64
    %12 = llvm.load %8 : !llvm.ptr<i64>
    llvm.store %11, %8 : !llvm.ptr<i64>
    %13 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %14 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %15 = llvm.getelementptr %14[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %16 = llvm.load %15 : !llvm.ptr<i64>
    llvm.store %c21_i64, %15 : !llvm.ptr<i64>
    %17 = llvm.load %15 : !llvm.ptr<i64>
    %18 = call @"____$module-0__foo"(%17) : (i64) -> i64
    %19 = llvm.getelementptr %13[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %20 = llvm.load %19 : !llvm.ptr<i64>
    llvm.store %18, %19 : !llvm.ptr<i64>
    %21 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %22 = llvm.load %19 : !llvm.ptr<i64>
    %23 = llvm.load %5 : !llvm.ptr<i64>
    %24 = addi %22, %23 : i64
    %25 = llvm.getelementptr %21[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %26 = llvm.load %25 : !llvm.ptr<i64>
    llvm.store %24, %25 : !llvm.ptr<i64>
    %27 = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr<i64>
    %28 = llvm.load %25 : !llvm.ptr<i64>
    %29 = llvm.load %8 : !llvm.ptr<i64>
    %30 = addi %28, %29 : i64
    %31 = llvm.getelementptr %27[%c0_i32] : (!llvm.ptr<i64>, i32) -> !llvm.ptr<i64>
    %32 = llvm.load %31 : !llvm.ptr<i64>
    llvm.store %30, %31 : !llvm.ptr<i64>
    %33 = llvm.load %31 : !llvm.ptr<i64>
    return %33 : i64
  }
}
