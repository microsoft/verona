// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

include "mlir/IR/OpBase.td"

def RegionCheckInterface : OpInterface<"RegionCheckInterface"> {
  let methods = [
    InterfaceMethod<
      "Add facts", "void", "add_facts", (ins "FactEvaluator&":$facts)
    >,
  ];
}

def RegionCheck : DeclareOpInterfaceMethods<RegionCheckInterface>;
