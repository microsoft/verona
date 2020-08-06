// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  /// Perform typechecking on the given operation.
  ///
  /// For every operation contained within `op` (including `op` itself), if the
  /// operation implements `TypecheckInterface`, the `typecheck` implementation
  /// of that operation will be executed.
  ///
  /// Returns a successful result if all operations typecheck correctly.
  LogicalResult typecheck(Operation* op);

  /// TypecheckerPass wraps the `typecheck` function into a conventional MLIR
  /// pass, so it can easily be interleaved with other passes in a PassManager.
  class TypecheckerPass : public PassWrapper<TypecheckerPass, OperationPass<>>
  {
    void runOnOperation() override;
  };

  /// Returns true if `lhs` is a subtype of `rhs`.
  /// `lhs` and `rhs` should be in normal form already.
  bool isSubtype(Type lhs, Type rhs);

  /// Check whether `lhs` is a subtype of `rhs`. If it isn't, an error is
  /// emitted and a failure is returned.
  LogicalResult checkSubtype(Operation* op, Type lhs, Type rhs);
}
