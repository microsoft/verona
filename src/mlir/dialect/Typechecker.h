#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  LogicalResult typecheck(Operation* op);

  /// TypecheckerPass wraps the `typecheck` function into a conventional MLIR
  /// pass, so it can easily be interleaved with other passes in a PassManager.
  class TypecheckerPass : public PassWrapper<TypecheckerPass, OperationPass<>>
  {
    void runOnOperation() override;
  };

  bool isSubtype(Type lhs, Type rhs);
  LogicalResult checkSubtype(Location loc, Type lhs, Type rhs);
}
