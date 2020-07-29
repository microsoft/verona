#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  LogicalResult typecheck(Operation* op);

  class TypecheckerPass : public PassWrapper<TypecheckerPass, OperationPass<>>
  {
    void runOnOperation() override;
  };
}
