#include "Typechecker.h"

#include "TypecheckInterface.h"

namespace mlir::verona
{
  LogicalResult typecheck(Operation* op)
  {
    auto callback = [](TypecheckInterface innerOp) -> WalkResult {
      // If typecheck fails, WalkResult::interrupt is returned.
      return innerOp.typecheck();
    };

    if (op->walk(callback).wasInterrupted())
      return failure();
    else
      return success();
  }

  void TypecheckerPass::runOnOperation()
  {
    if (failed(typecheck(getOperation())))
    {
      signalPassFailure();
    }

    // Typechecking does not modify the IR, so all analysis are preserved.
    markAllAnalysesPreserved();
  }
}
