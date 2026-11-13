#include "compile.h"

#include "lang.h"

#include <trieste/rewriter.h>

namespace virc
{
  std::vector<Pass> pipeline(std::shared_ptr<Compilation> state)
  {
    return {
      assign_ids(state),
      validate_ids(state),
      typecheck(state),
      optimize(state),
      liveness(state),
    };
  }

  CompileResult compile(Node reified_vir)
  {
    auto state = std::make_shared<Compilation>();
    auto process = Rewriter{"VIRC", pipeline(state), wfIR}.rewrite(reified_vir);
    return {std::move(process), std::move(state)};
  }
}
