// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "error.h"

namespace mlir::verona
{
  char RuntimeError::ID = 0;
  llvm::Error runtimeError(llvm::StringRef desc)
  {
    return llvm::make_error<RuntimeError>(desc);
  }
}
