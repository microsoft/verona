// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "error.h"

namespace mlir::verona
{
  char RuntimeError::ID = 0;
  llvm::Error runtimeError(llvm::StringRef desc)
  {
    return llvm::make_error<llvm::StringError>(
      desc, llvm::inconvertibleErrorCode());
  }

  char ParsingError::ID = 0;
  llvm::Error parsingError(llvm::StringRef desc, mlir::Location loc)
  {
    return llvm::make_error<ParsingError>(desc, loc);
  }
}
