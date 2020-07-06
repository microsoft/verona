// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/IR/Location.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

// Error Handling
//
// This file implements the various types of errors that the generator methods
// can return when trying to parse Verona files to MLIR, and then lowering to
// LLVM IR.
//
// They all must have a description. Parsing related errors must also have a
// file location, pointing to the original place the failing function is
// trying to work with. Stacking multiple errors generate a stack trace.
namespace mlir::verona
{
  // RuntimeError: Errors that have no relation to the source file
  class RuntimeError : public llvm::ErrorInfo<RuntimeError>
  {
    std::string desc;

  public:
    static char ID;
    RuntimeError(llvm::StringRef desc) : desc(desc) {}
    void log(llvm::raw_ostream& OS) const override
    {
      OS << desc;
    }
    std::error_code convertToErrorCode() const override
    {
      return llvm::inconvertibleErrorCode();
    }
  };
  // Create a parsing error and return
  llvm::Error runtimeError(llvm::StringRef desc);

  // Parsing Error: Errors when parsing Verona into MLIR
  class ParsingError : public llvm::ErrorInfo<ParsingError>
  {
    std::string desc;
    mlir::Location loc;

  public:
    static char ID;
    ParsingError(llvm::StringRef desc, mlir::Location loc)
    : desc(desc), loc(loc)
    {}
    void log(llvm::raw_ostream& OS) const override
    {
      OS << desc << " at ";
      loc.print(OS);
    }
    std::error_code convertToErrorCode() const override
    {
      return llvm::inconvertibleErrorCode();
    }
  };
  // Create a parsing error and return
  llvm::Error parsingError(llvm::StringRef desc, mlir::Location loc);
}
