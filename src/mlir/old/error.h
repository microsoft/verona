// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/IR/Location.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

/**
 * Error Handling
 *
 * This file implements the various types of errors that the generator methods
 * can return when trying to parse Verona files to MLIR, and then lowering to
 * LLVM IR.
 *
 * See: http://llvm.org/docs/ProgrammersManual.html#error-handling
 *
 * They all must have a description. Parsing related errors must also have a
 * file location, pointing to the original place the failing function is
 * trying to work with. Stacking multiple errors generate a stack trace.
 *
 * FIXME: The descriptions are initialised by the caller directly with s
 * string value, which doesn't work when the error could be emitted in
 * multiple languages. We need a mechanism where strings could come from
 * any language with a formatted output for a list of arguments, filled in
 * by the caller.
 */
namespace mlir::verona
{
  /// RuntimeError: Errors that have no relation to the source file
  class RuntimeError : public llvm::ErrorInfo<RuntimeError>
  {
    /// Human-readable description of the error
    std::string desc;

  public:
    /**
     * Internal ID to control matching. The value of this field is never used,
     * its address is used as a unique identifier.
     */
    static char ID;
    RuntimeError(llvm::StringRef desc) : desc(desc) {}

    /**
     * Public interface defined by `ErrorInfo`, used for error handlers to
     * report the human-readable error message. This version displays the
     * message description and the source location to which it corresponds.
     */
    void log(llvm::raw_ostream& OS) const override
    {
      OS << desc;
    }
    /**
     * Public ErrorInfo interface, converts to error_code for
     * interoperation with other error handling mechanisms that use the C++11
     * error types.
     * FIXME: For now, we don't care about that, but once we start working
     * with other error handling we should implement this for real.
     */
    std::error_code convertToErrorCode() const override
    {
      return llvm::inconvertibleErrorCode();
    }
  };

  /**
   * Create and return a new runtime error.
   * The returned error will own a copy of the string passed as `desc`.
   */
  llvm::Error runtimeError(llvm::StringRef desc);

  // Parsing Error: Errors when parsing Verona into MLIR
  class ParsingError : public llvm::ErrorInfo<ParsingError>
  {
    /// Human-readable description of the error
    std::string desc;
    /// File location where the error occurred
    mlir::Location loc;

  public:
    /**
     * Internal ID to control matching. The value of this field is never used,
     * its address is used as a unique identifier.
     */
    static char ID;
    ParsingError(llvm::StringRef desc, mlir::Location loc)
    : desc(desc), loc(loc)
    {}
    /**
     * Public interface defined by `ErrorInfo`, used for error handlers to
     * report the human-readable error message. This version displays the
     * message description and the source location to which it corresponds.
     */
    void log(llvm::raw_ostream& OS) const override
    {
      OS << desc << " at ";
      loc.print(OS);
    }
    /**
     * Public ErrorInfo interface, converts to error_code for
     * interoperation with other error handling mechanisms that use the C++11
     * error types.
     * FIXME: For now, we don't care about that, but once we start working
     * with other error handling we should implement this for real.
     */
    std::error_code convertToErrorCode() const override
    {
      return llvm::inconvertibleErrorCode();
    }
  };

  /**
   * Create and return a new parsing error.
   * The returned error will own a copy of the string passed as `desc`.
   */
  llvm::Error parsingError(llvm::StringRef desc, mlir::Location loc);
}
