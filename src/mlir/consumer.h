// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "generator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "parser/ast.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::verona
{
  /// Field offset maps between field names, types and their relative position.
  struct FieldOffset
  {
    SmallVector<llvm::StringRef> fields;
    SmallVector<Type> types;
  };

  /**
   * AST Consumer
   */
  class ASTConsumer
  {
    /// This is temporary to make the passes work, we need to think of a better
    /// way out.
    friend struct ASTDeclarations;
    friend struct ASTDefinitions;

    ASTConsumer(MLIRContext* context) : gen(context) {}

    /// MLIR Generator
    MLIRGenerator gen;

    /// Map for each type which fields does it have.
    /// Use type.getAsOpaquePointer() for keys
    /// FIXME: Find better key than opaque pointer types
    using OpaqueType = const void*;
    std::map<OpaqueType, FieldOffset> classFields;

    // ===================================================== Helpers

    /// Get builder from generator.
    OpBuilder& builder()
    {
      return gen.getBuilder();
    }

    /// Get symbol table from generator.
    SymbolTableT& symbolTable()
    {
      return gen.getSymbolTable();
    }

    /// Get location of an ast node.
    Location getLocation(::verona::parser::NodeDef& ast);

    /// Mangle function names. If scope is not passed, use functionScope.
    std::string mangleName(
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> functionScope = {},
      llvm::ArrayRef<llvm::StringRef> callScope = {});

    /// Return the offset into the structure to load/store values into fields
    /// and the type of the field's value (if stored in a different container).
    std::tuple<size_t, Type, bool>
    getField(Type type, llvm::StringRef fieldName);

    /// Looks up a symbol with the ast's view.
    Value lookup(::verona::parser::Ast ast, bool lastContextOnly = false);

    /// Consumes a type.
    Type consumeType(::verona::parser::Type& ast);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef>
    lower(MLIRContext* context, ::verona::parser::Ast ast);
  };

  /*
   * Scope Cleanup helper
   *
   * Automatically invokes the callable object passed to the constructor on
   * destruction. This class is intended to provide lexically scoped cleanups,
   * for example:
   * ```c++
   * ScopedCleanup defer([&] { cleanup code here });
   * ```
   * The code in the lambda will be invoked when `defer` goes out of scope.
   */
  template<class T>
  class ScopeCleanup
  {
    /// Action to perform on destruction
    T cleanup;

  public:
    ScopeCleanup(T&& c) : cleanup(std::move(c)) {}

    /// Automatically applies the cleanup
    ~ScopeCleanup()
    {
      cleanup();
    }
  };
}
