// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "error.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR.h"
#include "parser/ast.h"
#include "symbol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <variant>

namespace mlir::verona
{
  /**
   * Return Value.
   *
   * Verona constructs (including functions, operations, statements) can return
   * 0, 1 or many values (tuples). Constructs that don't return values (like
   * lexical blocks, conditionals, loops, termination) can safely return an
   * empty "ReturnValue" with the guarantee (by the ast construction) that no
   * other construct will try to use their return value.
   *
   * Tuples in MLIR are best represented by a loose list of values:
   * https://mlir.llvm.org/docs/Rationale/Rationale/#tuple-types
   * So we keep track of all values returned along with their normalised and
   * canonicalised types (property distribution), and can access them via simply
   * taking the n-th value in the list.
   */
  class ReturnValue
  {
    using ResultTy = std::variant<Value, ModuleOp, FuncOp>;
    using ValuesTy = llvm::SmallVector<ResultTy, 1>;
    ValuesTy values;

  public:
    /// Default constructor, builds an empty return value.
    ReturnValue() {}
    /// Single value constructor.
    ReturnValue(Value value)
    {
      values.push_back(value);
    }
    /// Single value constructor.
    ReturnValue(ModuleOp mod)
    {
      values.push_back(mod);
    }
    /// Single value constructor.
    ReturnValue(FuncOp func)
    {
      values.push_back(func);
    }
    /// Multiple value constructor.
    ReturnValue(ResultRange& range)
    {
      values.insert(values.begin(), range.begin(), range.end());
    }
    /// Assignment operator for Value.
    ReturnValue& operator=(Value& value)
    {
      values.push_back(value);
      return *this;
    }

    /// Returns true if this return value has exactly `n` values.
    bool hasValue(size_t n = 1) const
    {
      assert(n != 0);
      return values.size() == n;
    }

    /// Access to the single value held, error if none or more than one.
    template<class T>
    T get() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() == 1 && "Direct access to multiple values");
      return std::get<T>(values[0]);
    }

    /// Access a specific value held, error if none or not enough.
    template<class T>
    T get(size_t n) const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() > n && "Not enough values");
      return std::get<T>(values[n - 1]);
    }

    /// Returns a copy of the values.
    llvm::ArrayRef<ResultTy> getAll() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      return values;
    }

    /// Add elements to the list
    void push_back(Value& value)
    {
      values.push_back(value);
    }
  };

  /**
   * MLIR Generator.
   */
  class Generator
  {
    Generator(MLIRContext* context) : context(context), builder(context) {}

    /// MLIR module.
    OwningModuleRef module;

    /// MLIR context. This is owned by the caller of Generator::lower.
    MLIRContext* context;

    /// MLIR builder.
    OpBuilder builder;

    /// Symbol tables for variables.
    SymbolTableT symbolTable;
    /// Symbol tables for functions.
    FunctionTableT functionTable;
    /// Symbol tables for classes.
    TypeTableT typeTable;
    /// Nested reference for head/exit blocks in loops.
    BasicBlockTableT loopTable;

    /// A list of types
    using Types = llvm::SmallVector<Type, 1>;
    /// AST aliases
    using Ast = ::verona::parser::Ast;
    using AstPath = ::verona::parser::AstPath;
    /// Name of the root module
    static const char* rootModuleName;

    // ===================================================== Helpers
    // Methods for symbols, location and other helpers for building
    // MLIR nodes.

    /// Get location of an ast node.
    Location getLocation(Ast ast);

    // ================================================================= Parsers
    // These methods parse the AST into MLIR constructs, then either return the
    // expected MLIR value or call the generators (see below) to do that for
    // them.

    /// Parses a module, the global context.
    llvm::Error parseModule(Ast ast);

    /// Parse a class declaration
    llvm::Expected<ReturnValue> parseClass(Ast ast);

    /// Generic node parser, calls other parse functions to handle each
    /// individual type.
    llvm::Expected<ReturnValue> parseNode(Ast ast);

    /// Parses a function definition.
    llvm::Expected<ReturnValue> parseFunction(Ast ast);

    /// Parses a lambda (function body).
    llvm::Expected<ReturnValue> parseLambda(Ast ast);

    /// Parses a literal.
    llvm::Expected<ReturnValue> parseLiteral(Ast ast);

    /// Parses a type.
    Type parseType(Ast ast);

    // ===================================================== MLIR Generators
    /// Generate a prototype, populating the symbol table
    llvm::Expected<mlir::FuncOp> generateProto(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Type> types,
      llvm::ArrayRef<mlir::Type> retTy);
    /// Generates an empty function (with the first basic block)
    llvm::Expected<mlir::FuncOp> generateEmptyFunction(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> args,
      llvm::ArrayRef<mlir::Type> types,
      llvm::ArrayRef<mlir::Type> retTy);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef> lower(MLIRContext* context, Ast ast);
  };
}
