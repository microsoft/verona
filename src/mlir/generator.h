// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "error.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Target/LLVMIR.h"
#include "symbol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <peglib.h>
#include <string>

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
    using ValuesTy = llvm::SmallVector<mlir::Value, 4>;
    ValuesTy values;

  public:
    /// Default constructor, builds an empty return value.
    ReturnValue() {}
    /// Single value constructor.
    ReturnValue(mlir::Value& value)
    {
      values.push_back(value);
    }
    /// Multiple value constructor.
    ReturnValue(mlir::ResultRange& range)
    {
      values.insert(values.begin(), range.begin(), range.end());
    }
    /// Assignment operator for ReturnValue.
    ReturnValue& operator=(ReturnValue& other)
    {
      values = other.values;
      return *this;
    }
    /// Assignment operator for mlir::Value.
    ReturnValue& operator=(mlir::Value& value)
    {
      values.push_back(value);
      return *this;
    }

    /// Returns true if this return value has exactly one value.
    bool hasValue() const
    {
      return values.size() == 1;
    }

    /// Access to the single value held, error if none or more than one.
    mlir::Value get() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() == 1 && "Direct access to multiple values");
      return values[0];
    }

    /// Returns true if this return value has one or more values.
    bool hasValues() const
    {
      return !values.empty();
    }

    /// Returns a copy of the values.
    ValuesTy getAll() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      return values;
    }

    /// Add elements to the list
    void push_back(mlir::Value& value)
    {
      values.push_back(value);
    }
  };

  /**
   * MLIR Generator.
   */
  struct Generator
  {
    /**
     * Convert an AST into a high-level MLIR module.
     *
     * Currently this generates opaque MLIR operations, which can't be processed
     * by the rest of the compiler, but we will slowly be transitioning to the
     * Verona dialect.
     *
     */
    static llvm::Expected<mlir::OwningModuleRef>
    lower(MLIRContext* context, const ::ast::Ast& ast);

  private:
    Generator(MLIRContext* context) : context(context), builder(context)
    {
      // Initialise known opaque types, for comparison.
      // TODO: Use Verona dialect types directly and isA<>.
      allocaTy = genOpaqueType("alloca");
      unkTy = genOpaqueType("unk");
      noneTy = genOpaqueType("none");
      boolTy = builder.getI1Type();
    }

    using Types = llvm::SmallVector<mlir::Type, 4>;

    /// MLIR module.
    mlir::OwningModuleRef module;

    /// MLIR context. This is owned by the caller of Generator::lower.
    mlir::MLIRContext* context;

    /// MLIR builder.
    mlir::OpBuilder builder;

    /// Symbol tables for variables.
    SymbolTableT symbolTable;
    /// Symbol tables for functions.
    FunctionTableT functionTable;
    /// Symbol tables for classes.
    TypeTableT typeTable;
    /// Nested reference for head/exit blocks in loops.
    BasicBlockTableT loopTable;

    /// Alloca types, before we start using Verona types with known sizes.
    mlir::Type allocaTy;
    /// Unknown types, will be defined during type inference.
    mlir::Type unkTy;
    /// Temporary type to hold no types at all (ex: return void).
    mlir::Type noneTy;
    /// MLIR boolean type (int1).
    mlir::Type boolTy;

    /// Get location of an ast node
    mlir::Location getLocation(const ::ast::Ast& ast);

    /// Parses a module, the global context.
    llvm::Error parseModule(const ::ast::Ast& ast);

    /// Parses the prototype (signature) of a function.
    llvm::Expected<mlir::FuncOp> parseProto(const ::ast::Ast& ast);
    /// Parses a function, from a top-level (module) view.
    llvm::Expected<mlir::FuncOp> parseFunction(const ::ast::Ast& ast);

    /// Recursive type parser, gathers all available information on the type
    /// and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

    /// Declares a new variable.
    void declareVariable(llvm::StringRef name, mlir::Value val);
    /// Updates am existing variable.
    void updateVariable(llvm::StringRef name, mlir::Value val);

    /// Generic node parser, calls other parse functions to handle each
    /// individual type.
    llvm::Expected<ReturnValue> parseNode(const ::ast::Ast& ast);

    /// Parses a block (multiple statements), return last value.
    llvm::Expected<ReturnValue> parseBlock(const ::ast::Ast& ast);
    /// Parses a value (constants, variables).
    llvm::Expected<ReturnValue> parseValue(const ::ast::Ast& ast);
    /// Parses an assign statement.
    llvm::Expected<ReturnValue> parseAssign(const ::ast::Ast& ast);
    /// Parses function calls and native operations.
    llvm::Expected<ReturnValue> parseCall(const ::ast::Ast& ast);
    /// Parses an if/else block.
    llvm::Expected<ReturnValue> parseCondition(const ::ast::Ast& ast);
    /// Parses a 'while' loop block.
    llvm::Expected<ReturnValue> parseWhileLoop(const ::ast::Ast& ast);
    /// Parses a 'continue' statement.
    llvm::Expected<ReturnValue> parseContinue(const ::ast::Ast& ast);
    /// Parses a 'break' statement.
    llvm::Expected<ReturnValue> parseBreak(const ::ast::Ast& ast);
    /// Parses a 'return' statement.
    llvm::Expected<ReturnValue> parseReturn(const ::ast::Ast& ast);

    // =============================================================== Temporary

    /// Wrapper for opaque operators before we use actual Verona dialect.
    llvm::Expected<ReturnValue> genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);

    /// Wrappers for opaque types before we use actual Verona dialect.
    mlir::OpaqueType genOpaqueType(llvm::StringRef name);
  };
}
