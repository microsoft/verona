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
    ReturnValue(mlir::Value value)
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
    Generator(MLIRContext* context)
    : context(context), builder(context), unkLoc(builder.getUnknownLoc())
    {
      // Initialise known opaque types, for comparison.
      // TODO: Use Verona dialect types directly and isA<>.
      allocaTy = genOpaqueType("alloca");
      unkTy = genOpaqueType("unk");
      boolTy = builder.getI1Type();
    }

    using Types = llvm::SmallVector<mlir::Type, 4>;

    /// MLIR module.
    mlir::OwningModuleRef module;

    /// MLIR context. This is owned by the caller of Generator::lower.
    mlir::MLIRContext* context;

    /// MLIR builder.
    mlir::OpBuilder builder;

    /// Unknown location, for compiler generated stuff.
    mlir::Location unkLoc;

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
    /// MLIR boolean type (int1).
    mlir::Type boolTy;

    // ===================================================== Helpers
    // Methods for symbols, location and other helpers for building
    // MLIR nodes.

    /// Get location of an ast node.
    mlir::Location getLocation(const ::ast::Ast& ast);

    /// Declares a new variable.
    void declareVariable(llvm::StringRef name, mlir::Value val);
    /// Updates an existing variable in the local context.
    void updateVariable(llvm::StringRef name, mlir::Value val);
    /// Declare a (compiler generated) function.
    void declareFunction(
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> types,
      llvm::StringRef retTy);

    // ================================================================= Parsers
    // These methods parse the AST into MLIR constructs, then either return the
    // expected MLIR value or call the generators (see below) to do that for
    // them.

    /// Parses a module, the global context.
    llvm::Error parseModule(const ::ast::Ast& ast);

    /// Parses a function, from a top-level (module) view.
    llvm::Expected<mlir::FuncOp> parseFunction(const ::ast::Ast& ast);

    /// Recursive type parser, gathers all available information on the type
    /// and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

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
    /// Parses a 'for' loop block.
    llvm::Expected<ReturnValue> parseForLoop(const ::ast::Ast& ast);
    /// Parses a 'continue' statement.
    llvm::Expected<ReturnValue> parseContinue(const ::ast::Ast& ast);
    /// Parses a 'break' statement.
    llvm::Expected<ReturnValue> parseBreak(const ::ast::Ast& ast);
    /// Parses a 'return' statement.
    llvm::Expected<ReturnValue> parseReturn(const ::ast::Ast& ast);

    // ============================================================== Generators
    // These methods build complex MLIR constructs from parameters either
    // acquired from the AST or built by the compiler as to mimic the AST.

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
    /// Generates a conditional branch, casting to i1 if necessary
    llvm::Error generateCondBranch(
      mlir::Location loc,
      mlir::Value cond,
      mlir::Block* ifBB,
      mlir::ValueRange ifArgs,
      mlir::Block* elseBB,
      mlir::ValueRange elseArgs);
    /// Generates an unconditional loop branch (continue, creak)
    llvm::Error
    generateLoopBranch(mlir::Location loc, llvm::StringRef blockName);

    // ======================================================= Generator Helpers
    // These are helpers for the generators, creating simple recurrent patterns
    // for basic constructs (casts, load/store, constants, etc).
    // They return mlir::Values because they're not supposed to fail.

    /// Generates a cast, trusting the AST did the right thing
    mlir::Value
    generateAutoCast(mlir::Location loc, mlir::Value value, mlir::Type type);
    /// Generates a verona constant with opaque type
    mlir::Value generateConstant(
      mlir::Location loc, llvm::StringRef value, llvm::StringRef typeName);
    /// Generates a verona alloca with special type (or allocaTy if none)
    // TODO: use defining operation instead of a special type, default to unkTy
    mlir::Value
    generateAlloca(mlir::Location loc, llvm::StringRef typeName = "");
    /// Generates a verona load (using address' type, or unkTy)
    mlir::Value generateLoad(mlir::Location loc, mlir::Value addr);
    /// Generates a verona store (using value's type, or unkTy)
    mlir::Value
    generateStore(mlir::Location loc, mlir::Value value, mlir::Value addr);

    // =============================================================== Temporary
    // These methods should disappear once the Verona dialect is more advanced.

    /// Wrapper for opaque operators before we use actual Verona dialect.
    mlir::Value genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);

    /// Wrappers for opaque types before we use actual Verona dialect.
    mlir::OpaqueType genOpaqueType(llvm::StringRef name);
  };
}
