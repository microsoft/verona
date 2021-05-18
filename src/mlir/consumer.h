// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "error.h"
#include "generator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "parser/ast.h"

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
    /// The list of values returned. Usually one or zero but could be more.
    llvm::SmallVector<Value, 1> values;

  public:
    /// Default constructor, builds an empty return value.
    ReturnValue() {}
    /// Constructor for single valued Value
    ReturnValue(Value value)
    {
      values.push_back(value);
    }
    /// Multiple value constructor (not necessarily same types)
    ReturnValue(ResultRange& range)
    {
      values.insert(values.begin(), range.begin(), range.end());
    }
    /// Assignment operator for Value.
    ReturnValue& operator=(Value& value)
    {
      values.clear();
      values.push_back(value);
      return *this;
    }

    /// Returns true if this return value has exactly one value.
    bool hasValue() const
    {
      return hasValues(1);
    }

    /// Returns true if this return value has exactly `n` values.
    bool hasValues(size_t n = 1) const
    {
      return values.size() == n;
    }

    /// Access to the single value held, asserts if none or more than one.
    Value get() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() == 1 && "Direct access to multiple values");
      return values[0];
    }

    /// Access a specific value held, asserts if none or not enough values.
    Value get(size_t n) const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() > n && "Not enough values");
      return values[n - 1];
    }

    /// Returns a reference to all the values.
    llvm::ArrayRef<Value> getAll() const
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
   * Field offset maps between field names, types and their relative position.
   *
   * FIXME: This will need to be better encapsulated, but we'll expand it when
   * we start implementing classes fully, and dynamic dispatch.
   */
  struct FieldOffset
  {
    llvm::SmallVector<llvm::StringRef, 1> fields;
    llvm::SmallVector<Type, 1> types;
  };

  /**
   * AST Consumer
   */
  class ASTConsumer
  {
    ASTConsumer(MLIRContext* context) : generator(context) {}

    /// MLIR Generator
    MLIRGenerator generator;

    /// Map for each type which fields does it have.
    std::map<llvm::StringRef, FieldOffset> classFields;

    /// Function scope, for mangling names. 3 because there will always be the
    /// root module, the current module and a class, at the very least.
    llvm::SmallVector<llvm::StringRef, 3> functionScope;

    /// AST aliases
    using Ast = ::verona::parser::Ast;
    using AstPath = ::verona::parser::AstPath;

    // ===================================================== Helpers
    // Methods for symbols, location and other helpers for building
    // MLIR nodes.

    /// Get builder from generator.
    OpBuilder& builder()
    {
      return generator.getBuilder();
    }

    /// Get symbol table from generator.
    SymbolTableT& symbolTable()
    {
      return generator.getSymbolTable();
    }

    /// Get location of an ast node.
    Location getLocation(Ast ast);

    /// Mangle function names. If scope is not passed, use functionScope.
    std::string mangleName(
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> scope = {});

    /// Return the offset into the structure to load/store values into fields
    /// and the type of the field's value (if stored in a different container).
    std::tuple<size_t, Type, bool>
    getField(Type type, llvm::StringRef fieldName);

    // ===================================================== Top-Level Consumers
    /// Consumes the top module.
    llvm::Error consumeRootModule(Ast ast);

    /// Consume a class declaration
    llvm::Error consumeClass(Ast ast);

    /// Consumes a function definition.
    llvm::Expected<FuncOp> consumeFunction(Ast ast);

    // ======================================================= General Consumers
    /// Generic node consumer, calls other consumer functions to handle each
    /// individual type.
    llvm::Expected<ReturnValue> consumeNode(Ast ast);

    /// Consumes a field definition.
    llvm::Expected<Type> consumeField(Ast ast);

    /// Consumes a lambda (function body).
    llvm::Expected<ReturnValue> consumeLambda(Ast ast);

    /// Consumes a select statement.
    llvm::Expected<ReturnValue> consumeSelect(Ast ast);

    /// Consumes a variable reference.
    llvm::Expected<ReturnValue> consumeRef(Ast ast);

    /// Consumes a let/var binding.
    llvm::Expected<ReturnValue> consumeLocalDecl(Ast ast);

    /// Consumes a type declaration.
    llvm::Expected<ReturnValue> consumeOfType(Ast ast);

    /// Consumes a variable assignment.
    llvm::Expected<ReturnValue> consumeAssign(Ast ast);

    /// Consumes a literal.
    llvm::Expected<ReturnValue> consumeLiteral(Ast ast);

    /// Consumes a string literal.
    llvm::Expected<ReturnValue> consumeString(Ast ast);

    /// Consumes a type.
    Type consumeType(Ast ast);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef> lower(MLIRContext* context, Ast ast);
  };
}
