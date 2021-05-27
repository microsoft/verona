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
#include "llvm/Support/Error.h"

#include <string>

namespace mlir::verona
{
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
    ASTConsumer(MLIRContext* context) : gen(context) {}

    /// MLIR Generator
    MLIRGenerator gen;

    /// Map for each type which fields does it have.
    std::map<llvm::StringRef, FieldOffset> classFields;

    /// Function scope, for mangling names. 3 because there will always be the
    /// root module, the current module and a class, at the very least.
    llvm::SmallVector<llvm::StringRef, 3> functionScope;

    /// HACK: This tracks assignment types for `select` functions without a
    /// return type.
    /// FIXME: Either `select` should have a type or we should track this in a
    /// context variable of sorts.
    Type assignTypeFromSelect;

    /// AST aliases
    using Ast = ::verona::parser::Ast;
    using AstPath = ::verona::parser::AstPath;

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

    /// Get node as a shared pointer of a sub-type
    template<class T>
    ::verona::parser::Node<T> nodeAs(::verona::parser::Ast from)
    {
      return std::make_shared<T>(from->as<T>());
    }

    /// Get location of an ast node.
    Location getLocation(Ast ast);

    /// Mangle function names. If scope is not passed, use functionScope.
    std::string mangleName(
      llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> scope = {});

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
    llvm::Expected<Value> consumeNode(Ast ast);

    /// Consumes a field definition.
    llvm::Expected<Type> consumeField(Ast ast);

    /// Consumes a lambda (function body).
    llvm::Expected<Value> consumeLambda(Ast ast);

    /// Consumes a select statement.
    llvm::Expected<Value> consumeSelect(Ast ast);

    /// Consumes a variable reference.
    llvm::Expected<Value> consumeRef(Ast ast);

    /// Consumes a let/var binding.
    llvm::Expected<Value> consumeLocalDecl(Ast ast);

    /// Consumes a type declaration.
    llvm::Expected<Value> consumeOfType(Ast ast);

    /// Consumes a variable assignment.
    llvm::Expected<Value> consumeAssign(Ast ast);

    /// Consumes a tuple declaration.
    llvm::Expected<Value> consumeTuple(Ast ast);

    /// Consumes a literal.
    llvm::Expected<Value> consumeLiteral(Ast ast);

    /// Consumes a string literal.
    llvm::Expected<Value> consumeString(Ast ast);

    /// Consumes a type.
    Type consumeType(Ast ast);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef> lower(MLIRContext* context, Ast ast);
  };
}
