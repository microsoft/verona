// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "error.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
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
   * MLIR Generator.
   *
   * Generates common patterns in MLIR, used by any verona consumer /
   * transformer that needs to create instructions in a canonical form.
   *
   * Automates the process of checking for all fields and properties, adding the
   * right sequence of required operations or asserting on incompatibility or
   * errors.
   */
  class MLIRGenerator
  {
    /// MLIR module.
    OwningModuleRef module;

    /// MLIR builder for all uses. This is the main builder and users should use
    /// getBuilder() to build stuff on their own.
    OpBuilder builder;

    /// Symbol tables for variables.
    SymbolTableT symbolTable;

  public:
    MLIRGenerator(MLIRContext* context) : builder(context)
    {
      module = ModuleOp::create(builder.getUnknownLoc());
    }

    /// Expose builder to users.
    OpBuilder& getBuilder()
    {
      return builder;
    }

    /// Expose symbol table to users.
    SymbolTableT& getSymbolTable()
    {
      return symbolTable;
    }

    /// Expose the symbol table lookup form the module.
    template<class OpTy>
    OpTy lookupSymbol(StringRef name)
    {
      return module->lookupSymbol<OpTy>(name);
    }

    /// Expose the function push-back in the module.
    void push_back(FuncOp func)
    {
      module->push_back(func);
    }

    /// Return the module with move semantics. No further actions can be taken
    /// on this generator after that.
    OwningModuleRef finish()
    {
      return std::move(module);
    }

    /// Convert (promote/demote) the value to the specified type. This
    /// automatically chooses promotion / demotion based on the types involved.
    Value typeConversion(Value val, Type ty);

    /// Promote the smallest (compatible) type and return the values to be used
    /// for arithmetic operations. If types are same, just return them, if not,
    /// return the cast operations that make them the same. If types are
    /// incompatible, assert.
    std::pair<Value, Value> typePromotion(Value lhs, Value rhs);

    /// Generate a prototype, populating the symbol table
    llvm::Expected<FuncOp> generateProto(
      Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<Type> types,
      llvm::ArrayRef<Type> retTy);

    /// Generates an empty function (with the first basic block)
    llvm::Expected<FuncOp> generateEmptyFunction(
      Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<Type> types,
      llvm::ArrayRef<Type> retTy);

    /// Generates a call to a static function (FIXME: implement dynamic calls)
    llvm::Expected<Value>
    generateCall(Location loc, FuncOp func, llvm::ArrayRef<Value> args);

    /// Generates arithmetic based on param types and op names
    llvm::Expected<Value> generateArithmetic(
      Location loc, llvm::StringRef opName, Value lhs, Value rhs);

    /// Generates an alloca (stack variable)
    Value generateAlloca(Location loc, Type ty);

    /// Generates an element pointer
    Value generateGEP(Location loc, Value addr, int offset = 0);

    /// Generates a load of an address
    Value generateLoad(Location loc, Value addr, int offset = 0);

    /// Generates a load if the expected type is not a pointer and is compatible
    /// with the element type (asserts if not)
    Value generateAutoLoad(Location loc, Value addr, Type ty, int offset = 0);

    /// Generates a store into an address
    void generateStore(Location loc, Value addr, Value val, int offset = 0);

    /// Generate a constant value of a certain type
    Value generateConstant(Type ty, std::variant<int, double> val);

    /// Generate a zero initialized value of a certain type
    Value generateZero(Type ty);
  };
}
