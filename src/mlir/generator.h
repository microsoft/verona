// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "symbol.h"

#include <optional>
#include <string>
#include <variant>

namespace mlir::verona
{
  /// LLVM aliases
  using StructType = mlir::LLVM::LLVMStructType;
  using PointerType = mlir::LLVM::LLVMPointerType;
  using ArrayType = mlir::LLVM::LLVMArrayType;

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

    /// Return the number of operators if arithmetic function is recognised
    size_t numArithmeticOps(llvm::StringRef name);

  public:
    MLIRGenerator(MLIRContext* context) : builder(context)
    {
      // There is only one module and it's created here. Everywhere else, the
      // module is a non-woning reference to this one. This is the global
      // module, not a specific Verona module, so there is no location for it.
      // Verona modules will end up embedded in this one via mangling, so no
      // need to create any additiona modules.
      module = ModuleOp::create(builder.getUnknownLoc());
    }

    // ====== Helpers to interface consumers and transformers with the generator

    /// Expose builder to users.
    OpBuilder& getBuilder();

    /// Expose symbol table to users.
    SymbolTableT& getSymbolTable();

    /// Expose the symbol table lookup form the module.
    template<class OpTy>
    OpTy lookupSymbol(StringRef name)
    {
      return module->lookupSymbol<OpTy>(name);
    }

    /// Expose the function push-back in the module.
    void push_back(FuncOp func);

    /// Return the module with move semantics. No further actions can be taken
    /// on this generator after that.
    OwningModuleRef finish();

    // ==================================== Generic helpers that manipulate MLIR

    /// Returns true if the basic block has a terminator
    static bool hasTerminator(Block* bb);

    /// Returns true if val is a pointer.
    static bool isPointer(Value val);

    /// Returns the element type if val is a pointer.
    static Type getPointedType(Value val);

    /// Returns true if val is a pointer to a structure.
    static bool isStructPointer(Value val);

    /// Returns StructType if the value has a pointer to a structure type.
    static StructType getPointedStructType(Value val, bool anonymous = false);

    /// Get the type of the strucure field at this offset
    static Type getFieldType(StructType type, int offset);

    // ==================================================== Top level generators

    /// Generate a prototype, populating the symbol table
    FuncOp Proto(
      Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<Type> types,
      llvm::ArrayRef<Type> retTy);

    /// Generates a definition, with a starting basic block
    FuncOp StartFunction(FuncOp& func);

    /// Generates a call to a static function
    /// FIXME: implement dynamic calls
    Value Call(Location loc, FuncOp func, llvm::ArrayRef<Value> args);

    // ==================================================== Low level generators

    /// Generates an alloca (stack variable)
    Value Alloca(Location loc, Type ty);

    /// Generates an element pointer
    Value GEP(Location loc, Value addr, std::optional<int> offset = {});

    /// Generates a load of an address
    Value Load(Location loc, Value addr, std::optional<int> offset = {});

    /// Generates a load if the expected type is not a pointer and is compatible
    /// with the element type (asserts if not)
    Value AutoLoad(
      Location loc,
      Value addr,
      Type ty = Type(),
      std::optional<int> offset = {});

    /// Generates a store into an address
    void
    Store(Location loc, Value addr, Value val, std::optional<int> offset = {});

    /// Mangle constant name to use the symbol table and avoid duplication
    std::string mangleConstantName(Type ty, std::variant<int, double> val);

    /// Generate a constant value of a certain type
    Value Constant(Type ty, std::variant<int, double> val);

    /// Generate a zero initialized value of a certain type
    Value Zero(Type ty);

    /// Generate a constant string as an LLVM global constant
    Value ConstantString(StringRef str, StringRef name = "");

    /// Generate an arithmetic call (known operation or intrinsic)
    Value Arithmetic(Location loc, StringRef name, Value ops, Type retTy);

    /// Generate a return on the current basic block of the function
    void Return(Location loc, FuncOp& func, Value ret);
  };
}
