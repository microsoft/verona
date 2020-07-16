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
  // MLIR Generator.
  //
  // There are two entry points: AST and MLIR, and two exit points: MLIR
  // and LLVM IR. The MLIR -> MLIR path is for self-validation and future
  // optimisation passes (mainly testing).
  //
  // The LLVM IR can be dumped or used in the next step of the compilation,
  // lowering to machine code by the LLVM library.
  //
  // For now, the error handling is crude and needs proper consideration,
  // especially aggregating all errors and context before sending it back to
  // the public API callers.
  struct Generator
  {
    Generator() : builder(&context)
    {
      // Opaque operations and types can only exist if we allow
      // unregistered dialects to co-exist. Full conversions later will
      // make sure we end up with onlt Verona dialect, then Standard
      // dialects, then LLVM dialect, before converting to LLVM IR.
      context.allowUnregisteredDialects();

      // Initialise known opaque types, for comparison.
      // TODO: Use Verona dialect types directly and isA<>.
      allocaTy = genOpaqueType("alloca", context);
      unkTy = genOpaqueType("unk", context);
      noneTy = genOpaqueType("none", context);
      boolTy = builder.getI1Type();
    }

    // Read AST/MLIR into MLIR module, returns false on failure.
    llvm::Error readAST(const ::ast::Ast& ast);
    llvm::Error readMLIR(const std::string& filename);

    // Transform the opaque MLIR format into Verona dialect and LLVM IR.
    llvm::Error emitMLIR(const llvm::StringRef filename, unsigned optLevel = 0);
    llvm::Error emitLLVM(const llvm::StringRef filename, unsigned optLevel = 0);

    using Types = llvm::SmallVector<mlir::Type, 4>;
    using Values = llvm::SmallVector<mlir::Value, 4>;

  private:
    // MLIR module, builder and context.
    mlir::OwningModuleRef module;
    mlir::OpBuilder builder;
    mlir::MLIRContext context;
    mlir::FuncOp currentFunc;

    // Symbol tables for variables, functions and classes.
    SymbolTableT symbolTable;
    FunctionTableT functionTable;
    TypeTableT typeTable;

    // Helper for types, before we start using actual Verona types
    mlir::Type allocaTy;
    mlir::Type unkTy;
    mlir::Type noneTy;
    mlir::Type boolTy;

    // Get location of an ast node
    mlir::Location getLocation(const ::ast::Ast& ast);

    // Parses a module, the global context.
    llvm::Error parseModule(const ::ast::Ast& ast);

    // Parses a function, from a top-level (module) view.
    llvm::Expected<mlir::FuncOp> parseProto(const ::ast::Ast& ast);
    llvm::Expected<mlir::FuncOp> parseFunction(const ::ast::Ast& ast);

    // Recursive type parser, gathers all available information on the type
    // and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

    // Declares/Updates a variable.
    void declareVariable(llvm::StringRef name, mlir::Value val);
    void updateVariable(llvm::StringRef name, mlir::Value val);

    // Generic block/node parser, calls other parse functions to handle each
    // individual type. Block returns last value, for return.
    llvm::Expected<mlir::Value> parseBlock(const ::ast::Ast& ast);
    llvm::Expected<mlir::Value> parseNode(const ::ast::Ast& ast);
    llvm::Expected<mlir::Value> parseValue(const ::ast::Ast& ast);

    // Specific parsers (there will be more).
    llvm::Expected<mlir::Value> parseAssign(const ::ast::Ast& ast);
    llvm::Expected<mlir::Value> parseCall(const ::ast::Ast& ast);
    llvm::Expected<mlir::Value> parseCondition(const ::ast::Ast& ast);

    // Wrappers for opaque operators/types before we use actual Verona dialect
    llvm::Expected<mlir::Value> genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);
    mlir::OpaqueType
    genOpaqueType(llvm::StringRef name, mlir::MLIRContext& context);
  };
}
