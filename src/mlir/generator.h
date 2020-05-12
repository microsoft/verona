// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR.h"
#include "symbol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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
    }

    // Read AST/MLIR into MLIR module, returns false on failure.
    void readAST(const ::ast::Ast& ast);
    void readMLIR(const std::string& filename);

    // Transform the opaque MLIR format into Verona dialect and LLVM IR.
    void emitMLIR(const llvm::StringRef filename = "", unsigned optLevel = 0);
    void emitLLVM(const llvm::StringRef filename = "", unsigned optLevel = 0);

    using Types = llvm::SmallVector<mlir::Type, 4>;
    using Values = llvm::SmallVector<mlir::Value, 4>;

  private:
    // MLIR module, builder and context.
    mlir::OwningModuleRef module;
    mlir::OpBuilder builder;
    mlir::MLIRContext context;

    // Symbol tables for variables, functions and classes.
    SymbolTableT symbolTable;
    FunctionTableT functionTable;
    TypeTableT typeTable;

    // Get location of an ast node
    mlir::Location getLocation(const ::ast::Ast& ast);

    // Parses a module, the global context.
    void parseModule(const ::ast::Ast& ast);

    // Parses a function, from a top-level (module) view.
    mlir::FuncOp parseProto(const ::ast::Ast& ast);
    mlir::FuncOp parseFunction(const ::ast::Ast& ast);

    // Recursive type parser, gathers all available information on the type
    // and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

    // Declares/Updates a variable.
    void declareVariable(llvm::StringRef name, mlir::Value val);
    void updateVariable(llvm::StringRef name, mlir::Value val);

    // Generic block/node parser, calls other parse functions to handle each
    // individual type. Block returns last value, for return.
    mlir::Value parseBlock(const ::ast::Ast& ast);
    mlir::Value parseNode(const ::ast::Ast& ast);
    mlir::Value parseValue(const ::ast::Ast& ast);

    // Specific parsers (there will be more).
    mlir::Value parseAssign(const ::ast::Ast& ast);
    mlir::Value parseCall(const ::ast::Ast& ast);

    // Wrappers for unary/binary operands
    mlir::Value genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);
  };
}
