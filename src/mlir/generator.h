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
   * MLIR Generator.
   *
   * There are two entry points: AST and MLIR, and two exit points: MLIR
   * and LLVM IR. The MLIR -> MLIR path is for self-validation and future
   * optimisation passes (mainly testing).
   *
   * The LLVM IR can be dumped or used in the next step of the compilation,
   * lowering to machine code by the LLVM library.
   *
   * For now, the error handling is crude and needs proper consideration,
   * especially aggregating all errors and context before sending it back to
   * the public API callers.
   */
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

    /// Read AST into MLIR module.
    llvm::Error readAST(const ::ast::Ast& ast);
    /// Read MLIR into MLIR module.
    llvm::Error readMLIR(const std::string& filename);

    /// Transform the opaque MLIR format into Verona dialect.
    llvm::Error emitMLIR(const llvm::StringRef filename, unsigned optLevel = 0);
    /// Transform the Verona dialect into LLVM IR.
    llvm::Error emitLLVM(const llvm::StringRef filename, unsigned optLevel = 0);

    using Types = llvm::SmallVector<mlir::Type, 4>;

  private:
    /// MLIR module.
    mlir::OwningModuleRef module;
    /// MLIR builder.
    mlir::OpBuilder builder;
    /// MLIR context.
    mlir::MLIRContext context;
    /// Current MLIR function.
    mlir::FuncOp currentFunc;

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
    llvm::Expected<mlir::Value> parseNode(const ::ast::Ast& ast);

    /// Parses a block (multiple statements), return last value.
    llvm::Expected<mlir::Value> parseBlock(const ::ast::Ast& ast);
    /// Parses a value (constants, variables).
    llvm::Expected<mlir::Value> parseValue(const ::ast::Ast& ast);
    /// Parses an assign statement.
    llvm::Expected<mlir::Value> parseAssign(const ::ast::Ast& ast);
    /// Parses function calls and native operations.
    llvm::Expected<mlir::Value> parseCall(const ::ast::Ast& ast);
    /// Parses an if/else block.
    llvm::Expected<mlir::Value> parseCondition(const ::ast::Ast& ast);
    /// Parses a 'while' loop block.
    llvm::Expected<mlir::Value> parseWhileLoop(const ::ast::Ast& ast);
    /// Parses a 'continue' statement.
    llvm::Expected<mlir::Value> parseContinue(const ::ast::Ast& ast);
    /// Parses a 'break' statement.
    llvm::Expected<mlir::Value> parseBreak(const ::ast::Ast& ast);
    /// Parses a 'return' statement.
    llvm::Expected<mlir::Value> parseReturn(const ::ast::Ast& ast);

    // =============================================================== Temporary

    /// Wrapper for opaque operators before we use actual Verona dialect.
    llvm::Expected<mlir::Value> genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);
    /// Wrappers for opaque types before we use actual Verona dialect.
    mlir::OpaqueType
    genOpaqueType(llvm::StringRef name, mlir::MLIRContext& context);
  };
}
