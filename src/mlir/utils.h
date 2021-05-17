// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "parser/ast.h"

namespace mlir::verona
{
  /// LLVM aliases
  using StructType = mlir::LLVM::LLVMStructType;
  using PointerType = mlir::LLVM::LLVMPointerType;

  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb);

  /// Return true if the value has a pointer type.
  bool isPointer(mlir::Value val);

  /// Return the element type if val is a pointer.
  mlir::Type getElementType(mlir::Value val);

  /// Return true if the value has a pointer to a structure type.
  bool isStructPointer(mlir::Value val);

  mlir::Type getFieldType(StructType type, int offset);

  /// Get node as a shared pointer of a sub-type
  template<class T>
  ::verona::parser::Node<T> nodeAs(::verona::parser::Ast from)
  {
    return std::make_shared<T>(from->as<T>());
  }
}
