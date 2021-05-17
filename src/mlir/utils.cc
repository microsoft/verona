// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "utils.h"

namespace mlir::verona
{
  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() &&
      bb->back().mightHaveTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Return true if the value has a pointer type.
  bool isPointer(mlir::Value val)
  {
    return val && val.getType().isa<PointerType>();
  }

  /// Return the element type if val is a pointer.
  mlir::Type getElementType(mlir::Value val)
  {
    assert(isPointer(val) && "Bad type");
    return val.getType().dyn_cast<PointerType>().getElementType();
  }

  /// Return true if the value has a pointer to a structure type.
  bool isStructPointer(mlir::Value val)
  {
    return isPointer(val) && getElementType(val).isa<StructType>();
  }

  /// Return the element type if val is a pointer.
  mlir::Type getFieldType(StructType type, int offset)
  {
    auto field = type.getBody().begin();
    std::advance(field, offset);
    return PointerType::get(*field);
  }
}
