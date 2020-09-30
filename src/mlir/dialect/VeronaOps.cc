// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaOps.h"

#include "Typechecker.h"
#include "VeronaDialect.h"
#include "VeronaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

/**
 * AllocateRegionOp and AllocateObjectOp share similar features, which we verify
 * here:
 * - There must be as many field names as there are operands.
 * - The set of field names must match those specified in the class type
 *   (TODO: not implemented yet)
 */
template<typename Op>
static LogicalResult verifyAllocationOp(Op op)
{
  if (op.field_names().size() != op.fields().size())
  {
    return op.emitError("The number of operands (")
      << op.fields().size() << ") for '" << op.getOperationName()
      << "' op does not match the number of field names ("
      << op.field_names().size() << ")";
  }

  return success();
}

static LogicalResult verify(verona::AllocateRegionOp op)
{
  return verifyAllocationOp(op);
}

static LogicalResult verify(verona::AllocateObjectOp op)
{
  return verifyAllocationOp(op);
}

namespace mlir::verona
{
  Type FieldReadOp::getFieldType()
  {
    return lookupFieldType(origin().getType(), field()).first;
  }

  std::pair<Type, Type> FieldWriteOp::getFieldType()
  {
    return lookupFieldType(origin().getType(), field());
  }

#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"

} // namespace mlir::verona
