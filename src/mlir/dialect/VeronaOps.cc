// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaOps.h"

#include "Typechecker.h"
#include "VeronaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

static ParseResult parseClassOp(OpAsmParser& parser, OperationState& state)
{
  Region* body = state.addRegion();

  StringAttr nameAttr;
  if (parser.parseSymbolName(
        nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  verona::ClassOp::ensureTerminator(*body, parser.getBuilder(), state.location);

  return success();
}

static void print(OpAsmPrinter& printer, verona::ClassOp op)
{
  printer << verona::ClassOp::getOperationName() << ' ';
  printer.printSymbolName(op.sym_name());
  printer.printOptionalAttrDict(
    op.getAttrs(), /*elidedAttrs=*/{SymbolTable::getSymbolAttrName()});
  printer.printRegion(
    op.body(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

static ParseResult parseWhileOp(OpAsmParser& parser, OperationState& state)
{
  Region* body = state.addRegion();

  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

static void print(OpAsmPrinter& printer, verona::WhileOp op)
{
  printer << verona::WhileOp::getOperationName() << ' ';
  printer.printOptionalAttrDict(
    op.getAttrs(), /*elidedAttrs=*/{SymbolTable::getSymbolAttrName()});
  printer.printRegion(
    op.body(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

/**
 * AllocateRegionOp and AllocateObjectOp share similar features, which we verify
 * here:
 * - The class referenced by the operation must exist.
 * - There must be as many field names as there are operands.
 * - The set of field names must match those specified in the class definition
 *   (TODO: not implemented yet)
 */
template<typename Op>
static LogicalResult verifyAllocationOp(Op op)
{
  auto className = op.class_name();
  auto classOp =
    SymbolTable::lookupNearestSymbolFrom<verona::ClassOp>(op, className);
  if (!classOp)
  {
    return op.emitOpError("class '")
      << className << "' not found in nearest symbol table";
  }

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

static LogicalResult verify(verona::ClassOp classOp)
{
  Block* body = classOp.getBody();

  // Verify that classes contain only fields, and that there are no duplicates.
  // TODO: once we add methods, verify that there are no duplicates either.
  llvm::StringSet<> fields;
  for (Operation& op : *body)
  {
    if (verona::FieldOp field_op = dyn_cast<verona::FieldOp>(op))
    {
      auto it = fields.insert(field_op.name());

      if (!it.second)
      {
        return op.emitError().append(
          "redefinition of field named '", field_op.name(), "'");
      }
    }
    else if (!isa<verona::ClassEndOp>(op))
    {
      return op.emitOpError("cannot be contained in class");
    }
  }

  return success();
}

static LogicalResult verify(verona::WhileOp whileOp)
{
  auto& body = whileOp.body();

  // While blocks must have a condition exit with a LoopExit operation.
  unsigned numExits = 0;
  for (Block& b : body.getBlocks())
  {
    // Until we find a loop_exit, keep searching
    for (Operation& op : b)
    {
      if (verona::LoopExitOp exit_op = dyn_cast<verona::LoopExitOp>(op))
      {
        numExits++;
        break;
      }
    }
  }

  // Can only have one loop_exit
  if (numExits == 1)
    return success();
  return failure();
}

namespace mlir::verona
{
  LogicalResult CopyOp::typecheck()
  {
    return checkSubtype(getLoc(), input().getType(), output().getType());
  }

#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"

} // namespace mlir::verona
