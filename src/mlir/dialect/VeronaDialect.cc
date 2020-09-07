// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaDialect.h"

#include "VeronaOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::verona;

//===----------------------------------------------------------------------===//
// Verona dialect.
//===----------------------------------------------------------------------===//

void VeronaDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "dialect/VeronaOps.cpp.inc"
    >();

  addTypes<
    MeetType,
    JoinType,
    IntegerType,
    CapabilityType,
    ClassType,
    FloatType,
    BoolType,
    ViewpointType>();

  allowUnknownOperations();
  allowUnknownTypes();
}

Type VeronaDialect::parseType(DialectAsmParser& parser) const
{
  return parseVeronaType(parser);
}

void VeronaDialect::printType(Type type, DialectAsmPrinter& os) const
{
  return printVeronaType(type, os);
}
