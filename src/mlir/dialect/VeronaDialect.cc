// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaDialect.h"

#include "VeronaOps.h"
#include "VeronaTypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::verona
{
  void VeronaDialect::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "dialect/VeronaOps.cpp.inc"
      >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/VeronaTypes.cpp.inc"
      >();

    // ClassType isn't defined by ODS yet.
    addTypes<ClassType>();

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
}
