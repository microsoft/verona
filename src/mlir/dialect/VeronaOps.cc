// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaOps.h"

#include "VeronaDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::verona
{
#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"

} // namespace mlir::verona
