// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  class RegionCheckerPass : public PassWrapper<RegionCheckerPass, FunctionPass>
  {
    void runOnFunction() override;
  };

  struct FactEvaluator;

#include "dialect/RegionChecker.h.inc"
}
