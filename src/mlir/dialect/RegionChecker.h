// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  class PrintRegionAnalysisPass
  : public PassWrapper<PrintRegionAnalysisPass, OperationPass<ModuleOp>>
  {
    void runOnOperation() override;
  };

  struct StableFacts;
  struct RegionAnalysis
  {
    RegionAnalysis(FuncOp operation);
    void print(llvm::raw_ostream& os);

    RegionAnalysis(const RegionAnalysis& other) = delete;
    RegionAnalysis& operator=(const RegionAnalysis& other) = delete;

  private:
    DenseMap<Block*, StableFacts> facts;
    FuncOp operation;
  };

  struct FactEvaluator;

#include "dialect/RegionChecker.h.inc"
}
