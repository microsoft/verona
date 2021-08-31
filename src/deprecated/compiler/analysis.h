// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/liveness.h"
#include "compiler/ir/builder.h"
#include "compiler/regionck/region_graph.h"
#include "compiler/typecheck/typecheck.h"

namespace verona::compiler
{
  struct FnAnalysis
  {
    std::unique_ptr<MethodIR> ir;
    std::unique_ptr<InferResults> inference;
    std::unique_ptr<TypecheckResults> typecheck;
    std::unique_ptr<LivenessAnalysis> liveness;
    std::unique_ptr<RegionGraphs> region_graphs;
  };

  struct AnalysisResults
  {
    std::unordered_map<const Method*, FnAnalysis> functions;
    bool ok;
  };

  std::unique_ptr<AnalysisResults> analyse(Context& context, Program* program);

  void dump_ast(Context& context, Program* program, const std::string& name);
}
