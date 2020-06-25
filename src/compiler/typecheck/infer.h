// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/context.h"
#include "compiler/dataflow/liveness.h"
#include "compiler/ir/ir.h"
#include "compiler/typecheck/constraint.h"

namespace verona::compiler
{
  typedef std::unordered_map<Variable, TypePtr> TypeAssignment;

  template<typename T>
  using TypeArgumentsMap = std::unordered_map<TypeArgumentsId, T>;

  struct InferResults
  {
    Constraints constraints;
    std::unordered_map<const BasicBlock*, TypeAssignment> types;
    TypeArgumentsMap<InferableTypeSequence> type_arguments;

    void dump(Context& context, const Method& method);
  };

  std::unique_ptr<InferResults> infer(
    Context& context,
    const Program& program,
    const Method& method,
    const MethodIR& ir,
    const LivenessAnalysis& liveness);

  void dump_types(
    Context& context,
    const Method& method,
    std::string_view name,
    std::string_view title,
    const std::unordered_map<const BasicBlock*, TypeAssignment>& types);
}
