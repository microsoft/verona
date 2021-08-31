// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/typecheck/infer.h"

namespace verona::compiler
{
  struct TypecheckResults
  {
    std::unordered_map<const BasicBlock*, TypeAssignment> types;
    TypeArgumentsMap<TypeList> type_arguments;
  };

  std::unique_ptr<TypecheckResults> typecheck(
    Context& context, const Method* method, const InferResults& inference);
}
