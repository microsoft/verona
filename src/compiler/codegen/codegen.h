// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "compiler/analysis.h"

namespace verona::compiler
{
  /**
   * Generate bytecode for the program.
   *
   * Any errors during codegen will be reported in the context.
   */
  std::vector<uint8_t> codegen(
    Context& context, const Program& program, const AnalysisResults& analysis);
}
