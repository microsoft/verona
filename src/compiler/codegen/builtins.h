// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "compiler/codegen/function.h"

namespace verona::compiler
{
  /**
   * Generate code for builtin functions.
   *
   * These are all static methods of the verona `Builtin` class.
   */
  class BuiltinGenerator : public FunctionGenerator
  {
    using FunctionGenerator::FunctionGenerator;

  public:
    void generate_builtin(std::string_view name);

  private:
    void builtin_print();
    void builtin_create_sleeping_cown();
    void builtin_trace_region();
    void builtin_fulfill_sleeping_cown();
    void builtin_binop(bytecode::BinaryOperator op);
  };
}
