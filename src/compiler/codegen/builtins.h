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
    static void generate(
      Context& context, Generator& gen, const CodegenItem<Method>& method);

  private:
    void generate_builtin(std::string_view entity, std::string_view method);

    void builtin_print();
    void builtin_binop(bytecode::BinaryOperator op);
    void builtin_create_sleeping_cown();
    void builtin_fulfill_sleeping_cown();
    void builtin_freeze();
    void builtin_trace_region();
  };
}
