// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
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
    void builtin_freeze();
    void builtin_trace_region();
    void builtin_binop(bytecode::BinaryOperator op);
    void builtin_cown_create();
    void builtin_cown_create_sleeping();
    void builtin_cown_fulfill_sleeping();
  };
}
