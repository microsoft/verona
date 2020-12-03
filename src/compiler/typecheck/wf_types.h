// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::compiler
{
  struct Program;
  class Context;
  bool check_wf_types(Context& context, Program* program);
}
