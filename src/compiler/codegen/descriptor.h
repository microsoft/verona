// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/codegen/reachability.h"
#include "compiler/codegen/selector.h"

namespace verona::compiler
{
  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& main);
};
