// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "compiler/codegen/reachability.h"
#include "compiler/codegen/selector.h"

namespace verona::compiler
{
  void emit_descriptor(
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& entity,
    const EntityReachability& reachability);
};
