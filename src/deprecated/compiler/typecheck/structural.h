// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/instantiation.h"
#include "compiler/typecheck/constraint.h"
#include "compiler/visitor.h"

namespace verona::compiler
{
  void add_structural_constraints(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const EntityTypePtr& super);

  bool solve_has_method(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasMethodTypePtr& super);

  bool solve_has_applied_method(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasAppliedMethodTypePtr& super);

  bool solve_has_field(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasFieldTypePtr& super);
}
