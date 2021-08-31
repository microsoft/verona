// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/fixpoint.h"

namespace verona::compiler
{
  /**
   * This mapper expands a set of mutually recursive definitions into fully
   * expanded types, using the fixpoint operator to encode the recursion.
   *
   * TODO: this is currently specific to eliminating indirect types, but it
   * could be generalised to handle any kind of mutually recursive definitions,
   * eg. for expanding type aliases.
   */
  struct SimplifyIndirectTypes : public RecursiveTypeMapper
  {
    SimplifyIndirectTypes(
      Context& context,
      const std::unordered_map<const BasicBlock*, TypeAssignment>& types)
    : RecursiveTypeMapper(context), types_(types)
    {}

    TypePtr expand(const BasicBlock* bb, Variable variable)
    {
      auto [it, inserted] =
        current_variables_.insert({{bb, variable}, {depth_, false}});
      if (inserted)
      {
        TypePtr replacement = types_.at(bb).at(variable);

        // TODO: this currently maps over the types up three times, once for
        // shift up, once for apply(), and potentially once for shift down.
        //
        // Shift up and apply() can probably be merged in a single pass. I don't
        // know if we can easily fold the shift down into it as well.
        depth_++;
        TypePtr result = apply(shift_fixpoint(context(), replacement, 1));
        depth_--;

        if (it->second.fixpoint_needed)
          result = context().mk_fixpoint(result);
        else
          result = shift_fixpoint(context(), result, -1);

        current_variables_.erase(it);
        return result;
      }
      else
      {
        it->second.fixpoint_needed = true;
        return context().mk_fixpoint_variable(depth_ - it->second.depth - 1);
      }
    }

  private:
    TypePtr visit_indirect_type(const IndirectTypePtr& ty)
    {
      return expand(ty->block, ty->variable);
    }

    TypePtr visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      depth_++;
      TypePtr result = RecursiveTypeMapper::visit_fixpoint_type(ty);
      depth_--;
      return result;
    }

    const std::unordered_map<const BasicBlock*, TypeAssignment>& types_;

    struct VariableState
    {
      size_t depth;
      bool fixpoint_needed;
    };
    std::map<std::pair<const BasicBlock*, Variable>, VariableState>
      current_variables_;

    size_t depth_ = 0;
  };
}
