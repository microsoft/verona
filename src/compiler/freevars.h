// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/visitor.h"

namespace verona::compiler
{
  struct FreeVariables
  {
    explicit FreeVariables() = default;
    explicit FreeVariables(InferTypePtr ty)
    {
      inference.insert(ty);
      has_inferable_regions = true;
    };
    explicit FreeVariables(UnboundedTypeSequence seq)
    {
      sequences.insert(seq);
    };

    static FreeVariables fixpoint_variable(size_t depth)
    {
      FreeVariables result;
      result.maximum_fixpoint_variable = depth + 1;
      return result;
    }

    static FreeVariables region(const Region& region)
    {
      FreeVariables result;
      if (auto rv = std::get_if<RegionVariable>(&region))
        result.region_variables.insert(rv->variable);
      return result;
    }

    static FreeVariables indirect_type()
    {
      FreeVariables result;
      result.has_indirect_type = true;
      return result;
    }

    std::set<InferTypePtr> inference;
    std::set<Variable> region_variables;
    std::set<UnboundedTypeSequence> sequences;
    size_t maximum_fixpoint_variable = 0;

    bool has_inferable_regions = false;
    bool has_indirect_type = false;

    FreeVariables& merge(const FreeVariables& other)
    {
      inference.insert(other.inference.begin(), other.inference.end());
      region_variables.insert(
        other.region_variables.begin(), other.region_variables.end());
      has_inferable_regions |= other.has_inferable_regions;

      maximum_fixpoint_variable =
        std::max(other.maximum_fixpoint_variable, maximum_fixpoint_variable);

      return *this;
    }

    bool contains(const InferTypePtr& ty) const
    {
      return inference.find(ty) != inference.end();
    }

    bool contains_region(Variable region) const
    {
      return region_variables.find(region) != region_variables.end();
    }

    bool is_fixpoint_closed() const
    {
      return maximum_fixpoint_variable == 0;
    }

    FreeVariables without_inferrable_regions() const
    {
      FreeVariables result = *this;
      result.has_inferable_regions = false;
      return result;
    }
  };

  /**
   * Compute the free inference variables of a type.
   */
  class FreeVariablesVisitor : private TypeVisitor<FreeVariables>
  {
  public:
    const FreeVariables& free_variables(const TypePtr& type)
    {
      auto it = cache_.insert({type, FreeVariables()});
      if (it.second)
      {
        it.first->second = visit_type(type);
      }
      return it.first->second;
    }

  private:
    FreeVariables visit_infer(const InferTypePtr& ty) final
    {
      return FreeVariables(ty);
    }

    FreeVariables visit_type_parameter(const TypeParameterPtr& ty) final
    {
      return FreeVariables();
    }

    FreeVariables visit_capability(const CapabilityTypePtr& ty) final
    {
      return FreeVariables::region(ty->region);
    }

    FreeVariables visit_apply_region(const ApplyRegionTypePtr& ty) final
    {
      return free_variables(ty->type).without_inferrable_regions().merge(
        FreeVariables::region(ty->region));
    }

    FreeVariables visit_entity_type(const EntityTypePtr& ty) final
    {
      return combine(ty->arguments).without_inferrable_regions();
    }

    FreeVariables visit_static_type(const StaticTypePtr& ty) final
    {
      return combine(ty->arguments).without_inferrable_regions();
      ;
    }

    FreeVariables visit_intersection(const IntersectionTypePtr& ty) final
    {
      return combine(ty->elements);
    }

    FreeVariables visit_union(const UnionTypePtr& ty) final
    {
      return combine(ty->elements);
    }

    FreeVariables visit_range_type(const RangeTypePtr& ty) final
    {
      return combine(ty->lower, ty->upper);
    }

    FreeVariables visit_viewpoint_type(const ViewpointTypePtr& ty) final
    {
      return free_variables(ty->right);
    }

    FreeVariables visit_has_field_type(const HasFieldTypePtr& ty) final
    {
      return combine(ty->view, ty->read_type, ty->write_type)
        .without_inferrable_regions();
    }

    FreeVariables
    visit_delayed_field_view_type(const DelayedFieldViewTypePtr& ty) final
    {
      return free_variables(ty->type).without_inferrable_regions();
    }

    FreeVariables visit_has_method_type(const HasMethodTypePtr& ty) final
    {
      return combine(
               ty->signature.receiver,
               ty->signature.arguments,
               ty->signature.return_type)
        .without_inferrable_regions();
    }

    FreeVariables
    visit_has_applied_method_type(const HasAppliedMethodTypePtr& ty) final
    {
      return combine(
               ty->application,
               ty->signature.receiver,
               ty->signature.arguments,
               ty->signature.return_type)
        .without_inferrable_regions();
    }

    FreeVariables visit_is_entity_type(const IsEntityTypePtr& ty) final
    {
      return FreeVariables();
    }

    FreeVariables visit_unit_type(const UnitTypePtr& ty) final
    {
      return FreeVariables();
    }

    FreeVariables visit_string_type(const StringTypePtr& ty) final
    {
      return FreeVariables();
    }

    FreeVariables visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      FreeVariables result = free_variables(ty->inner);
      if (result.maximum_fixpoint_variable > 0)
        result.maximum_fixpoint_variable -= 1;
      return result;
    }

    FreeVariables
    visit_fixpoint_variable_type(const FixpointVariableTypePtr& ty) final
    {
      return FreeVariables::fixpoint_variable(ty->depth);
    }

    FreeVariables visit_entity_of_type(const EntityOfTypePtr& ty) final
    {
      return free_variables(ty->inner).without_inferrable_regions();
    }

    FreeVariables
    visit_variable_renaming_type(const VariableRenamingTypePtr& ty) final
    {
      return free_variables(ty->type);
    }

    FreeVariables
    visit_path_compression_type(const PathCompressionTypePtr& ty) final
    {
      FreeVariables freevars = combine(ty->compression, ty->type);
      for (auto [dead_variable, _] : ty->compression)
      {
        freevars.region_variables.erase(dead_variable);
      }
      return freevars;
    }

    FreeVariables visit_indirect_type(const IndirectTypePtr& ty) final
    {
      return FreeVariables::indirect_type();
    }

    FreeVariables visit_not_child_of_type(const NotChildOfTypePtr& ty) final
    {
      return FreeVariables();
    }

    FreeVariables visit_inferable_type_sequence(const BoundedTypeSequence& seq)
    {
      return combine(seq.types);
    }

    FreeVariables
    visit_inferable_type_sequence(const UnboundedTypeSequence& seq)
    {
      return FreeVariables(seq);
    }

    template<typename... Args>
    FreeVariables combine(const Args&... args)
    {
      FreeVariables result;
      (combine_into(&result, args), ...);
      return result;
    }
    void combine_into(FreeVariables* dst, const FreeVariables& src)
    {
      dst->merge(src);
    }
    void combine_into(FreeVariables* dst, const InferableTypeSequence& src)
    {
      combine_into(
        dst,
        std::visit(
          [&](const auto& inner) {
            return visit_inferable_type_sequence(inner);
          },
          src));
    }
    void combine_into(FreeVariables* dst, const TypePtr& ty)
    {
      combine_into(dst, free_variables(ty));
    }
    void combine_into(FreeVariables* dst, const TypeSet& types)
    {
      for (const auto& ty : types)
      {
        combine_into(dst, ty);
      }
    }
    void combine_into(FreeVariables* dst, const TypeList& types)
    {
      for (const auto& ty : types)
      {
        combine_into(dst, ty);
      }
    }
    void combine_into(FreeVariables* dst, const PathCompressionMap& types)
    {
      for (const auto& [_, ty] : types)
      {
        combine_into(dst, ty);
      }
    }

    std::unordered_map<TypePtr, FreeVariables> cache_;
  };
}
