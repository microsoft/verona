// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/capability_predicate.h"

#include "compiler/printing.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  PredicateSet::PredicateSet(CapabilityPredicate predicate)
  : values_(static_cast<underlying_type>(predicate))
  {}

  PredicateSet::PredicateSet(underlying_type values) : values_(values) {}

  PredicateSet PredicateSet::empty()
  {
    return PredicateSet(0);
  }

  PredicateSet PredicateSet::all()
  {
    return CapabilityPredicate::Readable | CapabilityPredicate::Writable |
      CapabilityPredicate::NonWritable | CapabilityPredicate::NonLinear |
      CapabilityPredicate::Sendable;
  }

  bool PredicateSet::contains(CapabilityPredicate predicate) const
  {
    return (values_ & static_cast<underlying_type>(predicate)) != 0;
  }

  PredicateSet PredicateSet::operator|(PredicateSet other) const
  {
    return PredicateSet(values_ | other.values_);
  }

  PredicateSet PredicateSet::operator&(PredicateSet other) const
  {
    return PredicateSet(values_ & other.values_);
  }

  PredicateSet operator|(CapabilityPredicate lhs, CapabilityPredicate rhs)
  {
    return PredicateSet(lhs) | PredicateSet(rhs);
  }

  struct CapabilityPredicateVisitor : public TypeVisitor<PredicateSet>
  {
    PredicateSet visit_base_type(const TypePtr& type) final
    {
      return PredicateSet::empty();
    }

    PredicateSet visit_capability(const CapabilityTypePtr& type) final
    {
      switch (type->kind)
      {
        case CapabilityKind::Isolated:
          if (std::holds_alternative<RegionNone>(type->region))
          {
            return CapabilityPredicate::Readable |
              CapabilityPredicate::Writable | CapabilityPredicate::Sendable;
          }
          else
          {
            return CapabilityPredicate::Readable |
              CapabilityPredicate::Writable | CapabilityPredicate::NonLinear;
          }

        case CapabilityKind::Mutable:
          return CapabilityPredicate::Readable | CapabilityPredicate::Writable |
            CapabilityPredicate::NonLinear;

        case CapabilityKind::Subregion:
          return CapabilityPredicate::Readable | CapabilityPredicate::Writable |
            CapabilityPredicate::NonLinear;

        case CapabilityKind::Immutable:
          return CapabilityPredicate::Readable |
            CapabilityPredicate::NonWritable | CapabilityPredicate::Sendable |
            CapabilityPredicate::NonLinear;

          EXHAUSTIVE_SWITCH;
      }
    }

    /**
     * Visits each member of elements and merges the results using the given
     * BinaryOp.
     */
    template<typename BinaryOp>
    PredicateSet combine_elements(
      const TypeSet& elements, PredicateSet initial, BinaryOp combine)
    {
      // std::transform_reduce isn't available yet in libstdc++7 :(
      PredicateSet result = initial;
      for (const auto& elem : elements)
      {
        result = combine(result, visit_type(elem));
      }
      return result;
    }

    PredicateSet visit_static_type(const StaticTypePtr& type) final
    {
      return CapabilityPredicate::Sendable | CapabilityPredicate::NonLinear;
    }
    PredicateSet visit_string_type(const StringTypePtr& type) final
    {
      return CapabilityPredicate::Sendable | CapabilityPredicate::NonLinear;
    }

    PredicateSet visit_union(const UnionTypePtr& type) final
    {
      return combine_elements(
        type->elements, PredicateSet::all(), std::bit_and<PredicateSet>());
    }

    PredicateSet visit_intersection(const IntersectionTypePtr& type) final
    {
      return combine_elements(
        type->elements, PredicateSet::empty(), std::bit_or<PredicateSet>());
    }
    PredicateSet
    visit_variable_renaming_type(const VariableRenamingTypePtr& type) final
    {
      return visit_type(type->type);
    }
    PredicateSet
    visit_path_compression_type(const PathCompressionTypePtr& type) final
    {
      return visit_type(type->type);
    }
    PredicateSet visit_fixpoint_type(const FixpointTypePtr& type) final
    {
      return visit_type(type->inner);
    }
    PredicateSet
    visit_fixpoint_variable_type(const FixpointVariableTypePtr& type) final
    {
      return PredicateSet::all();
    }
  };

  PredicateSet predicates_for_type(TypePtr type)
  {
    // This is pretty cheap, but it's likely we'll be doing it a lot in the
    // future. We might want to cache results.
    return CapabilityPredicateVisitor().visit_type(type);
  }

  std::ostream& operator<<(std::ostream& out, CapabilityPredicate predicate)
  {
    switch (predicate)
    {
      case CapabilityPredicate::Readable:
        return out << "readable";
      case CapabilityPredicate::Writable:
        return out << "writable";
      case CapabilityPredicate::NonWritable:
        return out << "non-writable";
      case CapabilityPredicate::Sendable:
        return out << "sendable";
      case CapabilityPredicate::NonLinear:
        return out << "non-linear";

        EXHAUSTIVE_SWITCH;
    }
  }
}
