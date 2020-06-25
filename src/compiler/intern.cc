// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/intern.h"

#include "compiler/format.h"
#include "compiler/ir/print.h"
#include "compiler/printing.h"
#include "ds/helpers.h"

#include <fmt/ostream.h>
#include <iostream>
#include <typeindex>

using std::placeholders::_1;

namespace verona::compiler
{
  CapabilityTypePtr
  TypeInterner::mk_capability(CapabilityKind capability, Region region)
  {
    return intern(CapabilityType(capability, region));
  }

  CapabilityTypePtr TypeInterner::mk_mutable(Region region)
  {
    return mk_capability(CapabilityKind::Mutable, region);
  }

  CapabilityTypePtr TypeInterner::mk_subregion(Region region)
  {
    return mk_capability(CapabilityKind::Subregion, region);
  }

  CapabilityTypePtr TypeInterner::mk_isolated(Region region)
  {
    return mk_capability(CapabilityKind::Isolated, region);
  }

  CapabilityTypePtr TypeInterner::mk_immutable()
  {
    return mk_capability(CapabilityKind::Immutable, RegionNone());
  }

  EntityTypePtr
  TypeInterner::mk_entity_type(const Entity* definition, TypeList arguments)
  {
    assert(definition != nullptr);
    assert(is_interned(arguments));
    return intern(EntityType(definition, arguments));
  }

  StaticTypePtr
  TypeInterner::mk_static_type(const Entity* definition, TypeList arguments)
  {
    assert(definition != nullptr);
    assert(is_interned(arguments));
    return intern(StaticType(definition, arguments));
  }

  TypeParameterPtr TypeInterner::mk_type_parameter(
    const TypeParameterDef* definition, TypeParameter::Expanded expanded)
  {
    assert(definition != nullptr);
    return intern(TypeParameter(definition, expanded));
  }

  template<typename T>
  void TypeInterner::flatten_connective(
    TypeSet elements, std::set<typename T::DualPtr>* duals, TypeSet* others)
  {
    auto includes = [](const auto& left, const auto& right) {
      return std::includes(
        left->elements.begin(),
        left->elements.end(),
        right->elements.begin(),
        right->elements.end());
    };
    auto contains = [](const auto& container, const auto& elem) {
      return container->elements.find(elem) != container->elements.end();
    };
    auto none_of = [](const auto& seq, const auto& fn) {
      return std::none_of(seq->begin(), seq->end(), fn);
    };

    for (const TypePtr& child : elements)
    {
      if (auto nested = child->dyncast<T>())
      {
        flatten_connective<T>(nested->elements, duals, others);
      }
      else if (auto dual = child->dyncast<typename T::Dual>())
      {
        if (
          none_of(duals, std::bind(includes, dual, _1)) &&
          none_of(others, std::bind(contains, dual, _1)))
        {
          for (auto it = duals->begin(); it != duals->end();)
          {
            if (includes(*it, dual))
              it = duals->erase(it);
            else
              it++;
          }
          duals->insert(dual);
        }
      }
      else
      {
        bool inserted = others->insert(child).second;

        if (inserted)
        {
          for (auto it = duals->begin(); it != duals->end();)
          {
            if (contains(*it, child))
              it = duals->erase(it);
            else
              it++;
          }
        }
      }
    }
  }

  template<typename T>
  TypeSet TypeInterner::flatten_connective(TypeSet elements)
  {
    std::set<typename T::DualPtr> duals;
    TypeSet others;
    flatten_connective<T>(elements, &duals, &others);
    others.insert(duals.begin(), duals.end());

    return others;
  }

  template<typename T>
  TypePtr TypeInterner::mk_connective(TypeSet elements)
  {
    assert(is_interned(elements));
    elements = flatten_connective<T>(elements);

    /**
     * When building an intersection, if two different class entities are
     * present we can replace it with bottom.
     *
     * This is used in pattern matching, since intersecting the pattern with the
     * input will likely eliminate some alternatives of a union.
     *
     * TODO: do the same thing for incompatible capabilities.
     * TODO: This currently ignores type arguments. For what T and U is
     * `Array[T] & Array[U] = bottom`?
     */
    if constexpr (std::is_same_v<T, IntersectionType>)
    {
      const Entity* class_definition = nullptr;
      for (const auto& elem : elements)
      {
        if (auto entity = elem->dyncast<EntityType>();
            entity && entity->definition->kind->value() == Entity::Class)
        {
          if (
            class_definition != nullptr &&
            entity->definition != class_definition)
          {
            return mk_bottom();
          }
          else
          {
            class_definition = entity->definition;
          }
        }
      }
    }

    if (elements.size() > 0)
    {
      TypePtr absorbing_element = mk_connective<typename T::Dual>({});
      if (elements.find(absorbing_element) != elements.end())
        return absorbing_element;
    }

    switch (elements.size())
    {
      case 1:
        return *elements.begin();
      default:
        return intern(T(elements));
    }
  }

  template TypePtr TypeInterner::mk_connective<UnionType>(TypeSet);
  template TypePtr TypeInterner::mk_connective<IntersectionType>(TypeSet);

  TypePtr TypeInterner::mk_union(TypeSet elements)
  {
    return mk_connective<UnionType>(elements);
  }

  TypePtr TypeInterner::mk_union(TypePtr first, TypePtr second)
  {
    return mk_union({first, second});
  }

  TypePtr TypeInterner::mk_intersection(TypeSet elements)
  {
    return mk_connective<IntersectionType>(elements);
  }

  TypePtr TypeInterner::mk_intersection(TypePtr first, TypePtr second)
  {
    return mk_intersection({first, second});
  }

  // mk_apply_region, mk_unapply_region and mk_viewpoint are quite a mess
  // We should look into a better "rewrite framework" to use
  //
  // Also should we do these normalizations here when interning, or later
  // (eg. when polarizing). Doing it here means we have no way of representing
  // the non-normalized form.
  TypePtr TypeInterner::mk_apply_region(
    ApplyRegionType::Mode mode, Region region, TypePtr type)
  {
    assert(is_interned(type));

    if (auto isect = type->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_apply_region(mode, region, elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = type->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_apply_region(mode, region, elem));
      }
      return mk_union(result);
    }

    if (auto fixpoint = type->dyncast<FixpointType>())
    {
      return mk_fixpoint(mk_apply_region(mode, region, fixpoint->inner));
    }
    else if (type->dyncast<FixpointVariableType>())
    {
      return type;
    }

    if (auto range = type->dyncast<RangeType>())
    {
      return mk_range(
        mk_apply_region(mode, region, range->lower),
        mk_apply_region(mode, region, range->upper));
    }

    if (auto viewpoint = type->dyncast<ViewpointType>())
    {
      return mk_viewpoint(
        viewpoint->capability,
        viewpoint->variables,
        mk_apply_region(mode, region, viewpoint->right));
    }

    if (auto capability = type->dyncast<CapabilityType>())
    {
      switch (capability->kind)
      {
        case CapabilityKind::Mutable:
          assert(std::holds_alternative<RegionHole>(capability->region));
          switch (mode)
          {
            case ApplyRegionType::Mode::Adapt:
            case ApplyRegionType::Mode::Extract:
              return mk_mutable(region);

            case ApplyRegionType::Mode::Under:
              return mk_subregion(region);

              EXHAUSTIVE_SWITCH;
          }
          break;

        case CapabilityKind::Subregion:
          assert(std::holds_alternative<RegionHole>(capability->region));
          return mk_subregion(region);

        case CapabilityKind::Isolated:
          assert(std::holds_alternative<RegionHole>(capability->region));

          switch (mode)
          {
            case ApplyRegionType::Mode::Adapt:
            case ApplyRegionType::Mode::Under:
              return mk_isolated(region);
            case ApplyRegionType::Mode::Extract:
              return mk_isolated(RegionNone());

              EXHAUSTIVE_SWITCH;
          }

        case CapabilityKind::Immutable:
          assert(std::holds_alternative<RegionNone>(capability->region));
          return mk_immutable();

          EXHAUSTIVE_SWITCH
      }
    }

    if (
      type->dyncast<DelayedFieldViewType>() || type->dyncast<EntityOfType>() ||
      type->dyncast<EntityType>() || type->dyncast<HasAppliedMethodType>() ||
      type->dyncast<HasFieldType>() || type->dyncast<HasMethodType>() ||
      type->dyncast<StringType>() || type->dyncast<IsEntityType>() ||
      type->dyncast<UnitType>())
      return type;

    if (type->dyncast<TypeParameter>() || type->dyncast<InferType>())
      return intern(ApplyRegionType(mode, region, type));

    fmt::print(std::cerr, "Bad ApplyRegionType({}, {})\n", region, *type);
    abort();
  }

  TypePtr TypeInterner::mk_unapply_region(TypePtr type)
  {
    assert(is_interned(type));

    if (auto isect = type->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_unapply_region(elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = type->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_unapply_region(elem));
      }
      return mk_union(result);
    }

    if (auto range = type->dyncast<RangeType>())
    {
      return mk_range(
        mk_unapply_region(range->lower), mk_unapply_region(range->upper));
    }

    if (auto viewpoint = type->dyncast<ViewpointType>())
    {
      return mk_viewpoint(
        viewpoint->capability,
        viewpoint->variables,
        mk_unapply_region(viewpoint->right));
    }

    if (auto capability = type->dyncast<CapabilityType>())
    {
      switch (capability->kind)
      {
        case CapabilityKind::Mutable:
          return mk_mutable(RegionHole());

        case CapabilityKind::Subregion:
          return mk_subregion(RegionHole());

        case CapabilityKind::Isolated:
          // TODO: should this be iso() or iso(.)? mk_unapply_region should
          // probably carry the mode so we can decide.
          return mk_isolated(RegionHole());

        case CapabilityKind::Immutable:
          return mk_immutable();
      }
    }

    if (auto compression = type->dyncast<PathCompressionType>())
      return mk_unapply_region(compression->type);

    if (auto renaming = type->dyncast<VariableRenamingType>())
      return mk_unapply_region(renaming->type);

    if (auto apply = type->dyncast<ApplyRegionType>())
      return apply->type;

    if (
      type->dyncast<DelayedFieldViewType>() || type->dyncast<EntityOfType>() ||
      type->dyncast<EntityType>() || type->dyncast<HasAppliedMethodType>() ||
      type->dyncast<HasFieldType>() || type->dyncast<HasMethodType>() ||
      type->dyncast<StringType>() || type->dyncast<IsEntityType>() ||
      type->dyncast<UnitType>())
      return type;

    // TODO: We never actually create UnapplyRegionType anymore, so we could
    // delete the underlying class.

    std::cerr << "Bad UnapplyRegionType(" << *type << ")" << std::endl;
    abort();
  }

  TypePtr TypeInterner::mk_viewpoint(TypePtr left, TypePtr right)
  {
    assert(is_interned(left));
    assert(is_interned(right));

    if (left->dyncast<EntityType>() || left->dyncast<EntityOfType>())
      return mk_top();

    if (right->dyncast<EntityType>() || right->dyncast<StringType>())
      return right;

    if (auto isect = left->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_viewpoint(elem, right));
      }
      return mk_intersection(result);
    }

    if (auto union_ = left->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_viewpoint(elem, right));
      }
      return mk_union(result);
    }

    if (auto apply_region = left->dyncast<ApplyRegionType>())
    {
      return mk_viewpoint(apply_region->type, right);
    }
    if (auto renaming = left->dyncast<VariableRenamingType>())
    {
      return mk_viewpoint(renaming->type, right);
    }
    if (auto compression = left->dyncast<PathCompressionType>())
    {
      return mk_viewpoint(compression->type, right);
    }

    if (auto isect = right->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_viewpoint(left, elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = right->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_viewpoint(left, elem));
      }
      return mk_union(result);
    }

    if (auto range = right->dyncast<RangeType>())
    {
      return mk_range(
        mk_viewpoint(left, range->lower), mk_viewpoint(left, range->upper));
    }

    if (auto right_cap = right->dyncast<CapabilityType>())
    {
      if (right_cap->kind == CapabilityKind::Immutable)
        return mk_immutable();

      if (auto left_cap = left->dyncast<CapabilityType>();
          left_cap && left_cap->kind == CapabilityKind::Immutable)
        return mk_immutable();
    }

    if (auto compression = left->dyncast<PathCompressionType>())
    {
      // TODO: This assumes path compatibility doesn't change capabilities, or
      // at least not in a way that affects viewpoints.
      //
      // eg. compress(x: iso(y), mut(x)) = sub(y), but sub and mut have the same
      // impact (aka none) on viewpoint adaptation.
      //
      // However we have discussed having `compressed(x: imm, mut(x)) = imm`
      // before, would would impact viewpoint adaptation.
      return mk_viewpoint(compression->type, right);
    }

    if (
      right->dyncast<CapabilityType>() || right->dyncast<TypeParameter>() ||
      right->dyncast<InferType>() || right->dyncast<ApplyRegionType>() ||
      right->dyncast<UnapplyRegionType>())
    {
      if (auto param = left->dyncast<TypeParameter>())
      {
        return intern(ViewpointType(std::nullopt, {param}, right));
      }

      if (auto capability = left->dyncast<CapabilityType>())
      {
        switch (capability->kind)
        {
          case CapabilityKind::Mutable:
          case CapabilityKind::Subregion:
          case CapabilityKind::Isolated:
            return right;

          case CapabilityKind::Immutable:
            return intern(
              ViewpointType(capability->kind, TypeParameterSet(), right));

            EXHAUSTIVE_SWITCH;
        }
      }

      if (auto viewpoint = left->dyncast<ViewpointType>())
      {
        if (auto typeparam = viewpoint->right->dyncast<TypeParameter>())
        {
          TypeParameterSet variables = viewpoint->variables;
          variables.insert(typeparam);
          return intern(ViewpointType(viewpoint->capability, variables, right));
        }
      }
    }

    std::cerr << "Bad Viewpoint(" << *left << ", " << *right << ")"
              << std::endl;
    abort();
  }

  template<typename T>
  TypePtr TypeInterner::mk_viewpoint(
    std::optional<CapabilityKind> capability,
    const std::set<std::shared_ptr<const T>>& types,
    TypePtr right)
  {
    assert(is_interned(types));
    assert(is_interned(right));

    TypePtr result = right;
    if (capability.has_value())
    {
      switch (*capability)
      {
        case CapabilityKind::Mutable:
          result = mk_viewpoint(mk_mutable(RegionHole()), result);
          break;

        case CapabilityKind::Subregion:
          result = mk_viewpoint(mk_subregion(RegionHole()), result);
          break;

        case CapabilityKind::Isolated:
          result = mk_viewpoint(mk_isolated(RegionHole()), result);
          break;

        case CapabilityKind::Immutable:
          result = mk_viewpoint(mk_immutable(), result);
          break;
      }
    }

    for (auto elem : types)
    {
      result = mk_viewpoint(elem, result);
    }

    return result;
  }

  template TypePtr TypeInterner::mk_viewpoint<Type>(
    std::optional<CapabilityKind>, const TypeSet&, TypePtr);
  template TypePtr TypeInterner::mk_viewpoint<TypeParameter>(
    std::optional<CapabilityKind>, const TypeParameterSet&, TypePtr);

  InferTypePtr TypeInterner::mk_infer(
    uint64_t index, std::optional<uint64_t> subindex, Polarity polarity)
  {
    return intern(InferType(index, subindex, polarity));
  }

  TypePtr TypeInterner::mk_range(TypePtr lower, TypePtr upper)
  {
    assert(is_interned(lower));
    assert(is_interned(upper));

    return intern(RangeType(lower, upper));
  }

  TypePtr
  TypeInterner::mk_infer_range(uint64_t index, std::optional<uint64_t> subindex)
  {
    return mk_range(
      mk_infer(index, subindex, Polarity::Negative),
      mk_infer(index, subindex, Polarity::Positive));
  }

  UnitTypePtr TypeInterner::mk_unit()
  {
    return intern(UnitType());
  }

  TypePtr TypeInterner::mk_top()
  {
    return intern(IntersectionType(TypeSet()));
  }

  TypePtr TypeInterner::mk_bottom()
  {
    return intern(UnionType(TypeSet()));
  }

  HasFieldTypePtr TypeInterner::mk_has_field(
    TypePtr view, std::string name, TypePtr read_type, TypePtr write_type)
  {
    assert(is_interned(view));
    assert(is_interned(read_type));
    assert(is_interned(write_type));

    return intern(HasFieldType(view, name, read_type, write_type));
  }

  DelayedFieldViewTypePtr
  TypeInterner::mk_delayed_field_view(std::string name, TypePtr type)
  {
    assert(is_interned(type));

    return intern(DelayedFieldViewType(name, type));
  }

  HasMethodTypePtr
  TypeInterner::mk_has_method(std::string name, TypeSignature signature)
  {
    assert(is_interned(signature));

    return intern(HasMethodType(name, signature));
  }

  HasAppliedMethodTypePtr TypeInterner::mk_has_applied_method(
    std::string name,
    InferableTypeSequence application,
    TypeSignature signature)
  {
    // TODO: check that application is interned
    assert(is_interned(signature));

    return intern(HasAppliedMethodType(name, application, signature));
  }

  IsEntityTypePtr TypeInterner::mk_is_entity()
  {
    return intern(IsEntityType());
  }

  StringTypePtr TypeInterner::mk_string_type()
  {
    return intern(StringType());
  }

  FixpointTypePtr TypeInterner::mk_fixpoint(TypePtr inner)
  {
    assert(is_interned(inner));
    return intern(FixpointType(inner));
  }

  FixpointVariableTypePtr TypeInterner::mk_fixpoint_variable(uint64_t depth)
  {
    return intern(FixpointVariableType(depth));
  }

  TypePtr TypeInterner::mk_entity_of(TypePtr inner)
  {
    assert(is_interned(inner));

    if (auto isect = inner->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_entity_of(elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = inner->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_entity_of(elem));
      }
      return mk_union(result);
    }

    if (auto range = inner->dyncast<RangeType>())
    {
      return mk_range(mk_entity_of(range->lower), mk_entity_of(range->upper));
    }

    if (auto fixpoint = inner->dyncast<FixpointType>())
    {
      // entity-of is idempotent, making it fine to push down inside the
      // fixpoint.
      return mk_fixpoint(mk_entity_of(fixpoint->inner));
    }

    if (auto viewpoint = inner->dyncast<ViewpointType>())
    {
      return mk_entity_of(viewpoint->right);
    }

    if (auto apply = inner->dyncast<ApplyRegionType>())
    {
      return mk_entity_of(apply->type);
    }

    if (auto renaming = inner->dyncast<VariableRenamingType>())
    {
      return mk_entity_of(renaming->type);
    }

    if (auto compression = inner->dyncast<PathCompressionType>())
    {
      return mk_entity_of(compression->type);
    }

    if (auto capability = inner->dyncast<CapabilityType>())
    {
      return mk_top();
    }

    // These are all entity-like types already.
    if (
      inner->dyncast<EntityType>() || inner->dyncast<StringType>() ||
      inner->dyncast<UnitType>() || inner->dyncast<EntityOfType>() ||
      inner->dyncast<HasFieldType>() || inner->dyncast<HasMethodType>() ||
      inner->dyncast<HasAppliedMethodType>() ||
      inner->dyncast<DelayedFieldViewType>())
      return inner;

    if (
      inner->dyncast<InferType>() || inner->dyncast<TypeParameter>() ||
      inner->dyncast<FixpointType>())
      return intern(EntityOfType(inner));

    std::cerr << "Bad EntityOfType(" << *inner << ")" << std::endl;
    abort();
  }

  TypePtr
  TypeInterner::mk_variable_renaming(VariableRenaming renaming, TypePtr type)
  {
    assert(is_interned(type));

    auto map_region = [&](const Region& region) -> Region {
      if (const RegionVariable* rv = std::get_if<RegionVariable>(&region))
        return RegionVariable{renaming.apply(rv->variable)};
      else
        return region;
    };

    if (auto isect = type->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_variable_renaming(renaming, elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = type->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_variable_renaming(renaming, elem));
      }
      return mk_union(result);
    }

    if (auto range = type->dyncast<RangeType>())
    {
      return mk_range(
        mk_variable_renaming(renaming, range->lower),
        mk_variable_renaming(renaming, range->upper));
    }

    if (auto viewpoint = type->dyncast<ViewpointType>())
    {
      return mk_viewpoint(
        viewpoint->capability,
        viewpoint->variables,
        mk_variable_renaming(renaming, viewpoint->right));
    }

    if (auto capability = type->dyncast<CapabilityType>())
    {
      assert(!std::holds_alternative<RegionHole>(capability->region));
      return mk_capability(capability->kind, map_region(capability->region));
    }

    if (auto apply = type->dyncast<ApplyRegionType>())
    {
      // apply->type can only have RegionHole, so there's no need to apply the
      // renaming to it.
      return mk_apply_region(
        apply->mode, map_region(apply->region), apply->type);
    }

    if (auto not_child_of = type->dyncast<NotChildOfType>())
    {
      return mk_not_child_of(map_region(not_child_of->region));
    }

    if (auto unapply = type->dyncast<UnapplyRegionType>())
    {
      return mk_unapply_region(mk_variable_renaming(renaming, unapply->type));
    }

    if (auto inner = type->dyncast<VariableRenamingType>())
    {
      return mk_variable_renaming(
        renaming.compose(inner->renaming), inner->type);
    }

    // This are all entity-like types
    if (
      type->dyncast<DelayedFieldViewType>() || type->dyncast<EntityOfType>() ||
      type->dyncast<EntityType>() || type->dyncast<HasAppliedMethodType>() ||
      type->dyncast<HasFieldType>() || type->dyncast<HasMethodType>() ||
      type->dyncast<StringType>() || type->dyncast<IsEntityType>() ||
      type->dyncast<StaticType>() || type->dyncast<UnitType>() ||
      type->dyncast<TypeParameter>())
      return type;

    if (
      type->dyncast<InferType>() || type->dyncast<PathCompressionType>() ||
      type->dyncast<FixpointType>() || type->dyncast<FixpointVariableType>() ||
      type->dyncast<IndirectType>())
      return intern(VariableRenamingType(renaming, type));

    fmt::print(std::cerr, "Bad VariableRenamingType({})\n", *type);
    abort();
  }

  TypePtr TypeInterner::mk_path_compression(
    PathCompressionMap compression, TypePtr type)
  {
    assert(is_interned(type));

    if (compression.empty())
      return type;

    if (auto isect = type->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        result.insert(mk_path_compression(compression, elem));
      }
      return mk_intersection(result);
    }

    if (auto union_ = type->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        result.insert(mk_path_compression(compression, elem));
      }
      return mk_union(result);
    }

#if 0
    if (auto fixpoint = type->dyncast<FixpointType>())
    {
      return mk_fixpoint(mk_path_compression(compression, fixpoint->inner));
    }
    else if (type->dyncast<FixpointVariableType>())
    {
      return type;
    }
#endif

    if (auto range = type->dyncast<RangeType>())
    {
      return mk_range(
        mk_path_compression(compression, range->lower),
        mk_path_compression(compression, range->upper));
    }

    /**
     * compress(x: T, σU) =
     *  (σ - { (y, x) | forall y }) (compress(σ⁻¹(x): σ⁻¹(T), U))
     */
    if (auto variable_renaming = type->dyncast<VariableRenamingType>())
    {
      VariableRenaming inverse = variable_renaming->renaming.invert();
      PathCompressionMap renamed_compression;
      for (auto [dead_variable, dead_type] : compression)
      {
        renamed_compression.insert({inverse.apply(dead_variable),
                                    mk_variable_renaming(inverse, dead_type)});
      }

      VariableRenaming compressed_renaming =
        variable_renaming->renaming.filter([&](Variable from, Variable to) {
          return compression.find(to) == compression.end();
        });

      return mk_variable_renaming(
        compressed_renaming,
        mk_path_compression(renamed_compression, variable_renaming->type));
    }

    if (auto inner = type->dyncast<PathCompressionType>())
    {
      PathCompressionMap composed_compression = inner->compression;
      for (auto [dead_variable, dead_type] : compression)
      {
        // inner->compression may already contain dead_variable, in which case
        // this does nothing.
        composed_compression.insert({dead_variable, dead_type});
      }

      return mk_path_compression(composed_compression, inner->type);
    }

    // These are all entity-like types
    if (
      type->dyncast<DelayedFieldViewType>() || type->dyncast<EntityOfType>() ||
      type->dyncast<EntityType>() || type->dyncast<HasAppliedMethodType>() ||
      type->dyncast<HasFieldType>() || type->dyncast<HasMethodType>() ||
      type->dyncast<StringType>() || type->dyncast<IsEntityType>() ||
      type->dyncast<StaticType>() || type->dyncast<UnitType>() ||
      type->dyncast<TypeParameter>())
      return type;

    if (
      type->dyncast<InferType>() || type->dyncast<FixpointVariableType>() ||
      type->dyncast<FixpointType>())
      return intern(PathCompressionType(compression, type));

    if (auto capability = type->dyncast<CapabilityType>())
    {
      return unfold_compression(compression, capability->region, type);
    }
    else if (auto apply_region = type->dyncast<ApplyRegionType>())
    {
      return unfold_compression(compression, apply_region->region, type);
    }

    fmt::print(
      std::cerr,
      "Bad PathCompression([{}], {})\n",
      format::comma_sep(
        compression,
        [](const auto& x) {
          return fmt::format("{}: {}", x.first, *x.second);
        }),
      *type);

    abort();
  }

  TypePtr TypeInterner::unfold_compression(
    const PathCompressionMap& compression, const Region& region, TypePtr type)
  {
    const auto* rv = std::get_if<RegionVariable>(&region);
    if (!rv)
      return type;

    auto it = compression.find(rv->variable);
    if (it == compression.end())
      return type;

    return unfold_compression(compression, it->first, type);
  }

  TypePtr TypeInterner::unfold_compression(
    PathCompressionMap compression, Variable dead_variable, TypePtr type)
  {
    TypePtr replacement = compression.at(dead_variable);
    if (auto isect = replacement->dyncast<IntersectionType>())
    {
      TypeSet result;
      for (auto elem : isect->elements)
      {
        compression.at(dead_variable) = elem;
        result.insert(unfold_compression(compression, dead_variable, type));
      }
      return mk_intersection(result);
    }

    if (auto union_ = replacement->dyncast<UnionType>())
    {
      TypeSet result;
      for (auto elem : union_->elements)
      {
        compression.at(dead_variable) = elem;
        result.insert(unfold_compression(compression, dead_variable, type));
      }
      return mk_union(result);
    }

    if (replacement->dyncast<EntityType>())
      return mk_top();

    // TODO: There's potentially some normalization we could do here if
    // `replacement` is a CapabilityType (or even an ApplyRegionType).
    return intern(PathCompressionType(compression, type));
  }

  TypePtr
  TypeInterner::mk_indirect_type(const BasicBlock* block, Variable variable)
  {
    return intern(IndirectType(block, variable));
  }

  TypePtr TypeInterner::mk_not_child_of(Region region)
  {
    return intern(NotChildOfType(region));
  }

  bool TypeInterner::is_interned(const TypePtr& ty)
  {
    if (!ty)
      return false;

    auto it = types_.find(ty);
    return it != types_.end() && *it == ty;
  }

  bool TypeInterner::is_interned(const TypeList& tys)
  {
    return std::all_of(
      tys.begin(), tys.end(), [&](auto ty) { return is_interned(ty); });
  }

  bool TypeInterner::is_interned(const TypeSignature& signature)
  {
    return is_interned(signature.receiver) &&
      is_interned(signature.arguments) && is_interned(signature.return_type);
  }

  bool TypeInterner::is_interned(const PathCompressionMap& compression)
  {
    return std::all_of(
      compression.begin(), compression.end(), [&](const auto& entry) {
        return is_interned(entry.second);
      });
  }

  template<typename T>
  std::shared_ptr<const T> TypeInterner::intern(T value)
  {
    // We split the lookup and insert steps, avoiding the allocation of the
    // shared_ptr if the lookup succeeds. To avoid searching the map twice,
    // we use lower_bound to perform the lookup, giving us a hint on where
    // to insert even if the lookup fails.
    // <Some rant about how unergonomic C++ containers are>

    // After lower_bound, one of these must hold:
    // - `it` is `types_.end()`
    // - `*it` is strictly greater than `value`
    // - `*it` is equal to `value`
    // In the first two cases, value is not in the map, so we insert it.
    auto it = types_.lower_bound(value);
    if (it == types_.end() || LessTypes()(value, *it))
    {
      it = types_.emplace_hint(it, std::make_shared<T>(value));
    }

    assert(!LessTypes()(*it, value) && !LessTypes()(value, *it));

    // At this point, `*it` is equal to `value`, making the cast to a
    // shared_ptr<T> safe.
    return std::static_pointer_cast<const T>(*it);
  }

  /**
   * Shallow by-value comparison of arbitrary types.
   *
   * Each subclass of Type has an operator< which can be used to compare it
   * to values of the same type. To extend this to arbitrary pairs of types,
   * we compare them lexicographically with
   * (typeof(left), dataof(left)) < (typeof(right), dataof(right))
   *
   * Only if the types match do we need to compare the data, which we can do
   * using the existing operator<.
   */
  bool
  TypeInterner::LessTypes::operator()(const Type& left, const Type& right) const
  {
    const std::type_info& info = typeid(left);
    std::type_index ti_left(info);
    std::type_index ti_right(typeid(right));
    if (ti_left < ti_right)
    {
      return true;
    }
    else if (ti_left > ti_right)
    {
      return false;
    }

    // At this point we know the two arguments have the same type.
#define DISPATCH(ty) \
  if (info == typeid(ty)) \
  { \
    const auto& left_ = static_cast<const ty&>(left); \
    const auto& right_ = static_cast<const ty&>(right); \
    return left_ < right_; \
  }
    DISPATCH(ApplyRegionType);
    DISPATCH(CapabilityType);
    DISPATCH(DelayedFieldViewType);
    DISPATCH(EntityOfType);
    DISPATCH(EntityType);
    DISPATCH(FixpointType);
    DISPATCH(FixpointVariableType);
    DISPATCH(HasAppliedMethodType);
    DISPATCH(HasFieldType);
    DISPATCH(HasMethodType);
    DISPATCH(IndirectType);
    DISPATCH(InferType);
    DISPATCH(IntersectionType);
    DISPATCH(IsEntityType);
    DISPATCH(NotChildOfType);
    DISPATCH(PathCompressionType);
    DISPATCH(RangeType);
    DISPATCH(StaticType);
    DISPATCH(StringType);
    DISPATCH(TypeParameter);
    DISPATCH(UnapplyRegionType);
    DISPATCH(UnionType);
    DISPATCH(UnitType);
    DISPATCH(VariableRenamingType);
    DISPATCH(ViewpointType);
#undef DISPATCH

    fmt::print(std::cerr, "TypeInterner dispatch failed on {}\n", info.name());
    abort();
  }

  bool TypeInterner::LessTypes::operator()(
    const TypePtr& left, const TypePtr& right) const
  {
    return (*this)(*left, *right);
  }

  bool TypeInterner::LessTypes::operator()(
    const TypePtr& left, const Type& right) const
  {
    return (*this)(*left, right);
  }

  bool TypeInterner::LessTypes::operator()(
    const Type& left, const TypePtr& right) const
  {
    return (*this)(left, *right);
  }
}
