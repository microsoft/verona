// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/mapper.h"

#include "compiler/context.h"
#include "ds/helpers.h"

namespace verona::compiler
{
  TypeSignature
  apply_mapper(TypeMapper<>& mapper, const TypeSignature& signature)
  {
    return TypeSignature(
      mapper.apply(signature.receiver),
      mapper.apply(signature.arguments),
      mapper.apply(signature.return_type));
  }

  TypePtr RecursiveTypeMapper::apply_one(const TypePtr& type)
  {
    if (modifies_type(type))
      return visit_type(type);
    else
      return type;
  }

  TypePtr RecursiveTypeMapper::visit_entity_type(const EntityTypePtr& ty)
  {
    return context().mk_entity_type(ty->definition, apply(ty->arguments));
  }

  TypePtr RecursiveTypeMapper::visit_static_type(const StaticTypePtr& ty)
  {
    return context().mk_static_type(ty->definition, this->apply(ty->arguments));
  }

  TypePtr RecursiveTypeMapper::visit_type_parameter(const TypeParameterPtr& ty)
  {
    return context().mk_type_parameter(ty->definition, ty->expanded);
  }

  TypePtr RecursiveTypeMapper::visit_capability(const CapabilityTypePtr& ty)
  {
    return context().mk_capability(ty->kind, ty->region);
  }

  TypePtr RecursiveTypeMapper::visit_apply_region(const ApplyRegionTypePtr& ty)
  {
    return context().mk_apply_region(ty->mode, ty->region, apply(ty->type));
  }

  TypePtr
  RecursiveTypeMapper::visit_unapply_region(const UnapplyRegionTypePtr& ty)
  {
    return context().mk_unapply_region(apply(ty->type));
  }

  TypePtr RecursiveTypeMapper::visit_union(const UnionTypePtr& ty)
  {
    return context().mk_union(apply(ty->elements));
  }

  TypePtr RecursiveTypeMapper::visit_intersection(const IntersectionTypePtr& ty)
  {
    return context().mk_intersection(apply(ty->elements));
  }

  TypePtr RecursiveTypeMapper::visit_unit_type(const UnitTypePtr& ty)
  {
    return context().mk_unit();
  }

  TypePtr RecursiveTypeMapper::visit_infer(const InferTypePtr& ty)
  {
    return context().mk_infer(ty->index, ty->subindex, ty->polarity);
  }

  TypePtr RecursiveTypeMapper::visit_range_type(const RangeTypePtr& ty)
  {
    return context().mk_range(apply(ty->lower), apply(ty->upper));
  }

  TypePtr RecursiveTypeMapper::visit_viewpoint_type(const ViewpointTypePtr& ty)
  {
    TypeSet variables;
    for (const auto& it : ty->variables)
    {
      variables.insert(apply(it));
    }

    return context().mk_viewpoint(ty->capability, variables, apply(ty->right));
  }

  TypePtr RecursiveTypeMapper::visit_has_field_type(const HasFieldTypePtr& ty)
  {
    return context().mk_has_field(
      apply(ty->view), ty->name, apply(ty->read_type), apply(ty->write_type));
  }

  TypePtr RecursiveTypeMapper::visit_delayed_field_view_type(
    const DelayedFieldViewTypePtr& ty)
  {
    return context().mk_delayed_field_view(ty->name, apply(ty->type));
  }

  TypePtr RecursiveTypeMapper::visit_has_method_type(const HasMethodTypePtr& ty)
  {
    return context().mk_has_method(ty->name, apply(ty->signature));
  }

  TypePtr RecursiveTypeMapper::visit_has_applied_method_type(
    const HasAppliedMethodTypePtr& ty)
  {
    return context().mk_has_applied_method(
      ty->name, apply(ty->application), apply(ty->signature));
  }

  TypePtr RecursiveTypeMapper::visit_is_entity_type(const IsEntityTypePtr& ty)
  {
    return context().mk_is_entity();
  }

  TypePtr RecursiveTypeMapper::visit_fixpoint_type(const FixpointTypePtr& ty)
  {
    return context().mk_fixpoint(apply(ty->inner));
  }

  TypePtr RecursiveTypeMapper::visit_fixpoint_variable_type(
    const FixpointVariableTypePtr& ty)
  {
    return context().mk_fixpoint_variable(ty->depth);
  }

  TypePtr RecursiveTypeMapper::visit_entity_of_type(const EntityOfTypePtr& ty)
  {
    return context().mk_entity_of(apply(ty->inner));
  }

  TypePtr RecursiveTypeMapper::visit_variable_renaming_type(
    const VariableRenamingTypePtr& ty)
  {
    return context().mk_variable_renaming(ty->renaming, apply(ty->type));
  }

  TypePtr RecursiveTypeMapper::visit_path_compression_type(
    const PathCompressionTypePtr& ty)
  {
    return context().mk_path_compression(
      apply(ty->compression), apply(ty->type));
  }

  TypePtr RecursiveTypeMapper::visit_indirect_type(const IndirectTypePtr& ty)
  {
    return context().mk_indirect_type(ty->block, ty->variable);
  }

  TypePtr
  RecursiveTypeMapper::visit_not_child_of_type(const NotChildOfTypePtr& ty)
  {
    return context().mk_not_child_of(ty->region);
  }

  InferableTypeSequence
  RecursiveTypeMapper::apply_sequence(const InferableTypeSequence& seq)
  {
    return std::visit(
      [&](const auto& inner) { return visit_sequence(inner); }, seq);
  }

  InferableTypeSequence
  RecursiveTypeMapper::visit_sequence(const BoundedTypeSequence& seq)
  {
    return BoundedTypeSequence(apply(seq.types));
  }

  InferableTypeSequence
  RecursiveTypeMapper::visit_sequence(const UnboundedTypeSequence& seq)
  {
    return UnboundedTypeSequence(seq.index);
  }

  TypePtr Flatten::visit_range_type(const RangeTypePtr& ty)
  {
    return apply(ty->upper);
  }

  TypePtr Flatten::visit_infer(const InferTypePtr& ty)
  {
    switch (ty->polarity)
    {
      case Polarity::Positive:
        return context().mk_bottom();
      case Polarity::Negative:
        return context().mk_top();

        EXHAUSTIVE_SWITCH;
    }
  }
}
