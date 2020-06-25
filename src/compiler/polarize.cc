// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/polarize.h"

#include "compiler/freevars.h"
#include "compiler/printing.h"
#include "ds/helpers.h"

namespace verona::compiler
{
  TypePtr Polarizer::apply_one(const TypePtr& type, Polarity polarity)
  {
    auto& cache =
      polarity == Polarity::Positive ? positive_cache_ : negative_cache_;

    auto it = cache.find(type);
    if (it != cache.end())
    {
      return it->second;
    }
    else
    {
      TypePtr normalized = visit_type(type, polarity);
      TypePtr repeat = visit_type(normalized, polarity);

      if (normalized != repeat)
      {
        std::cerr << "Not idempotent for" << std::endl;
        std::cerr << " " << *type << std::endl;
        std::cerr << " " << *normalized << std::endl;
        std::cerr << " " << *repeat << std::endl;
        abort();
      }

      cache.insert({type, normalized});
      cache.insert({normalized, normalized});
      return normalized;
    }
  }

  TypePtr Polarizer::visit_base_type(const TypePtr& ty, Polarity polarity)
  {
    return ty;
  }

  TypePtr Polarizer::visit_infer(const InferTypePtr& ty, Polarity polarity)
  {
    assert(ty->polarity == polarity);
    return ty;
  }

  TypePtr Polarizer::visit_range_type(const RangeTypePtr& ty, Polarity polarity)
  {
    switch (polarity)
    {
      case Polarity::Negative:
        return apply(ty->lower, polarity);
      case Polarity::Positive:
        return apply(ty->upper, polarity);

        EXHAUSTIVE_SWITCH;
    }
  }

  TypeSignature
  Polarizer::visit_signature(const TypeSignature& signature, Polarity polarity)
  {
    // Receiver and arguments are co-variant, return type is contra-variant
    return TypeSignature(
      apply(signature.receiver, reverse_polarity(polarity)),
      apply(signature.arguments, reverse_polarity(polarity)),
      apply(signature.return_type, polarity));
  }

  TypePtr Polarizer::visit_has_method_type(
    const HasMethodTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);

    return context_.mk_has_method(
      ty->name, visit_signature(ty->signature, polarity));
  }

  TypePtr Polarizer::visit_has_applied_method_type(
    const HasAppliedMethodTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);

    // Type arguments are invariant
    return context_.mk_has_applied_method(
      ty->name, ty->application, visit_signature(ty->signature, polarity));
  }

  TypePtr
  Polarizer::visit_has_field_type(const HasFieldTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);

    // Field read is co-variant, Field write is contra-variant
    // TODO: what should the view be? It's currently conservatively defined as
    // invariant.
    return context_.mk_has_field(
      ty->view,
      ty->name,
      apply(ty->read_type, Polarity::Negative),
      apply(ty->write_type, Polarity::Positive));
  }

  TypePtr Polarizer::visit_delayed_field_view_type(
    const DelayedFieldViewTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);

    // Field view is co-variant
    return context_.mk_delayed_field_view(
      ty->name, apply(ty->type, Polarity::Negative));
  }

  TypePtr
  Polarizer::visit_is_entity_type(const IsEntityTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);
    return context_.mk_is_entity();
  }

  TypePtr Polarizer::visit_union(const UnionTypePtr& ty, Polarity polarity)
  {
    return context_.mk_union(apply(ty->elements, polarity));
  }

  TypePtr
  Polarizer::visit_fixpoint_type(const FixpointTypePtr& ty, Polarity polarity)
  {
    TypePtr inner = apply(ty->inner, polarity);

    // If inner is closed, the variable this fixpoint binds is never mentions
    // and we can just elide it.
    //
    // This normalization could be done in the interner, since it does not
    // depend on polarity, but it's awkward to access free variables in there
    // with the current layering.
    //
    // This is conservative, in that a type (fixpoint (fixpoint-variable 1))
    // could be normalized to (fixpoint-variable 0) but isn't, because that
    // would require more information than just "closed or not".
    if (context_.free_variables(inner).is_fixpoint_closed())
      return inner;
    else
      return context_.mk_fixpoint(inner);
  }

  TypePtr Polarizer::visit_intersection(
    const IntersectionTypePtr& ty, Polarity polarity)
  {
    // Put in DNF
    return normalize<UnionType>(ty, polarity);
  }

  TypePtr
  Polarizer::visit_apply_region(const ApplyRegionTypePtr& ty, Polarity polarity)
  {
    return context_.mk_apply_region(
      ty->mode, ty->region, apply(ty->type, polarity));
  }

  TypePtr Polarizer::visit_unapply_region(
    const UnapplyRegionTypePtr& ty, Polarity polarity)
  {
    return context_.mk_unapply_region(apply(ty->type, polarity));
  }

  TypePtr
  Polarizer::visit_type_parameter(const TypeParameterPtr& ty, Polarity polarity)
  {
    /**
     * Rather than have a rule X <: Δ(X), we expand type parameters from X
     * to `X & Δ(X)`.
     */
    if (
      polarity == Polarity::Positive &&
      ty->expanded == TypeParameter::Expanded::No)
    {
      // TODO:
      // The `expanded` flag will prevent infinite recursion and should be
      // sound, but may not preserve all the information. For more complex
      // cycles we probably to be smarter and do something with fixpoint types.
      //
      // For example given `X: Equatable[X]`, we currently expand `X` to just
      // `X & Equatable[X]`, but it should expand to something along the lines
      // of `μβ. (X & Equatable[β])`.
      //
      TypePtr expanded = context_.mk_type_parameter(
        ty->definition, TypeParameter::Expanded::Yes);
      return apply(
        context_.mk_intersection(expanded, ty->definition->bound), polarity);
    }
    else
    {
      return ty;
    }
  }

  TypePtr Polarizer::visit_variable_renaming_type(
    const VariableRenamingTypePtr& ty, Polarity polarity)
  {
    return context_.mk_variable_renaming(
      ty->renaming, apply(ty->type, polarity));
  }

  TypePtr Polarizer::visit_path_compression_type(
    const PathCompressionTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Positive);
    return context_.mk_path_compression(
      apply(ty->compression, polarity), apply(ty->type, polarity));
  }

  TypePtr Polarizer::visit_not_child_of_type(
    const NotChildOfTypePtr& ty, Polarity polarity)
  {
    assert(polarity == Polarity::Negative);
    return ty;
  }

  namespace
  {
    /*
     * Extract a specific subclass of Type from a TypeSet.
     *
     * It returns a pair where the first item of the desired subclass, and the
     * second item is all the other elements. Only one element of that subclass
     * is extracted, the second item could contain more of these.
     *
     * If no element of the set matches this class, the first item is nullopt
     * and the second is the original set.
     */
    template<typename T>
    std::pair<std::optional<std::shared_ptr<const T>>, TypeSet>
    extract_specific_subclass(TypeSet elements)
    {
      for (auto it = elements.begin(); it != elements.end(); it++)
      {
        if (std::shared_ptr<const T> selected = (*it)->dyncast<T>())
        {
          elements.erase(it);
          return {selected, elements};
        }
      }

      // Nothing in the set matches T
      return {std::nullopt, elements};
    }
  }

  InferableTypeSequence
  Polarizer::apply_sequence(const InferableTypeSequence& seq, Polarity polarity)
  {
    return std::visit(
      [&](const auto& inner) { return visit_sequence(inner, polarity); }, seq);
  }

  InferableTypeSequence
  Polarizer::visit_sequence(const BoundedTypeSequence& seq, Polarity polarity)
  {
    return BoundedTypeSequence(apply(seq.types, polarity));
  }

  InferableTypeSequence
  Polarizer::visit_sequence(const UnboundedTypeSequence& seq, Polarity polarity)
  {
    return UnboundedTypeSequence(seq.index);
  }

  /*
   * If T is UnionType, convert to DNF
   * If T is IntersectionType, convert to CNF
   */
  template<typename T>
  TypePtr Polarizer::normalize(const typename T::DualPtr& ty, Polarity polarity)
  {
    TypeSet elements = apply(ty->elements, polarity);

    // Look to see if there's a nested T we need to distribute.
    auto extracted = extract_specific_subclass<T>(elements);
    if (extracted.first)
    {
      // One of the elements is of the T class.
      // Distribute it over the rest of the elements.
      TypeSet expanded;
      for (auto elem : (*extracted.first)->elements)
      {
        TypeSet current = extracted.second;
        current.insert(elem);

        // extracted.second could contain some more T items.
        // We need to recurse to make it NF as well.
        TypePtr current_ty = context_.mk_connective<typename T::Dual>(current);
        expanded.insert(apply(current_ty, polarity));
      }
      return context_.mk_connective<T>(expanded);
    }
    else
    {
      // No T element we could distribute, it's fine to return a top-level
      // T::Dual.
      return context_.mk_connective<typename T::Dual>(extracted.second);
    }
  }
}
