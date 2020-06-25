// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/context.h"
#include "compiler/mapper.h"
#include "compiler/visitor.h"

namespace verona::compiler
{
  /**
   * The polarizer performs some normalization on types, based on which side of
   * the subtyping constraint they appear (i.e. what polarity they have).
   *
   * For example, we can simplify type ranges to only select the side we care
   * about. This helps reduce the size of types and their set of free-variables,
   * leading to faster substitution.
   *
   * The polarizer also encodes some subtyping relations through normalization
   * rather than explicit rules in constraint.cc, based on [0].
   *
   * For example distributivity of intersections of unions, described by the
   * subtyping rule below, is implemented by putting the left-hand side in DNF:
   *
   * -----------------------------------------
   *  T1 & (T2 | T3) <: (T1 | T2) & (T1 | T3)
   *
   * Polarizer is not a RecursiveTypeMapper because we want to control how the
   * Polarity changes when recursing. For example is a HasMethodType, the
   * arguments are contra-variant and therefore have their polarity reversed.
   *
   * The base case, if no method is provided for a certain construct, is to do
   * nothing and return the type unmodified. This is always sound, but may cause
   * the solver to be slower and/or incomplete.
   *
   * Because the same types get polarized over and over again during solving,
   * the Polarizer memoizes results.
   *
   * [0]: Muehlboeck, Fabian, and Ross Tate. "Empowering union and intersection
   *      types with integrated subtyping."
   *      Proceedings of the ACM on Programming Languages 2.OOPSLA (2018): 112.
   *      https://www.cs.cornell.edu/~ross/publications/empower/empower-oopsla18.pdf
   */
  class Polarizer : public TypeMapper<Polarity>,
                    private TypeVisitor<TypePtr, Polarity>
  {
  public:
    Polarizer(Context& context) : context_(context), TypeMapper(context) {}

  private:
    TypePtr apply_one(const TypePtr& type, Polarity polarity) final;
    InferableTypeSequence
    apply_sequence(const InferableTypeSequence& seq, Polarity polarity) final;

    TypePtr visit_base_type(const TypePtr& ty, Polarity polarity) final;
    TypePtr visit_infer(const InferTypePtr& ty, Polarity polarity) final;
    TypePtr visit_range_type(const RangeTypePtr& ty, Polarity polarity) final;
    TypePtr visit_union(const UnionTypePtr& ty, Polarity polarity) final;
    TypePtr
    visit_intersection(const IntersectionTypePtr& ty, Polarity polarity) final;

    TypePtr
    visit_apply_region(const ApplyRegionTypePtr& ty, Polarity polarity) final;
    TypePtr visit_unapply_region(
      const UnapplyRegionTypePtr& ty, Polarity polarity) final;

    TypePtr
    visit_fixpoint_type(const FixpointTypePtr& ty, Polarity polarity) final;

    TypeSignature
    visit_signature(const TypeSignature& signature, Polarity polarity);
    TypePtr
    visit_has_method_type(const HasMethodTypePtr& ty, Polarity polarity) final;
    TypePtr visit_has_applied_method_type(
      const HasAppliedMethodTypePtr& ty, Polarity polarity) final;
    TypePtr
    visit_has_field_type(const HasFieldTypePtr& ty, Polarity polarity) final;
    TypePtr visit_delayed_field_view_type(
      const DelayedFieldViewTypePtr& ty, Polarity polarity) final;
    TypePtr
    visit_is_entity_type(const IsEntityTypePtr& ty, Polarity polarity) final;

    TypePtr
    visit_type_parameter(const TypeParameterPtr& ty, Polarity polarity) final;

    TypePtr visit_variable_renaming_type(
      const VariableRenamingTypePtr& ty, Polarity polarity) final;

    TypePtr visit_path_compression_type(
      const PathCompressionTypePtr& ty, Polarity polarity) final;

    TypePtr visit_not_child_of_type(
      const NotChildOfTypePtr& ty, Polarity polarity) final;

    InferableTypeSequence
    visit_sequence(const BoundedTypeSequence& seq, Polarity polarity);
    InferableTypeSequence
    visit_sequence(const UnboundedTypeSequence& seq, Polarity polarity);

    template<typename T>
    TypePtr normalize(const typename T::DualPtr& ty, Polarity polarity);

    Context& context_;

    std::unordered_map<TypePtr, TypePtr> positive_cache_;
    std::unordered_map<TypePtr, TypePtr> negative_cache_;
  };
}
