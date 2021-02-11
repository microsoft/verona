// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/type.h"

#include <algorithm>
#include <optional>
#include <set>

/**
 * Type interner.
 *
 * To allow for fast equality checks between two Type objects, we intern all of
 * them in a single TypeInterner. The interner uses shallow comparison, through
 * the operator< method defined by each kind of Type, to check if a type had
 * already been interned. After interning, comparison and hashing can be done
 * directly on the pointer value.
 *
 * Type objects are never created manually. The various mk_* methods of the
 * interner should be used instead.
 *
 * Interning does perform some amount of normalization on the types. For
 * instance unions and intersections will be flattened, such that
 * ((A & B) & C) becomes (A & B & C) for example.
 *
 * In general, mk_foo methods return a FooPtr. Methods that do perform
 * normalization may however return a TypePtr, since the normalization could
 * have changed the head constructor.
 *
 * All mk_ methods require their arguments to already be normalized. This is
 * enforced with debug-mode assertions.
 */
namespace verona::compiler
{
  class TypeInterner
  {
  public:
    TypeInterner() {}

    EntityTypePtr mk_entity_type(const Entity* definition, TypeList arguments);
    StaticTypePtr mk_static_type(const Entity* definition, TypeList arguments);

    CapabilityTypePtr mk_capability(CapabilityKind kind, Region region);
    CapabilityTypePtr mk_mutable(Region region);
    CapabilityTypePtr mk_subregion(Region region);
    CapabilityTypePtr mk_isolated(Region region);
    CapabilityTypePtr mk_immutable();

    TypeParameterPtr mk_type_parameter(
      const TypeParameterDef* definition, TypeParameter::Expanded expanded);

    TypePtr
    mk_apply_region(ApplyRegionType::Mode mode, Region region, TypePtr type);
    TypePtr mk_unapply_region(TypePtr type);

    TypePtr mk_bottom();
    TypePtr mk_union(TypeSet elements);
    TypePtr mk_union(TypePtr first, TypePtr second);

    TypePtr mk_top();
    TypePtr mk_intersection(TypeSet elements);
    TypePtr mk_intersection(TypePtr first, TypePtr second);

    // Generic method to create either a UnionType or IntersectionType.
    template<typename T>
    TypePtr mk_connective(TypeSet elements);

    TypePtr mk_viewpoint(TypePtr left, TypePtr right);

    template<typename T>
    TypePtr mk_viewpoint(
      std::optional<CapabilityKind> capability,
      const std::set<std::shared_ptr<const T>>& types,
      TypePtr right);

    InferTypePtr mk_infer(
      uint64_t index, std::optional<uint64_t> subindex, Polarity polarity);
    TypePtr mk_range(TypePtr lower, TypePtr upper);
    TypePtr mk_infer_range(uint64_t index, std::optional<uint64_t> subindex);

    HasFieldTypePtr mk_has_field(
      TypePtr view, std::string name, TypePtr read_type, TypePtr write_type);

    DelayedFieldViewTypePtr
    mk_delayed_field_view(std::string name, TypePtr type);

    HasMethodTypePtr mk_has_method(std::string name, TypeSignature signature);

    HasAppliedMethodTypePtr mk_has_applied_method(
      std::string name, InferableTypeSequence tyargs, TypeSignature signature);

    IsEntityTypePtr mk_is_entity();

    UnitTypePtr mk_unit();

    FixpointTypePtr mk_fixpoint(TypePtr inner);
    FixpointVariableTypePtr mk_fixpoint_variable(uint64_t depth);

    TypePtr mk_entity_of(TypePtr inner);

    TypePtr mk_variable_renaming(VariableRenaming renaming, TypePtr type);
    TypePtr mk_path_compression(PathCompressionMap compression, TypePtr type);
    TypePtr mk_indirect_type(const BasicBlock* block, Variable variable);

    TypePtr mk_not_child_of(Region region);

    // It's important that we only have one interner, but C++ makes it easy
    // to accidentally make copies. Protect against that.
    TypeInterner(const TypeInterner&) = delete;
    TypeInterner& operator=(const TypeInterner&) = delete;

  private:
    bool is_interned(const TypePtr& ty);

    /**
     * Templated so it works on stuff like InferTypeSet in addition to TypeSet.
     */
    template<typename T>
    bool is_interned(const std::set<std::shared_ptr<const T>>& tys)
    {
      return std::all_of(
        tys.begin(), tys.end(), [&](auto ty) { return is_interned(ty); });
    }

    bool is_interned(const TypeList& tys);
    bool is_interned(const TypeSignature& signature);
    bool is_interned(const PathCompressionMap& signature);

    template<typename T>
    TypeSet flatten_connective(TypeSet elements);

    template<typename T>
    void flatten_connective(
      TypeSet elements, std::set<typename T::DualPtr>* duals, TypeSet* others);

    template<typename T>
    std::shared_ptr<const T> intern(T value);

    /**
     * Shallow by-value comparison of types.
     *
     * The set contains shared_ptr<Type>, but we don't want to allocate a
     * shared_ptr everytime we lookup something. For this we make LessTypes
     * a "transparent functor", allowing heterogenous lookup into the
     * interning set.
     *
     */
    struct LessTypes
    {
      using is_transparent = void;

      // Main comparison operator. We define the others in term of this one.
      bool operator()(const Type& left, const Type& right) const;

      bool operator()(const TypePtr& left, const TypePtr& right) const;
      bool operator()(const TypePtr& left, const Type& right) const;
      bool operator()(const Type& left, const TypePtr& right) const;
    };

    TypePtr unfold_compression(
      const PathCompressionMap& compression,
      const Region& region,
      TypePtr type);
    TypePtr unfold_compression(
      PathCompressionMap compression, Variable dead_variable, TypePtr type);

    std::set<TypePtr, LessTypes> types_;
  };
}
