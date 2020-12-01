// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/type_visitor.h"

namespace verona::compiler
{
  /*
   * Utility class to transform types.
   *
   * Subclasses should override apply_one. TypeMapper provides an apply() method
   * which is overloaded to work on TypePtr, anything with an apply_mapper
   * method, or any collections of these.
   */
  template<typename... Args>
  class TypeMapper
  {
  public:
    TypeMapper(Context& context) : context_(context) {}

    template<typename T>
    auto apply(const T& value, Args... args)
    {
      return apply_mapper(*this, value, args...);
    }

    Context& context() const
    {
      return context_;
    }

    friend TypePtr apply_mapper(
      TypeMapper<Args...>& mapper, const TypePtr& value, Args... args)
    {
      return mapper.apply_one(value, std::forward<Args>(args)...);
    }

    friend InferableTypeSequence apply_mapper(
      TypeMapper<Args...>& mapper,
      const InferableTypeSequence& value,
      Args... args)
    {
      return mapper.apply_sequence(value, std::forward<Args>(args)...);
    }

  protected:
    virtual TypePtr apply_one(const TypePtr& type, Args... args) = 0;
    virtual InferableTypeSequence
    apply_sequence(const InferableTypeSequence& seq, Args... args) = 0;

  private:
    Context& context_;
  };

  // Vector-like collections of things that can be applied.
  template<typename T, typename... Args>
  std::enable_if_t<traits::is_vector_like_v<T>, T> apply_mapper(
    TypeMapper<Args...>& mapper,
    const T& collection,
    std::common_type_t<Args>... args)
  {
    T result;
    for (const auto& it : collection)
    {
      result.push_back(mapper.apply(it, args...));
    }
    return result;
  }

  // Set-like collections of things that can be applied.
  template<typename T, typename... Args>
  std::enable_if_t<traits::is_set_like_v<T>, T> apply_mapper(
    TypeMapper<Args...>& mapper,
    const T& collection,
    std::common_type_t<Args>... args)
  {
    T result;
    for (const auto& it : collection)
    {
      result.insert(mapper.apply(it, args...));
    }
    return result;
  }

  // Map-like collections of things that can be applied.
  // Note that it only modifies the values, not the keys
  template<typename T, typename... Args>
  std::enable_if_t<traits::is_map_like_v<T>, T> apply_mapper(
    TypeMapper<Args...>& mapper,
    const T& collection,
    std::common_type_t<Args>... args)
  {
    T result;
    for (const auto& it : collection)
    {
      auto applied = mapper.apply(it.second, args...);
      result.insert({it.first, applied});
    }
    return result;
  }

  // Specialization of apply_mapper(TypeMapper&, const T&) for types which have
  // a apply_mapper(TypeMapper&) method. This allows the custom implementation
  // to be defined as a member method rather than a free function
  // specialization.
  template<typename T, typename... Args>
  decltype(std::declval<const T&>().apply_mapper(
    std::declval<TypeMapper<Args...>&>(), std::declval<Args&>()...))
  apply_mapper(
    TypeMapper<Args...>& mapper,
    const T& value,
    std::common_type_t<Args>... args)
  {
    return value.apply_mapper(mapper, args...);
  }

  template<typename... Ts, typename... Args>
  std::variant<Ts...> apply_mapper(
    TypeMapper<Args...>& mapper,
    std::variant<Ts...> values,
    std::common_type_t<Args>... args)
  {
    return std::visit(
      [&](const auto& v) -> std::variant<Ts...> {
        return apply_mapper(mapper, v, args...);
      },
      values);
  }

  TypeSignature
  apply_mapper(TypeMapper<>& mapper, const TypeSignature& signature);

  /*
   * By default, this mapper will return the input type unchanged. Subclasses of
   * RecursiveTypeMapper can provide individual visit_XXX to perform transforms
   * where desired.
   *
   * RecursiveTypeMapper uses the interner to create the types it returns, even
   * when these do not necessitate any modification. This has a useful
   * side-effect of interning the tree structure if it wasn't, which is how the
   * name-resolution pass interns the output of the parser.
   */
  class RecursiveTypeMapper : public TypeMapper<>, private TypeVisitor<TypePtr>
  {
  public:
    RecursiveTypeMapper(Context& context) : TypeMapper(context) {}

  protected:
    /**
     * Check whether the mapper modifies a type at all.
     *
     * This can be used to avoid unnecessary work, and made efficient by caching
     * some property about the type (eg. its free variables).
     */
    virtual bool modifies_type(const TypePtr& ty) const
    {
      return true;
    }

    TypePtr apply_one(const TypePtr& type) override;
    InferableTypeSequence
    apply_sequence(const InferableTypeSequence& seq) override;

    TypePtr visit_entity_type(const EntityTypePtr& ty) override;
    TypePtr visit_static_type(const StaticTypePtr& ty) override;
    TypePtr visit_type_parameter(const TypeParameterPtr& ty) override;
    TypePtr visit_capability(const CapabilityTypePtr& ty) override;
    TypePtr visit_apply_region(const ApplyRegionTypePtr& ty) override;
    TypePtr visit_unapply_region(const UnapplyRegionTypePtr& ty) override;
    TypePtr visit_union(const UnionTypePtr& ty) override;
    TypePtr visit_intersection(const IntersectionTypePtr& ty) override;
    TypePtr visit_unit_type(const UnitTypePtr& ty) override;
    TypePtr visit_infer(const InferTypePtr& ty) override;
    TypePtr visit_range_type(const RangeTypePtr& ty) override;
    TypePtr visit_viewpoint_type(const ViewpointTypePtr& ty) override;
    TypePtr visit_has_field_type(const HasFieldTypePtr& ty) override;
    TypePtr
    visit_delayed_field_view_type(const DelayedFieldViewTypePtr& ty) override;
    TypePtr visit_has_method_type(const HasMethodTypePtr& ty) override;
    TypePtr
    visit_has_applied_method_type(const HasAppliedMethodTypePtr& ty) override;
    TypePtr visit_is_entity_type(const IsEntityTypePtr& ty) override;
    TypePtr visit_fixpoint_type(const FixpointTypePtr& ty) override;
    TypePtr
    visit_fixpoint_variable_type(const FixpointVariableTypePtr& ty) override;
    TypePtr visit_entity_of_type(const EntityOfTypePtr& ty) override;
    TypePtr
    visit_variable_renaming_type(const VariableRenamingTypePtr& ty) override;
    TypePtr
    visit_path_compression_type(const PathCompressionTypePtr& ty) override;
    TypePtr visit_indirect_type(const IndirectTypePtr& ty) override;
    TypePtr visit_not_child_of_type(const NotChildOfTypePtr& ty) override;

    virtual InferableTypeSequence
    visit_sequence(const BoundedTypeSequence& seq);

    virtual InferableTypeSequence
    visit_sequence(const UnboundedTypeSequence& seq);
  };

  class Flatten : private RecursiveTypeMapper
  {
  public:
    Flatten(Context& context) : RecursiveTypeMapper(context) {}

    template<typename T>
    static T apply(Context& context, T value)
    {
      Flatten v(context);
      return v.apply(value);
    }

  private:
    using RecursiveTypeMapper::apply;

    TypePtr visit_range_type(const RangeTypePtr& ty) override;
    TypePtr visit_infer(const InferTypePtr& ty) override;
  };
}
