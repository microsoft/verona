// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/traits.h"
#include "compiler/type.h"
#include "ds/error.h"

#include <iostream>

/**
 * Visitors for the Type representation.
 *
 * This follows the same patterns as the expression visitor.
 *
 * Each class exposes a public visit_foo(Foo) method, which will dispatch to
 * the appropriate virtual visit_XXX method based which concrete subclass of
 * Foo the argument is.
 *
 * Implementations of the visitors should either provide all the visit_XXX
 * virtual methods, or implement only some of them but provide a visit_base_foo
 * method as a default case. If a visit_XXX method is missing but no
 * visit_base_foo method is provided, the visitor will abort at runtime.
 *
 * The dispatch is done using dynamic_cast followed by static dispatch, rather
 * than dynamic dispatch through a vtable, because we want to be generic over
 * the arguments and return type of the visitor.
 *
 * By default, visitors does not recurse into children nodes. It is up to
 * each implementation to do so. For expression nodes, the
 * Expression::visit_children method will call visit_expr on each child
 * expression node. For types, the TypeMapper class from mapper.h can be used.
 */
namespace verona::compiler
{
  template<typename Return = void, typename... Args>
  class TypeVisitor
  {
  public:
    template<typename Ts>
    void visit_types(const Ts& types, Args... args)
    {
      static_assert(std::is_void_v<Return>);
      for (const auto& ty : types)
      {
        visit_type(ty, args...);
      }
    }

  public:
    virtual Return visit_type(const TypePtr& ty, Args... args)
    {
      if (auto ty_ = ty->dyncast<EntityType>())
      {
        return visit_entity_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<TypeParameter>())
      {
        return visit_type_parameter(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<StaticType>())
      {
        return visit_static_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<CapabilityType>())
      {
        return visit_capability(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<ApplyRegionType>())
      {
        return visit_apply_region(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<UnapplyRegionType>())
      {
        return visit_unapply_region(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<UnionType>())
      {
        return visit_union(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<IntersectionType>())
      {
        return visit_intersection(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<InferType>())
      {
        return visit_infer(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<UnitType>())
      {
        return visit_unit_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<RangeType>())
      {
        return visit_range_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<ViewpointType>())
      {
        return visit_viewpoint_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<HasFieldType>())
      {
        return visit_has_field_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<DelayedFieldViewType>())
      {
        return visit_delayed_field_view_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<HasMethodType>())
      {
        return visit_has_method_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<HasAppliedMethodType>())
      {
        return visit_has_applied_method_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<IsEntityType>())
      {
        return visit_is_entity_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<FixpointType>())
      {
        return visit_fixpoint_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<FixpointVariableType>())
      {
        return visit_fixpoint_variable_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<EntityOfType>())
      {
        return visit_entity_of_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<VariableRenamingType>())
      {
        return visit_variable_renaming_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<PathCompressionType>())
      {
        return visit_path_compression_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<IndirectType>())
      {
        return visit_indirect_type(ty_, std::forward<Args>(args)...);
      }
      else if (auto ty_ = ty->dyncast<NotChildOfType>())
      {
        return visit_not_child_of_type(ty_, std::forward<Args>(args)...);
      }
      else
      {
        // Silence a warning about typeid(*ty) having a potential side-effect
        const Type& ty_ref = *ty;
        InternalError::print(
          "TypeVisitor dispatch failed on {}\n", typeid(ty_ref).name());
      }
    }

  private:
    virtual Return visit_base_type(const TypePtr& ty, Args... args)
    {
      // Silence a warning about typeid(*ty) having a potential side-effect
      const Type& ty_ref = *ty;
      InternalError::print(
        "Unhandled case {} in visitor {}.\n",
        typeid(ty_ref).name(),
        typeid(*this).name());
    }
    virtual Return visit_entity_type(const EntityTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_type_parameter(const TypeParameterPtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_static_type(const StaticTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_capability(const CapabilityTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_apply_region(const ApplyRegionTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_unapply_region(const UnapplyRegionTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_union(const UnionTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_intersection(const IntersectionTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_infer(const InferTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_unit_type(const UnitTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_range_type(const RangeTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_viewpoint_type(const ViewpointTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_has_field_type(const HasFieldTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_delayed_field_view_type(
      const DelayedFieldViewTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_has_method_type(const HasMethodTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_has_applied_method_type(
      const HasAppliedMethodTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_is_entity_type(const IsEntityTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_fixpoint_type(const FixpointTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_fixpoint_variable_type(
      const FixpointVariableTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_entity_of_type(const EntityOfTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_variable_renaming_type(
      const VariableRenamingTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_path_compression_type(const PathCompressionTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return visit_indirect_type(const IndirectTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
    virtual Return
    visit_not_child_of_type(const NotChildOfTypePtr& ty, Args... args)
    {
      return visit_base_type(ty, std::forward<Args>(args)...);
    }
  };

  template<typename... Args>
  class RecursiveTypeVisitor : public TypeVisitor<void, Args...>
  {
  protected:
    void visit_entity_type(const EntityTypePtr& ty, Args... args) override
    {
      this->visit_types(ty->arguments, args...);
    }

    void visit_static_type(const StaticTypePtr& ty, Args... args) override
    {
      this->visit_types(ty->arguments, args...);
    }

    void visit_capability(const CapabilityTypePtr& ty, Args... args) override {}

    void visit_union(const UnionTypePtr& ty, Args... args) override
    {
      this->visit_types(ty->elements, args...);
    }

    void
    visit_intersection(const IntersectionTypePtr& ty, Args... args) override
    {
      this->visit_types(ty->elements, args...);
    }

    void visit_range_type(const RangeTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->lower, args...);
      this->visit_type(ty->upper, args...);
    }

    void visit_has_field_type(const HasFieldTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->view, args...);
      this->visit_type(ty->read_type, args...);
      this->visit_type(ty->write_type, args...);
    }

    void visit_delayed_field_view_type(
      const DelayedFieldViewTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->type, args...);
    }

    void visit_fixpoint_type(const FixpointTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->inner, args...);
    }

    void visit_entity_of_type(const EntityOfTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->inner, args...);
    }

    void visit_type_parameter(const TypeParameterPtr& ty, Args... args) override
    {}
    void visit_unit_type(const UnitTypePtr& ty, Args... args) override {}
    void visit_is_entity_type(const IsEntityTypePtr& ty, Args... args) override
    {}
    void visit_infer(const InferTypePtr& ty, Args... args) override {}
    void visit_fixpoint_variable_type(
      const FixpointVariableTypePtr& ty, Args... args) override
    {}

    void visit_apply_region(const ApplyRegionTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->type, args...);
    }

    void visit_viewpoint_type(const ViewpointTypePtr& ty, Args... args) override
    {
      for (const auto& v : ty->variables)
      {
        this->visit_type(v, args...);
      }
      this->visit_type(ty->right, args...);
    }

    void visit_variable_renaming_type(
      const VariableRenamingTypePtr& ty, Args... args) override
    {
      this->visit_type(ty->type, args...);
    }

    void visit_path_compression_type(
      const PathCompressionTypePtr& ty, Args... args) override
    {
      for (const auto& [v, replacement] : ty->compression)
      {
        this->visit_type(replacement, args...);
      }
      this->visit_type(ty->type, args...);
    }

    void visit_indirect_type(const IndirectTypePtr& ty, Args... args) override
    {}
  };
}
