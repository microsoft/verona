// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/visitor.h"

namespace verona::compiler
{
  template<typename... Args>
  class RecursiveExprVisitor : public ExprVisitor<void, Args...>
  {
  protected:
    void visit_field(FieldExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.expr, args...);
    }
    void visit_assign_local(AssignLocalExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.right, args...);
    }
    void visit_assign_field(AssignFieldExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.expr, args...);
      this->visit_expr(*expr.right, args...);
    }
    void visit_seq(SeqExpr& expr, Args... args) override
    {
      for (const auto& e : expr.elements)
      {
        this->visit_expr(*e, args...);
      }
      this->visit_expr(*expr.last, args...);
    }
    void visit_call(CallExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.receiver, args...);
      for (const auto& arg : expr.arguments)
      {
        this->visit_expr(*arg->inner, args...);
      }
    }
    void visit_when(WhenExpr& expr, Args... args) override
    {
      for (const auto& cown : expr.cowns)
      {
        if (auto argument = dynamic_cast<WhenArgumentAs*>(cown.get()))
        {
          this->visit_expr(*argument->inner, args...);
        }
      }
      this->visit_expr(*expr.body, args...);
    }
    void visit_while(WhileExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.condition, args...);
      this->visit_expr(*expr.body, args...);
    }
    void visit_if(IfExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.condition, args...);
      this->visit_expr(*expr.then_block, args...);
      if (expr.else_block)
      {
        this->visit_expr(*expr.else_block->body, args...);
      }
    }
    void visit_block(BlockExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.inner, args...);
    }
    void visit_define_local(DefineLocalExpr& expr, Args... args) override
    {
      if (expr.right)
      {
        this->visit_expr(*expr.right, args...);
      }
    }
    void visit_match_expr(MatchExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.expr, args...);
      for (auto& arm : expr.arms)
      {
        this->visit_expr(*arm->expr, args...);
      }
    }
    void visit_view_expr(ViewExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.expr, args...);
    }
    void visit_new_expr(NewExpr& expr, Args... args) override {}

    void visit_symbol(SymbolExpr& expr, Args... args) override {}
    void visit_empty(EmptyExpr& expr, Args... args) override {}
    void
    visit_integer_literal_expr(IntegerLiteralExpr& expr, Args... args) override
    {}
    void
    visit_string_literal_expr(StringLiteralExpr& expr, Args... args) override
    {}
    void
    visit_binary_operator_expr(BinaryOperatorExpr& expr, Args... args) override
    {
      this->visit_expr(*expr.left, args...);
      this->visit_expr(*expr.right, args...);
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

    void visit_string_type(const StringTypePtr& ty, Args... args) override {}
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
