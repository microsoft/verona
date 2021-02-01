// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/traits.h"

#include <fmt/ostream.h>
#include <iostream>

/**
 * Visitors for the AST.
 *
 * We define two visitors, one for expression nodes, the other one for types,
 * both following the same pattern.
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
  class ExprVisitor
  {
  public:
    virtual Return visit_expr(Expression& expr, Args... args)
    {
      if (auto expr_ = dynamic_cast<DefineLocalExpr*>(&expr))
      {
        return visit_define_local(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<SymbolExpr*>(&expr))
      {
        return visit_symbol(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<AssignLocalExpr*>(&expr))
      {
        return visit_assign_local(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<FieldExpr*>(&expr))
      {
        return visit_field(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<AssignFieldExpr*>(&expr))
      {
        return visit_assign_field(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<SeqExpr*>(&expr))
      {
        return visit_seq(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<CallExpr*>(&expr))
      {
        return visit_call(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<WhileExpr*>(&expr))
      {
        return visit_while(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<WhenExpr*>(&expr))
      {
        return visit_when(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<IfExpr*>(&expr))
      {
        return visit_if(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<BlockExpr*>(&expr))
      {
        return visit_block(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<EmptyExpr*>(&expr))
      {
        return visit_empty(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<MatchExpr*>(&expr))
      {
        return visit_match_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<NewExpr*>(&expr))
      {
        return visit_new_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<IntegerLiteralExpr*>(&expr))
      {
        return visit_integer_literal_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<StringLiteralExpr*>(&expr))
      {
        return visit_string_literal_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<ViewExpr*>(&expr))
      {
        return visit_view_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<BinaryOperatorExpr*>(&expr))
      {
        return visit_binary_operator_expr(*expr_, std::forward<Args>(args)...);
      }
      else
      {
        InternalError::print(
          "ExprVisitor dispatch failed on {}\n", typeid(expr).name());
      }
    }

  private:
    virtual Return visit_base_expr(Expression& expr, Args... args)
    {
      InternalError::print(
        "Unhandled case {} in visitor {}.\n",
        typeid(expr).name(),
        typeid(*this).name());
    }
    virtual Return visit_symbol(SymbolExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_field(FieldExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_assign_local(AssignLocalExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_assign_field(AssignFieldExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_seq(SeqExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_call(CallExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_when(WhenExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_while(WhileExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_if(IfExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_block(BlockExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_empty(EmptyExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_define_local(DefineLocalExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_match_expr(MatchExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_new_expr(NewExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return
    visit_integer_literal_expr(IntegerLiteralExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return
    visit_string_literal_expr(StringLiteralExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return
    visit_binary_operator_expr(BinaryOperatorExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
    virtual Return visit_view_expr(ViewExpr& expr, Args... args)
    {
      return visit_base_expr(expr, std::forward<Args>(args)...);
    }
  };

  template<typename Return = void, typename... Args>
  class TypeExpressionVisitor
  {
  public:
    virtual Return visit_type_expression(TypeExpression& te, Args... args)
    {
      if (auto expr_ = dynamic_cast<SymbolTypeExpr*>(&te))
      {
        return visit_symbol_type_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<UnionTypeExpr*>(&te))
      {
        return visit_union_type_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<IntersectionTypeExpr*>(&te))
      {
        return visit_intersection_type_expr(
          *expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<ViewpointTypeExpr*>(&te))
      {
        return visit_viewpoint_type_expr(*expr_, std::forward<Args>(args)...);
      }
      else if (auto expr_ = dynamic_cast<CapabilityTypeExpr*>(&te))
      {
        return visit_capability_type_expr(*expr_, std::forward<Args>(args)...);
      }
      else
      {
        InternalError::print(
          "TypeExpressionVisitor dispatch failed on {}\n", typeid(te).name());
      }
    }

  private:
    virtual Return visit_base_type_expression(TypeExpression& te, Args... args)
    {
      InternalError::print(
        "Unhandled case {} in visitor {}.\n",
        typeid(te).name(),
        typeid(*this).name());
    }
    virtual Return visit_symbol_type_expr(SymbolTypeExpr& te, Args... args)
    {
      return visit_base_type_expression(te, std::forward<Args>(args)...);
    }
    virtual Return
    visit_capability_type_expr(CapabilityTypeExpr& te, Args... args)
    {
      return visit_base_type_expression(te, std::forward<Args>(args)...);
    }
    virtual Return visit_union_type_expr(UnionTypeExpr& te, Args... args)
    {
      return visit_base_type_expression(te, std::forward<Args>(args)...);
    }
    virtual Return
    visit_intersection_type_expr(IntersectionTypeExpr& te, Args... args)
    {
      return visit_base_type_expression(te, std::forward<Args>(args)...);
    }
    virtual Return
    visit_viewpoint_type_expr(ViewpointTypeExpr& te, Args... args)
    {
      return visit_base_type_expression(te, std::forward<Args>(args)...);
    }
  };

  template<typename Return = void, typename... Args>
  class MemberVisitor
  {
  public:
    void visit_members(ASTList<Member>& members, Args... args)
    {
      static_assert(std::is_void_v<Return>);
      for (const auto& def : members)
      {
        visit_member(def.get(), args...);
      }
    }

    virtual Return visit_member(Member* member, Args... args)
    {
      if (auto method = member->get_as<Method>())
        return visit_method(method, std::forward<Args>(args)...);
      else if (auto field = member->get_as<Field>())
        return visit_field(field, std::forward<Args>(args)...);
      else
        abort();
    }

    virtual Return visit_method(Method* method, Args... args) = 0;
    virtual Return visit_field(Field* field, Args... args) = 0;
  };
}
