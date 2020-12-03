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
}
