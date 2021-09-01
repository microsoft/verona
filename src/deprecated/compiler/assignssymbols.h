// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "recursive_visitor.h"

#include <map>
#include <tuple>
#include <unordered_set>

namespace verona::compiler
{
  /*
   *  This class provides a memoised lookup for the following query
   *    bool assigns(Expression* expr, LocalID local)
   *  Does the the expression expr define `local`?
   *
   *  It uses a recursive visitor pattern, and will visit each
   *  syntax node at most once. If queries are only top down.
   */
  class ExprAssignsSymbol : private RecursiveExprVisitor<>
  {
  private:
    std::set<std::pair<Expression*, LocalID>> cache;
    std::unordered_set<Expression*> explored;
    std::vector<Expression*> stack;

    void visit_define_local(DefineLocalExpr& expr)
    {
      if (expr.right)
      {
        for (auto v : stack)
        {
          cache.insert({v, expr.local});
        }
      }
      RecursiveExprVisitor<>::visit_define_local(expr);
    }

    void visit_assign_local(AssignLocalExpr& expr)
    {
      for (auto v : stack)
      {
        cache.insert({v, expr.local});
      }
      RecursiveExprVisitor<>::visit_assign_local(expr);
    }

    void visit_while(WhileExpr& expr)
    {
      stack.push_back(&expr);
      RecursiveExprVisitor<>::visit_while(expr);
      stack.pop_back();
      explored.insert(&expr);
    }

    void visit_if(IfExpr& expr)
    {
      stack.push_back(&expr);
      RecursiveExprVisitor<>::visit_if(expr);
      stack.pop_back();
      explored.insert(&expr);
    }

  public:
    void precalc(Expression* expr)
    {
      if (explored.find(expr) == explored.end())
      {
        visit_expr(*expr);
      }
    }

    bool assigns(Expression* expr, LocalID local)
    {
      if (explored.find(expr) == explored.end())
        abort();

      bool result = cache.find({expr, local}) != cache.end();
      return result;
    }
  };
}
