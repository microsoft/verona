// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ident.h"
#include "lookup.h"
#include "resolve.h"
#include "rewrite.h"

namespace verona::parser::anf
{
  struct ANF : Pass<ANF>
  {
    AST_PASS;

    struct State
    {
      Node<Lambda> lambda;
      List<Expr> anf;
      size_t hygienic;
    };

    std::vector<State> state_stack;
    Ident ident;

    void make_trivial()
    {
      // Do nothing if the expression occurs outside of a lambda.
      // TODO: these are initexprs, need to account for them somehow.
      if (state_stack.empty())
        return;

      auto& state = state_stack.back();
      auto expr = std::static_pointer_cast<Expr>(stack.back());

      // Don't change refs.
      if (expr->kind() == Kind::Ref)
        return;

      if (parent()->kind() == Kind::Lambda)
      {
        // The result of this expression isn't used. Add it as-is.
        state.anf.push_back(expr);
        return;
      }

      if (expr->kind() == Kind::Oftype)
      {
        // Lift oftype nodes and leave their contents in place.
        state.anf.push_back(expr);
        rewrite(stack, expr->as<Oftype>().expr);
        return;
      }

      if (is_kind(expr, {Kind::Let, Kind::Var}))
      {
        // Lift variable declarations and leave a reference in their place.
        state.anf.push_back(expr);
        auto ref = std::make_shared<Ref>();
        ref->location = expr->location;
        rewrite(stack, ref);
        return;
      }

      if (parent()->kind() == Kind::Assign)
      {
        // If this expression is the right-hand side of an assignment, leave
        // it in place.
        auto& asn = parent()->as<Assign>();

        if (asn.right == expr)
          return;
      }

      // Append (assign (let $x) expr) to the body.
      auto let = std::make_shared<Let>();
      let->location = ident();
      state.lambda->symbol_table()->set(let->location, let);

      auto asn = std::make_shared<Assign>();
      asn->location = expr->location;
      asn->left = let;
      asn->right = expr;
      state.anf.push_back(asn);

      // Replace expr with (ref $x)
      auto ref = std::make_shared<Ref>();
      ref->location = let->location;

      rewrite(stack, ref);
    }

    void post(Expr& expr)
    {
      make_trivial();
    }

    void pre(Lambda& lambda)
    {
      state_stack.push_back(
        {std::static_pointer_cast<Lambda>(stack.back()), {}, ident.hygienic});
      ident.hygienic = 0;
    }

    void post(Lambda& lambda)
    {
      auto& state = state_stack.back();
      lambda.body = state.anf;
      ident.hygienic = state.hygienic;
      state_stack.pop_back();
      make_trivial();
    }
  };

  bool run(Ast& ast, std::ostream& out)
  {
    ANF r;
    r.set_error(out);
    return r << ast;
  }

  struct WF : Pass<WF>
  {
    AST_PASS;
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
