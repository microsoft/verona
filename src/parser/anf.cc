// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "anf.h"

#include "ident.h"
#include "lookup.h"
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
    Location name_eq;
    Location name_requires;

    ANF()
    {
      name_eq = ident("==");
      name_requires = ident("requires");
    }

    void make_trivial()
    {
      auto& state = state_stack.back();
      auto expr = std::static_pointer_cast<Expr>(stack.back());

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
        rewrite(expr->as<Oftype>().expr);
        return;
      }

      if (is_kind(expr, {Kind::Let, Kind::Var}))
      {
        // Lift variable declarations and leave a reference in their place.
        state.anf.push_back(expr);
        auto ref = std::make_shared<Ref>();
        ref->location = expr->location;
        rewrite(ref);
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

      rewrite(ref);
    }

    void post(Param& param)
    {
      // No changes.
    }

    void post(Ref& ref)
    {
      // Do nothing if it's a top level reference. This removes it.
      if (parent()->kind() == Kind::Lambda)
        return;

      // Check if it's a local variable or parameter.
      auto& state = state_stack.back();
      auto def = state.lambda->st.get(ref.location);

      if (def)
        return;

      // Insert free variable declarations.
      auto defs = look_up(stack, ref.location);
      Node<Expr> fr;

      switch (defs.front().back()->kind())
      {
        case Kind::Param:
        case Kind::Let:
        {
          fr = std::make_shared<FreeLet>();
          break;
        }

        case Kind::Var:
        {
          fr = std::make_shared<FreeVar>();
          break;
        }

        default:
          return;
      }

      fr->location = ref.location;
      state.anf.push_back(fr);
      state.lambda->symbol_table()->set(fr->location, fr);
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

      // Turn patterns into parameters.
      auto& state = state_stack.back();

      for (auto& expr : lambda.params)
      {
        if (expr->kind() == Kind::Param)
          continue;

        auto param = std::make_shared<Param>();
        param->location = ident();
        lambda.symbol_table()->set(param->location, param);

        auto ref = std::make_shared<Ref>();
        ref->location = param->location;

        auto eq = std::make_shared<TypeName>();
        eq->location = name_eq;

        auto eq_sel = std::make_shared<Select>();
        eq_sel->expr = expr;
        eq_sel->location = expr->location;
        eq_sel->typenames.push_back(eq);
        eq_sel->args = ref;

        auto req = std::make_shared<TypeName>();
        req->location = name_requires;

        auto req_sel = std::make_shared<Select>();
        req_sel->location = expr->location;
        req_sel->typenames.push_back(req);
        req_sel->args = eq_sel;
        state.anf.push_back(req_sel);

        expr = param;
      }

      lambda.body.insert(
        lambda.body.begin(), state.anf.begin(), state.anf.end());
      state.anf.clear();
    }

    void post(Lambda& lambda)
    {
      auto& state = state_stack.back();
      lambda.body = state.anf;
      ident.hygienic = state.hygienic;
      state_stack.pop_back();

      if (!state_stack.empty())
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
