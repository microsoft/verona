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

    Node<Expr> expr()
    {
      return current<Expr>();
    }

    bool top()
    {
      return parent()->kind() == Kind::Lambda;
    }

    bool last()
    {
      return parent<Lambda>()->body.back() == expr();
    }

    void add()
    {
      add(expr());
    }

    void add(Node<Expr> expr)
    {
      state_stack.back().anf.push_back(expr);

      if (is_kind(expr, {Kind::Let, Kind::Var, Kind::FreeLet, Kind::FreeVar}))
        state_stack.back().lambda->symbol_table()->set(expr->location, expr);
    }

    void make_trivial()
    {
      // Append (let $x) (assign (ref $x) expr) to the body.
      auto let = std::make_shared<Let>();
      let->location = ident();
      add(let);

      auto ref = std::make_shared<Ref>();
      ref->location = let->location;

      auto e = expr();
      auto asn = std::make_shared<Assign>();
      asn->location = e->location;
      asn->left = ref;
      asn->right = e;
      add(asn);

      // Replace expr with (ref $x)
      if (!top())
        rewrite(ref);
    }

    void post(Param& param)
    {
      // No changes.
    }

    void post(Let& let)
    {
      // This handles Let and Var.
      add();

      // This is already ANF.
      if (top())
        return;

      // Lift the variable declaration and leave a reference in it's place.
      auto ref = std::make_shared<Ref>();
      ref->location = let.location;
      rewrite(ref);
    }

    void post(Oftype& oftype)
    {
      // Lift oftype nodes.
      add();

      // Leave the contents in place if it isn't a top level node.
      if (!top())
        rewrite(oftype.expr);
    }

    void post(Throw& thr)
    {
      if (!top())
      {
        error() << thr.location
                << "A throw can't be used as an intermediate value."
                << text(thr.location);
        return;
      }
      else if (!last())
      {
        error() << thr.location
                << "A throw must be the last expression in a lambda."
                << text(thr.location);
        return;
      }

      add();
    }

    void post(Assign& asn)
    {
      if (top() && !last())
      {
        // Keep this assign at the top level.
        add();
        return;
      }

      // (assign (ref x) e) ->
      // (let $0)
      // (assign (ref $0) (ref x))
      // (assign (ref x) e)
      // rewrite (ref $0)
      auto let = std::make_shared<Let>();
      let->location = ident();
      add(let);

      auto ref = std::make_shared<Ref>();
      ref->location = let->location;

      auto asn2 = std::make_shared<Assign>();
      asn2->location = asn.location;
      asn2->left = ref;
      asn2->right = asn.left;
      add(asn2);

      add();

      if (last())
        add(ref);
      else
        rewrite(ref);
    }

    void post(Ref& ref)
    {
      if (top() && !last())
      {
        error() << ref.location << "This is an unused reference."
                << text(ref.location);
        return;
      }

      // Check if it's a local variable or parameter.
      auto def = state_stack.back().lambda->st.get(ref.location);

      if (!def)
      {
        // Insert a free variable declaration.
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
        add(fr);
      }

      // Add it to the ANF if it's top level.
      if (top())
        add();
    }

    void post(Expr&)
    {
      make_trivial();
    }

    void pre(Lambda& lambda)
    {
      state_stack.push_back({current<Lambda>(), {}, ident.hygienic});
      ident.hygienic = 0;

      // Turn patterns into parameters.
      for (auto& expr : lambda.params)
      {
        if (expr->kind() == Kind::Param)
          continue;

        auto param = std::make_shared<Param>();
        param->location = ident();
        lambda.symbol_table()->set(param->location, param);
        expr = param;

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
        add(req_sel);
      }

      auto& state = state_stack.back();
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

      if (!lambda.body.empty())
      {
        auto& expr = lambda.body.back();

        switch (expr->kind())
        {
          case Kind::Ref:
          case Kind::Throw:
            break;

          case Kind::Oftype:
          {
            lambda.body.push_back(expr->as<Oftype>().expr);
            break;
          }

          case Kind::Assign:
          {
            lambda.body.push_back(expr->as<Assign>().left);
            break;
          }

          default:
          {
            error() << expr->location << "This is not a valid return expression"
                    << text(expr->location);
            break;
          }
        }
      }

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
