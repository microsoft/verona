// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "anf.h"

#include "ident.h"
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

      if (is_kind(expr, {Kind::Let, Kind::Var}))
        state_stack.back().lambda->symbol_table()->set(expr->location, expr);
    }

    void make_trivial()
    {
      auto e = expr();
      auto id = ident();

      if ((parent()->kind() == Kind::Assign) && (parent<Assign>()->left == e))
      {
        // (var $x) if this is an lvalue.
        auto var = std::make_shared<Var>();
        var->location = id;
        var->type = std::make_shared<InferType>();
        add(var);
      }
      else
      {
        // (let $x) if this is an rvalue.
        auto let = std::make_shared<Let>();
        let->location = id;
        let->type = std::make_shared<InferType>();
        add(let);
      }

      // (assign (ref $x) expr)
      auto ref = std::make_shared<Ref>();
      ref->location = id;

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

      if (!last())
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
      let->type = std::make_shared<InferType>();
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
      // Check if it's a local variable or parameter.
      auto def = state_stack.back().lambda->st.get(ref.location);

      if (!def)
      {
        // Insert a free variable declaration.
        auto fr = std::make_shared<Free>();
        fr->location = ref.location;
        add(fr);
      }

      // Add it to the ANF if it's top level.
      if (top())
      {
        add();

        if (!last())
        {
          error() << ref.location << "This is an unused reference."
                  << text(ref.location);
        }
      }
    }

    void post(Expr&)
    {
      make_trivial();
    }

    void pre(Lambda& lambda)
    {
      if (state_stack.empty())
        ident.hygienic = 0;

      state_stack.push_back({current<Lambda>(), {}});

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

        auto eq_tr = std::make_shared<TypeRef>();
        eq_tr->location = expr->location;
        eq_tr->typenames.push_back(eq);

        auto eq_sel = std::make_shared<Select>();
        eq_sel->location = expr->location;
        eq_sel->expr = expr;
        eq_sel->typeref = eq_tr;
        eq_sel->args = ref;

        auto req = std::make_shared<TypeName>();
        req->location = name_requires;

        auto req_tr = std::make_shared<TypeRef>();
        req_tr->location = expr->location;
        req_tr->typenames.push_back(req);

        auto req_sel = std::make_shared<Select>();
        req_sel->location = expr->location;
        req_sel->typeref = req_tr;
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

      if (
        !state_stack.empty() && !is_kind(parent(), {Kind::Param, Kind::Field}))
      {
        make_trivial();
      }
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

    void post(Oftype& oftype)
    {
      if (oftype.expr->kind() != Kind::Ref)
      {
        error() << oftype.expr->location << "Unexpected oftype LHS."
                << text(oftype.expr->location);
      }
    }

    void post(Throw& thr)
    {
      if (thr.expr->kind() != Kind::Ref)
      {
        error() << thr.expr->location << "Unexpected throw expression."
                << text(thr.expr->location);
      }
    }

    void post(Assign& asn)
    {
      if (asn.left->kind() != Kind::Ref)
      {
        error() << asn.left->location << "Unexpected assignment LHS."
                << text(asn.left->location);
      }

      if (!is_kind(
            asn.right,
            {Kind::Ref,
             Kind::Tuple,
             Kind::Select,
             Kind::New,
             Kind::ObjectLiteral,
             Kind::Lambda,
             Kind::Match,
             Kind::Try,
             Kind::When,
             Kind::Bool,
             Kind::Binary,
             Kind::Int,
             Kind::Hex,
             Kind::Float,
             Kind::Character,
             Kind::EscapedString,
             Kind::UnescapedString}))
      {
        error() << asn.right->location << "Unexpected assignment RHS."
                << text(asn.right->location);
      }
    }

    void post(Lambda& lambda)
    {
      for (auto& e : lambda.body)
      {
        if (!is_kind(
              e,
              {Kind::Ref,
               Kind::Assign,
               Kind::Let,
               Kind::Var,
               Kind::Free,
               Kind::Oftype,
               Kind::Throw}))
        {
          error() << e->location << "Unexpected expression at the top level."
                  << text(e->location);
        }
      }
    }
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
