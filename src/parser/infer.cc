// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "infer.h"

#include "dnf.h"
#include "ident.h"
#include "lookup.h"
#include "print.h"
#include "rewrite.h"

namespace verona::parser::infer
{
  struct Infer : Pass<Infer>
  {
    AST_PASS;

    Ident ident;
    Location name_imm = ident("imm");
    Location name_bool = ident("Bool");
    Location name_int = ident("Integer");
    Location name_float = ident("Float");

    Node<Imm> type_imm;
    Node<Type> type_bool;
    Node<Type> type_int;
    Node<Type> type_float;

    std::unordered_multimap<Node<Type>, Node<Type>> con;

    Infer()
    {
      type_imm = std::make_shared<Imm>();
      type_imm->location = name_imm;

      type_bool = make_constant_type(name_bool);
      type_int = make_constant_type(name_int);
      type_float = make_constant_type(name_float);
    }

    Node<Type> make_constant_type(const Location& name)
    {
      auto n = std::make_shared<TypeName>();
      n->location = name;

      auto t = std::make_shared<TypeRef>();
      t->location = name;
      t->typenames.push_back(n);

      auto i = std::make_shared<IsectType>();
      i->types.push_back(t);
      i->types.push_back(type_imm);

      return i;
    }

    Node<Let> g(const Location& name)
    {
      return std::static_pointer_cast<Let>(lookup::name(symbols(), name));
    }

    Location lhs()
    {
      return parent<Assign>()->left->location;
    }

    void constraint(Node<Type> lhs, Node<Type> rhs)
    {
      if (!lhs || !rhs)
        return;

      auto [cur, end] = con.equal_range(lhs);

      for (; cur != end; ++cur)
      {
        if (cur->second == rhs)
          return;
      }

      // lhs <: rhs
      // TODO:
      // std::cout << "Constraint: " << lhs << " <: " << rhs << std::endl;
      con.insert({lhs, rhs});
    }

    Node<FunctionType> function_type(Lambda& lambda)
    {
      auto f = std::make_shared<FunctionType>();
      f->right = lambda.result;

      if (lambda.params.size() == 1)
      {
        f->left = lambda.params.front()->as<Param>().type;
      }
      else if (lambda.params.size() > 1)
      {
        auto t = std::make_shared<TupleType>();
        f->left = t;

        for (auto& p : lambda.params)
          t->types.push_back(p->as<Param>().type);
      }

      return f;
    }

    void post(Free& fr)
    {
      auto l = g(fr.location);

      if (!l->assigned)
      {
        error() << fr.location
                << "Free variables can't be captured if they haven't been "
                   "assigned to."
                << text(fr.location) << l->location << "Definition is here."
                << text(l->location);
      }
    }

    void post(Ref& ref)
    {
      // Allow an unassigned ref in an Oftype node.
      if (parent()->kind() == Kind::Oftype)
        return;

      auto l = g(ref.location);

      if (parent()->kind() == Kind::Assign)
      {
        auto& asn = parent()->as<Assign>();

        // Allow an unassigned ref on the left-hand side of an assignment.
        if (asn.left == current<Expr>())
          return;

        constraint(l->type, g(asn.left->location)->type);
      }
      else if (parent()->kind() == Kind::Lambda)
      {
        constraint(l->type, parent<Lambda>()->result);
      }

      if (!l->assigned)
      {
        error() << ref.location << "Variable used before assignment"
                << text(ref.location);
      }
    }

    void post(Oftype& oftype)
    {
      constraint(g(oftype.expr->location)->type, oftype.type);
    }

    void post(Throw& thr)
    {
      auto t = dnf::throwtype(g(thr.expr->location)->type);
      constraint(t, parent<Lambda>()->result);
    }

    void post(Assign& asn)
    {
      auto l = g(asn.left->location);

      if (!l->assigned || (l->kind() == Kind::Var))
      {
        l->assigned = true;
      }
      else
      {
        error() << asn.right->location << "This expression can't be assigned"
                << text(asn.right->location) << asn.left->location
                << "This local has already been assigned to"
                << text(asn.left->location);
      }
    }

    void post(Tuple& tuple)
    {
      auto t = std::make_shared<TupleType>();

      for (auto& e : tuple.seq)
        t->types.push_back(g(e->location)->type);

      g(lhs())->type = t;
    }

    void post(Select& sel)
    {
      // TODO: a select with a result that is always a throw should only be
      // allowed at the end of a lambda

      auto def = lookup::typenames(symbols(), sel.typenames);

      if (def)
      {
        // TODO: argument constraints, typearg constraints
        // auto& lambda = def->as<Function>().lambda->as<Lambda>();
        // auto f = function_type(lambda);
        // g(lhs())->type = f->right;
        // return;
      }

      // TODO: dynamic dispatch
    }

    void post(New& nw)
    {
      // TODO:
    }

    void post(ObjectLiteral& obj)
    {
      // TODO:
    }

    void post(Match& match)
    {
      // TODO:
    }

    void post(When& when)
    {
      // TODO:
    }

    void post(EscapedString& s)
    {
      // TODO:
    }

    void post(Int& i)
    {
      constraint(g(lhs())->type, type_int);
    }

    void post(Float& f)
    {
      constraint(g(lhs())->type, type_float);
    }

    void post(Bool& b)
    {
      constraint(g(lhs())->type, type_bool);
    }

    void post(Lambda& lambda)
    {
      switch (parent()->kind())
      {
        case Kind::Assign:
        {
          constraint(function_type(lambda), g(lhs())->type);
          break;
        }

        case Kind::Param:
        {
          constraint(lambda.result, parent<Param>()->type);
          break;
        }

        default:
        {
          // Do nothing.
          break;
        }
      }
    }
  };

  bool run(Ast& ast, std::ostream& out)
  {
    Infer r;
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
