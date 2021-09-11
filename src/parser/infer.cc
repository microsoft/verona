// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "infer.h"

#include "dnf.h"
#include "ident.h"
#include "lookup.h"
#include "print.h"
#include "rewrite.h"
#include "subtype.h"

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

    Subtype subtype;
    Lookup lookup;

    Infer()
    : lookup([this]() -> std::ostream& { return error(); }, &subtype.bounds)
    {
      type_imm = std::make_shared<Imm>();
      type_imm->location = name_imm;
      subtype.name_apply = ident("apply");
    }

    Node<Type> make_constant_type(const Location& name)
    {
      auto n = std::make_shared<TypeName>();
      n->location = name;

      auto t = std::make_shared<TypeRef>();
      t->location = name;
      t->typenames.push_back(n);

      if (!lookup.typeref(symbols(), *t))
        return {};

      auto i = std::make_shared<IsectType>();
      i->location = name;
      i->types.push_back(t);
      i->types.push_back(type_imm);

      return i;
    }

    Node<Let> g(const Location& name)
    {
      return std::static_pointer_cast<Let>(
        symbols()->symbol_table()->get_scope(name));
    }

    Location lhs()
    {
      return parent<Assign>()->left->location;
    }

    void unpack_type(Node<TupleType>& to, Node<Type>& from)
    {
      if (!from)
        return;

      if (from->kind() == Kind::TupleType)
      {
        auto t = from->as<TupleType>();

        for (auto& e : t.types)
          to->types.push_back(e);
      }
      else
      {
        to->types.push_back(from);
      }
    }

    Node<FunctionType> call_type(Node<Expr>& left, Node<Expr>& right)
    {
      auto f = std::make_shared<FunctionType>();

      if (left && !right)
      {
        f->left = g(left->location)->type;
      }
      else if (!left && right)
      {
        f->left = g(right->location)->type;
      }
      else if (left && right)
      {
        auto lt = g(left->location)->type;
        auto rt = g(right->location)->type;
        assert(lt && rt);

        auto t = std::make_shared<TupleType>();
        t->location = lt->location;
        unpack_type(t, lt);
        unpack_type(t, rt);
        f->left = t;
      }

      f->right = g(lhs())->type;
      return f;
    }

    Node<Expr> receiver_expr(Node<Expr>& left, Node<Expr>& right)
    {
      if (left)
      {
        if (left->kind() == Kind::Tuple)
          return left->as<Tuple>().seq.front();

        return left;
      }

      if (right)
      {
        if (right->kind() == Kind::Tuple)
          return right->as<Tuple>().seq.front();

        return right;
      }

      return {};
    }

    Node<Expr> call_args(Node<Expr>& left, Node<Expr>& right)
    {
      auto args = std::make_shared<Tuple>();

      if (left)
      {
        args->location = left->location;

        if (left->kind() == Kind::Tuple)
        {
          for (auto e : left->as<Tuple>().seq)
            args->seq.push_back(e);
        }
        else
        {
          args->seq.push_back(left);
        }
      }

      if (right)
      {
        args->location.extend(right->location);

        if (right->kind() == Kind::Tuple)
        {
          for (auto e : right->as<Tuple>().seq)
            args->seq.push_back(e);
        }
        else
        {
          args->seq.push_back(right);
        }
      }

      return args;
    }

    Node<Expr> call_receiver(Node<Expr>& args)
    {
      assert(args->kind() == Kind::Tuple);
      auto t = args->as<Tuple>();

      if (t.seq.empty())
        return {};

      auto r = t.seq.front();
      t.seq.erase(t.seq.begin());
      return r;
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

    void post(LookupRef& find)
    {
      // Type arguments must be a subtype of the type parameter upper bounds.
      for (auto& [wparam, arg] : find.subs)
      {
        auto param = wparam.lock();

        if (param)
          subtype(arg, param->upper);
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

        subtype(l->type, g(asn.left->location)->type);
      }
      else if (parent()->kind() == Kind::Lambda)
      {
        auto& type = parent<Lambda>()->result;

        if (!subtype(l->type, type))
        {
          error() << ref.location
                  << "The return value is not a subtype of the result type."
                  << text(ref.location) << type->location
                  << "The result type is here." << text(type->location);
        }
      }

      if (!l->assigned)
      {
        error() << ref.location << "Variable used before assignment"
                << text(ref.location);
      }
    }

    void post(Oftype& oftype)
    {
      subtype(g(oftype.expr->location)->type, oftype.type);
    }

    void post(Throw& thr)
    {
      auto t = dnf::throwtype(g(thr.expr->location)->type);
      subtype(t, parent<Lambda>()->result);
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
      t->location = tuple.location;

      for (auto& e : tuple.seq)
        t->types.push_back(g(e->location)->type);

      g(lhs())->type = t;
    }

    void post(Select& sel)
    {
      // TODO: a select with a result that is always a throw should only be
      // allowed at the end of a lambda

      auto call = call_type(sel.expr, sel.args);

      // TODO: `apply` on a functiontype receiver

      if (call && call->left && (sel.typeref->typenames.size() == 1))
      {
        // Dynamic dispatch.
        auto rt = receiver_type(call->left);
        auto find = lookup.member(rt, sel.typeref->typenames.front());

        // A->B <: C->D <=> C <: A /\ B <: D
        // The member(s) we find must be a subtype of the call, not the other
        // way around. This means they are substitutable for the call. If the
        // call has an inferred result, it will receive a lower bound, which is
        // what we want. Each LookupRef in `find` modifies the receiver in
        // `call` to be `receiver & lookupref.self`.
        if (find && subtype.dynamic(find, call))
        {
          // Rewrite as dynamic dispatch.
          // TODO: can be static if `rt` is a concrete type
          auto dc = std::make_shared<DynamicCall>();
          dc->location = sel.location;
          dc->lookup = find;
          dc->args = call_args(sel.expr, sel.args);
          dc->receiver = call_receiver(dc->args);
          rewrite(dc);
          return;
        }
      }

      // Static dispatch.
      auto find = lookup.typeref(symbols(), *sel.typeref);

      if (!find)
      {
        error() << sel.location << "Couldn't find this function."
                << text(sel.location);
        return;
      }

      if (find->kind() != Kind::LookupRef)
      {
        error() << sel.location << "Expected a function but found "
                << kindname(find->kind()) << text(sel.location);
        return;
      }

      subtype(find, call);

      // Rewrite as static dispatch.
      auto sc = std::make_shared<StaticCall>();
      sc->location = sel.location;
      sc->lookup = find;
      sc->args = call_args(sel.expr, sel.args);
      rewrite(sc);
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
      auto t = make_constant_type(name_int);

      if (!t)
      {
        error() << i.location << "No type Integer in scope."
                << text(i.location);
        return;
      }

      subtype(g(lhs())->type, t);
    }

    void post(Float& f)
    {
      auto t = make_constant_type(name_float);

      if (!t)
      {
        error() << f.location << "No type Float in scope." << text(f.location);
        return;
      }

      subtype(g(lhs())->type, t);
    }

    void post(Bool& b)
    {
      auto t = make_constant_type(name_bool);

      if (!t)
      {
        error() << b.location << "No type Bool in scope." << text(b.location);
        return;
      }

      subtype(t, g(lhs())->type);
    }

    void post(Lambda& lambda)
    {
      switch (parent()->kind())
      {
        case Kind::Assign:
        {
          subtype(function_type(lambda), g(lhs())->type);
          break;
        }

        case Kind::Param:
        {
          assert(lambda.typeparams.size() == 0);
          assert(lambda.params.size() == 0);
          // TODO: check that there is some instantiation of the param type
          // that this default argument would satisfy. This isn't necessary for
          // soundness, but it would produce a useful early error message.
          // subtype(lambda.result, parent<Param>()->type);
          break;
        }

        case Kind::Field:
        {
          assert(lambda.typeparams.size() == 0);
          assert(lambda.params.size() == 0);
          auto& type = parent<Field>()->type;

          if (!subtype(lambda.result, type))
          {
            error()
              << lambda.location
              << "The field initialiser is not a subtype of the field type."
              << text(lambda.location) << type->location
              << "Field type is here." << text(type->location);
          }
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
    r.subtype.set_error(out);
    r << ast;
    return r && r.subtype;
  }

  struct WF : Pass<WF>
  {
    AST_PASS;

    void post(InferType& infer)
    {
      // TODO:
      // error() << infer.location << "Unresolved type." <<
      // text(infer.location);
    }

    void post(Select& sel)
    {
      error() << sel.location << "Unresolved select." << text(sel.location);
    }
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
