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

    Infer() : lookup([this]() -> std::ostream& { return error(); })
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

    Node<Type> args_type(Node<Expr>& lhs, Node<Expr>& rhs)
    {
      if (!lhs && !rhs)
        return {};

      if (lhs && !rhs)
        return g(lhs->location)->type;

      if (!lhs && rhs)
        return g(rhs->location)->type;

      auto lt = g(lhs->location)->type;
      auto rt = g(rhs->location)->type;

      if (!lt)
        return rt;

      if (!rt)
        return lt;

      auto t = std::make_shared<TupleType>();
      t->location = lt->location;
      unpack_type(t, lt);
      unpack_type(t, rt);
      return t;
    }

    Node<Type> receiver_type(Node<Type>& args)
    {
      if (!args)
        return {};

      if (args->kind() == Kind::TupleType)
        return args->as<TupleType>().types.front();

      return args;
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

    void post(TypeRef& tr)
    {
      // Type arguments must be a subtype of the type parameter upper bounds.
      // TODO: not in a dynamic dispatch node
      for (auto& [wparam, arg] : tr.subs)
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
        subtype(l->type, parent<Lambda>()->result);
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

      // TODO: rewrite the node to be static or dynamic dispatch
      // include a precise reference to the selected function
      assert(!sel.typeref->resolved);
      auto args = args_type(sel.expr, sel.args);

      // TODO: apply on a functiontype receiver

      // Dynamic dispatch.
      if (args && (sel.typeref->typenames.size() == 1))
      {
        // TODO: lookup on receiver
        // need to check infertype
        Ast receiver = receiver_type(args);
        Ast def;

        if (receiver->kind() == Kind::InferType)
        {
          auto find = subtype.bounds.find(receiver);

          if (find == subtype.bounds.end())
          {
            error() << sel.location
                    << "Can't look this up, no bounds for the receiver."
                    << text(sel.location);
            return;
          }

          for (auto& upper : find->second.upper)
          {
            // TODO: union of parameters, isect of results
            Ast context = upper;
            def = lookup.member(
              context, upper, sel.typeref->typenames.front()->location);
          }

          for (auto& lower : find->second.lower)
          {
            // TODO: isect of parameters, union of results
            // must be present in every lower bound
            Ast context = lower;
            def = lookup.member(
              context, lower, sel.typeref->typenames.front()->location);
          }
        }
        else
        {
          Ast context = receiver;
          def = lookup.member(
            context, receiver, sel.typeref->typenames.front()->location);

          if (context->kind() == Kind::Class)
          {
            // TODO: use static dispatch
          }
        }

        lookup.substitutions(
          sel.typeref->subs, def, sel.typeref->typenames.front()->typeargs);

        if (def)
        {
          sel.typeref->context = receiver;
          sel.typeref->def = def;

          switch (def->kind())
          {
            case Kind::Field:
            {
              // TODO: view and extract types
              // could have additional args, at which point it's an apply
              // call on the field
              error() << sel.location << "Fields not handled yet."
                      << text(sel.location);
              return;
            }

            case Kind::Function:
            {
              auto self = clone(sel.typeref->subs, receiver);
              auto f = function_type(def->as<Function>().lambda->as<Lambda>());
              f = clone(sel.typeref->subs, f, self);
              sel.typeref->resolved = f;

              g(lhs())->type = f->right;
              subtype(args, f->left);
              return;
            }

            default:
              break;
          }
        }
      }

      // Static dispatch.
      auto def = lookup.typeref(symbols(), sel.typeref->as<TypeRef>());

      if (!def || (def->kind() != Kind::Function))
      {
        error() << sel.location << "Couldn't find this function."
                << text(sel.location);
        return;
      }

      // Resolve the static function, with substitutions and a Self type.
      auto self = clone(sel.typeref->subs, sel.typeref->context.lock());
      auto f = function_type(def->as<Function>().lambda->as<Lambda>());
      f = clone(sel.typeref->subs, f, self);
      sel.typeref->resolved = f;

      g(lhs())->type = f->right;
      subtype(args, f->left);
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
          subtype(lambda.result, parent<Field>()->type);
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
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
