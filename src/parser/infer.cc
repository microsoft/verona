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

      // TODO: rewrite the node to be static or dynamic dispatch
      // include a precise reference to the selected function
      assert(!sel.typeref->resolved);
      auto call = call_type(sel.expr, sel.args);

      // TODO: apply on a functiontype receiver

      // Dynamic dispatch.
      if (dynamic_dispatch(sel, call))
        return;

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
      subtype(f, call);
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

    bool dynamic_dispatch(Select& sel, Node<FunctionType>& call)
    {
      if (!call || !call->left || (sel.typeref->typenames.size() != 1))
        return false;

      Ast receiver = receiver_type(call->left);
      auto& tn = sel.typeref->typenames.front();
      auto sym = receiver;
      auto def = lookup.member(sym, receiver, tn->location);

      if (!def)
        return false;

      lookup.substitutions(sel.typeref->subs, def, tn->typeargs);
      sel.typeref->context = sym;
      sel.typeref->def = def;

      if (sym->kind() == Kind::Class)
      {
        // TODO: we know the method statically
      }

      return check_dispatch_type(receiver, call, def, sel.typeref->subs, sel);
    }

    bool check_dispatch_type(
      Ast& receiver,
      Node<FunctionType>& call,
      Ast& def,
      Substitutions& subs,
      Select& sel)
    {
      switch (def->kind())
      {
        case Kind::Field:
        {
          // TODO: view and extract types
          // could have additional args, at which point it's an apply
          // call on the field
          error() << sel.location << "Fields not handled yet."
                  << text(sel.location);
          return false;
        }

        case Kind::Function:
        {
          auto& lambda = def->as<Function>().lambda->as<Lambda>();
          auto f = function_type(lambda);

          // TODO: don't use whole receiver type as self?
          // discard capabilities? specialise to the dispatch type?
          // but when we check subtyping, `receiver & dispatch-type` for the
          // first argument of `call`, instead of just `receiver`
          // return clone(subs, f, receiver);
          std::cerr << f << std::endl;
          std::cerr << call << std::endl;
          return subtype(f, call);
        }

        case Kind::LookupUnion:
        {
          auto& un = def->as<LookupUnion>();
          bool ok = true;

          for (auto& t : un.list)
            ok &= check_dispatch_type(receiver, call, t, subs, sel);

          return ok;
        }

        case Kind::LookupIsect:
        {
          auto& isect = def->as<LookupIsect>();
          subtype.show = false;
          size_t ok = 0;

          for (auto& t : isect.list)
          {
            if (check_dispatch_type(receiver, call, t, subs, sel))
              ok++;
          }

          subtype.show = true;

          if (ok == 0)
          {
            // Do it again and show error messages.
            for (auto& t : isect.list)
              check_dispatch_type(receiver, call, t, subs, sel);
          }

          return ok > 0;
        }

        default:
          return false;
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
