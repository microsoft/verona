// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

#include "ident.h"
#include "lookup.h"
#include "print.h"
#include "rewrite.h"

namespace verona::parser::resolve
{
  struct Resolve : Pass<Resolve>
  {
    AST_PASS;

    Ident ident;
    Location name_create = ident("create");
    Lookup lookup;

    Resolve() : lookup([this]() -> std::ostream& { return error(); }) {}

    void post(Using& use)
    {
      // The contained TypeRef node is fully resolved. Add it to the local
      // scope as a resolve target.
      parent()->symbol_table()->use.push_back(current<Using>());
    }

    void post(TypeRef& tr)
    {
      auto def = lookup.typeref(symbols(), tr);

      // Handle this in Select instead.
      if (parent()->kind() == Kind::Select)
        return;

      if (!def)
      {
        error() << tr.location << "Couldn't find a definition of this type."
                << text(tr.location);
        return;
      }

      if (!is_kind(
            def,
            {Kind::Class, Kind::Interface, Kind::TypeAlias, Kind::TypeParam}))
      {
        error() << tr.location << "Expected a type, but got a "
                << kindname(def->kind()) << text(tr.location) << def->location
                << "Definition is here" << text(def->location);
        return;
      }
    }

    void post(Select& select)
    {
      // TODO: multiple definitions of functions for arity-based overloading

      // If it's a single element name with any arguments, it can be a dynamic
      // member select.
      bool dynamic =
        (select.expr || select.args) && (select.typeref->typenames.size() == 1);

      // Find all definitions of the selector.
      auto def = select.typeref->def.lock();

      if (!def)
      {
        if (!dynamic)
        {
          error() << select.typeref->location
                  << "Couldn't find a definition for this."
                  << text(select.typeref->location);
        }
        return;
      }

      switch (def->kind())
      {
        case Kind::Class:
        case Kind::Interface:
        case Kind::TypeAlias:
        case Kind::TypeParam:
        {
          // We found a type as a selector, so we'll turn it into a constructor.
          auto create = std::make_shared<TypeName>();
          create->location = name_create;
          select.typeref->typenames.push_back(create);

          auto sym = select.typeref->context.lock();
          def = lookup.member(sym, def, name_create);

          select.typeref->context = select.typeref->def;
          select.typeref->def = def;

          if (!def || (def->kind() != Kind::Function))
          {
            error() << select.typeref->location
                    << "Couldn't find a create function for this."
                    << text(select.typeref->location);
            return;
          }

          // If this was a selector after a selector, rewrite it to be the
          // right-hand side of the previous selector.
          auto expr = select.expr;

          if (expr && (expr->kind() == Kind::Select))
          {
            auto& lhs = expr->as<Select>();

            if (!lhs.args)
            {
              auto sel = std::make_shared<Select>();
              sel->location = select.location;
              sel->typeref = select.typeref;
              sel->args = select.args;
              lhs.args = sel;
              rewrite(expr);
            }
          }
          break;
        }

        case Kind::Function:
          break;

        default:
        {
          if (!dynamic)
          {
            error() << select.typeref->location
                    << "Expected a function, but got a "
                    << kindname(def->kind()) << text(select.typeref->location)
                    << def->location << "Definition is here"
                    << text(def->location);
          }
          break;
        }
      }
    }

    void post(Tuple& tuple)
    {
      // Collapse unnecessary tuple nodes.
      if (tuple.seq.size() == 0)
        rewrite({});
      else if (tuple.seq.size() == 1)
        rewrite(tuple.seq.front());
    }

    void post(TupleType& tuple)
    {
      // Collapse unnecessary tuple type nodes.
      if (tuple.types.size() == 0)
        rewrite({});
      else if (tuple.types.size() == 1)
        rewrite(tuple.types.front());
    }

    void post(Function& func)
    {
      // Cache a function type for use in type checking.
      func.type = function_type(func.lambda->as<Lambda>());
    }

    void post(Self& self)
    {
      // If this occurs in a class, replace it with Class. If it occurs in
      // an interface, replace it with (Self & Interface).
      Ast type;

      for (auto it = stack.rbegin(); it != stack.rend(); ++it)
      {
        if (is_kind(*it, {Kind::Class, Kind::Interface}))
        {
          type = *it;
          break;
        }
      }

      if (!type)
      {
        error() << self.location << "No enclosing type." << text(self.location);
        return;
      }

      // Build a reference to the enclosing type.
      auto& iface = type->as<Interface>();
      auto tn = std::make_shared<TypeName>();
      tn->location = iface.location;

      for (auto& tp : iface.typeparams)
      {
        if (tp->kind() == Kind::TypeParamList)
        {
          auto tl = std::make_shared<TypeList>();
          tl->location = tp->location;
          tn->typeargs.push_back(tl);
        }
        else
        {
          auto tna = std::make_shared<TypeName>();
          tna->location = tp->location;

          auto ta = std::make_shared<TypeRef>();
          ta->location = self.location;
          ta->typenames.push_back(tna);
          lookup.typeref(symbols(), *ta);

          tn->typeargs.push_back(ta);
        }
      }

      auto tr = std::make_shared<TypeRef>();
      tr->location = self.location;
      tr->typenames.push_back(tn);

      lookup.typeref(symbols(), *tr);

      if (type->kind() == Kind::Class)
      {
        // Replace Self with a reference to the enclosing type.
        rewrite(tr);
      }
      else
      {
        // Replace Self with an intersection of Self and the enclosing type.
        assert(current()->kind() == Kind::Self);
        auto p = parent();

        if (p->kind() == Kind::IsectType)
        {
          p->as<IsectType>().types.push_back(tr);
        }
        else
        {
          auto isect = std::make_shared<IsectType>();
          isect->location = self.location;
          isect->types.push_back(tr);
          isect->types.push_back(current<Self>());
          rewrite(isect);
        }
      }
    }
  };

  bool run(Ast& ast, std::ostream& out)
  {
    Resolve r;
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
