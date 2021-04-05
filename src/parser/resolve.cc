// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

#include "ident.h"
#include "lookup.h"
#include "rewrite.h"

namespace verona::parser::resolve
{
  struct Resolve : Pass<Resolve>
  {
    AST_PASS;

    Ident ident;
    Location name_create = ident("create");

    void post(Using& use)
    {
      // The contained TypeRef node is fully resolved. Add it to the local
      // scope as a resolve target.
      parent()->symbol_table()->use.push_back(current<Using>());
    }

    void post(TypeRef& tr)
    {
      auto def = lookup::typenames(symbols(), tr.typenames);

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
      }
    }

    void post(Select& select)
    {
      // TODO: multiple definitions of functions for arity-based overloading

      // If it's a single element name with any arguments, it can be a dynamic
      // member select.
      bool dynamic =
        (select.expr || select.args) && (select.typenames.size() == 1);

      // Find all definitions of the selector.
      auto def = lookup::typenames(symbols(), select.typenames);

      if (!def)
      {
        if (!dynamic)
        {
          error() << select.typenames.front()->location
                  << "Couldn't find a definition for this."
                  << text(select.typenames.front()->location);
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
          select.typenames.push_back(create);

          // Resolve this again as a function.
          lookup::reset(select.typenames);
          def = lookup::typenames(symbols(), select.typenames);

          if (!def || (def->kind() != Kind::Function))
          {
            error() << select.typenames.front()->location
                    << "Couldn't find a create function for this."
                    << text(select.typenames.front()->location);
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
              sel->typenames = select.typenames;
              sel->args = select.args;
              lhs.args = sel;
              rewrite(expr);
            }
          }
          break;
        }

        case Kind::Field:
        case Kind::Function:
          break;

        default:
        {
          if (!dynamic)
          {
            error() << select.typenames.front()->location
                    << "Expected a field or a function, but got a "
                    << kindname(def->kind())
                    << text(select.typenames.front()->location) << def->location
                    << "Definition is here" << text(def->location);
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
