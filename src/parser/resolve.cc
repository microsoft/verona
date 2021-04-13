// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

#include "ident.h"
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
      auto def = typeref(symbols(), tr);

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
          def = def->symbol_table()->get(name_create);
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

    Ast typeref(Ast sym, TypeRef& tr)
    {
      // Each element will have a definition. This will point to a Class,
      // Interface, TypeAlias, Field, or Function.
      auto def = tr.def.lock();

      if (def)
        return def;

      // Lookup the first element in the current symbol context.
      def = name(sym, tr.typenames.front()->location);

      if (!def)
        return {};

      substitutions(tr.subs, def, tr.typenames.front()->typeargs);

      // Look in the current definition for the next name.
      for (size_t i = 1; i < tr.typenames.size(); i++)
      {
        def = member(sym, def, tr.typenames.at(i)->location);
        substitutions(tr.subs, def, tr.typenames.at(i)->typeargs);
      }

      tr.def = def;
      return def;
    }

    // This looks up `name` in the lexical scope, following `using` statements.
    Ast name(Ast sym, const Location& name)
    {
      while (sym)
      {
        auto st = sym->symbol_table();
        assert(st != nullptr);

        auto def = st->get(name);

        if (def)
          return def;

        for (auto it = st->use.rbegin(); it != st->use.rend(); ++it)
        {
          auto& use = *it;

          // Only accept `using` statements in the same file.
          if (use->location.source->origin != name.source->origin)
            continue;

          // Only accept `using` statements that are earlier in scope.
          if (use->location.start > name.start)
            continue;

          // Find `name` in the used TypeRef, using the current symbol table.
          def = member(sym, use->type, name);

          if (def)
            return def;
        }

        sym = st->parent.lock();
      }

      return {};
    }

    // This looks up `name` as a member of `node`. If `node` is not a Class or
    // an Interface, `node` is first resolved in the symbol context.
    Ast member(Ast& sym, Ast node, const Location& name)
    {
      if (!node)
        return {};

      switch (node->kind())
      {
        case Kind::Class:
        case Kind::Interface:
        {
          // Update the symbol context.
          sym = node;
          return node->symbol_table()->get(name);
        }

        case Kind::TypeAlias:
        {
          // Look in the type we are aliasing.
          return member(sym, node->as<TypeAlias>().inherits, name);
        }

        case Kind::TypeParam:
        {
          // Look in our upper bounds.
          return member(sym, node->as<TypeParam>().upper, name);
        }

        case Kind::ExtractType:
        case Kind::ViewType:
        {
          // This is the result of a `using`, a type alias, or a type parameter.
          // Look in the right-hand side.
          return member(sym, node->as<TypePair>().right, name);
        }

        case Kind::TypeRef:
        {
          // This is the result of a `using`, a type alias, or a type parameter.
          // Look in the resolved type.
          auto& tr = node->as<TypeRef>();
          auto def = typeref(sym, tr);

          if (!def)
            return {};

          // Update the symbol context.
          if (is_kind(def, {Kind::Class, Kind::Interface, Kind::TypeAlias}))
            sym = def;

          return member(sym, def, name);
        }

        case Kind::IsectType:
        {
          // Look in all conjunctions.
          // TODO: what if we find it more than once?
          auto& isect = node->as<IsectType>();

          for (auto& type : isect.types)
          {
            auto def = member(sym, type, name);

            if (def)
              return def;
          }

          return {};
        }

        case Kind::UnionType:
        {
          // Look in all disjunctions.
          // TODO: must be present everywhere
          return {};
        }

        default:
        {
          // No lookup in Field, Function, ThrowType, FunctionType, TupleType,
          // TypeList, or a capability.
          // TODO: Self
          return {};
        }
      }
    }

    void substitutions(Substitutions& subs, Ast& def, List<Type>& typeargs)
    {
      if (!def)
        return;

      List<TypeParam>* typeparams;

      switch (def->kind())
      {
        case Kind::Class:
        case Kind::Interface:
        case Kind::TypeAlias:
        {
          typeparams = &def->as<Interface>().typeparams;
          break;
        }

        case Kind::Function:
        {
          typeparams = &def->as<Function>().lambda->as<Lambda>().typeparams;
          break;
        }

        default:
          return;
      }

      size_t i = 0;

      while ((i < typeparams->size()) && (i < typeargs.size()))
      {
        subs.emplace(typeparams->at(i), typeargs.at(i));
        i++;
      }

      if (i < typeparams->size())
      {
        while (i < typeparams->size())
        {
          subs.emplace(typeparams->at(i), std::make_shared<InferType>());
          i++;
        }
      }
      else if (i < typeargs.size())
      {
        error() << typeargs.at(i)->location
                << "Too many type arguments supplied."
                << text(typeargs.at(i)->location);
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
