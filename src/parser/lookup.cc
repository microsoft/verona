// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

namespace verona::parser
{
  Ast Lookup::typeref(Ast sym, TypeRef& tr)
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
    Ast context = sym;

    for (size_t i = 1; i < tr.typenames.size(); i++)
    {
      context = def;
      def = member(sym, def, tr.typenames.at(i)->location);
      substitutions(tr.subs, def, tr.typenames.at(i)->typeargs);
    }

    // Don't set up tr.resolved here. That needs all typerefs to already have
    // their context and def set.
    tr.context = context;
    tr.def = def;
    return def;
  }

  // This looks up `name` in the lexical scope, following `using` statements.
  Ast Lookup::name(Ast& sym, const Location& name)
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

        if (!name.source->origin.empty())
        {
          // Only accept `using` statements in the same file.
          if (use->location.source->origin != name.source->origin)
            continue;

          // Only accept `using` statements that are earlier in scope.
          if (use->location.start > name.start)
            continue;
        }

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
  Ast Lookup::member(Ast& sym, Ast node, const Location& name)
  {
    if (!node)
      return {};

    switch (node->kind())
    {
      case Kind::Class:
      case Kind::Interface:
      {
        auto def = node->symbol_table()->get(name);

        // Update the symbol context.
        if (def)
          sym = node;

        return def;
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
        auto isym = sym;
        auto def = typeref(isym, tr);

        if (!def)
          return {};

        // Update the symbol context.
        if (is_kind(def, {Kind::Class, Kind::Interface, Kind::TypeAlias}))
          isym = def;

        def = member(isym, def, name);

        if (def)
          sym = isym;

        return def;
      }

      case Kind::IsectType:
      {
        // Look in all conjunctions. Fail if we find more than one.
        auto& isect = node->as<IsectType>();
        Ast isym = sym;
        Ast def;

        for (auto& type : isect.types)
        {
          auto lsym = sym;
          auto ldef = member(lsym, type, name);

          if (ldef)
          {
            if (def)
              return {};

            isym = lsym;
            def = ldef;
          }
        }

        sym = isym;
        return def;
      }

      default:
      {
        // No lookup in Field, Function, UnionType, ThrowType, FunctionType,
        // TupleType, TypeList, or a capability.
        return {};
      }
    }
  }

  void
  Lookup::substitutions(Substitutions& subs, Ast& def, List<Type>& typeargs)
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
      error() << typeargs.at(i)->location << "Too many type arguments supplied."
              << text(typeargs.at(i)->location);
    }
  }
}
