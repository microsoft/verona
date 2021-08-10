// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include "dnf.h"

namespace verona::parser
{
  Node<Type> Lookup::typeref(Ast sym, TypeRef& tr)
  {
    Substitutions subs;
    return typeref(subs, sym, tr);
  }

  Node<Type> Lookup::typeref(Substitutions& subs, Ast sym, TypeRef& tr)
  {
    auto def = tr.lookup;

    if (def)
      return def;

    // Lookup the first element in the current symbol context.
    def = name(subs, sym, tr.typenames.front());

    // Look in the current definition for the next name.
    for (size_t i = 1; i < tr.typenames.size(); i++)
      def = member(subs, sym, def, tr.typenames.at(i));

    tr.lookup = def;
    return def;
  }

  Node<Type> Lookup::name(Substitutions& subs, Ast sym, Node<TypeName>& tn)
  {
    auto name = tn->location;

    while (sym)
    {
      auto find = lexical(subs, sym, tn, true);

      if (find)
        return find;

      auto st = sym->symbol_table();

      if (!st)
        return {};

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
        find = member(subs, sym, use->type, tn);

        if (find)
          return find;
      }

      sym = st->parent.lock();
    }

    return {};
  }

  Node<Type>
  Lookup::lexical(Substitutions& subs, Ast node, Node<TypeName>& tn, bool up)
  {
    // This comes from `name` and is looking inside a symbol table, or it
    // comes from looking inside a previous lookup result.
    auto def = node->symbol_table()->get(tn->location);

    if (!def)
      return {};

    // Don't return typeparams here. This is for when we're looking inside
    // something, not looking up lexically.
    if (!up && is_kind(def, {Kind::TypeParam, Kind::TypeParamList}))
      return {};

    // A LookupRef always references a Class, Interface, TypeAlias, TypeParam,
    // TypeParamList, Field, or Function. The `self` member is a reference to
    // the enclosing scope if the target has no symbol table.
    auto res = std::make_shared<LookupRef>();
    res->def = def;
    res->subs = substitutions(subs, def, tn->typeargs);

    if (!def->symbol_table())
    {
      auto self = std::make_shared<LookupRef>();
      self->def = node;
      self->subs = subs;
      res->self = self;
    }

    return res;
  }

  Node<Type> Lookup::member(Ast node, Node<TypeName>& tn)
  {
    Substitutions subs;
    return member(subs, node, node, tn);
  }

  Node<Type>
  Lookup::member(Substitutions& subs, Ast sym, Ast node, Node<TypeName>& tn)
  {
    // `sym` is the context in which TypeRef names are resolved. It's changed
    // when we lookup through a LookupOne or a TypeAlias.

    // When called from `name`, `node` is always a Class, Interface, or a
    // TypeRef that comes from a `using` directive. When called from `typeref`,
    // `node` is always a Lookup result. When called for dynamic dispatch
    // lookup, `node` is a Node<Type>.
    if (!node)
      return {};

    switch (node->kind())
    {
      case Kind::LookupRef:
      {
        // We previously found a Class, Interface, TypeAlias, Field, or
        // Function. Look inside it, using the previous substitutions. Any
        // current substitutions can be discarded.
        auto& find = node->as<LookupRef>();
        auto def = find.def.lock();
        return member(find.subs, def, def, tn);
      }

      case Kind::Class:
      case Kind::Interface:
      {
        return lexical(subs, node, tn, false);
      }

      case Kind::TypeAlias:
      {
        // This comes from looking inside a previous lookup result.
        // Look in the type we are aliasing. Names are looked up in the context
        // of the type alias. Substitutions for this TypeAlias were already
        // created by the LookupOne result.
        return member(subs, node, node->as<TypeAlias>().inherits, tn);
      }

      case Kind::TypeRef:
      {
        // Get the lookup for the TypeRef and look inside it.
        auto def = typeref(subs, sym, node->as<TypeRef>());
        return member(subs, sym, def, tn);
      }

      case Kind::TypeParam:
      {
        // Look in our upper bounds.
        return member(subs, sym, node->as<TypeParam>().upper, tn);
      }

      case Kind::ExtractType:
      case Kind::ViewType:
      {
        // This is the result of a `using`, a type alias, or a type parameter.
        // Look in the right-hand side.
        return member(subs, sym, node->as<TypePair>().right, tn);
      }

      case Kind::IsectType:
      {
        // Look in all conjunctions.
        auto& l = node->as<UnionType>();
        Node<Type> def;

        for (auto& t : l.types)
        {
          auto ldef = member(subs, sym, t, tn);
          def = dnf::conjunction(def, ldef);
        }

        return def;
      }

      case Kind::UnionType:
      {
        // Look in all disjunctions. Fail if it isn't present in every branch.
        auto& l = node->as<UnionType>();
        Node<Type> def;

        for (auto& t : l.types)
        {
          auto ldef = member(subs, sym, t, tn);

          if (!ldef)
            return {};

          def = dnf::disjunction(def, ldef);
        }

        return def;
      }

      case Kind::InferType:
      {
        if (!bounds)
          return {};

        auto find = bounds->find(node);

        if (find == bounds->end())
          return {};

        // We are a supertype of all lower bounds, so treat the list of lower
        // bounds as a union type.
        auto lower = member(subs, sym, find->second.lower, tn);

        // We are a subtype of all upper bounds, so treat the list of upper
        // bounds as an intersection type.
        auto upper = member(subs, sym, find->second.upper, tn);

        // We conform to both, so return the conjunction.
        return dnf::conjunction(lower, upper);
      }

      default:
      {
        // No lookup in Field, Function, ThrowType, FunctionType, TupleType,
        // TypeList, or a capability.
        return {};
      }
    }
  }

  Substitutions
  Lookup::substitutions(Substitutions& subs, Ast def, List<Type>& typeargs)
  {
    auto ret = subs;

    if (!def)
      return ret;

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

      case Kind::Field:
      case Kind::TypeParam:
        return ret;

      default:
      {
        assert(false);
        return ret;
      }
    }

    size_t i = 0;

    while ((i < typeparams->size()) && (i < typeargs.size()))
    {
      ret.emplace(typeparams->at(i), typeargs.at(i));
      i++;
    }

    if (i < typeparams->size())
    {
      while (i < typeparams->size())
      {
        ret.emplace(typeparams->at(i), std::make_shared<InferType>());
        i++;
      }
    }
    else if (i < typeargs.size())
    {
      error() << typeargs.at(i)->location << "Too many type arguments supplied."
              << text(typeargs.at(i)->location);
    }

    return ret;
  }
}
