// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

namespace verona::parser
{
  Ast Lookup::typeref(Ast sym, TypeRef& tr)
  {
    // Each element will have a definition. This will point to a Class,
    // Interface, TypeAlias, Field, Function, LookupUnion, or LookupIsect.
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
        // Look in all conjunctions. Only set sym if we find a single result.
        return isect_member(sym, node->as<IsectType>().types, name);
      }

      case Kind::UnionType:
      {
        // Look in all disjunctions. Fail if it isn't present in every branch.
        // Only set sym if we find a single result.
        return union_member(sym, node->as<UnionType>().types, name);
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
        auto def = union_member(sym, find->second.lower, name);

        // We are a subtype of all upper bounds, so treat the list of upper
        // bounds as an intersection type.
        if (!def)
          def = isect_member(sym, find->second.upper, name);

        return def;
      }

      default:
      {
        // No lookup in LookupIsect, LookupUnion, Field, Function, ThrowType,
        // FunctionType, TupleType, TypeList, or a capability.
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

      case Kind::LookupUnion:
      {
        auto& l = def->as<LookupUnion>();

        for (auto& t : l.list)
          substitutions(subs, t, typeargs);
        return;
      }

      case Kind::LookupIsect:
      {
        auto& l = def->as<LookupIsect>();

        for (auto& t : l.list)
          substitutions(subs, t, typeargs);
        return;
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

  Ast Lookup::union_member(Ast& sym, List<Type>& list, const Location& name)
  {
    // Look in all disjunctions. Fail if it isn't present in every branch.
    // Only set sym if we find a single result.
    Ast def;
    Ast isym;

    for (auto& type : list)
    {
      auto lsym = sym;
      auto ldef = member(lsym, type, name);

      if (!ldef)
        return {};

      isym = lsym;
      def = disjunction(def, ldef);
    }

    if (def && !is_kind(def, {Kind::LookupUnion, Kind::LookupIsect}))
      sym = isym;

    return def;
  }

  Ast Lookup::isect_member(Ast& sym, List<Type>& list, const Location& name)
  {
    // Look in all conjunctions. Only set sym if we find a single result.
    Ast def;
    Ast isym;

    for (auto& type : list)
    {
      auto lsym = sym;
      auto ldef = member(lsym, type, name);

      if (ldef)
      {
        if (lsym->kind() == Kind::Class)
        {
          // We must be this concrete type, so ignore other elements.
          sym = lsym;
          return ldef;
        }

        isym = lsym;
        def = conjunction(def, ldef);
      }
    }

    if (def && !is_kind(def, {Kind::LookupUnion, Kind::LookupIsect}))
      sym = isym;

    return def;
  }

  Ast Lookup::disjunction(Ast left, Ast right)
  {
    if (!left)
      return right;

    if (!right)
      return left;

    if (left == right)
      return left;

    if (left->kind() == Kind::LookupUnion)
    {
      auto& lhs = left->as<LookupUnion>();

      if (right->kind() == Kind::LookupUnion)
      {
        // (A | B) | (C | D) -> (A | B | C | D)
        auto& rhs = right->as<LookupUnion>();

        for (auto& t : rhs.list)
        {
          if (std::find(lhs.list.begin(), lhs.list.end(), t) == lhs.list.end())
            lhs.list.push_back(t);
        }
      }
      else
      {
        // (A | B) | C -> (A | B | C)
        if (
          std::find(lhs.list.begin(), lhs.list.end(), right) == lhs.list.end())
        {
          lhs.list.push_back(right);
        }
      }

      return left;
    }

    if (right->kind() == Kind::LookupUnion)
      return disjunction(right, left);

    auto r = std::make_shared<LookupUnion>();
    r->list.push_back(left);
    r->list.push_back(right);
    return r;
  }

  Ast Lookup::conjunction(Ast left, Ast right)
  {
    if (!left)
      return right;

    if (!right)
      return left;

    if (left == right)
      return left;

    if (left->kind() == Kind::LookupUnion)
    {
      auto& lhs = left->as<LookupUnion>();
      auto un = std::make_shared<LookupUnion>();

      if (right->kind() == Kind::LookupUnion)
      {
        // (A | B) & (C | D) -> (A & C) | (A & D) | (B & C) | (B & D)
        auto& rhs = right->as<LookupUnion>();

        for (auto& l : lhs.list)
        {
          for (auto& r : rhs.list)
            un->list.push_back(conjunction(l, r));
        }
      }
      else
      {
        // (A | B) & C -> (A & C) | (B & C)
        for (auto& t : lhs.list)
          un->list.push_back(conjunction(t, right));
      }

      return un;
    }

    if (right->kind() == Kind::LookupUnion)
      return conjunction(right, left);

    if (left->kind() == Kind::LookupIsect)
    {
      auto& lhs = left->as<LookupIsect>();

      if (right->kind() == Kind::LookupIsect)
      {
        // (A & B) & (C & D) -> (A & B & C & D)
        auto& rhs = right->as<LookupIsect>();

        for (auto& t : rhs.list)
        {
          if (std::find(lhs.list.begin(), lhs.list.end(), t) == lhs.list.end())
            lhs.list.push_back(t);
        }
      }
      else
      {
        // (A & B) & C -> (A & B & C)
        if (
          std::find(lhs.list.begin(), lhs.list.end(), right) == lhs.list.end())
        {
          lhs.list.push_back(right);
        }
      }

      return left;
    }

    if (right->kind() == Kind::LookupIsect)
      return conjunction(right, left);

    auto r = std::make_shared<LookupIsect>();
    r->list.push_back(left);
    r->list.push_back(right);
    return r;
  }
}
