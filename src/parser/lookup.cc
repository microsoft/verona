// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

namespace verona::parser
{
  // Helper functions for looking up in unions and intersections.
  template<typename T>
  Node<LookupResult> member_union(
    Lookup* lookup,
    Substitutions& subs,
    Ast& sym,
    List<T>& list,
    Node<TypeName>& tn)
  {
    // Look in all disjunctions. Fail if it isn't present in every branch.
    Node<LookupResult> def;

    for (auto& element : list)
    {
      auto ldef = lookup->member(subs, sym, element, tn);

      if (!ldef)
        return {};

      def = disjunction(def, ldef);
    }

    return def;
  }

  template<typename T>
  Node<LookupResult> member_isect(
    Lookup* lookup,
    Substitutions& subs,
    Ast& sym,
    List<T>& list,
    Node<TypeName>& tn)
  {
    Node<LookupResult> def;

    for (auto& element : list)
    {
      auto ldef = lookup->member(subs, sym, element, tn);
      def = conjunction(def, ldef);
    }

    return def;
  }

  // DNF over lookup values.
  Node<LookupResult>
  disjunction(Node<LookupResult> left, Node<LookupResult> right)
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

  Node<LookupResult>
  conjunction(Node<LookupResult> left, Node<LookupResult> right)
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
          auto& rhs = right->as<LookupOne>();
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

  Node<LookupResult> Lookup::typeref(Ast sym, TypeRef& tr)
  {
    Substitutions subs;
    return typeref(subs, sym, tr);
  }

  Node<LookupResult> Lookup::typeref(Substitutions& subs, Ast sym, TypeRef& tr)
  {
    // Each element will have a definition. This will be a LookupOne,
    // LookupIsect, or LookupUnion. A LookupOne will point to a Class,
    // Interface, TypeAlias, Field, or Function, and carries TypeParam
    // substitutions.
    auto def = tr.lookup;

    if (def)
      return def;

    // Lookup the first element in the current symbol context.
    def = name(subs, sym, tr.typenames.front());

    // Look in the current definition for the next name.
    for (size_t i = 1; i < tr.typenames.size(); i++)
      def = member(subs, sym, def, tr.typenames.at(i));

    // Don't set up tr.resolved here. That needs all typerefs to already have
    // their lookup set.
    tr.lookup = def;
    return def;
  }

  Node<LookupResult>
  Lookup::name(Substitutions& subs, Ast sym, Node<TypeName>& tn)
  {
    auto name = tn->location;

    while (sym)
    {
      auto find = member(subs, sym, sym, tn);

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
        auto find = member(subs, sym, use->type, tn);

        if (find)
          return find;
      }

      sym = st->parent.lock();
    }

    return {};
  }

  Node<LookupResult> Lookup::member(Ast node, Node<TypeName>& tn)
  {
    Substitutions subs;
    return member(subs, node, node, tn);
  }

  Node<LookupResult>
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
      case Kind::LookupOne:
      {
        // We previously found a Class, Interface, TypeAlias, Field, or
        // Function. Look inside it, using the previous substitutions. Any
        // current substitutions can be discarded.
        auto& find = node->as<LookupOne>();
        auto def = find.def.lock();
        return member(find.subs, def, def, tn);
      }

      case Kind::LookupIsect:
      {
        // Look in all conjunctions.
        return member_isect(this, subs, sym, node->as<LookupIsect>().list, tn);
      }

      case Kind::LookupUnion:
      {
        // Look in all disjunctions.
        return member_union(this, subs, sym, node->as<LookupUnion>().list, tn);
      }

      case Kind::Class:
      case Kind::Interface:
      {
        // This comes from `name` and is looking inside a symbol table, or it
        // comes from looking inside a previous lookup result.
        auto def = node->symbol_table()->get(tn->location);

        if (!def)
          return {};

        // A LookupOne always references a Class, Interface, TypeAlias, Field,
        // or Function. The `self` member is a reference to the enclosing type
        // for a Field or Function, or to `def` otherwise.
        auto res = std::make_shared<LookupOne>();
        res->def = def;
        res->subs = substitutions(subs, def, tn->typeargs);

        if (is_kind(def, {Kind::Field, Kind::Function}))
          res->self = node;
        else
          res->self = def;

        return res;
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
        return member_isect(this, subs, sym, node->as<IsectType>().types, tn);
      }

      case Kind::UnionType:
      {
        // Look in all disjunctions. Fail if it isn't present in every branch.
        return member_union(this, subs, sym, node->as<UnionType>().types, tn);
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
        auto lower = member_union(this, subs, sym, find->second.lower, tn);

        // We are a subtype of all upper bounds, so treat the list of upper
        // bounds as an intersection type.
        auto upper = member_isect(this, subs, sym, find->second.upper, tn);

        // We conform to both, so return the conjunction.
        return conjunction(lower, upper);
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
