// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "subtype.h"

#include "dnf.h"
#include "print.h"
#include "rewrite.h"

#include <iostream>

namespace verona::parser
{
  bool Subtype::operator()(Node<Type> lhs, Node<Type> rhs)
  {
    auto prev = show;
    show = true;
    auto r = constraint(lhs, rhs);
    show = prev;
    return r;
  }

  bool Subtype::constraint(Node<Type>& lhs, Node<Type>& rhs)
  {
    // lhs <: rhs
    if (!lhs || !rhs)
      return false;

    // Don't repeat checks. Initially assume the check succeeds.
    auto [it, fresh] = checked.try_emplace(std::pair{lhs, rhs}, true);

    // This check has already been done.
    if (!fresh)
      return returnval(it);

    // Keep a stack of iterators into the cache.
    auto prev = current;
    current = it;

    // This does the work.
    t_sub_t(lhs, rhs);

    current = prev;
    auto r = returnval(it);

    // TODO: debugging output
    if (r)
      std::cout << "Constraint: " << lhs << " <: " << rhs << std::endl;

    return r;
  }

  void Subtype::t_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    // Check InferTypes, lhs first.
    if (lhs->kind() == Kind::InferType)
    {
      infer_sub_t(lhs, rhs);
      return;
    }

    if (rhs->kind() == Kind::InferType)
    {
      t_sub_infer(lhs, rhs);
      return;
    }

    // Check UnionTypes, lhs first.
    if (lhs->kind() == Kind::UnionType)
    {
      union_sub_t(lhs, rhs);
      return;
    }

    if (rhs->kind() == Kind::UnionType)
    {
      t_sub_union(lhs, rhs);
      return;
    }

    // Check ThrowTypes.
    if (rhs->kind() == Kind::ThrowType)
    {
      t_sub_throw(lhs, rhs);
      return;
    }

    // Check IsectTypes, rhs first.
    if (rhs->kind() == Kind::IsectType)
    {
      // The lhs isn't an InferType, UnionType, or ThrowType.
      t_sub_isect(lhs, rhs);
      return;
    }

    if (lhs->kind() == Kind::IsectType)
    {
      // The rhs isn't an InferType, UnionType, ThrowType, or IsectType.
      isect_sub_t(lhs, rhs);
      return;
    }

    // Expand TypeAlias and TypeParam on the lhs.
    if (lhs->kind() == Kind::TypeRef)
    {
      auto& tr = lhs->as<TypeRef>();
      auto def = tr.def.lock();

      switch (def->kind())
      {
        case Kind::TypeAlias:
        {
          // Check the aliased type.
          if (!tr.resolved)
            tr.resolved = clone(tr.subs, def->as<TypeAlias>().inherits);

          result(constraint(tr.resolved, rhs));
          return;
        }

        case Kind::TypeParam:
        {
          // Treat this as (T & Bounds) by checking the upper bounds first.
          auto noshow = NoShow(this);

          if (constraint(def->as<TypeParam>().upper, rhs))
          {
            result(true);
            return;
          }
          break;
        }

        default:
          // It's a Class or an Interface. Fall through.
          break;
      }
    }

    // Neither side is an inference variable, union, throw, or isect.
    // The lhs is not a type alias or type param reference.

    // Check TupleTypes.
    if (rhs->kind() == Kind::TupleType)
    {
      t_sub_tuple(lhs, rhs);
      return;
    }

    // Check capability types.
    if (is_kind(rhs, {Kind::Iso, Kind::Mut, Kind::Imm}))
    {
      sub_same(lhs, rhs);
      return;
    }

    // TODO: view, extract, typelist.

    // Check Self.
    if ((lhs->kind() == Kind::Self) || (rhs->kind() == Kind::Self))
    {
      sub_same(lhs, rhs);
      return;
    }

    // Check FunctionTypes.
    if (rhs->kind() == Kind::FunctionType)
    {
      t_sub_function(lhs, rhs);
      return;
    }

    // Check TypeRefs.
    if (rhs->kind() == Kind::TypeRef)
    {
      t_sub_typeref(lhs, rhs);
      return;
    }

    unexpected();
  }

  void Subtype::infer_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    auto& bnd = bounds[lhs];
    bnd.upper.push_back(rhs);
    bool ok = true;

    for (auto& lower : bnd.lower)
      ok &= constraint(lower, rhs);

    result(ok);
  }

  void Subtype::isect_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    // TODO: If the lhs has multiple interfaces, they could combine to fulfill
    // an interface on the rhs.
    // could build a synthetic combination interface
    auto& l = lhs->as<IsectType>();

    {
      // Don't show error messages.
      auto noshow = NoShow(this);
      size_t ok = 0;

      // Some element of the lhs must be a subtype of the rhs.
      for (auto& t : l.types)
      {
        if (constraint(t, rhs))
          ok++;
      }

      if (ok > 0)
      {
        result(true);
        return;
      }
    }

    if (show)
    {
      // Go back and show error messages.
      for (auto& t : l.types)
        constraint(t, rhs);
    }

    result(false);
  }

  void Subtype::union_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    // Every element of the lhs must be a subtype of the rhs.
    auto& l = lhs->as<UnionType>();
    bool ok = true;

    for (auto& t : l.types)
      ok &= constraint(t, rhs);

    result(ok);
  }

  void Subtype::t_sub_infer(Node<Type>& lhs, Node<Type>& rhs)
  {
    auto& bnd = bounds[rhs];
    bnd.lower.push_back(lhs);
    bool ok = true;

    for (auto& upper : bnd.upper)
      ok &= constraint(lhs, upper);

    result(ok);
  }

  void Subtype::t_sub_union(Node<Type>& lhs, Node<Type>& rhs)
  {
    auto& r = rhs->as<UnionType>();

    {
      // Don't show error messages.
      auto noshow = NoShow(this);
      size_t ok = 0;

      // The lhs must be a subtype of some element of the rhs.
      for (auto& t : r.types)
      {
        if (constraint(lhs, t))
          ok++;
      }

      if (ok > 0)
      {
        result(true);
        return;
      }
    }

    if (show)
    {
      // Go back and show error messages.
      for (auto& t : r.types)
        constraint(lhs, t);
    }

    result(false);
  }

  void Subtype::t_sub_throw(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::ThrowType)
    {
      kinderror(lhs, rhs);
      return;
    }

    result(constraint(lhs->as<ThrowType>().type, rhs->as<ThrowType>().type));
  }

  void Subtype::t_sub_isect(Node<Type>& lhs, Node<Type>& rhs)
  {
    // The lhs must be a subtype of every element of the rhs.
    auto& r = rhs->as<IsectType>();
    bool ok = true;

    for (auto& t : r.types)
      ok &= constraint(lhs, t);

    result(ok);
  }

  void Subtype::t_sub_tuple(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::TupleType)
    {
      kinderror(lhs, rhs);
      return;
    }

    auto& l = lhs->as<TupleType>();
    auto& r = rhs->as<TupleType>();

    if (l.types.size() != r.types.size())
    {
      error() << l.location << "A tuple of arity/" << l.types.size()
              << " isn't a subtype of a tuple of arity/" << r.types.size()
              << text(l.location) << r.location << "The supertype is here."
              << text(r.location);
      return;
    }

    bool ok = true;

    for (size_t i = 0; i < l.types.size(); i++)
      ok &= constraint(l.types.at(i), r.types.at(i));

    result(ok);
  }

  void Subtype::t_sub_function(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() == Kind::TypeRef)
    {
      // The lhs must be a class or interface. An alias or a typeparam has
      // already been checked.
      auto& l = lhs->as<TypeRef>();
      auto def = l.def.lock();

      if (is_kind(def, {Kind::Class, Kind::Interface}))
      {
        // Apply methods can fulfill function types.
        auto apply = def->symbol_table()->get(name_apply);

        if (!apply || (apply->kind() != Kind::Function))
        {
          error() << lhs->location
                  << "This type doesn't have an apply function."
                  << text(lhs->location) << rhs->location
                  << "The supertype is here." << text(rhs->location);
          return;
        }

        auto t = clone(l.subs, apply->as<Function>().type);
        result(constraint(t, rhs));
        return;
      }
    }

    if (lhs->kind() != Kind::FunctionType)
    {
      kinderror(lhs, rhs);
      return;
    }

    auto& l = lhs->as<FunctionType>();
    auto& r = rhs->as<FunctionType>();

    bool ok = constraint(r.left, l.left);
    ok &= constraint(l.right, r.right);
    result(ok);
  }

  void Subtype::t_sub_typeref(Node<Type>& lhs, Node<Type>& rhs)
  {
    // The rhs is a Class, Interface, TypeAlias, or TypeParam.
    auto& r = rhs->as<TypeRef>();
    auto def = r.def.lock();

    switch (def->kind())
    {
      case Kind::Class:
      case Kind::TypeParam:
      {
        t_sub_class(lhs, rhs);
        return;
      }

      case Kind::Interface:
      {
        t_sub_iface(lhs, rhs);
        return;
      }

      case Kind::TypeAlias:
      {
        if (!r.resolved)
          r.resolved = clone(r.subs, def->as<TypeAlias>().inherits);

        result(constraint(lhs, r.resolved));
        return;
      }

      default:
        unexpected();
        return;
    }
  }

  void Subtype::t_sub_class(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::TypeRef)
    {
      kinderror(lhs, rhs);
      return;
    }

    auto& l = lhs->as<TypeRef>();
    auto ldef = l.def.lock();

    auto& r = rhs->as<TypeRef>();
    auto rdef = r.def.lock();

    if (ldef != rdef)
    {
      error() << lhs->location
              << "A class or type parameter must be an exact match."
              << text(lhs->location) << rhs->location
              << "The type being matched is here." << text(rhs->location);
      return;
    }

    if (l.subs.size() != r.subs.size())
    {
      error() << lhs->location << "Type argument count doesn't match."
              << text(lhs->location) << rhs->location
              << "The supertype is here." << text(rhs->location);
      return;
    }

    bool ok = true;

    for (auto& sub : r.subs)
    {
      auto def = sub.first.lock();
      auto find = l.subs.find(def);

      if (find == l.subs.end())
      {
        error() << def->location << "Type argument not present in the subtype."
                << text(def->location) << lhs->location
                << "The subtype is here." << text(lhs->location)
                << rhs->location << "The supertype is here."
                << text(rhs->location);
        return;
      }

      // Invariant type args.
      ok &= constraint(find->second, sub.second);
      ok &= constraint(sub.second, find->second);
    }

    result(ok);
  }

  void Subtype::t_sub_iface(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::TypeRef)
    {
      kinderror(lhs, rhs);
      return;
    }

    auto& l = lhs->as<TypeRef>();
    auto ldef = l.def.lock();

    auto& r = rhs->as<TypeRef>();
    auto rdef = r.def.lock();
    assert(rdef->kind() == Kind::Interface);

    if (is_kind(ldef, {Kind::Class, Kind::Interface}))
    {
      bool ok = true;

      for (auto& rm : rdef->symbol_table()->map)
      {
        // TODO: should we be skipping other members?
        if (!is_kind(rm.second, {Kind::Field, Kind::Function}))
          continue;

        auto lm = ldef->symbol_table()->get(rm.first);

        if (!lm)
        {
          error() << lhs->location << "This type doesn't have a member "
                  << rm.first << text(lhs->location) << rhs->location
                  << "The supertype is here." << text(rhs->location);
          ok = false;
          continue;
        }

        if (lm->kind() != rm.second->kind())
        {
          error() << lhs->location << "This type's member " << rm.first
                  << " is a " << kindname(lm->kind())
                  << " which is not a subtype of a "
                  << kindname(rm.second->kind()) << "." << text(lhs->location)
                  << rhs->location << "The supertype is here."
                  << text(rhs->location);
          ok = false;
          continue;
        }

        switch (rm.second->kind())
        {
          case Kind::Field:
          {
            auto& lf = lm->as<Field>();
            auto& rf = rm.second->as<Field>();
            auto lt = clone(l.subs, lf.type, lhs);
            auto rt = clone(r.subs, rf.type, lhs);
            ok &= constraint(lt, rt);
            break;
          }

          case Kind::Function:
          {
            auto& lf = lm->as<Function>();
            auto& rf = rm.second->as<Function>();
            auto lt = clone(l.subs, lf.type, lhs);
            auto rt = clone(r.subs, rf.type, lhs);
            ok &= constraint(lt, rt);
            break;
          }

          default:
          {
            unexpected();
            return;
          }
        }
      }

      if (!ok)
      {
        error() << lhs->location << "This isn't a subtype of this interface."
                << text(lhs->location) << rhs->location
                << "The supertype is here." << text(rhs->location);
      }
      return;
    }

    // TODO: what else can be a subtype of an interface?
    kinderror(lhs, rhs);
  }

  void Subtype::sub_same(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != rhs->kind())
    {
      kinderror(lhs, rhs);
      return;
    }

    result(true);
  }

  bool Subtype::returnval(Cache::iterator& it)
  {
    if (auto errmsg = std::get_if<std::stringstream>(&it->second))
    {
      if (show)
      {
        // We haven't printed this cached error message before. Print it now,
        // then remove it from the cache.
        error() << errmsg->str();
        it->second = false;
      }

      return false;
    }

    return std::get<bool>(it->second);
  }

  void Subtype::kinderror(Node<Type>& lhs, Node<Type>& rhs)
  {
    error() << lhs->location << "A " << kindname(lhs->kind())
            << " isn't a subtype of a " << kindname(rhs->kind()) << "."
            << text(lhs->location) << rhs->location << "The supertype is here."
            << text(rhs->location);
  }

  void Subtype::unexpected()
  {
    error() << current->first.first->location
            << "Reached an unexpected constraint. Left-hand side:"
            << text(current->first.first->location)
            << current->first.second->location
            << "Right-hand side:" << text(current->first.second->location);
  }
}
