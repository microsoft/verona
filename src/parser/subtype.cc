// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "subtype.h"

#include "dnf.h"
#include "print.h"

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
    if ((lhs->kind() == Kind::ThrowType) || (rhs->kind() == Kind::ThrowType))
    {
      sub_throw(lhs, rhs);
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

    // Check TupleTypes.
    if ((lhs->kind() == Kind::TupleType) || (rhs->kind() == Kind::TupleType))
    {
      sub_tuple(lhs, rhs);
      return;
    }

    // Check capability types.
    auto types = {Kind::Iso, Kind::Mut, Kind::Imm};

    if (is_kind(lhs, types) || is_kind(rhs, types))
    {
      sub_same(lhs, rhs);
      return;
    }

    // TODO: view, extract, self, typelist.

    // Check FunctionTypes.
    if (
      (lhs->kind() == Kind::FunctionType) ||
      (rhs->kind() == Kind::FunctionType))
    {
      // TODO: typeref can be if it is a typeparam or typealias
      sub_function(lhs, rhs);
      return;
    }

    // Check TypeRefs, lhs first.
    if (lhs->kind() == Kind::TypeRef)
    {
      auto& tr = lhs->as<TypeRef>();
      auto def = tr.def.lock();

      switch (def->kind())
      {
        case Kind::TypeAlias:
        {
          // Check the aliased type.
          result(constraint(def->as<TypeAlias>().inherits, rhs));
          return;
        }

        case Kind::TypeParam:
        {
          // Treat this as (T & Bounds).
          auto isect = dnf::conjunction(lhs, def->as<TypeParam>().upper);
          result(constraint(isect, rhs));
          return;
        }

        default:
          // It's a Class or an Interface. Fall through.
          break;
      }
    }

    if (rhs->kind() == Kind::TypeRef)
    {
      // The lhs is a TypeRef to a Class or Interface.
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
    // could also rule out uninhabitable types?
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

  void Subtype::sub_function(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (
      (lhs->kind() != Kind::FunctionType) ||
      (rhs->kind() != Kind::FunctionType))
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

  void Subtype::t_sub_isect(Node<Type>& lhs, Node<Type>& rhs)
  {
    // The lhs must be a subtype of every element of the rhs.
    auto& r = rhs->as<IsectType>();
    bool ok = true;

    for (auto& t : r.types)
      ok &= constraint(lhs, t);

    result(ok);
  }

  void Subtype::t_sub_typeref(Node<Type>& lhs, Node<Type>& rhs)
  {
    // The rhs is a Class, Interface, TypeAlias, or TypeParam.
    auto def = rhs->as<TypeRef>().def.lock();

    switch (def->kind())
    {
      case Kind::Class:
      case Kind::TypeParam:
      {
        if (
          (lhs->kind() != Kind::TypeRef) ||
          (lhs->as<TypeRef>().def.lock() != def))
        {
          error() << lhs->location
                  << "A class or type parameter must be an exact match."
                  << text(lhs->location) << rhs->location
                  << "The type being matched is here." << text(rhs->location);
          return;
        }

        result(true);
        break;
      }

      case Kind::Interface:
      {
        // TODO:
        unexpected();
        break;
      }

      case Kind::TypeAlias:
        result(constraint(lhs, def->as<TypeAlias>().inherits));
        return;

      default:
        unexpected();
        break;
    }
  }

  void Subtype::sub_throw(Node<Type>& lhs, Node<Type>& rhs)
  {
    if ((lhs->kind() != Kind::ThrowType) || (rhs->kind() != Kind::ThrowType))
    {
      kinderror(lhs, rhs);
      return;
    }

    result(constraint(lhs->as<ThrowType>().type, rhs->as<ThrowType>().type));
  }

  void Subtype::sub_tuple(Node<Type>& lhs, Node<Type>& rhs)
  {
    if ((lhs->kind() != Kind::TupleType) || (rhs->kind() != Kind::TupleType))
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
