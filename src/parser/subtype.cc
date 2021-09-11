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

  bool Subtype::dynamic(Node<Type> lhs, Node<Type> rhs)
  {
    auto pshow = show;
    auto pdyn = dyn;
    show = false;
    dyn = true;
    auto r = constraint(lhs, rhs);
    show = pshow;
    dyn = pdyn;
    return r;
  }

  bool Subtype::constraint(Node<Type>& lhs, Node<Type>& rhs)
  {
    // lhs <: rhs
    if (!lhs || !rhs)
      return true;

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
    if (infer_sub_t(lhs, rhs))
      return;

    if (t_sub_infer(lhs, rhs))
      return;

    // Check UnionTypes, lhs first.
    if (union_sub_t(lhs, rhs))
      return;

    if (t_sub_union(lhs, rhs))
      return;

    // Check ThrowTypes.
    if (t_sub_throw(lhs, rhs))
      return;

    // Check IsectTypes, rhs first.
    // The lhs isn't an InferType, UnionType, or ThrowType.
    if (t_sub_isect(lhs, rhs))
      return;

    // The rhs isn't an InferType, UnionType, ThrowType, or IsectType.
    if (isect_sub_t(lhs, rhs))
      return;

    // Check TypeRefs, lhs first.
    if (typeref_sub_t(lhs, rhs))
      return;

    if (t_sub_typeref(lhs, rhs))
      return;

    // Check LookupRef, lhs first.
    if (lookupref_sub_t(lhs, rhs))
      return;

    if (t_sub_lookupref(lhs, rhs))
      return;

    // TODO: view, extract.

    // Check TupleTypes.
    if (t_sub_tuple(lhs, rhs))
      return;

    // Check reference capability types and Self.
    if (is_kind(rhs, {Kind::Iso, Kind::Mut, Kind::Imm, Kind::Self}))
    {
      sub_same(lhs, rhs);
      return;
    }

    // Check TypeLists.
    if (t_sub_typelist(lhs, rhs))
      return;

    // Check FunctionTypes.
    if (t_sub_function(lhs, rhs))
      return;

    unexpected();
  }

  bool Subtype::infer_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::InferType)
      return false;

    // Add the rhs as a new upper bound. All of our lower bounds must be a
    // subtype of the new upper bound.
    auto& bnd = bounds[lhs];
    bnd.upper = dnf::conjunction(bnd.upper, rhs);
    result(constraint(bnd.lower, rhs));
    return true;
  }

  bool Subtype::union_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::UnionType)
      return false;

    // Every element of the lhs must be a subtype of the rhs.
    auto& l = lhs->as<UnionType>();
    bool ok = true;

    for (auto& t : l.types)
      ok &= constraint(t, rhs);

    result(ok);
    return true;
  }

  bool Subtype::isect_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::IsectType)
      return false;

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
        return true;
      }
    }

    if (show)
    {
      // Go back and show error messages.
      for (auto& t : l.types)
        constraint(t, rhs);
    }

    result(false);
    return true;
  }

  bool Subtype::typeref_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::TypeRef)
      return false;

    Node<Type> t = lhs->as<TypeRef>().lookup;
    result(constraint(t, rhs));
    return true;
  }

  bool Subtype::lookupref_sub_t(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (lhs->kind() != Kind::LookupRef)
      return false;

    resolve_lookupref(lhs);

    auto& l = lhs->as<LookupRef>();
    auto ldef = l.def.lock();

    switch (ldef->kind())
    {
      case Kind::TypeAlias:
      {
        // Check the aliased type.
        result(constraint(l.resolved, rhs));
        return true;
      }

      case Kind::TypeParam:
      {
        // Treat this as (T & Bounds) by checking the upper bounds first.
        auto noshow = NoShow(this);

        if (constraint(l.resolved, rhs))
        {
          result(true);
          return true;
        }
        break;
      }

      default:
        // It's a Class, Interface, Function, or Field. Fall through.
        break;
    }

    return false;
  }

  bool Subtype::t_sub_infer(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::InferType)
      return false;

    // Add the lhs as a new lower bound. The new lower bound must be a subtype
    // of all upper bounds.
    auto& bnd = bounds[rhs];
    bnd.lower = dnf::disjunction(bnd.lower, lhs);
    result(constraint(lhs, bnd.upper));
    return true;
  }

  bool Subtype::t_sub_union(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::UnionType)
      return false;

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
        return true;
      }
    }

    if (show)
    {
      // Go back and show error messages.
      for (auto& t : r.types)
        constraint(lhs, t);
    }

    result(false);
    return true;
  }

  bool Subtype::t_sub_throw(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::ThrowType)
      return false;

    if (lhs->kind() != Kind::ThrowType)
    {
      kinderror(lhs, rhs);
      return true;
    }

    result(constraint(lhs->as<ThrowType>().type, rhs->as<ThrowType>().type));
    return true;
  }

  bool Subtype::t_sub_isect(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::IsectType)
      return false;

    // The lhs must be a subtype of every element of the rhs.
    auto& r = rhs->as<IsectType>();
    bool ok = true;

    for (auto& t : r.types)
      ok &= constraint(lhs, t);

    result(ok);
    return true;
  }

  bool Subtype::t_sub_typeref(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::TypeRef)
      return false;

    Node<Type> t = rhs->as<TypeRef>().lookup;
    result(constraint(lhs, t));
    return true;
  }

  bool Subtype::t_sub_tuple(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::TupleType)
      return false;

    if (lhs->kind() != Kind::TupleType)
    {
      kinderror(lhs, rhs);
      return true;
    }

    auto& l = lhs->as<TupleType>();
    auto& r = rhs->as<TupleType>();

    if (l.types.size() != r.types.size())
    {
      error() << l.location << "A tuple of arity/" << l.types.size()
              << " isn't a subtype of a tuple of arity/" << r.types.size()
              << text(l.location) << r.location << "The supertype is here."
              << text(r.location);
      return true;
    }

    bool ok = true;

    for (size_t i = 0; i < l.types.size(); i++)
      ok &= constraint(l.types.at(i), r.types.at(i));

    result(ok);
    return true;
  }

  bool Subtype::t_sub_typelist(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::TypeList)
      return false;

    if (lhs->kind() != Kind::TypeList)
    {
      kinderror(lhs, rhs);
    }
    else if (lhs->as<TypeList>().def.lock() != rhs->as<TypeList>().def.lock())
    {
      error() << lhs->location << "A type list must be an exact match."
              << text(lhs->location) << rhs->location
              << "The supertype is here." << text(rhs->location);
    }

    return true;
  }

  bool Subtype::t_sub_function(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::FunctionType)
      return false;

    auto& r = rhs->as<FunctionType>();

    if (lhs->kind() == Kind::LookupRef)
    {
      resolve_lookupref(lhs);

      // The lhs must be a class, interface, field, or function. A typeparam
      // will already have had its upper bounds checked, so we can safely fail
      // here. A typealias will already have had its rhs checked.
      auto& l = lhs->as<LookupRef>();
      auto def = l.def.lock();

      switch (def->kind())
      {
        case Kind::Class:
        case Kind::Interface:
        {
          // TODO: can apply methods can fulfill function types?
          // leave it as no for now, as it can be written explicitly:
          // x ~ apply
          break;
        }

        case Kind::Field:
        {
          // TODO:
          error() << lhs->location << "Fields aren't implemented yet"
                  << text(lhs->location);
          return true;
        }

        case Kind::Function:
        {
          if (dyn)
          {
            // Modify the receiver in the rhs to be `receiver & self` if we're
            // checking dynamic dispatch.
            auto self = l.self.lock();
            auto f = receiver_self(rhs, self);
            result(constraint(l.resolved, f));
          }
          else
          {
            result(constraint(l.resolved, rhs));
          }
          return true;
        }

        case Kind::FunctionType:
        {
          result(constraint(l.resolved, rhs));
          return true;
        }

        default:
          break;
      }
    }

    if (lhs->kind() != Kind::FunctionType)
    {
      kinderror(lhs, rhs);
      return true;
    }

    auto& l = lhs->as<FunctionType>();
    bool ok = constraint(r.left, l.left);
    ok &= constraint(l.right, r.right);
    result(ok);
    return true;
  }

  bool Subtype::t_sub_lookupref(Node<Type>& lhs, Node<Type>& rhs)
  {
    if (rhs->kind() != Kind::LookupRef)
      return false;

    resolve_lookupref(rhs);

    auto& r = rhs->as<LookupRef>();
    auto rdef = r.def.lock();

    switch (rdef->kind())
    {
      case Kind::Class:
      case Kind::TypeParam:
        return t_sub_class(lhs, rhs);

      case Kind::Interface:
        return t_sub_iface(lhs, rhs);

      case Kind::TypeAlias:
      {
        result(constraint(lhs, r.resolved));
        return true;
      }

      default:
      {
        kinderror(lhs, rhs);
        return true;
      }
    }
  }

  bool Subtype::t_sub_class(Node<Type>& lhs, Node<Type>& rhs)
  {
    // We've already checked rhs.
    if (lhs->kind() != Kind::LookupRef)
    {
      kinderror(lhs, rhs);
      return true;
    }

    auto& l = lhs->as<LookupRef>();
    auto ldef = l.def.lock();

    auto& r = rhs->as<LookupRef>();
    auto rdef = r.def.lock();

    if (ldef != rdef)
    {
      error() << lhs->location
              << "A class or type parameter must be an exact match."
              << text(lhs->location) << rhs->location
              << "The type being matched is here." << text(rhs->location);
      return true;
    }

    if (l.subs.size() != r.subs.size())
    {
      error() << lhs->location << "Type argument count doesn't match."
              << text(lhs->location) << rhs->location
              << "The supertype is here." << text(rhs->location);
      return true;
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
        return true;
      }

      // Invariant type args.
      ok &= constraint(find->second, sub.second);
      ok &= constraint(sub.second, find->second);
    }

    result(ok);
    return true;
  }

  bool Subtype::t_sub_iface(Node<Type>& lhs, Node<Type>& rhs)
  {
    // We've already checked rhs.
    {
      // Check for an exact match first.
      auto noshow = NoShow(this);

      if (t_sub_class(lhs, rhs) && returnval(current))
        return true;

      // No exact match, reset the return value and continue.
      result(true);
    }

    // TODO: a function can be a subtype of an interface that only has an apply
    // method
    if (lhs->kind() != Kind::LookupRef)
    {
      kinderror(lhs, rhs);
      return true;
    }

    auto& l = lhs->as<LookupRef>();
    auto ldef = l.def.lock();

    auto& r = rhs->as<LookupRef>();
    auto rdef = r.def.lock();
    bool ok = true;

    assert(rdef->kind() == Kind::Interface);
    assert(is_kind(ldef, {Kind::Class, Kind::Interface}));

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
          // Field types must be invariant for lhs to be a subtype of rhs.
          auto& lf = lm->as<Field>();
          auto& rf = rm.second->as<Field>();
          auto lt = clone(l.subs, lf.type, lhs);
          auto rt = clone(r.subs, rf.type, lhs);
          ok &= constraint(lt, rt);
          ok &= constraint(rt, lt);
          break;
        }

        case Kind::Function:
        {
          // A function in lhs must be a subtype of the function in rhs.
          auto& lf = lm->as<Function>();
          auto& rf = rm.second->as<Function>();

          // Contravariant type parameters.
          List<TypeParam>& ltp = lf.lambda->as<Lambda>().typeparams;
          List<TypeParam>& rtp = rf.lambda->as<Lambda>().typeparams;

          if (ltp.size() != rtp.size())
          {
            error() << lf.location
                    << "This type's function has a different type parameter "
                       "count than the supertype."
                    << text(lf.location) << rf.location
                    << "The supertype function is here." << text(rf.location);
            ok = false;
            break;
          }

          for (size_t i = 0; i < ltp.size(); i++)
            ok &= constraint(rtp.at(i)->upper, ltp.at(i)->upper);

          // Apply substitutions and replace Self with the lhs for both sides.
          auto lt = clone(l.subs, lf.type, lhs);
          auto rt = clone(r.subs, rf.type, lhs);

          // Substitute a reference to the lhs typeparams for references to any
          // rhs typeparams.
          if (ltp.size() > 0)
          {
            Substitutions subs;

            for (size_t i = 0; i < ltp.size(); i++)
              subs.emplace(rtp.at(i), typeparamref(ltp.at(i)));

            rt = clone(subs, rt);
          }

          ok &= constraint(lt, rt);
          break;
        }

        default:
        {
          unexpected();
          return true;
        }
      }
    }

    if (!ok)
    {
      error() << lhs->location << "This isn't a subtype of this interface."
              << text(lhs->location) << rhs->location
              << "The supertype is here." << text(rhs->location);
    }

    return true;
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

  void Subtype::resolve_lookupref(Node<Type>& t)
  {
    auto& l = t->as<LookupRef>();

    if (l.resolved)
      return;

    auto ldef = l.def.lock();

    switch (ldef->kind())
    {
      case Kind::TypeAlias:
      {
        l.resolved = clone(l.subs, ldef->as<TypeAlias>().inherits);
        break;
      }

      case Kind::TypeParam:
      {
        l.resolved = clone(l.subs, ldef->as<TypeParam>().upper);
        break;
      }

      case Kind::Function:
      {
        auto self = l.self.lock();
        l.resolved = clone(l.subs, ldef->as<Function>().type, self);
        break;
      }

      case Kind::FunctionType:
      {
        auto self = l.self.lock();
        l.resolved =
          clone(l.subs, std::static_pointer_cast<FunctionType>(ldef), self);
        break;
      }

      default:
        // TODO:
        // It's a Class, Interface, or Field.
        break;
    }
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
        ok = false;
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
