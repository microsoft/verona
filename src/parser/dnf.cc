// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "dnf.h"

#include "pass.h"

namespace verona::parser::dnf
{
  Location range(Node<Type>& left, Node<Type>& right)
  {
    return left->location.range(right->location);
  }

  Node<Type> single_single(Node<Type>& left, Node<Type>& right)
  {
    // A & B
    auto isect = std::make_shared<IsectType>();
    isect->location = range(left, right);
    isect->types.push_back(left);
    isect->types.push_back(right);
    return isect;
  }

  Node<Type> throws_throws(Node<Type>& left, Node<Type>& right)
  {
    // throw A & throw B -> throw (A & B)
    auto& lhs = left->as<ThrowType>();
    auto& rhs = right->as<ThrowType>();

    auto isect = std::make_shared<IsectType>();
    isect->location = range(left, right);
    isect->types.push_back(lhs.type);
    isect->types.push_back(rhs.type);

    auto th = std::make_shared<ThrowType>();
    th->location = isect->location;
    th->type = isect;
    return th;
  }

  Node<Type> isect_single(Node<Type>& left, Node<Type>& right)
  {
    // (A & B) & C -> A & B & C
    auto& lhs = left->as<IsectType>();

    auto isect = std::make_shared<IsectType>();
    isect->location = range(left, right);
    isect->types = lhs.types;
    isect->types.push_back(right);
    return isect;
  }

  Node<Type> isect_isect(Node<Type>& left, Node<Type>& right)
  {
    // (A & B) & (C & D) -> A & B & C & D
    auto& lhs = left->as<IsectType>();
    auto& rhs = right->as<IsectType>();

    auto isect = std::make_shared<IsectType>();
    isect->location = range(left, right);
    isect->types = lhs.types;
    isect->types.insert(isect->types.end(), rhs.types.begin(), rhs.types.end());
    return isect;
  }

  Node<Type> union_other(Node<Type>& left, Node<Type>& right)
  {
    // (A | B) & C -> (A & B) | (A & C)
    auto& lhs = left->as<UnionType>();
    auto un = std::make_shared<UnionType>();
    un->location = range(left, right);

    for (auto& type : lhs.types)
    {
      auto conj = conjunction(type, right);

      if (conj)
        un->types.push_back(conj);
    }

    if (un->types.empty())
      return {};

    return un;
  }

  Node<Type> union_union(Node<Type>& left, Node<Type>& right)
  {
    // (A | B) & (C | D) -> (A & C) | (A & D) | (B & C) | (B & D)
    auto& lhs = left->as<UnionType>();
    auto& rhs = right->as<UnionType>();

    auto un = std::make_shared<UnionType>();
    un->location = range(left, right);

    for (auto& ltype : lhs.types)
    {
      for (auto& rtype : rhs.types)
      {
        auto conj = conjunction(ltype, rtype);

        if (conj)
          un->types.push_back(conj);
      }
    }

    if (un->types.empty())
      return {};

    return un;
  }

  Node<Type> conjunction(Node<Type>& left, Node<Type>& right)
  {
    if (!left)
      return right;

    if (!right)
      return left;

    switch (left->kind())
    {
      case Kind::IsectType:
      {
        switch (right->kind())
        {
          case Kind::IsectType:
            return isect_isect(left, right);

          case Kind::ThrowType:
            return {};

          case Kind::UnionType:
            return union_other(right, left);

          default:
            return isect_single(left, right);
        }
      }

      case Kind::ThrowType:
      {
        switch (right->kind())
        {
          case Kind::ThrowType:
            return throws_throws(left, right);

          case Kind::UnionType:
            return union_other(right, left);

          default:
            return {};
        }
      }

      case Kind::UnionType:
      {
        switch (right->kind())
        {
          case Kind::UnionType:
            return union_union(left, right);

          default:
            return union_other(left, right);
        }
      }

      default:
      {
        switch (right->kind())
        {
          case Kind::IsectType:
            return isect_single(right, left);

          case Kind::ThrowType:
            return {};

          case Kind::UnionType:
            return union_other(right, left);

          default:
            return single_single(left, right);
        }
      }
    }
  }

  Node<Type> throwtype(Node<Type>& type)
  {
    if (!type)
      return {};

    if (type->kind() == Kind::UnionType)
    {
      auto& un = type->as<UnionType>();
      auto res = std::make_shared<UnionType>();
      res->location = type->location;

      for (size_t i = 0; i < un.types.size(); i++)
      {
        auto& ty = un.types[i];

        if (ty->kind() != Kind::ThrowType)
        {
          auto th = std::make_shared<ThrowType>();
          th->location = ty->location;
          th->type = ty;
          res->types.push_back(th);
        }
        else
        {
          res->types.push_back(ty);
        }
      }

      return res;
    }

    if (type->kind() != Kind::ThrowType)
    {
      auto th = std::make_shared<ThrowType>();
      th->location = type->location;
      th->type = type;
      return th;
    }

    return type;
  }

  Node<Type> disjunction(Node<Type>& left, Node<Type>& right)
  {
    if (!left)
      return right;

    if (!right)
      return left;

    auto un = std::make_shared<UnionType>();
    un->location = range(left, right);

    if (left->kind() == Kind::UnionType)
    {
      auto& lhs = left->as<UnionType>().types;
      un->types.insert(un->types.end(), lhs.begin(), lhs.end());
    }
    else
    {
      un->types.push_back(left);
    }

    if (right->kind() == Kind::UnionType)
    {
      auto& rhs = right->as<UnionType>().types;
      un->types.insert(un->types.end(), rhs.begin(), rhs.end());
    }
    else
    {
      un->types.push_back(right);
    }

    return un;
  }

  bool wellformed(Ast& ast, std::ostream& out)
  {
    struct WF : Pass<WF>
    {
      AST_PASS;

      void pre(UnionType& un)
      {
        for (auto& ty : un.types)
        {
          if (ty->kind() == Kind::UnionType)
          {
            error() << loc()
                    << "Union type should not contain another union type"
                    << line();
            return;
          }
        }
      }

      void pre(ThrowType& tt)
      {
        if (tt.type->kind() == Kind::UnionType)
        {
          error() << loc() << "Throw type should not contain a union type"
                  << line();
          return;
        }

        if (tt.type->kind() == Kind::ThrowType)
        {
          error() << loc() << "Throw type should not contain another throw type"
                  << line();
          return;
        }
      }

      void pre(IsectType& isect)
      {
        for (auto& ty : isect.types)
        {
          if (ty->kind() == Kind::UnionType)
          {
            error() << loc() << "Isect type should not contain a union type"
                    << line();
            return;
          }

          if (ty->kind() == Kind::ThrowType)
          {
            error() << loc() << "Isect type should not contain a throw type"
                    << line();
            return;
          }

          if (ty->kind() == Kind::IsectType)
          {
            error() << loc()
                    << "Isect type should not contain another isect type"
                    << line();
            return;
          }
        }
      }
    };

    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
