// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "dnf.h"

#include "pass.h"

namespace verona::parser::dnf
{
  Node<Type> single_union(Node<Type>& left, UnionType& right, Location& loc)
  {
    // A & (B | C) -> (A & B) | (A & C)
    auto un = std::make_shared<UnionType>();
    un->location = loc;

    for (auto& type : right.types)
    {
      auto isect = std::make_shared<IsectType>();
      isect->location = type->location;
      isect->types.push_back(left);
      isect->types.push_back(type);
      un->types.push_back(isect);
    }

    return un;
  }

  Node<Type> isect_union(IsectType& left, UnionType& right, Location& loc)
  {
    // (A & B) & (C | D) -> (A & B & C) | (A & B & D)
    auto un = std::make_shared<UnionType>();
    un->location = loc;

    for (auto& type : right.types)
    {
      auto isect = std::make_shared<IsectType>();
      isect->location = type->location;
      isect->types = left.types;
      isect->types.push_back(type);
      un->types.push_back(isect);
    }

    return un;
  }

  Node<Type> union_union(UnionType& left, UnionType& right, Location& loc)
  {
    // (A | B) & (C | D) -> (A & C) | (A & D) | (B & C) | (B & D)
    auto un = std::make_shared<UnionType>();
    un->location = loc;

    for (auto& ltype : left.types)
    {
      for (auto& rtype : right.types)
      {
        auto isect = std::make_shared<IsectType>();
        isect->location = ltype->location;
        auto& types = isect->types;

        if (ltype->kind() == Kind::IsectType)
          types = ltype->as<IsectType>().types;
        else
          types.push_back(ltype);

        if (rtype->kind() == Kind::IsectType)
        {
          auto& rtypes = rtype->as<IsectType>().types;
          types.insert(types.end(), rtypes.begin(), rtypes.end());
        }
        else
        {
          types.push_back(rtype);
        }

        un->types.push_back(isect);
      }
    }

    return un;
  }
}

namespace verona::parser
{
  Node<Type> intersect(Node<Type>& left, Node<Type>& right, Location& loc)
  {
    switch (left->kind())
    {
      case Kind::IsectType:
      {
        auto lhs = left->as<IsectType>();

        switch (right->kind())
        {
          case Kind::IsectType:
          {
            auto rhs = right->as<IsectType>();

            for (auto& type : rhs.types)
              lhs.types.push_back(type);

            return left;
          }

          case Kind::UnionType:
            return dnf::isect_union(lhs, right->as<UnionType>(), loc);

          default:
          {
            lhs.types.push_back(right);
            return left;
          }
        }
      }

      case Kind::UnionType:
      {
        auto lhs = left->as<UnionType>();

        switch (right->kind())
        {
          case Kind::IsectType:
            return dnf::isect_union(right->as<IsectType>(), lhs, loc);

          case Kind::UnionType:
            return dnf::union_union(lhs, right->as<UnionType>(), loc);

          default:
            return dnf::single_union(right, lhs, loc);
        }
      }

      default:
      {
        switch (right->kind())
        {
          case Kind::IsectType:
          {
            auto rhs = right->as<IsectType>();
            rhs.types.push_back(left);
            return right;
          }

          case Kind::UnionType:
            return dnf::single_union(left, right->as<UnionType>(), loc);

          default:
          {
            auto isect = std::make_shared<IsectType>();
            isect->location = loc;
            isect->types.push_back(left);
            isect->types.push_back(right);
            return isect;
          }
        }
      }
    }
  }

  bool wellformed(Node<Type>& type)
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
            error() << loc() << "Union type should not contain another union type"
            << line();
            return;
          }
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

          if (ty->kind() == Kind::IsectType)
          {
            error() << loc() << "Isect type should not contain another isect type"
            << line();
            return;
          }
        }
      }
    };

    return WF() << type;
  }
}
