// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

#include <vector>

namespace verona::parser
{
  enum class Kind
  {
    Expr,
    Type,
    UnionType,
    IsectType,
    TupleType,
    FunctionType,
    ViewType,
    ExtractType,
    TypeRef,
    TypeParam,
    Open,
    TypeAlias,
    Interface,
    Class,
    Field,
    Signature,
    Function,
    Method,
  };

  using ID = std::string;

  struct NodeDef;

  template<typename T>
  using Node = std::shared_ptr<T>;

  template<typename T>
  using List = std::vector<Node<T>>;

  struct NodeDef
  {
    Location location;

    virtual ~NodeDef() = default;
    virtual Kind kind() = 0;
  };

  struct Expr : NodeDef
  {
    // TODO: expr
    Kind kind()
    {
      return Kind::Expr;
    }
  };

  struct Type : NodeDef
  {
    // TODO: module ref, anonymous interface
    Kind kind()
    {
      return Kind::Type;
    }
  };

  struct TypeOp : Type
  {
    List<Type> types;
  };

  struct UnionType : TypeOp
  {
    Kind kind()
    {
      return Kind::UnionType;
    }
  };

  struct IsectType : TypeOp
  {
    Kind kind()
    {
      return Kind::IsectType;
    }
  };

  struct TupleType : TypeOp
  {
    Kind kind()
    {
      return Kind::TupleType;
    }
  };

  struct TypePair : Type
  {
    Node<Type> left;
    Node<Type> right;
  };

  struct FunctionType : TypePair
  {
    Kind kind()
    {
      return Kind::FunctionType;
    }
  };

  struct ViewType : TypePair
  {
    Kind kind()
    {
      return Kind::ViewType;
    }
  };

  struct ExtractType : TypePair
  {
    Kind kind()
    {
      return Kind::ExtractType;
    }
  };

  struct TypeName
  {
    ID id;
    List<Expr> typeargs;
  };

  struct TypeRef : Type
  {
    std::vector<TypeName> typenames;

    Kind kind()
    {
      return Kind::TypeRef;
    }
  };

  struct TypeParam : NodeDef
  {
    // TODO: value-dependent types
    ID id;
    Node<Type> type;
    Node<Type> init;

    Kind kind()
    {
      return Kind::TypeParam;
    }
  };

  struct Member : NodeDef
  {};

  struct Open : Member
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::Open;
    }
  };

  struct Entity : Member
  {
    ID id;
    List<TypeParam> typeparams;
    Node<Type> inherits;
  };

  struct TypeAlias : Entity
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::TypeAlias;
    }
  };

  struct Interface : Entity
  {
    List<Member> members;

    Kind kind()
    {
      return Kind::Interface;
    }
  };

  struct Class : Interface
  {
    Kind kind()
    {
      return Kind::Class;
    }
  };

  struct Field : Member
  {
    ID id;
    Node<Type> type;
    Node<Expr> init;

    Kind kind()
    {
      return Kind::Field;
    }
  };

  struct Signature : NodeDef
  {
    List<TypeParam> typeparams;
    List<Field> params;
    Node<Type> result;
    Node<Type> throws;

    Kind kind()
    {
      return Kind::Signature;
    }
  };

  struct Function : Member
  {
    ID id;
    Node<Signature> signature;
    Node<Expr> body;

    Kind kind()
    {
      return Kind::Function;
    }
  };

  struct Method : Function
  {
    Kind kind()
    {
      return Kind::Method;
    }
  };
}
