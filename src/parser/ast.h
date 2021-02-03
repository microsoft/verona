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
    Constraint,
    Open,
    TypeAlias,
    Interface,
    Class,
    Module,
    Field,
    Signature,
    Function,
    Method,
  };

  using ID = Location;

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

  struct Constraint : NodeDef
  {
    // TODO: value-dependent types
    ID id;
    Node<Type> type;
    Node<Type> init;

    Kind kind()
    {
      return Kind::Constraint;
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
    std::vector<ID> typeparams;
    Node<Type> inherits;
    List<Constraint> constraints;
  };

  struct NamedEntity : Entity
  {
    ID id;
  };

  struct TypeAlias : NamedEntity
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::TypeAlias;
    }
  };

  struct Interface : NamedEntity
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

  struct Module : Entity
  {
    Kind kind()
    {
      return Kind::Module;
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
    std::vector<ID> typeparams;
    List<Field> params;
    Node<Type> result;
    Node<Type> throws;
    List<Constraint> constraints;

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
