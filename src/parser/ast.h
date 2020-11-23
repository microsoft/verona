// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

#include <vector>

namespace verona::parser
{
  enum class Kind
  {
    // Definitions
    Constraint,
    Open,
    TypeAlias,
    Interface,
    Class,
    Module,
    Field,
    Param,
    Signature,
    Function,
    Method,

    // Types
    Type,
    UnionType,
    IsectType,
    TupleType,
    FunctionType,
    ViewType,
    ExtractType,
    TypeRef,

    // Expressions
    Tuple,
    Block,
    When,
    Conditional,
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
  {};

  struct Atom : Expr
  {};

  struct Tuple : Expr
  {
    List<Expr> seq;

    Kind kind()
    {
      return Kind::Tuple;
    }
  };

  struct Block : Tuple
  {
    Kind kind()
    {
      return Kind::Block;
    }
  };

  struct When : Expr
  {
    Node<Tuple> waitfor;
    Node<Block> behaviour;

    Kind kind()
    {
      return Kind::When;
    }
  };

  struct Conditional : Expr
  {
    Node<Tuple> cond;
    Node<Block> on_true;
    Node<Block> on_false;

    Kind kind()
    {
      return Kind::Conditional;
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

  struct Param : Field
  {
    Kind kind()
    {
      return Kind::Param;
    }
  };

  struct Signature : NodeDef
  {
    std::vector<ID> typeparams;
    List<Param> params;
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
    Node<Block> body;

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
