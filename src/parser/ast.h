// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"
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
    While,
    For,
    Case,
    Match,
    If,
    Lambda,
    Break,
    Continue,
    Return,
    Yield,
    Assign,
    Infix,
    Prefix,
    Preblock,
    Select,
    Specialise,
    Apply,
    Ref,
    SymRef,
    StaticRef,
    Constant,
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

    template<typename T>
    T& as()
    {
      if (T().kind() != kind())
        abort();

      return static_cast<T&>(*this);
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

  struct Expr : NodeDef
  {};

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

  struct Param : NodeDef
  {
    ID id;
    Node<Type> type;
    Node<Expr> init;

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

  struct Tuple : Expr
  {
    List<Expr> seq;
    Node<Type> type;

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

  struct While : Expr
  {
    Node<Tuple> cond;
    Node<Block> body;

    Kind kind()
    {
      return Kind::While;
    }
  };

  struct For : Expr
  {
    Node<Expr> left;
    Node<Expr> right;
    Node<Block> body;

    Kind kind()
    {
      return Kind::For;
    }
  };

  struct Case : NodeDef
  {
    Node<Expr> pattern;
    Node<Expr> guard;
    Node<Block> body;

    Kind kind()
    {
      return Kind::Case;
    }
  };

  struct Match : Expr
  {
    Node<Tuple> cond;
    List<Case> cases;

    Kind kind()
    {
      return Kind::Match;
    }
  };

  struct If : Expr
  {
    Node<Tuple> cond;
    Node<Block> on_true;
    Node<Block> on_false;

    Kind kind()
    {
      return Kind::If;
    }
  };

  struct Lambda : Expr
  {
    Node<Signature> signature;
    Node<Expr> body;

    Kind kind()
    {
      return Kind::Lambda;
    }
  };

  struct Break : Expr
  {
    Kind kind()
    {
      return Kind::Break;
    }
  };

  struct Continue : Expr
  {
    Kind kind()
    {
      return Kind::Continue;
    }
  };

  struct Return : Expr
  {
    Node<Expr> expr;

    Kind kind()
    {
      return Kind::Return;
    }
  };

  struct Yield : Return
  {
    Kind kind()
    {
      return Kind::Yield;
    }
  };

  struct Assign : Expr
  {
    Node<Expr> left;
    Node<Expr> right;

    Kind kind()
    {
      return Kind::Assign;
    }
  };

  struct Infix : Expr
  {
    Node<Expr> op;
    Node<Expr> left;
    Node<Expr> right;

    Kind kind()
    {
      return Kind::Infix;
    }
  };

  struct Prefix : Expr
  {
    Node<Expr> op;
    Node<Expr> expr;

    Kind kind()
    {
      return Kind::Prefix;
    }
  };

  struct Preblock : Prefix
  {
    Kind kind()
    {
      return Kind::Preblock;
    }
  };

  struct Select : Expr
  {
    Node<Expr> expr;
    Token member;

    Kind kind()
    {
      return Kind::Select;
    }
  };

  struct Specialise : Expr
  {
    Node<Expr> expr;
    List<Expr> args;

    Kind kind()
    {
      return Kind::Specialise;
    }
  };

  struct Apply : Expr
  {
    Node<Expr> expr;
    Node<Tuple> args;

    Kind kind()
    {
      return Kind::Apply;
    }
  };

  struct Ref : Expr
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::Ref;
    }
  };

  struct SymRef : Expr
  {
    Kind kind()
    {
      return Kind::SymRef;
    }
  };

  struct StaticRef : Expr
  {
    std::vector<Token> ref;

    Kind kind()
    {
      return Kind::StaticRef;
    }
  };

  struct Constant : Expr
  {
    Kind kind()
    {
      return Kind::Constant;
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

  struct Function : Member
  {
    Token name;
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
