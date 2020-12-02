// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"
#include "source.h"

#include <cassert>
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
    UnionType,
    IsectType,
    TupleType,
    FunctionType,
    ViewType,
    ExtractType,
    TypeName,
    TypeRef,

    // Expressions
    Tuple,
    Block,
    When,
    While,
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
    Inblock,
    Preblock,
    Select,
    Specialise,
    Apply,
    Ref,
    SymRef,
    StaticRef,
    Let,
    Var,
    Constant,
    New,
    ObjectLiteral,
    Concat,
  };

  using ID = Location;

  struct NodeDef;

  template<typename T>
  using Node = std::shared_ptr<T>;

  template<typename T>
  using List = std::vector<Node<T>>;

  struct SymbolTable;

  struct NodeDef
  {
    Location location;

    virtual ~NodeDef() = default;
    virtual Kind kind() = 0;

    virtual SymbolTable* symbol_table()
    {
      return nullptr;
    }

    template<typename T>
    T& as()
    {
      assert(T().kind() == kind());
      return static_cast<T&>(*this);
    }
  };

  struct SymbolTable
  {
    std::unordered_map<ID, Node<NodeDef>> map;
  };

  // TODO: anonymous interface

  struct Type : NodeDef
  {};

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

  struct BlockExpr : Expr
  {
    SymbolTable st;

    SymbolTable* symbol_table()
    {
      return &st;
    }
  };

  struct Block : BlockExpr
  {
    List<Expr> seq;

    Kind kind()
    {
      return Kind::Block;
    }
  };

  struct When : BlockExpr
  {
    Node<Expr> waitfor;
    Node<Block> behaviour;

    Kind kind()
    {
      return Kind::When;
    }
  };

  struct While : BlockExpr
  {
    Node<Expr> cond;
    Node<Block> body;

    Kind kind()
    {
      return Kind::While;
    }
  };

  struct Case : BlockExpr
  {
    Node<Expr> pattern;
    Node<Expr> guard;
    Node<Expr> body;

    Kind kind()
    {
      return Kind::Case;
    }
  };

  struct Match : BlockExpr
  {
    Node<Expr> cond;
    List<Case> cases;

    Kind kind()
    {
      return Kind::Match;
    }
  };

  struct If : BlockExpr
  {
    Node<Expr> cond;
    Node<Block> on_true;
    Node<Block> on_false;

    Kind kind()
    {
      return Kind::If;
    }
  };

  struct Lambda : BlockExpr
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

  struct Inblock : Infix
  {
    Kind kind()
    {
      return Kind::Inblock;
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
    List<Expr> typeargs;

    Kind kind()
    {
      return Kind::Specialise;
    }
  };

  struct Apply : Expr
  {
    Node<Expr> expr;
    Node<Expr> args;

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

  struct Constant : Expr
  {
    Token value;

    Kind kind()
    {
      return Kind::Constant;
    }
  };

  struct Let : Expr
  {
    Node<Expr> decl;

    Kind kind()
    {
      return Kind::Let;
    }
  };

  struct Var : Let
  {
    Kind kind()
    {
      return Kind::Var;
    }
  };

  struct New : Expr
  {
    Node<Expr> args;
    ID in;

    Kind kind()
    {
      return Kind::New;
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

  struct TypeName : NodeDef
  {
    ID id;
    List<Expr> typeargs;

    Kind kind()
    {
      return Kind::TypeName;
    }
  };

  struct TypeRef : Type
  {
    List<TypeName> typenames;

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
    SymbolTable st;
    List<Member> members;

    SymbolTable* symbol_table()
    {
      return &st;
    }

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
    SymbolTable st;
    Token name;
    Node<Signature> signature;
    Node<Block> body;

    SymbolTable* symbol_table()
    {
      return &st;
    }

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

  struct ObjectLiteral : BlockExpr
  {
    // TODO: put a Class in here?
    Node<Type> inherits;
    List<Member> members;
    ID in;

    Kind kind()
    {
      return Kind::ObjectLiteral;
    }
  };

  struct Concat : Expr
  {
    List<Expr> list;

    Kind kind()
    {
      return Kind::Concat;
    }
  };

  struct StaticRef : Expr
  {
    Node<Type> path;
    Token ref;

    Kind kind()
    {
      return Kind::StaticRef;
    }
  };
}
