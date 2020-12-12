// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"

namespace verona::parser
{
  enum class Kind
  {
    // Definitions
    Open,
    TypeAlias,
    Interface,
    Class,
    Module,
    Field,
    Param,
    TypeParam,
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
    ModuleName,
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
    Select,
    Specialise,
    Apply,
    Ref,
    StaticRef,
    Let,
    Var,
    New,
    ObjectLiteral,

    // Constants
    EscapedString,
    UnescapedString,
    Character,
    Int,
    Float,
    Hex,
    Binary,
    True,
    False,
  };

  struct NodeDef;

  template<typename T>
  using Node = std::shared_ptr<T>;

  template<typename T>
  using List = std::vector<Node<T>>;

  using Ast = Node<NodeDef>;
  using AstPath = List<NodeDef>;
  using AstPaths = std::vector<AstPath>;

  struct SymbolTable;

  const char* kindname(Kind kind);

  bool is_kind(Kind kind, const std::initializer_list<Kind>& kinds);

  std::pair<AstPath::iterator, Ast> get_sym(AstPath& stack, const Location& id);

  struct NodeDef
  {
    Location location;

    virtual ~NodeDef() = default;
    virtual Kind kind() = 0;

    virtual SymbolTable* symbol_table()
    {
      return nullptr;
    }

    Ast get_sym(const Location& id);

    template<typename T>
    T& as()
    {
      return static_cast<T&>(*this);
    }
  };

  struct SymbolTable
  {
    std::unordered_map<Location, Ast> map;
  };

  // TODO: anonymous interface

  struct Type : NodeDef
  {};

  struct Expr : NodeDef
  {};

  struct Param : NodeDef
  {
    Node<Type> type;
    Node<Expr> init;

    Kind kind()
    {
      return Kind::Param;
    }
  };

  struct TypeParam : NodeDef
  {
    // TODO: value-dependent types
    Node<Type> type;
    Node<Type> init;

    Kind kind()
    {
      return Kind::TypeParam;
    }
  };

  struct Signature : NodeDef
  {
    List<TypeParam> typeparams;
    List<Param> params;
    Node<Type> result;
    Node<Type> throws;

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
    Node<Expr> behaviour;

    Kind kind()
    {
      return Kind::When;
    }
  };

  struct While : BlockExpr
  {
    Node<Expr> cond;
    Node<Expr> body;

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
    Node<Expr> on_true;
    Node<Expr> on_false;

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
    bool block;

    Kind kind()
    {
      return Kind::Infix;
    }
  };

  struct Prefix : Expr
  {
    Node<Expr> op;
    Node<Expr> expr;
    bool block;

    Kind kind()
    {
      return Kind::Prefix;
    }
  };

  struct Select : Expr
  {
    Node<Expr> expr;
    Location member;

    Kind kind()
    {
      return Kind::Select;
    }
  };

  struct Specialise : Expr
  {
    Node<Expr> expr;
    List<Type> typeargs;

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

  struct Let : Expr
  {
    Node<Type> type;

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
    Location in;

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
    List<Type> typeargs;

    Kind kind()
    {
      return Kind::TypeName;
    }
  };

  struct ModuleName : TypeName
  {
    Kind kind()
    {
      return Kind::ModuleName;
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
    List<TypeParam> typeparams;
    Node<Type> inherits;
  };

  struct NamedEntity : Entity
  {
    Location id;
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
    Location name;
    Node<Signature> signature;
    Node<Expr> body;

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
    Location in;

    Kind kind()
    {
      return Kind::ObjectLiteral;
    }
  };

  struct StaticRef : Expr
  {
    List<TypeName> typenames;
    bool maybe_member = false;

    Kind kind()
    {
      return Kind::StaticRef;
    }
  };

  struct Constant : Expr
  {};

  struct EscapedString : Constant
  {
    Kind kind()
    {
      return Kind::EscapedString;
    }
  };

  struct UnescapedString : Constant
  {
    Kind kind()
    {
      return Kind::UnescapedString;
    }
  };

  struct Character : Constant
  {
    Kind kind()
    {
      return Kind::Character;
    }
  };

  struct Int : Constant
  {
    Kind kind()
    {
      return Kind::Int;
    }
  };

  struct Float : Constant
  {
    Kind kind()
    {
      return Kind::Float;
    }
  };

  struct Hex : Constant
  {
    Kind kind()
    {
      return Kind::Hex;
    }
  };

  struct Binary : Constant
  {
    Kind kind()
    {
      return Kind::Binary;
    }
  };

  struct True : Constant
  {
    Kind kind()
    {
      return Kind::True;
    }
  };

  struct False : Constant
  {
    Kind kind()
    {
      return Kind::False;
    }
  };
}
