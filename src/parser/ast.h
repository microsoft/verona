// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"

#include <unordered_map>
#include <vector>

namespace verona::parser
{
  enum class Kind
  {
    // Definitions
    Using,
    TypeAlias,
    Interface,
    Class,
    Module,
    Field,
    Param,
    TypeParam,
    Function,

    // Types
    ThrowType,
    UnionType,
    IsectType,
    TupleType,
    FunctionType,
    ViewType,
    ExtractType,
    TypeName,
    ModuleName,
    TypeRef,
    Iso,
    Mut,
    Imm,
    Self,

    // Expressions
    Oftype,
    Tuple,
    When,
    Try,
    Match,
    Lambda,
    Assign,
    Infix,
    Apply,
    Select,
    Specialise,
    StaticSelect,
    Ref,
    StaticRef,
    Let,
    Var,
    Throw,
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

  bool is_kind(Ast ast, const std::initializer_list<Kind>& kinds);

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
      return static_cast<T&>(*this);
    }
  };

  // TODO: anonymous interface

  struct Type : NodeDef
  {};

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

  struct Iso : Type
  {
    Kind kind()
    {
      return Kind::Iso;
    }
  };

  struct Mut : Type
  {
    Kind kind()
    {
      return Kind::Mut;
    }
  };

  struct Imm : Type
  {
    Kind kind()
    {
      return Kind::Imm;
    }
  };

  struct Self : Type
  {
    Kind kind()
    {
      return Kind::Self;
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

  struct Using : Member
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::Using;
    }
  };

  struct SymbolTable
  {
    std::unordered_map<Location, Ast> map;
    std::vector<Node<Using>> use;
  };

  struct Expr : NodeDef
  {};

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

  struct Param : Expr
  {
    Node<Type> type;
    Node<Expr> init;

    Kind kind()
    {
      return Kind::Param;
    }
  };

  struct Oftype : Expr
  {
    Node<Expr> expr;
    Node<Type> type;

    Kind kind()
    {
      return Kind::Oftype;
    }
  };

  struct Tuple : Expr
  {
    List<Expr> seq;

    Kind kind()
    {
      return Kind::Tuple;
    }
  };

  struct Scope : Expr
  {
    SymbolTable st;

    SymbolTable* symbol_table()
    {
      return &st;
    }
  };

  struct When : Scope
  {
    Node<Expr> waitfor;
    Node<Expr> behaviour;

    Kind kind()
    {
      return Kind::When;
    }
  };

  struct Try : Scope
  {
    Node<Expr> body;
    List<Expr> catches;

    Kind kind()
    {
      return Kind::Try;
    }
  };

  struct Match : Scope
  {
    Node<Expr> test;
    List<Expr> cases;

    Kind kind()
    {
      return Kind::Match;
    }
  };

  struct Lambda : Scope
  {
    List<TypeParam> typeparams;
    List<Expr> params;
    List<Expr> body;

    Kind kind()
    {
      return Kind::Lambda;
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

  struct Apply : Expr
  {
    Node<Expr> expr;
    Node<Expr> args;

    Kind kind()
    {
      return Kind::Apply;
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

  struct StaticSelect : Expr
  {
    Node<Expr> expr;
    List<TypeName> typenames;

    Kind kind()
    {
      return Kind::StaticSelect;
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

  struct Ref : Expr
  {
    Kind kind()
    {
      return Kind::Ref;
    }
  };

  struct Let : Expr
  {
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

  struct Throw : Expr
  {
    Node<Expr> expr;

    Kind kind()
    {
      return Kind::Throw;
    }
  };

  struct New : Expr
  {
    Location in;
    Node<Expr> args;

    Kind kind()
    {
      return Kind::New;
    }
  };

  struct TypeOp : Type
  {
    List<Type> types;
  };

  struct ThrowType : Type
  {
    Node<Type> type;

    Kind kind()
    {
      return Kind::ThrowType;
    }
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

  struct TypeAlias : Member
  {
    Location id;
    List<TypeParam> typeparams;
    Node<Type> type;

    Kind kind()
    {
      return Kind::TypeAlias;
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
    List<TypeParam> typeparams;
    List<Expr> params;
    Node<Type> result;
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

  struct ObjectLiteral : Scope
  {
    // TODO: put a Class in here?
    Location in;
    Node<Type> inherits;
    List<Member> members;

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
