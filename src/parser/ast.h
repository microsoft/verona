// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"

#include <optional>
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
    TypeParamList,
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
    TypeList,
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
    Select,
    Ref,
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
    Bool,
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

    Kind kind() override
    {
      return Kind::TypeName;
    }
  };

  struct ModuleName : TypeName
  {
    Kind kind() override
    {
      return Kind::ModuleName;
    }
  };

  struct Iso : Type
  {
    Kind kind() override
    {
      return Kind::Iso;
    }
  };

  struct Mut : Type
  {
    Kind kind() override
    {
      return Kind::Mut;
    }
  };

  struct Imm : Type
  {
    Kind kind() override
    {
      return Kind::Imm;
    }
  };

  struct Self : Type
  {
    Kind kind() override
    {
      return Kind::Self;
    }
  };

  struct TypeRef : Type
  {
    List<TypeName> typenames;

    Kind kind() override
    {
      return Kind::TypeRef;
    }
  };

  struct TypeList : Type
  {
    Kind kind() override
    {
      return Kind::TypeList;
    }
  };

  struct Member : NodeDef
  {};

  struct Using : Member
  {
    Node<Type> type;

    Kind kind() override
    {
      return Kind::Using;
    }
  };

  struct SymbolTable
  {
    std::unordered_map<Location, Ast> map;
    std::vector<Node<Using>> use;

    std::optional<Location> set(const Location& id, Ast node)
    {
      auto find = map.find(id);

      if (find != map.end())
        return find->second->location;

      map.emplace(id, node);
      return {};
    }
  };

  struct Expr : NodeDef
  {};

  struct TypeParam : NodeDef
  {
    // TODO: value-dependent types
    Node<Type> type;
    Node<Type> init;

    Kind kind() override
    {
      return Kind::TypeParam;
    }
  };

  struct TypeParamList : TypeParam
  {
    Kind kind() override
    {
      return Kind::TypeParamList;
    }
  };

  struct Param : Expr
  {
    Node<Type> type;
    Node<Expr> init;

    Kind kind() override
    {
      return Kind::Param;
    }
  };

  struct Oftype : Expr
  {
    Node<Expr> expr;
    Node<Type> type;

    Kind kind() override
    {
      return Kind::Oftype;
    }
  };

  struct Tuple : Expr
  {
    List<Expr> seq;

    Kind kind() override
    {
      return Kind::Tuple;
    }
  };

  struct Scope : Expr
  {
    SymbolTable st;

    SymbolTable* symbol_table() override
    {
      return &st;
    }
  };

  struct When : Scope
  {
    Node<Expr> waitfor;
    Node<Expr> behaviour;

    Kind kind() override
    {
      return Kind::When;
    }
  };

  struct Try : Scope
  {
    Node<Expr> body;
    List<Expr> catches;

    Kind kind() override
    {
      return Kind::Try;
    }
  };

  struct Match : Scope
  {
    Node<Expr> test;
    List<Expr> cases;

    Kind kind() override
    {
      return Kind::Match;
    }
  };

  struct Lambda : Scope
  {
    List<TypeParam> typeparams;
    List<Expr> params;
    List<Expr> body;

    Kind kind() override
    {
      return Kind::Lambda;
    }
  };

  struct Assign : Expr
  {
    Node<Expr> left;
    Node<Expr> right;

    Kind kind() override
    {
      return Kind::Assign;
    }
  };

  struct Select : Expr
  {
    Node<Expr> expr;
    List<TypeName> typenames;
    Node<Expr> args;

    Kind kind() override
    {
      return Kind::Select;
    }
  };

  struct Ref : Expr
  {
    Kind kind() override
    {
      return Kind::Ref;
    }
  };

  struct Let : Expr
  {
    Kind kind() override
    {
      return Kind::Let;
    }
  };

  struct Var : Let
  {
    Kind kind() override
    {
      return Kind::Var;
    }
  };

  struct Throw : Expr
  {
    Node<Expr> expr;

    Kind kind() override
    {
      return Kind::Throw;
    }
  };

  struct New : Expr
  {
    Location in;
    Node<Expr> args;

    Kind kind() override
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

    Kind kind() override
    {
      return Kind::ThrowType;
    }
  };

  struct UnionType : TypeOp
  {
    Kind kind() override
    {
      return Kind::UnionType;
    }
  };

  struct IsectType : TypeOp
  {
    Kind kind() override
    {
      return Kind::IsectType;
    }
  };

  struct TupleType : TypeOp
  {
    Kind kind() override
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
    Kind kind() override
    {
      return Kind::FunctionType;
    }
  };

  struct ViewType : TypePair
  {
    Kind kind() override
    {
      return Kind::ViewType;
    }
  };

  struct ExtractType : TypePair
  {
    Kind kind() override
    {
      return Kind::ExtractType;
    }
  };

  struct TypeAlias : Member
  {
    Location id;
    List<TypeParam> typeparams;
    Node<Type> type;

    Kind kind() override
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

    SymbolTable* symbol_table() override
    {
      return &st;
    }

    Kind kind() override
    {
      return Kind::Interface;
    }
  };

  struct Class : Interface
  {
    Kind kind() override
    {
      return Kind::Class;
    }
  };

  struct Module : Entity
  {
    Kind kind() override
    {
      return Kind::Module;
    }
  };

  struct Field : Member
  {
    Node<Type> type;
    Node<Expr> init;

    Kind kind() override
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

    SymbolTable* symbol_table() override
    {
      return &st;
    }

    Kind kind() override
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

    Kind kind() override
    {
      return Kind::ObjectLiteral;
    }
  };

  struct Constant : Expr
  {};

  struct EscapedString : Constant
  {
    Kind kind() override
    {
      return Kind::EscapedString;
    }
  };

  struct UnescapedString : Constant
  {
    Kind kind() override
    {
      return Kind::UnescapedString;
    }
  };

  struct Character : Constant
  {
    Kind kind() override
    {
      return Kind::Character;
    }
  };

  struct Int : Constant
  {
    Kind kind() override
    {
      return Kind::Int;
    }
  };

  struct Float : Constant
  {
    Kind kind() override
    {
      return Kind::Float;
    }
  };

  struct Hex : Constant
  {
    Kind kind() override
    {
      return Kind::Hex;
    }
  };

  struct Binary : Constant
  {
    Kind kind() override
    {
      return Kind::Binary;
    }
  };

  struct Bool : Constant
  {
    Kind kind() override
    {
      return Kind::Bool;
    }
  };
}
