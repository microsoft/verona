// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ident.h"
#include "lexer.h"

#include <map>
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
    InferType,
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
    Free, // Generated in the ANF pass
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

    // Lookup
    LookupUnion,
    LookupIsect,
  };

  struct NodeDef;

  template<typename T>
  using Node = std::shared_ptr<T>;

  template<typename T>
  using Weak = std::weak_ptr<T>;

  template<typename T>
  using List = std::vector<Node<T>>;

  using Ast = Node<NodeDef>;
  using AstWeak = Weak<NodeDef>;

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

  struct TypeParam : NodeDef
  {
    // TODO: value-dependent types
    Node<Type> upper;
    Node<Type> dflt;

    Kind kind()
    {
      return Kind::TypeParam;
    }
  };

  struct TypeParamList : TypeParam
  {
    Kind kind()
    {
      return Kind::TypeParamList;
    }
  };

  using Substitutions =
    std::map<Weak<TypeParam>, Node<Type>, std::owner_less<>>;

  struct TypeRef : Type
  {
    List<TypeName> typenames;
    AstWeak context;
    AstWeak def;
    Substitutions subs;
    Node<Type> resolved;

    Kind kind() override
    {
      return Kind::TypeRef;
    }
  };

  struct TypeList : Type
  {
    AstWeak def;

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
    AstWeak parent;

    Ast get(const Location& name)
    {
      auto find = map.find(name);

      if (find != map.end())
        return find->second;

      return {};
    }

    Ast get_scope(const Location& name)
    {
      auto def = get(name);

      if (def)
        return def;

      auto node = parent.lock();

      if (node)
        return node->symbol_table()->get_scope(name);

      return {};
    }

    bool set(const Location& name, Ast node)
    {
      auto find = map.find(name);

      if (find != map.end())
        return false;

      map.emplace(name, node);
      return true;
    }
  };

  struct Expr : NodeDef
  {};

  struct Let : Expr
  {
    Node<Type> type;
    bool assigned = false;

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

  struct Free : Expr
  {
    Kind kind()
    {
      return Kind::Free;
    }
  };

  struct Param : Let
  {
    Node<Expr> dflt;

    Param()
    {
      assigned = true;
    }

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

  struct When : Expr
  {
    Node<Expr> waitfor;
    Node<Expr> behaviour;

    Kind kind() override
    {
      return Kind::When;
    }
  };

  struct Try : Expr
  {
    Node<Expr> body;
    List<Expr> catches;

    Kind kind() override
    {
      return Kind::Try;
    }
  };

  struct Match : Expr
  {
    Node<Expr> test;
    List<Expr> cases;

    Kind kind() override
    {
      return Kind::Match;
    }
  };

  struct Lambda : Expr
  {
    List<TypeParam> typeparams;
    List<Expr> params;
    Node<Type> result;
    List<Expr> body;

    SymbolTable st;

    SymbolTable* symbol_table()
    {
      return &st;
    }

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
    Node<TypeRef> typeref;
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

  struct InferType : Type
  {
    static Ident ident;

    InferType()
    {
      location = ident();
    }

    Kind kind()
    {
      return Kind::InferType;
    }
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

  struct Interface : Member
  {
    List<TypeParam> typeparams;
    Node<Type> inherits;
    List<Member> members;

    SymbolTable st;

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

  struct TypeAlias : Interface
  {
    Kind kind()
    {
      return Kind::TypeAlias;
    }
  };

  struct Module : Member
  {
    List<TypeParam> typeparams;
    Node<Type> inherits;

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
    Location name;
    Node<Expr> lambda;
    Node<Type> type;

    Kind kind() override
    {
      return Kind::Function;
    }
  };

  struct ObjectLiteral : Expr
  {
    // TODO: put a Class in here?
    Location in;
    Node<Type> inherits;
    List<Member> members;

    SymbolTable st;

    SymbolTable* symbol_table()
    {
      return &st;
    }

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

  struct UnescapedString : EscapedString
  {
    Kind kind() override
    {
      return Kind::UnescapedString;
    }
  };

  struct Int : Constant
  {
    Kind kind() override
    {
      return Kind::Int;
    }
  };

  struct Character : Int
  {
    Kind kind() override
    {
      return Kind::Character;
    }
  };

  struct Hex : Int
  {
    Kind kind() override
    {
      return Kind::Hex;
    }
  };

  struct Binary : Int
  {
    Kind kind() override
    {
      return Kind::Binary;
    }
  };

  struct Float : Constant
  {
    Kind kind() override
    {
      return Kind::Float;
    }
  };

  struct Bool : Constant
  {
    Kind kind()
    {
      return Kind::Bool;
    }
  };

  struct LookupUnion : NodeDef
  {
    List<NodeDef> list;

    Kind kind()
    {
      return Kind::LookupUnion;
    }
  };

  struct LookupIsect : LookupUnion
  {
    Kind kind()
    {
      return Kind::LookupIsect;
    }
  };
}
