// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  template<typename F, typename... Args>
  auto dispatch(F& f, const Ast& node, Args&... args) -> decltype(f(args...))
  {
    if (!node)
      return f(args...);

    switch (node->kind())
    {
      // Definitions
      case Kind::Using:
        return f(node->as<Using>(), args...);

      case Kind::TypeAlias:
        return f(node->as<TypeAlias>(), args...);

      case Kind::Interface:
        return f(node->as<Interface>(), args...);

      case Kind::Class:
        return f(node->as<Class>(), args...);

      case Kind::Module:
        return f(node->as<Module>(), args...);

      case Kind::Field:
        return f(node->as<Field>(), args...);

      case Kind::Param:
        return f(node->as<Param>(), args...);

      case Kind::TypeParam:
        return f(node->as<TypeParam>(), args...);

      case Kind::TypeParamList:
        return f(node->as<TypeParamList>(), args...);

      case Kind::Function:
        return f(node->as<Function>(), args...);

      // Types
      case Kind::ThrowType:
        return f(node->as<ThrowType>(), args...);

      case Kind::UnionType:
        return f(node->as<UnionType>(), args...);

      case Kind::IsectType:
        return f(node->as<IsectType>(), args...);

      case Kind::TupleType:
        return f(node->as<TupleType>(), args...);

      case Kind::FunctionType:
        return f(node->as<FunctionType>(), args...);

      case Kind::ViewType:
        return f(node->as<ViewType>(), args...);

      case Kind::ExtractType:
        return f(node->as<ExtractType>(), args...);

      case Kind::TypeName:
        return f(node->as<TypeName>(), args...);

      case Kind::ModuleName:
        return f(node->as<ModuleName>(), args...);

      case Kind::TypeRef:
        return f(node->as<TypeRef>(), args...);

      case Kind::TypeList:
        return f(node->as<TypeList>(), args...);

      case Kind::Iso:
        return f(node->as<Iso>(), args...);

      case Kind::Mut:
        return f(node->as<Mut>(), args...);

      case Kind::Imm:
        return f(node->as<Imm>(), args...);

      case Kind::Self:
        return f(node->as<Self>(), args...);

      // Expressions
      case Kind::Oftype:
        return f(node->as<Oftype>(), args...);

      case Kind::Tuple:
        return f(node->as<Tuple>(), args...);

      case Kind::When:
        return f(node->as<When>(), args...);

      case Kind::Try:
        return f(node->as<Try>(), args...);

      case Kind::Match:
        return f(node->as<Match>(), args...);

      case Kind::Lambda:
        return f(node->as<Lambda>(), args...);

      case Kind::Assign:
        return f(node->as<Assign>(), args...);

      case Kind::Select:
        return f(node->as<Select>(), args...);

      case Kind::Ref:
        return f(node->as<Ref>(), args...);

      case Kind::Let:
        return f(node->as<Let>(), args...);

      case Kind::Var:
        return f(node->as<Var>(), args...);

      case Kind::Throw:
        return f(node->as<Throw>(), args...);

      case Kind::New:
        return f(node->as<New>(), args...);

      case Kind::ObjectLiteral:
        return f(node->as<ObjectLiteral>(), args...);

      case Kind::EscapedString:
        return f(node->as<EscapedString>(), args...);

      case Kind::UnescapedString:
        return f(node->as<UnescapedString>(), args...);

      case Kind::Character:
        return f(node->as<Character>(), args...);

      case Kind::Int:
        return f(node->as<Int>(), args...);

      case Kind::Float:
        return f(node->as<Float>(), args...);

      case Kind::Hex:
        return f(node->as<Hex>(), args...);

      case Kind::Binary:
        return f(node->as<Binary>(), args...);

      case Kind::Bool:
        return f(node->as<Bool>(), args...);
    }

    // This is unreachable, and is only to suppress an MSVC error.
    return f(args...);
  }
}
