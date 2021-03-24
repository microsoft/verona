// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ast.h"

namespace verona::parser
{
  const char* kindname(Kind kind)
  {
    switch (kind)
    {
      // Definitions
      case Kind::Using:
        return "using";

      case Kind::TypeAlias:
        return "typealias";

      case Kind::Interface:
        return "interface";

      case Kind::Class:
        return "class";

      case Kind::Module:
        return "module";

      case Kind::Field:
        return "field";

      case Kind::Param:
        return "param";

      case Kind::TypeParam:
        return "typeparam";

      case Kind::TypeParamList:
        return "typeparamlist";

      case Kind::Function:
        return "function";

      // Types
      case Kind::ThrowType:
        return "throwtype";

      case Kind::UnionType:
        return "uniontype";

      case Kind::IsectType:
        return "isecttype";

      case Kind::TupleType:
        return "tupletype";

      case Kind::FunctionType:
        return "functiontype";

      case Kind::ViewType:
        return "viewtype";

      case Kind::ExtractType:
        return "extracttype";

      case Kind::TypeName:
        return "typename";

      case Kind::ModuleName:
        return "modulename";

      case Kind::TypeRef:
        return "typeref";

      case Kind::TypeList:
        return "typelist";

      case Kind::Iso:
        return "iso";

      case Kind::Mut:
        return "mut";

      case Kind::Imm:
        return "imm";

      case Kind::Self:
        return "Self";

      // Expressions
      case Kind::Oftype:
        return "oftype";

      case Kind::Tuple:
        return "tuple";

      case Kind::When:
        return "when";

      case Kind::Try:
        return "try";

      case Kind::Match:
        return "match";

      case Kind::Lambda:
        return "lambda";

      case Kind::Assign:
        return "assign";

      case Kind::Select:
        return "select";

      case Kind::Ref:
        return "ref";

      case Kind::Let:
        return "let";

      case Kind::Var:
        return "var";

      case Kind::Throw:
        return "throw";

      case Kind::New:
        return "new";

      case Kind::ObjectLiteral:
        return "object";

      // Constants
      case Kind::EscapedString:
        return "string";

      case Kind::UnescapedString:
        return "string";

      case Kind::Character:
        return "char";

      case Kind::Int:
        return "int";

      case Kind::Float:
        return "float";

      case Kind::Hex:
        return "hex";

      case Kind::Binary:
        return "binary";

      case Kind::Bool:
        return "bool";
    }

    // This is unreachable, and is only to suppress an MSVC error.
    return "ERROR";
  }

  bool is_kind(Ast ast, const std::initializer_list<Kind>& kinds)
  {
    for (auto kind : kinds)
    {
      if (ast->kind() == kind)
        return true;
    }

    return false;
  }
}
