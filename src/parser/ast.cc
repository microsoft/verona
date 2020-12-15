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

      case Kind::Signature:
        return "signature";

      case Kind::Function:
        return "function";

      case Kind::Method:
        return "method";

      // Types
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

      // Expressions
      case Kind::Tuple:
        return "tuple";

      case Kind::Block:
        return "block";

      case Kind::When:
        return "when";

      case Kind::While:
        return "while";

      case Kind::Case:
        return "case";

      case Kind::Match:
        return "match";

      case Kind::If:
        return "if";

      case Kind::Lambda:
        return "lambda";

      case Kind::Break:
        return "break";

      case Kind::Continue:
        return "continue";

      case Kind::Return:
        return "return";

      case Kind::Yield:
        return "yield";

      case Kind::Assign:
        return "assign";

      case Kind::Infix:
        return "infix";

      case Kind::Prefix:
        return "prefix";

      case Kind::Select:
        return "select";

      case Kind::Specialise:
        return "specialise";

      case Kind::Apply:
        return "apply";

      case Kind::Ref:
        return "ref";

      case Kind::StaticRef:
        return "staticref";

      case Kind::Let:
        return "let";

      case Kind::Var:
        return "var";

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

      case Kind::True:
        return "true";

      case Kind::False:
        return "false";
    }
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
