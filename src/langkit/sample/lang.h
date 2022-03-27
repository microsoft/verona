// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <langkit/driver.h>

namespace verona::lang
{
  using namespace langkit;

  // Parsing structure.
  const auto Comment = Token("comment");
  const auto Paren = Token("paren");
  const auto Square = Token("square");
  const auto Brace = Token("brace");
  const auto List = Token("list");
  const auto Equals = Token("equals");

  // Parsing literals.
  const auto Dot = Token("dot");
  const auto Ellipsis = Token("ellipsis");
  const auto Colon = Token("colon");
  const auto DoubleColon = Token("doublecolon");
  const auto FatArrow = Token("fatarrow");
  const auto Bool = Token("bool", flag::print);
  const auto Hex = Token("hex", flag::print);
  const auto Bin = Token("bin", flag::print);
  const auto Int = Token("int", flag::print);
  const auto HexFloat = Token("hexfloat", flag::print);
  const auto Float = Token("float", flag::print);
  const auto Char = Token("char", flag::print);
  const auto Escaped = Token("escaped", flag::print);
  const auto String = Token("string", flag::print);
  const auto Symbol = Token("symbol", flag::print);
  const auto Ident = Token("ident", flag::print);

  // Parsing keywords.
  const auto Private = Token("private");
  const auto Package = Token("package");
  const auto Using = Token("using");
  const auto Typealias = Token("typealias");
  const auto Class = Token("class", flag::symtab);
  const auto Var = Token("var");
  const auto Let = Token("let");
  const auto Throw = Token("throw");
  const auto Iso = Token("iso");
  const auto Imm = Token("imm");
  const auto Mut = Token("mut");

  // Semantic structure.
  const auto Classbody = Token("classbody");
  const auto Typeparams = Token("typeparams");
  const auto Typeparam = Token("typeparam");
  const auto Params = Token("params");
  const auto Param = Token("param", flag::symtab | flag::defbeforeuse);
  const auto Funcbody = Token("funcbody");
  const auto Function = Token("function", flag::symtab | flag::defbeforeuse);
  const auto FieldLet = Token("fieldlet", flag::symtab | flag::defbeforeuse);
  const auto FieldVar = Token("fieldvar", flag::symtab | flag::defbeforeuse);

  // Type structure.
  const auto Type = Token("type");
  const auto TypeTuple = Token("typetuple");
  const auto TypeRef = Token("typeref");
  const auto TypeView = Token("typeview");
  const auto TypeFunc = Token("typefunc");
  const auto TypeThrow = Token("typethrow");
  const auto TypeIsect = Token("typeisect");
  const auto TypeUnion = Token("typeunion");

  // Expression structure.
  const auto Expr = Token("expr");
  const auto Typeargs = Token("typeargs");
  const auto Lambda = Token("lambda", flag::symtab | flag::defbeforeuse);
  const auto Tuple = Token("tuple");
  const auto Assign = Token("assign");
  const auto RefVar = Token("refvar");
  const auto RefLet = Token("reflet");
  const auto RefParam = Token("refparam");
  const auto RefType = Token("reftype");
  const auto RefClass = Token("refclass");
  const auto RefFunction = Token("reffunc");
  const auto Scoped = Token("scoped");
  const auto Selector = Token("selector");
  const auto Call = Token("call");

  // Rewrite identifiers.
  const auto id = Token("id");
  const auto lhs = Token("lhs");
  const auto rhs = Token("rhs");
  const auto op = Token("op");

  // Sythetic locations.
  const auto apply = Location("apply");
  const auto create = Location("create");

  // Passes.
  const auto Imports = Token("imports");
  const auto Structure = Token("structure");
  const auto References = Token("references");
  const auto Selectors = Token("selectors");
  const auto ReverseApp = Token("reverseapp");
  const auto Expressions = Token("expressions");

  Driver& driver();
}
