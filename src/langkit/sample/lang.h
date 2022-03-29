// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <langkit/driver.h>

namespace verona::lang
{
  using namespace langkit;

  // Parsing structure.
  constexpr auto Comment = Token("comment");
  constexpr auto Paren = Token("paren");
  constexpr auto Square = Token("square");
  constexpr auto Brace = Token("brace");
  constexpr auto List = Token("list");
  constexpr auto Equals = Token("equals");

  // Parsing literals.
  constexpr auto Dot = Token("dot");
  constexpr auto Ellipsis = Token("ellipsis");
  constexpr auto Colon = Token("colon");
  constexpr auto DoubleColon = Token("doublecolon");
  constexpr auto FatArrow = Token("fatarrow");
  constexpr auto Bool = Token("bool", flag::print);
  constexpr auto Hex = Token("hex", flag::print);
  constexpr auto Bin = Token("bin", flag::print);
  constexpr auto Int = Token("int", flag::print);
  constexpr auto HexFloat = Token("hexfloat", flag::print);
  constexpr auto Float = Token("float", flag::print);
  constexpr auto Char = Token("char", flag::print);
  constexpr auto Escaped = Token("escaped", flag::print);
  constexpr auto String = Token("string", flag::print);
  constexpr auto Symbol = Token("symbol", flag::print);
  constexpr auto Ident = Token("ident", flag::print);

  // Parsing keywords.
  constexpr auto Private = Token("private");
  constexpr auto Package = Token("package");
  constexpr auto Use = Token("use");
  constexpr auto Typealias = Token("typealias", flag::symtab);
  constexpr auto Class = Token("class", flag::symtab);
  constexpr auto Var = Token("var");
  constexpr auto Let = Token("let");
  constexpr auto Throw = Token("throw");
  constexpr auto Iso = Token("iso");
  constexpr auto Imm = Token("imm");
  constexpr auto Mut = Token("mut");

  // Semantic structure.
  constexpr auto Classbody = Token("classbody");
  constexpr auto Typeparams = Token("typeparams");
  constexpr auto Typeparam = Token("typeparam");
  constexpr auto Params = Token("params");
  constexpr auto Param = Token("param", flag::symtab | flag::defbeforeuse);
  constexpr auto Funcbody = Token("funcbody");
  constexpr auto Function = Token("function", flag::symtab | flag::defbeforeuse);
  constexpr auto FieldLet = Token("fieldlet", flag::symtab | flag::defbeforeuse);
  constexpr auto FieldVar = Token("fieldvar", flag::symtab | flag::defbeforeuse);

  // Type structure.
  constexpr auto Type = Token("type");
  constexpr auto TypeTuple = Token("typetuple");
  constexpr auto TypeView = Token("typeview");
  constexpr auto TypeFunc = Token("typefunc");
  constexpr auto TypeThrow = Token("typethrow");
  constexpr auto TypeIsect = Token("typeisect");
  constexpr auto TypeUnion = Token("typeunion");

  // Expression structure.
  constexpr auto Expr = Token("expr");
  constexpr auto Typeargs = Token("typeargs");
  constexpr auto Lambda = Token("lambda", flag::symtab | flag::defbeforeuse);
  constexpr auto Tuple = Token("tuple");
  constexpr auto Assign = Token("assign");
  constexpr auto RefVar = Token("refvar");
  constexpr auto RefLet = Token("reflet");
  constexpr auto RefParam = Token("refparam");
  constexpr auto RefTypeparam = Token("reftypeparam");
  constexpr auto RefType = Token("reftype");
  constexpr auto RefClass = Token("refclass");
  constexpr auto RefFunction = Token("reffunc");
  constexpr auto Scoped = Token("scoped");
  constexpr auto Selector = Token("selector");
  constexpr auto Call = Token("call");

  // Rewrite identifiers.
  constexpr auto id = Token("id");
  constexpr auto lhs = Token("lhs");
  constexpr auto rhs = Token("rhs");
  constexpr auto op = Token("op");

  // Passes.
  constexpr auto Imports = Token("imports");
  constexpr auto Structure = Token("structure");
  constexpr auto References = Token("references");
  constexpr auto Selectors = Token("selectors");
  constexpr auto ReverseApp = Token("reverseapp");
  constexpr auto Application = Token("application");

  // Sythetic locations.
  const auto apply = Location("apply");
  const auto create = Location("create");

  Driver& driver();
}
