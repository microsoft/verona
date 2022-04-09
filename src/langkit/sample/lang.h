// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <langkit/driver.h>

namespace sample
{
  using namespace langkit;

  // Parsing structure.
  inline constexpr auto Paren = TokenDef("paren");
  inline constexpr auto Square = TokenDef("square");
  inline constexpr auto Brace = TokenDef("brace");
  inline constexpr auto List = TokenDef("list");
  inline constexpr auto Equals = TokenDef("equals");

  // Parsing literals.
  inline constexpr auto DontCare = TokenDef("dontcare");
  inline constexpr auto Dot = TokenDef("dot");
  inline constexpr auto Ellipsis = TokenDef("ellipsis");
  inline constexpr auto Colon = TokenDef("colon");
  inline constexpr auto DoubleColon = TokenDef("doublecolon");
  inline constexpr auto FatArrow = TokenDef("fatarrow");
  inline constexpr auto Bool = TokenDef("bool", flag::print);
  inline constexpr auto Hex = TokenDef("hex", flag::print);
  inline constexpr auto Bin = TokenDef("bin", flag::print);
  inline constexpr auto Int = TokenDef("int", flag::print);
  inline constexpr auto HexFloat = TokenDef("hexfloat", flag::print);
  inline constexpr auto Float = TokenDef("float", flag::print);
  inline constexpr auto Char = TokenDef("char", flag::print);
  inline constexpr auto Escaped = TokenDef("escaped", flag::print);
  inline constexpr auto String = TokenDef("string", flag::print);
  inline constexpr auto Symbol = TokenDef("symbol", flag::print);
  inline constexpr auto Ident = TokenDef("ident", flag::print);

  // Parsing keywords.
  inline constexpr auto Package = TokenDef("package");
  inline constexpr auto Use = TokenDef("use");
  inline constexpr auto Typealias =
    TokenDef("typealias", flag::print | flag::symtab);
  inline constexpr auto Class = TokenDef("class", flag::print | flag::symtab);
  inline constexpr auto Var = TokenDef("var", flag::print);
  inline constexpr auto Let = TokenDef("let", flag::print);
  inline constexpr auto Throw = TokenDef("throw");
  inline constexpr auto Iso = TokenDef("iso");
  inline constexpr auto Imm = TokenDef("imm");
  inline constexpr auto Mut = TokenDef("mut");

  // Semantic structure.
  inline constexpr auto Classbody = TokenDef("classbody");
  inline constexpr auto FieldLet =
    TokenDef("fieldlet", flag::print | flag::symtab | flag::defbeforeuse);
  inline constexpr auto FieldVar =
    TokenDef("fieldvar", flag::print | flag::symtab | flag::defbeforeuse);
  inline constexpr auto Function =
    TokenDef("function", flag::print | flag::symtab | flag::defbeforeuse);
  inline constexpr auto Typeparams = TokenDef("typeparams");
  inline constexpr auto Typeparam = TokenDef("typeparam", flag::print);
  inline constexpr auto Params = TokenDef("params");
  inline constexpr auto Param =
    TokenDef("param", flag::print | flag::symtab | flag::defbeforeuse);
  inline constexpr auto Funcbody = TokenDef("funcbody");

  // Type structure.
  inline constexpr auto Type = TokenDef("type");
  inline constexpr auto TypeTerm = TokenDef("typeterm");
  inline constexpr auto TypeTuple = TokenDef("typetuple");
  inline constexpr auto TypeView = TokenDef("typeview");
  inline constexpr auto TypeFunc = TokenDef("typefunc");
  inline constexpr auto TypeThrow = TokenDef("typethrow");
  inline constexpr auto TypeIsect = TokenDef("typeisect");
  inline constexpr auto TypeUnion = TokenDef("typeunion");
  inline constexpr auto TypeVar = TokenDef("typevar", flag::print);
  inline constexpr auto TypeTrait = TokenDef("typetrait", flag::symtab);

  // Expression structure.
  inline constexpr auto Expr = TokenDef("expr");
  inline constexpr auto Term = TokenDef("term");
  inline constexpr auto Typeargs = TokenDef("typeargs");
  inline constexpr auto Lambda =
    TokenDef("lambda", flag::symtab | flag::defbeforeuse);
  inline constexpr auto Tuple = TokenDef("tuple");
  inline constexpr auto Assign = TokenDef("assign");
  inline constexpr auto RefVar = TokenDef("refvar", flag::print);
  inline constexpr auto RefLet = TokenDef("reflet", flag::print);
  inline constexpr auto RefParam = TokenDef("refparam", flag::print);
  inline constexpr auto RefTypeparam = TokenDef("reftypeparam");
  inline constexpr auto RefTypealias = TokenDef("reftypealias");
  inline constexpr auto RefClass = TokenDef("refclass");
  inline constexpr auto RefFunction = TokenDef("reffunc");
  inline constexpr auto Selector = TokenDef("selector");
  inline constexpr auto DotSelector = TokenDef("dotselector");
  inline constexpr auto Call = TokenDef("call");
  inline constexpr auto Include = TokenDef("include");
  inline constexpr auto Lift = TokenDef("lift");

  // Indexing names.
  inline constexpr auto Bounds = TokenDef("bounds");
  inline constexpr auto Default = TokenDef("default");

  // Rewrite identifiers.
  inline constexpr auto id = TokenDef("id");
  inline constexpr auto lhs = TokenDef("lhs");
  inline constexpr auto rhs = TokenDef("rhs");
  inline constexpr auto op = TokenDef("op");

  // Sythetic locations.
  inline const auto apply = Location("apply");
  inline const auto create = Location("create");

  struct Found
  {
    Node def;
    std::map<Node, Node, std::owner_less<>> map;
  };

  Parse parser();
  Lookup<Found> lookup();
  Pass infer();
  Driver& driver();

  inline const auto look = lookup();
}
