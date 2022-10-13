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
  inline constexpr auto Class = TokenDef("class", flag::symtab);
  inline constexpr auto TypeAlias = TokenDef("typealias", flag::symtab);
  inline constexpr auto Use = TokenDef("use");
  inline constexpr auto Package = TokenDef("package");
  inline constexpr auto Var = TokenDef("var");
  inline constexpr auto Let = TokenDef("let");
  inline constexpr auto Ref = TokenDef("ref");
  inline constexpr auto Throw = TokenDef("throw");
  inline constexpr auto Iso = TokenDef("iso");
  inline constexpr auto Imm = TokenDef("imm");
  inline constexpr auto Mut = TokenDef("mut");

  // Semantic structure.
  inline constexpr auto TypeTrait = TokenDef("typetrait", flag::symtab);
  inline constexpr auto ClassBody = TokenDef("classbody");
  inline constexpr auto FieldLet =
    TokenDef("fieldlet", flag::symtab | flag::defbeforeuse);
  inline constexpr auto FieldVar =
    TokenDef("fieldvar", flag::symtab | flag::defbeforeuse);
  inline constexpr auto Function =
    TokenDef("function", flag::symtab | flag::defbeforeuse | flag::multidef);
  inline constexpr auto TypeParams = TokenDef("typeparams");
  inline constexpr auto TypeParam = TokenDef("typeparam");
  inline constexpr auto Params = TokenDef("params");
  inline constexpr auto Param =
    TokenDef("param", flag::symtab | flag::defbeforeuse);
  inline constexpr auto FuncBody = TokenDef("funcbody");

  // Type structure.
  inline constexpr auto Type = TokenDef("type");
  inline constexpr auto TypeUnit = TokenDef("typeunit");
  inline constexpr auto TypeList = TokenDef("typelist");
  inline constexpr auto TypeName = TokenDef("typename");
  inline constexpr auto TypeTuple = TokenDef("typetuple");
  inline constexpr auto TypeView = TokenDef("typeview");
  inline constexpr auto TypeFunc = TokenDef("typefunc");
  inline constexpr auto TypeThrow = TokenDef("typethrow");
  inline constexpr auto TypeIsect = TokenDef("typeisect");
  inline constexpr auto TypeUnion = TokenDef("typeunion");
  inline constexpr auto TypeVar = TokenDef("typevar", flag::print);

  // Expression structure.
  inline constexpr auto Expr = TokenDef("expr");
  inline constexpr auto ExprSeq = TokenDef("exprseq");
  inline constexpr auto TypeAssert = TokenDef("typeassert");
  inline constexpr auto TypeAssertOp = TokenDef("typeassertop");
  inline constexpr auto TypeArgs = TokenDef("typeargs");
  inline constexpr auto Lambda =
    TokenDef("lambda", flag::symtab | flag::defbeforeuse);
  inline constexpr auto Tuple = TokenDef("tuple");
  inline constexpr auto Assign = TokenDef("assign");
  inline constexpr auto RefVar = TokenDef("refvar", flag::print);
  inline constexpr auto RefLet = TokenDef("reflet", flag::print);
  inline constexpr auto FunctionName = TokenDef("funcname");
  inline constexpr auto Selector = TokenDef("selector");
  inline constexpr auto Call = TokenDef("call");
  inline constexpr auto Args = TokenDef("args");
  inline constexpr auto Include = TokenDef("include");
  inline constexpr auto TupleLHS = TokenDef("tuple-lhs");
  inline constexpr auto CallLHS = TokenDef("call-lhs");
  inline constexpr auto RefVarLHS = TokenDef("refvar-lhs", flag::print);
  inline constexpr auto TupleFlatten = TokenDef("tupleflatten");
  inline constexpr auto Bind = TokenDef("bind");

  // Indexing names.
  inline constexpr auto IdSym = TokenDef("idsym");
  inline constexpr auto Bounds = TokenDef("bounds");
  inline constexpr auto Default = TokenDef("default");

  // Rewrite identifiers.
  inline constexpr auto id = TokenDef("id");
  inline constexpr auto lhs = TokenDef("lhs");
  inline constexpr auto rhs = TokenDef("rhs");
  inline constexpr auto op = TokenDef("op");
  inline constexpr auto ltype = TokenDef("ltype");
  inline constexpr auto rtype = TokenDef("rtype");

  // Sythetic locations.
  inline const auto standard = Location("std");
  inline const auto ref = Location("ref");
  inline const auto cell = Location("cell");
  inline const auto create = Location("create");
  inline const auto apply = Location("apply");
  inline const auto load = Location("load");
  inline const auto store = Location("store");

  struct Found
  {
    Node def;
    NodeMap<Node> map;

    Found() = default;
    Found(const Found&) = default;
    Found& operator=(const Found&) = default;

    Found(Found&& that) : def(std::move(that.def)), map(std::move(that.map)) {}
    Found(Node def) : def(def) {}

    Found& operator|=(Found&& that)
    {
      def = std::move(that.def);
      map.insert(that.map.begin(), that.map.end());
      that.map.clear();
      return *this;
    }
  };

  Found resolve(Node typeName);
  Node lookdown(Found& found, Node id);

  Parse parser();
  Pass infer();
  Driver& driver();
}
