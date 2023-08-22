// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <trieste/driver.h>

namespace verona
{
  using namespace trieste;

  // Parsing structure.
  inline const auto Paren = TokenDef("paren");
  inline const auto Square = TokenDef("square");
  inline const auto Brace = TokenDef("brace");
  inline const auto List = TokenDef("list");
  inline const auto Equals = TokenDef("equals");

  // Parsing literals.
  inline const auto DontCare = TokenDef("dontcare");
  inline const auto Dot = TokenDef("dot");
  inline const auto Ellipsis = TokenDef("ellipsis");
  inline const auto Colon = TokenDef("colon");
  inline const auto DoubleColon = TokenDef("doublecolon");
  inline const auto TripleColon = TokenDef("triplecolon");
  inline const auto Arrow = TokenDef("arrow");
  inline const auto Bool = TokenDef("bool", flag::print);
  inline const auto Hex = TokenDef("hex", flag::print);
  inline const auto Bin = TokenDef("bin", flag::print);
  inline const auto Int = TokenDef("int", flag::print);
  inline const auto HexFloat = TokenDef("hexfloat", flag::print);
  inline const auto Float = TokenDef("float", flag::print);
  inline const auto Char = TokenDef("char", flag::print);
  inline const auto Escaped = TokenDef("escaped", flag::print);
  inline const auto String = TokenDef("string", flag::print);
  inline const auto Symbol = TokenDef("symbol", flag::print);
  inline const auto Ident = TokenDef("ident", flag::print);

  // Parsing keywords.
  inline const auto Class = TokenDef(
    "class", flag::symtab | flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto TypeAlias = TokenDef(
    "typealias",
    flag::symtab | flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto Inherit = TokenDef("inherit");
  inline const auto Where = TokenDef("where");
  inline const auto Use = TokenDef("use");
  inline const auto Package = TokenDef("package");
  inline const auto Var = TokenDef("var", flag::lookup | flag::shadowing);
  inline const auto Let = TokenDef("let", flag::lookup | flag::shadowing);
  inline const auto Ref = TokenDef("ref");
  inline const auto Self = TokenDef("Self");
  inline const auto If = TokenDef("if");
  inline const auto Else = TokenDef("else");
  inline const auto New = TokenDef("new");
  inline const auto Try = TokenDef("try");
  inline const auto Iso = TokenDef("iso");
  inline const auto Mut = TokenDef("mut");
  inline const auto Imm = TokenDef("imm");

  // Semantic structure.
  inline const auto TypeTrait = TokenDef(
    "typetrait",
    flag::symtab | flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto ClassBody = TokenDef("classbody");
  inline const auto FieldLet = TokenDef("fieldlet", flag::lookdown);
  inline const auto FieldVar = TokenDef("fieldvar", flag::lookdown);
  inline const auto Function = TokenDef(
    "function",
    flag::symtab | flag::defbeforeuse | flag::lookup | flag::lookdown);
  inline const auto TypeParams = TokenDef("typeparams");
  inline const auto TypeParam =
    TokenDef("typeparam", flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto ValueParam =
    TokenDef("valueparam", flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto Params = TokenDef("params");
  inline const auto Param = TokenDef("param", flag::lookup | flag::shadowing);
  inline const auto Block =
    TokenDef("block", flag::symtab | flag::defbeforeuse);

  // Type structure.
  inline const auto Type = TokenDef("type");
  inline const auto TypePred = TokenDef("typepred");
  inline const auto TypeList = TokenDef("typelist");
  inline const auto TypeClassName = TokenDef("typeclassname");
  inline const auto TypeAliasName = TokenDef("typealiasname");
  inline const auto TypeParamName = TokenDef("typeparamname");
  inline const auto TypeTraitName = TokenDef("typetraitname");
  inline const auto TypeTuple = TokenDef("typetuple");
  inline const auto TypeView = TokenDef("typeview");
  inline const auto TypeIsect = TokenDef("typeisect");
  inline const auto TypeUnion = TokenDef("typeunion");
  inline const auto TypeVar = TokenDef("typevar", flag::print);
  inline const auto TypeSubtype = TokenDef("typesubtype");
  inline const auto TypeTrue = TokenDef("typetrue");
  inline const auto TypeFalse = TokenDef("typefalse");
  inline const auto TypePath = TokenDef("typepath");
  inline const auto TypeParamBind = TokenDef("typeparambind");
  inline const auto FQType = TokenDef("fqtype");
  inline const auto FQFunction = TokenDef("fqfunction");

  // Expression structure.
  inline const auto Expr = TokenDef("expr");
  inline const auto ExprSeq = TokenDef("exprseq");
  inline const auto TypeAssert = TokenDef("typeassert");
  inline const auto TypeArgs = TokenDef("typeargs");
  inline const auto Lambda =
    TokenDef("lambda", flag::symtab | flag::defbeforeuse);
  inline const auto Tuple = TokenDef("tuple");
  inline const auto Unit = TokenDef("unit");
  inline const auto Assign = TokenDef("assign");
  inline const auto RefVar = TokenDef("refvar");
  inline const auto RefLet = TokenDef("reflet");
  inline const auto Selector = TokenDef("selector");
  inline const auto Call = TokenDef("call");
  inline const auto Args = TokenDef("args");
  inline const auto TupleLHS = TokenDef("tuple-lhs");
  inline const auto RefVarLHS = TokenDef("refvar-lhs");
  inline const auto TupleFlatten = TokenDef("tupleflatten");
  inline const auto Conditional = TokenDef("conditional");
  inline const auto FieldRef = TokenDef("fieldref");
  inline const auto Bind = TokenDef("bind", flag::lookup | flag::shadowing);
  inline const auto Move = TokenDef("move");
  inline const auto Copy = TokenDef("copy");
  inline const auto Drop = TokenDef("drop");
  inline const auto TypeTest = TokenDef("typetest");
  inline const auto Cast = TokenDef("cast");
  inline const auto Return = TokenDef("return");
  inline const auto NLRCheck = TokenDef("nlrcheck");

  // LLVM-specific.
  inline const auto LLVM = TokenDef("llvm", flag::print);
  inline const auto LLVMList = TokenDef("llvmlist");
  inline const auto LLVMFuncType = TokenDef("llvmfunctype");

  // Indexing names.
  inline const auto Default = TokenDef("default");

  // Rewrite identifiers.
  inline const auto Implicit = TokenDef("implicit");
  inline const auto Explicit = TokenDef("explicit");
  inline const auto Lhs = TokenDef("Lhs");
  inline const auto Rhs = TokenDef("Rhs");
  inline const auto Op = TokenDef("Op");

  // Sythetic locations.
  inline const auto l_typevar = Location("typevar");
  inline const auto l_param = Location("param");
  inline const auto l_trait = Location("trait");
  inline const auto l_class = Location("class");
  inline const auto l_self = Location("self");
  inline const auto l_new = Location("new");
  inline const auto l_apply = Location("apply");
  inline const auto l_create = Location("create");

  // Helper patterns.
  inline const auto IsImplicit = T(Implicit) / T(Explicit);
  inline const auto Hand = T(Lhs) / T(Rhs);
  inline const auto TypeStruct = In(Type) / In(TypeList) / In(TypeTuple) /
    In(TypeView) / In(TypeUnion) / In(TypeIsect) / In(TypeSubtype);
  inline const auto TypeCaps = T(Iso) / T(Mut) / T(Imm);
  inline const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) /
    T(Hex) / T(Bin) / T(Int) / T(Float) / T(HexFloat) / T(LLVM);
  inline const auto TypeElem = T(Type) / TypeCaps / T(FQType) / T(TypeTrait) /
    T(TypeTuple) / T(Self) / T(TypeList) / T(TypeView) / T(TypeIsect) /
    T(TypeUnion) / T(TypeVar) / T(Package) / T(TypeSubtype) / T(TypeTrue) /
    T(TypeFalse);
  inline const auto Object0 = Literal / T(RefVar) / T(RefVarLHS) / T(RefLet) /
    T(Unit) / T(Tuple) / T(Lambda) / T(Call) / T(NLRCheck) / T(Assign) /
    T(Expr) / T(ExprSeq) / T(DontCare) / T(Conditional) / T(TypeTest) / T(Cast);
  inline const auto Object = Object0 / (T(TypeAssert) << (Object0 * T(Type)));
  inline const auto Operator = T(FQFunction) / T(Selector);
  inline const auto RhsCall = T(Call)
    << ((T(Selector) << T(Rhs)) /
        (T(FQFunction) << (T(FQType) * (T(Selector) << T(Rhs)))));

  // Helper functions for generating AST fragments.
  Node err(NodeRange& r, const std::string& msg);
  Node err(Node node, const std::string& msg);
  Node typevar(Match& _);
  Node typevar(Match& _, const Token& t);
  Node inherit();
  Node inherit(Match& _, const Token& t);
  Node typepred();
  Node typepred(Match& _, const Token& t);
  Node nonlocal(Match& _);
  Node unittype();
  Node unit();
  Node cell();
  Node selector(Node name, Node ta = TypeArgs);
  Node selector(Location name, Node ta = TypeArgs);
  bool is_llvm_call(Node op);
  Node call(Node op, Node lhs = {}, Node rhs = {}, bool post_nlr = false);
  Node call_lhs(Node call);
  Node load(Node arg, bool post_nlr = false);
  bool is_implicit(Node n);

  // Pass definitions.
  Parse parser();
  PassDef modules();
  PassDef structure();
  PassDef memberconflict();
  PassDef typenames();
  PassDef typeview();
  PassDef typefunc();
  PassDef typealg();
  PassDef typeflat();
  PassDef typevalid();
  PassDef codereuse();
  PassDef conditionals();
  PassDef reference();
  PassDef reverseapp();
  PassDef application();
  PassDef assignlhs();
  PassDef localvar();
  PassDef assignment();
  PassDef nlrcheck();
  PassDef lambda();
  PassDef autofields();
  PassDef autorhs();
  PassDef autocreate();
  PassDef defaultargs();
  PassDef partialapp();
  PassDef traitisect();
  PassDef anf();
  PassDef defbeforeuse();
  PassDef drop();
  PassDef namearity();
  PassDef validtypeargs();

  struct Options : public trieste::Options
  {
    bool no_std = false;

    void configure(CLI::App& cli) override
    {
      cli.add_flag("--no-std", no_std, "Don't import the standard library.");
    }
  };

  Options& options();
  Driver& driver();
}
