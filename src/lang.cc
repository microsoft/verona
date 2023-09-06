// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

#include "lookup.h"
#include "subtype.h"
#include "wf.h"

namespace verona
{
  Node err(Node node, const std::string& msg)
  {
    return Error << (ErrorMsg ^ msg) << node;
  }

  Node typevar(Location loc)
  {
    return Type << (TypeVar ^ loc);
  }

  Node typevar(Node& node)
  {
    return typevar(node->fresh(l_typevar));
  }

  Node typevar(Match& _)
  {
    return typevar(_.fresh(l_typevar));
  }

  Node typevar(Match& _, const Token& t)
  {
    auto n = _(t);
    return n ? n : typevar(_);
  }

  Node inherit()
  {
    return Inherit << DontCare;
  }

  Node inherit(Match& _, const Token& t)
  {
    return Inherit << (_(t) || DontCare);
  }

  Node typepred()
  {
    return TypePred << (Type << TypeTrue);
  }

  Node typepred(Match& _, const Token& t)
  {
    auto n = _(t);
    return n ? n : typepred();
  }

  static Node int0()
  {
    static Location l_int0("0");
    return Int ^ l_int0;
  }

  static Node builtin_path()
  {
    static Location l_std("std");
    static Location l_builtin("builtin");
    return TypePath << (TypeClassName << (Ident ^ l_std) << TypeArgs)
                    << (TypeClassName << (Ident ^ l_builtin) << TypeArgs);
  }

  Node nonlocal(Match& _)
  {
    // Pin the type argument to a specific type variable.
    static Location l_nonlocal("nonlocal");
    return FQType << builtin_path()
                  << (TypeClassName << (Ident ^ l_nonlocal)
                                    << (TypeArgs << typevar(_)));
  }

  Node unittype()
  {
    static Location l_unit("unit");
    return FQType << builtin_path()
                  << (TypeClassName << (Ident ^ l_unit) << TypeArgs);
  }

  Node unit()
  {
    return call(FQFunction << unittype() << selector(l_create));
  }

  Node celltype()
  {
    static Location l_cell("cell");
    return FQType << builtin_path()
                  << (TypeClassName << (Ident ^ l_cell) << TypeArgs);
  }

  Node cell()
  {
    return call(FQFunction << celltype() << selector(l_create));
  }

  Node selector(Node name, Node ta)
  {
    return selector(name->location(), ta);
  }

  Node selector(Location name, Node ta)
  {
    if (!ta)
      ta = TypeArgs;

    return Selector << Rhs << (Ident ^ name) << int0() << ta;
  }

  bool is_llvm_call(Node op)
  {
    // `op` must already be in the AST in order to resolve the FQFunction.
    // If not, it won't be treated as an LLVM call.
    if (op != FQFunction)
      return false;

    auto l = resolve_fq(op);

    return l.def && (l.def == Function) &&
      ((l.def / LLVMFuncType) == LLVMFuncType);
  }

  static Node arg(Node args, Node arg)
  {
    if (arg)
    {
      if (arg == Tuple)
        args->push_back({arg->begin(), arg->end()});
      else if (arg == Expr)
        args << arg;
      else if (arg != Unit)
        args << (Expr << arg);
    }

    return args;
  }

  Node call(Node op, Node lhs, Node rhs, bool post_nlr)
  {
    assert(op->in({FQFunction, Selector}));
    auto args = arg(arg(Args, lhs), rhs);
    auto arity = Int ^ std::to_string(args->size());

    if (op == FQFunction)
      (op / Selector / Int) = arity;
    else
      (op / Int) = arity;

    auto ret = Call << op << args;

    if (!post_nlr)
      ret = NLRCheck << Explicit << ret;

    return ret;
  }

  Node call_lhs(Node call)
  {
    assert(call == Call);
    auto f = call / Selector;

    if (f == FQFunction)
      f = f / Selector;

    (f / Ref) = Lhs;
    return call;
  }

  Node load(Node arg, bool post_nlr)
  {
    static Location l_load("load");
    return call(selector(l_load), arg, {}, post_nlr);
  }

  bool is_implicit(Node n)
  {
    auto f = n->parent(Function);
    return f && ((f / Implicit) == Implicit);
  }

  static Token handed(Node& node)
  {
    assert(node->in({FieldLet, FieldVar, Function}));

    // Return Op to mean both.
    if (node == FieldVar)
      return Op;
    else if (node == FieldLet)
      return Lhs;
    else
      return (node / Ref)->type();
  }

  static std::pair<size_t, size_t> arity(Node& node)
  {
    assert(node->in({FieldLet, FieldVar, Function}));

    if (node != Function)
      return {1, 1};

    auto params = node / Params;
    auto arity_hi = params->size();
    auto arity_lo = 0;

    for (auto& param : *params)
    {
      if ((param / Default) != DontCare)
        break;

      arity_lo++;
    }

    return {arity_lo, arity_hi};
  }

  bool conflict(Node& a, Node& b)
  {
    // Check for handedness overlap.
    auto a_hand = handed(a);
    auto b_hand = handed(b);

    if ((a_hand != b_hand) && (a_hand != Op) && (b_hand != Op))
      return false;

    // Check for arity overlap.
    auto [a_lo, a_hi] = arity(a);
    auto [b_lo, b_hi] = arity(b);
    return (b_hi >= a_lo) && (a_hi >= b_lo);
  }

  Options& options()
  {
    static Options opts;
    return opts;
  }

  Driver& driver()
  {
    static Driver d(
      "Verona",
      &options(),
      parser(),
      wfParser,
      {
        {"modules", modules(), wfPassModules},
        {"structure", structure(), wfPassStructure},
        {"typenames", typenames(), wfPassTypeNames},
        {"typeview", typeview(), wfPassTypeView},
        {"typefunc", typefunc(), wfPassTypeFunc},
        {"typealg", typealg(), wfPassTypeAlg},
        {"typeflat", typeflat(), wfPassTypeFlat},
        {"typevalid", typevalid(), wfPassTypeFlat},
        {"reference", reference(), wfPassReference},
        {"codereuse", codereuse(), wfPassReference},
        {"memberconflict", memberconflict(), wfPassReference},
        {"conditionals", conditionals(), wfPassConditionals},
        {"reverseapp", reverseapp(), wfPassReverseApp},
        {"application", application(), wfPassApplication},
        {"assignlhs", assignlhs(), wfPassAssignLHS},
        {"localvar", localvar(), wfPassLocalVar},
        {"assignment", assignment(), wfPassAssignment},
        {"lambda", lambda(), wfPassLambda},
        {"autofields", autofields(), wfPassAutoFields},
        {"autorhs", autorhs(), wfPassAutoFields},
        {"autocreate", autocreate(), wfPassAutoCreate},
        {"defaultargs", defaultargs(), wfPassDefaultArgs},
        {"partialapp", partialapp(), wfPassDefaultArgs},
        {"traitisect", traitisect(), wfPassDefaultArgs},
        {"nlrcheck", nlrcheck(), wfPassNLRCheck},
        {"anf", anf(), wfPassANF},
        {"defbeforeuse", defbeforeuse(), wfPassANF},
        {"drop", drop(), wfPassDrop},
        {"validtypeargs", validtypeargs(), wfPassDrop},
        {"typeinfer", typeinfer(), wfPassDrop},
      });

    return d;
  }
}
