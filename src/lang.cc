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

  static Node builtin_type(const Location& name, Node ta = TypeArgs)
  {
    return FQType << builtin_path() << (TypeClassName << (Ident ^ name) << ta);
  }

  static Node call0(Node type, const Location& loc)
  {
    return call(FQFunction << type << selector(loc));
  }

  static Node create0(Node type)
  {
    return call0(type, l_create);
  }

  Node nonlocal(Match& _)
  {
    // Pin the type argument to a specific type variable.
    static Location l_nonlocal("nonlocal");
    return builtin_type(l_nonlocal, TypeArgs << typevar(_));
  }

  Node unittype()
  {
    static Location l_unit("unit");
    return builtin_type(l_unit);
  }

  Node unit()
  {
    return create0(unittype());
  }

  Node booltype()
  {
    static Location l_bool("Bool");
    return builtin_type(l_bool);
  }

  Node booltrue()
  {
    static Location l_true("make_true");
    return call0(booltype(), l_true);
  }

  Node boolfalse()
  {
    static Location l_false("make_false");
    return call0(booltype(), l_false);
  }

  Node celltype()
  {
    static Location l_cell("Cell");
    return builtin_type(l_cell);
  }

  Node cell()
  {
    return create0(celltype());
  }

  Node reftype(Node t)
  {
    static Location l_ref("Ref");
    return builtin_type(l_ref, TypeArgs << -t);
  }

  Node tuple_to_args(Node n)
  {
    assert(n == Tuple);
    if (n->size() == 0)
      return Unit;
    else if (n->size() == 1)
      return n->front();
    else
      return n;
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

  Node call(Node op, Node lhs, Node rhs)
  {
    assert(op->in({FQFunction, Selector}));
    auto args = arg(arg(Args, lhs), rhs);
    auto arity = Int ^ std::to_string(args->size());

    if (op == FQFunction)
      (op / Selector / Int) = arity;
    else
      (op / Int) = arity;

    return NLRCheck << (Call << op << args);
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

  Node load(Node arg)
  {
    static Location l_load("load");
    return call(selector(l_load), arg);
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

  static size_t arity(Node& node)
  {
    assert(node->in({FieldLet, FieldVar, Function}));
    return (node == Function) ? (node / Params)->size() : 1;
  }

  bool conflict(Node& a, Node& b)
  {
    assert(a->in({FieldLet, FieldVar, Function}));
    assert(b->in({FieldLet, FieldVar, Function}));

    // Check for handedness conflict.
    auto a_hand = handed(a);
    auto b_hand = handed(b);

    if ((a_hand != b_hand) && (a_hand != Op) && (b_hand != Op))
      return false;

    // Check for arity conflict.
    return arity(a) == arity(b);
  }

  Options& options()
  {
    static Options opts;
    return opts;
  }

  std::vector<Pass> passes()
  {
    return {
      modules(),       structure(),     reference(),     conditionals(),
      lambda(),        autocreate(),    defaultargs(),   typenames(),
      typeview(),      typefunc(),      typealg(),       typeflat(),
      typevalid(),     typereference(), codereuse(),     memberconflict(),
      resetimplicit(), reverseapp(),    application(),   assignlhs(),
      localvar(),      assignment(),    autofields(),    autorhs(),
      partialapp(),    traitisect(),    nlrcheck(),      anf(),
      defbeforeuse(),  drop(),          validtypeargs(), // typeinfer(),
    };
  }
}
