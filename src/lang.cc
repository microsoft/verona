// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

#include "lookup.h"
#include "subtype.h"
#include "wf.h"

namespace verona
{
  Node err(NodeRange& r, const std::string& msg)
  {
    return Error << (ErrorMsg ^ msg) << (ErrorAst << r);
  }

  Node err(Node node, const std::string& msg)
  {
    return Error << (ErrorMsg ^ msg) << ((ErrorAst ^ node) << node);
  }

  Node typevar(Match& _)
  {
    return Type << (TypeVar ^ _.fresh(l_typevar));
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

  Node builtin()
  {
    static Location l_standard("std");
    static Location l_builtin("builtin");
    return TypeClassName << (TypeClassName << DontCare << (Ident ^ l_standard)
                                           << TypeArgs)
                         << (Ident ^ l_builtin) << TypeArgs;
  }

  Node nonlocal(Match& _)
  {
    static Location l_nonlocal("nonlocal");
    return TypeClassName << builtin() << (Ident ^ l_nonlocal)
                         << (TypeArgs << typevar(_));
  }

  Node unittype()
  {
    static Location l_unit("unit");
    return TypeClassName << builtin() << (Ident ^ l_unit) << TypeArgs;
  }

  Node unit()
  {
    return (
      Call << (FunctionName << unittype() << (Ident ^ create) << TypeArgs)
           << Args);
  }

  Node cell()
  {
    static Location l_cell("cell");
    return Call << (FunctionName
                    << (TypeClassName << builtin() << (Ident ^ l_cell)
                                      << TypeArgs)
                    << (Ident ^ create) << TypeArgs)
                << Args;
  }

  Node apply_id()
  {
    static Location l_apply("apply");
    return Ident ^ l_apply;
  }

  Node apply(Node ta)
  {
    return Selector << apply_id() << ta;
  }

  Node makename(Node lhs, Node id, Node ta, bool func)
  {
    auto defs = lookup_scopedname_name(lhs, id, ta);

    if (defs.defs.size() == 0)
      return Error << (ErrorMsg ^ "unknown type name")
                   << ((ErrorAst ^ id) << lhs << id << ta);

    if (func)
    {
      if (std::any_of(defs.defs.begin(), defs.defs.end(), [](auto& def) {
            return (def.def->type() == Function) && !def.too_many_typeargs;
          }))
      {
        return FunctionName << lhs << id << ta;
      }
      else if (std::any_of(defs.defs.begin(), defs.defs.end(), [](auto& def) {
                 return (def.def->type() == Function) && def.too_many_typeargs;
               }))
      {
        return Error << (ErrorMsg ^ "too many function type arguments")
                     << ((ErrorAst ^ id) << lhs << id << ta);
      }
    }

    if (defs.defs.size() > 1)
    {
      auto err = Error << (ErrorMsg ^ "ambiguous type name")
                       << ((ErrorAst ^ id) << lhs << id << ta);

      for (auto& def : defs.defs)
        err << (ErrorAst ^ (def.def / Ident));

      return err;
    }

    if (std::all_of(defs.defs.begin(), defs.defs.end(), [](auto& def) {
          return def.too_many_typeargs;
        }))
    {
      return Error << (ErrorMsg ^ "too many type arguments")
                   << ((ErrorAst ^ id) << lhs << id << ta);
    }

    if (defs.one({Class}))
      return TypeClassName << lhs << id << ta;
    if (defs.one({TypeAlias}))
      return TypeAliasName << lhs << id << ta;
    if (defs.one({TypeParam}))
      return TypeParamName << lhs << id << ta;
    if (defs.one({TypeTrait}))
      return TypeTraitName << lhs << id << ta;

    return Error << (ErrorMsg ^ "not a type name")
                 << ((ErrorAst ^ id) << lhs << id << ta)
                 << (ErrorAst ^ (defs.defs.front().def / Ident));
  }

  bool is_llvm_call(Node op, size_t arity)
  {
    // `op` must already be in the AST in order to resolve the FunctionName.
    if (op->type() != FunctionName)
      return false;

    auto look = lookup_scopedname(op);

    for (auto& def : look.defs)
    {
      if (
        (def.def->type() == Function) &&
        ((def.def / Params)->size() == arity) &&
        ((def.def / LLVMFuncType)->type() == LLVMFuncType))
      {
        return true;
      }
    }

    return false;
  }

  Node arg(Node args, Node arg)
  {
    if (arg)
    {
      if (arg->type() == Tuple)
        args->push_back({arg->begin(), arg->end()});
      else if (arg->type() == Expr)
        args << arg;
      else if (arg->type() != Unit)
        args << (Expr << arg);
    }

    return args;
  }

  Node call(Node op, Node lhs, Node rhs)
  {
    auto args = arg(arg(Args, lhs), rhs);

    if (!is_llvm_call(op, args->size()))
      return NLRCheck << (Call << op << args);

    return Call << op << args;
  }

  Node load(Node arg)
  {
    static Location l_load("load");
    return Call << (Selector << (Ident ^ l_load) << TypeArgs)
                << (Args << (Expr << arg));
  }

  Node nlrexpand(Match& _, Node call, bool unwrap)
  {
    // Check the call result to see if it's a non-local return. If it is,
    // optionally unwrap it and return. Otherwise, continue execution.
    auto id = _.fresh();
    auto nlr = Type << nonlocal(_);
    Node ret = Cast << (Expr << (RefLet << (Ident ^ id))) << nlr;

    if (unwrap)
      ret = load(ret);

    return ExprSeq
      << (Expr << (Bind << (Ident ^ id) << typevar(_) << (Expr << call)))
      << (Expr
          << (Conditional << (Expr
                              << (TypeTest << (Expr << (RefLet << (Ident ^ id)))
                                           << clone(nlr)))
                          << (Block << (Return << (Expr << ret)))
                          << (Block << (Expr << (RefLet << (Ident ^ id))))));
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
        {"memberconflict", memberconflict(), wfPassStructure},
        {"typenames", typenames(), wfPassTypeNames},
        {"typeview", typeview(), wfPassTypeView},
        {"typefunc", typefunc(), wfPassTypeFunc},
        {"typealg", typealg(), wfPassTypeAlg},
        {"typeflat", typeflat(), wfPassTypeFlat},
        {"typevalid", typevalid(), wfPassTypeFlat},
        {"codereuse", codereuse(), wfPassTypeFlat},
        {"conditionals", conditionals(), wfPassConditionals},
        {"reference", reference(), wfPassReference},
        {"reverseapp", reverseapp(), wfPassReverseApp},
        {"application", application(), wfPassApplication},
        {"assignlhs", assignlhs(), wfPassAssignLHS},
        {"localvar", localvar(), wfPassLocalVar},
        {"assignment", assignment(), wfPassAssignment},
        {"nlrcheck", nlrcheck(), wfPassNLRCheck},
        {"lambda", lambda(), wfPassLambda},
        {"autofields", autofields(), wfPassAutoFields},
        {"autorhs", autorhs(), wfPassAutoFields},
        {"autocreate", autocreate(), wfPassAutoCreate},
        {"defaultargs", defaultargs(), wfPassDefaultArgs},
        {"partialapp", partialapp(), wfPassDefaultArgs},
        {"traitisect", traitisect(), wfPassDefaultArgs},
        {"anf", anf(), wfPassANF},
        {"defbeforeuse", defbeforeuse(), wfPassANF},
        {"drop", drop(), wfPassDrop},
        {"namearity", namearity(), wfPassNameArity},
        {"validtypeargs", validtypeargs(), wfPassNameArity},
      });

    return d;
  }
}
