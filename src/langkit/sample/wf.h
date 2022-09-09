// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace sample
{
  inline constexpr auto wfTypes = choice(
    RefType,
    TypeTuple,
    TypeView,
    TypeFunc,
    TypeThrow,
    TypeIsect,
    TypeUnion,
    TypeVar,
    TypeTrait,
    Iso,
    Imm,
    Mut);

  inline constexpr auto wfIdSym = field(IdSym, choice(Ident, Symbol));

  inline constexpr auto wf = wellformed(
    shape(Ident),

    // Class bodies.
    shape(Use, field(Type)),
    shape(
      Typealias,
      field(Ident),
      field(Typeparams),
      field(Bounds, Type),
      field(Default, Type)),
    shape(
      Class, field(Ident), field(Typeparams), field(Type), field(Classbody)),
    shape(Classbody, seq(Use, Class, Typealias, FieldLet, FieldVar, Function)),
    shape(FieldLet, field(Ident), field(Type), field(Expr)),
    shape(FieldVar, field(Ident), field(Type), field(Expr)),
    shape(
      Function,
      wfIdSym,
      field(Typeparams),
      field(Params),
      field(Type),
      field(Funcbody)),

    // Type parameters.
    shape(Typeparams, seq(Typeparam)),
    shape(Typeparam, field(Ident), field(Bounds, Type), field(Default, Type)),

    // Functions.
    shape(Params, seq(Param)),
    shape(Param, field(Ident), field(Type), field(Expr)),
    shape(Funcbody, undef()),

    // Types.
    shape(Type, field(Type, wfTypes)),
    shape(TypeTuple, seq(wfTypes)),
    shape(TypeView, field(lhs, wfTypes), field(rhs, wfTypes)),
    shape(TypeFunc, field(lhs, wfTypes), field(rhs, wfTypes)),
    shape(TypeThrow, field(Type, wfTypes)),
    shape(TypeIsect, seq(wfTypes)),
    shape(TypeUnion, seq(wfTypes)),
    shape(TypeVar, field(Ident)),
    shape(TypeTrait, field(Classbody)),
    shape(Iso),
    shape(Imm),
    shape(Mut),
    shape(Package, field(id, choice(String, Escaped))),
    shape(RefType, field(Ident), field(Typeargs)), // TODO: scoped

    shape(Var, field(Ident), field(Type)),
    shape(Let, field(Ident), field(Type)),
    shape(Throw, field(Expr)),

    // TODO:
    shape(Expr, undef()),

    shape(Typeargs, seq(wfTypes)),
    shape(Lambda, field(Typeparams), field(Params), field(Funcbody)),

    // TODO:
    shape(Tuple, undef()),
    shape(Assign, undef()),

    shape(RefVar),
    shape(RefVarLHS),
    shape(RefLet),
    shape(RefParam),

    shape(RefFunction, wfIdSym, field(Typeargs)),
    shape(Selector, wfIdSym, field(Typeargs)),

    // TODO:
    shape(Call, undef()),
    shape(CallLHS, undef()),

    shape(Include, field(Type)));
}
