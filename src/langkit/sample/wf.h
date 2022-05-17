// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace sample
{
  inline constexpr auto wf = wellformed(
    shape(Package, field(id, String, Escaped)),
    shape(Use, field(Type)),
    shape(
      Typealias, field(Typeparams), field(Bounds, Type), field(Default, Type)),
    shape(Class, field(Typeparams), field(Type), field(Classbody)),
    shape(Iso),
    shape(Imm),
    shape(Mut),

    shape(Classbody, seq(Use, Typealias, Class, FieldLet, FieldVar, Function)),
    shape(FieldLet, field(Type), field(Expr)),
    shape(FieldVar, field(Type), field(Expr)),
    shape(
      Function, field(Typeparams), field(Params), field(Type), field(Funcbody)),
    shape(Typeparams, seq(Typeparam)),
    shape(Typeparam, field(Bounds, Type), field(Default, Type)),
    shape(Params, seq(Param)),
    shape(Param, field(Type), field(Expr)),

    // TODO:
    shape(Funcbody, undef()),

    // TODO:
    shape(Type, undef()),
    shape(TypeTerm, undef()),
    shape(TypeTuple, undef()),
    shape(TypeView, undef()),
    shape(TypeFunc, undef()),
    shape(TypeThrow, undef()),
    shape(TypeIsect, undef()),
    shape(TypeUnion, undef()),
    shape(TypeVar, undef()),
    shape(TypeTrait, undef()),

    shape(Var, field(Type)),
    shape(Let, field(Type)),
    shape(Throw, field(Expr)),

    // TODO:
    shape(Expr, undef()),
    shape(Term, undef()),
    shape(Typeargs, undef()),

    shape(Lambda, field(Typeparams), field(Params), field(Funcbody)),

    // TODO:
    shape(Tuple, undef()),
    shape(Assign, undef()),

    shape(RefVar),
    shape(RefLet),
    shape(RefParam),

    // TODO: scoped
    shape(RefTypeparam, field(Ident), field(Typeargs)),
    shape(RefTypealias, field(Ident), field(Typeargs)),
    shape(RefClass, field(Ident), field(Typeargs)),

    shape(RefFunction, field(Ident), field(Typeargs)),
    shape(Selector, field(Ident), field(Typeargs)),

    // TODO:
    shape(Call, undef()),

    shape(
      Include,
      field(Type),
      field(
        Expr,
        Tuple,
        Lambda,
        Call,
        CallLHS,
        String,
        Escaped,
        Char,
        Bool,
        Hex,
        Bin,
        Int,
        Float,
        HexFloat)),

    // TODO:
    shape(Load),
    shape(Store, field(RefLet), field(RefLet)));
}
