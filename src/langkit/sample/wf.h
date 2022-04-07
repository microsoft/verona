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
    shape(Var),
    shape(Let),
    shape(Throw, field(Expr)),
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

    // TODO:
    shape(Expr, undef()),
    shape(Term, undef()),
    shape(Typeargs, undef()),

    shape(Lambda, field(Typeparams), field(Params), field(Funcbody)),

    // TODO:
    shape(Tuple, undef()),
    shape(Assign, undef()),

    shape(RefVar, field(Ident), field(Typeargs)),
    shape(RefLet, field(Ident), field(Typeargs)),
    shape(RefParam, field(Ident), field(Typeargs)),

    // TODO: scoped
    shape(RefTypeparam, field(Ident), field(Typeargs)),
    shape(RefTypealias, field(Ident), field(Typeargs)),
    shape(RefClass, field(Ident), field(Typeargs)),

    shape(RefFunction, field(Ident), field(Typeargs)),
    shape(Selector, field(Ident), field(Typeargs)),

    // TODO:
    shape(Call, undef()),

    shape(Include, field(Type)),

    // TODO: Let, Expr
    shape(Lift, undef()));
}