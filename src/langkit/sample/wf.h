// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace sample
{
  using namespace wf::ops;

  inline constexpr auto wfTypes = RefType | TypeTuple | TypeView | TypeFunc |
    TypeThrow | TypeIsect | TypeUnion | TypeVar | TypeTrait | Iso | Imm | Mut;

  inline constexpr auto wfIdSym = IdSym >>= Ident | Symbol;

  inline constexpr auto wfParseTokens = FatArrow | Package | Use | Typealias |
    Class | Var | Let | Ref | Throw | Iso | Imm | Mut | DontCare | Ident |
    Ellipsis | Dot | DoubleColon | Colon | Symbol;

  inline constexpr auto wfLiteral =
    Bool | Int | Hex | Bin | Float | HexFloat | Char | Escaped | String;

  inline constexpr auto wfInGroup =
    wfLiteral | Brace | Paren | Square | List | Equals | wfParseTokens;

  inline constexpr auto wfParser = wfLiteral | wfParseTokens |
    (Directory <<= (Directory | File)++) | (File <<= (Directory | Group)++) |
    (Brace <<= Group++) | (Paren <<= Group++) | (Square <<= Group++) |
    (List <<= Group++) | (Equals <<= Group++) | (Group <<= wfInGroup++);

  inline constexpr auto wf = Ident |
    (Typealias <<=
     Ident * Typeparams * (Bounds >>= Type) * (Default >>= Type)) |
    (Class <<= Ident * Typeparams * Type * Classbody) |
    (Classbody <<=
     (Use | Class | Typealias | FieldLet | FieldVar | Function)++) |

    (FieldLet <<= Ident * Type * Expr) | (FieldVar <<= Ident * Type * Expr) |
    (Function <<=
     wfIdSym * Typeparams * Params * (Type >>= wfTypes) * Funcbody) |

    (Params <<= Param++) | (Param <<= Ident * Type * Expr) |
    Funcbody | // TODO: define it

    (Type <<= (Type >>= wfTypes)) | (TypeTuple <<= wfTypes++) |
    (TypeView <<= (lhs >>= wfTypes) * (rhs >>= wfTypes)) |
    (TypeFunc <<= (lhs >>= wfTypes) * (rhs >>= wfTypes)) |
    (TypeThrow <<= (Type >>= wfTypes)) | (TypeIsect <<= wfTypes++) |
    (TypeUnion <<= wfTypes++) | TypeVar | // TODO:
    (TypeTrait <<= Classbody) | Iso | Imm | Mut |
    (Package <<= (id >>= String | Escaped)) |
    (RefType <<= Ident * Typeargs) | // TODO: scoped

    (Var <<= Ident * Type) | (Let <<= Ident * Type) | (Throw <<= Expr) |

    Expr | // TODO: define it
    (Typeargs <<= wfTypes++) | (Lambda <<= Typeparams * Params * Funcbody) |
    Tuple | // TODO: define it
    (Assign <<= Expr++);

  //   // These start containing an Ident and Typeargs, but after `refexpr`,
  //   // they're empty and their location is their variable.
  //   // TODO: should they keep an Ident node?
  //   shape(RefVar),
  //   shape(RefLet),
  //   shape(RefParam),

  //   shape(RefVarLHS),

  //   // TODO: scoped RefFunction
  //   shape(RefFunction, wfIdSym, field(Typeargs)),
  //   shape(Selector, wfIdSym, field(Typeargs)),

  //   // TODO:
  //   shape(Call, undef()),
  //   shape(CallLHS, undef()),

  //   shape(Include, field(Type)));
}
