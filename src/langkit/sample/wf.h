// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace sample
{
  using namespace wf::ops;

  inline constexpr auto wfType = RefType | TypeTuple | TypeView | TypeFunc |
    TypeThrow | TypeIsect | TypeUnion | TypeVar | TypeTrait | Iso | Imm | Mut;

  inline constexpr auto wfIdSym = IdSym >>= Ident | Symbol;

  inline constexpr auto wfTypesTokens = FatArrow | Use | Typealias | Class |
    Var | Let | Ref | Throw | Iso | Imm | Mut | DontCare | Ident | Ellipsis |
    Dot | DoubleColon | Symbol;

  inline constexpr auto wfParseTokens = wfTypesTokens | Colon | Package;

  inline constexpr auto wfLiteral =
    Bool | Int | Hex | Bin | Float | HexFloat | Char | Escaped | String;

  // clang-format off
  inline constexpr auto wfParseShape =
      (Brace <<= Group++)
    | (Paren <<= Group++)
    | (Square <<= Group++)
    | (List <<= Group++)
    | (Equals <<= Group++)
    ;

  inline constexpr auto wfParser =
      (Top <<= (Directory | File)++)
    | wfLiteral
    | wfParseTokens
    | wfParseShape
    | (Directory <<= (Directory | File)++)
    | (File <<= (Directory | Group)++)
    | (Group <<=
        (wfLiteral | wfParseTokens | Brace | Paren | Square | List | Equals)++)
    ;

  // Removes Directories and Files, adds TypeTraits.
  inline constexpr auto wfPassModules =
      (Top <<= Group++)
    | wfLiteral
    | wfParseTokens
    | wfParseShape
    | (TypeTrait <<= Classbody)
    | (Classbody <<= Group++) 
    | (Group <<=
        (wfLiteral | wfParseTokens | Brace | Paren | Square | List | Equals |
         TypeTrait)++)
    ;

  // Removes Colons, adds Types. Packages contain strings.
  inline constexpr auto wfPassTypes =
      (Top <<= Group++)
    | wfLiteral
    | wfTypesTokens
    | wfParseShape
    | (Type <<=
        (wfLiteral | wfTypesTokens | Paren | Square | List | Equals |
         TypeTrait | Type | Package)++)
    | (Package <<= String | Escaped)
    | (TypeTrait <<= Classbody)
    | (Classbody <<= Group++)
    | (Group <<=
        (wfLiteral | wfTypesTokens | Brace | Paren | Square | List | Equals |
         TypeTrait | Type | Package)++)
    ;

  inline constexpr auto wfPassStructure =
      (Top <<= Class++)
    | (Class - Ident <<= Ident * Typeparams * Type * Classbody)
    | (Classbody <<=
        (Use | Class | Typealias | FieldLet | FieldVar | Function)++)
    | (FieldLet - Ident <<= Ident * Type * Expr)
    | (FieldVar - Ident <<= Ident * Type * Expr)
    | (Function - IdSym <<= wfIdSym * Typeparams * Params * Type * Funcbody)
    | (Typeparams <<= Typeparam++)
    | (Typeparam - Ident <<= Ident * (Bounds >>= Type) * Type)
    | (Params <<= Param++)
    | (Param - Ident <<= Ident * Type * Expr)
    | (Typealias - Ident <<= Ident * Typeparams * (Bounds >>= Type) * Type)
    | (Use <<= Type)
    | (Type <<=
        (Type | TypeTuple | TypeTrait | TypeVar | Package | Throw |
         Ident | Symbol | DoubleColon)++)
    | (TypeTrait <<= Classbody)
    | (TypeTuple <<= Type++)
    | (Funcbody <<= Expr++)
    | (Expr <<=
        (Expr | Use | Tuple | Assign | Typeargs | Lambda | Let | Var | Throw |
         Ref | DontCare | Ident | Ellipsis | Dot | DoubleColon | Symbol |
         wfLiteral)++)
    | (Tuple <<= Expr++)
    | (Assign <<= Expr++)
    | (Typeargs <<= Type++)
    | (Lambda <<= Typeparams * Params * Funcbody)
    | (Let - Ident <<= Ident * (Type >>= (Type | TypeVar)))
    | (Var - Ident <<= Ident * (Type >>= (Type | TypeVar)))
    | (Throw <<= Expr)
    | Ref | DontCare | Ident | Ellipsis | Dot | DoubleColon | Symbol | wfLiteral
    ;

  inline constexpr auto wf =
      Ident
    | (Typealias <<= Ident * Typeparams * (Bounds >>= Type) * (Default >>= Type))
    | (Class <<= Ident * Typeparams * Type * Classbody)
    | (Classbody <<= (Use | Class | Typealias | FieldLet | FieldVar | Function)++)
    | (FieldLet <<= Ident * Type * Expr)
    | (FieldVar <<= Ident * Type * Expr)
    | (Function <<= wfIdSym * Typeparams * Params * (Type >>= wfType) * Funcbody)
    | (Params <<= Param++) | (Param <<= Ident * Type * Expr)
    | Funcbody // TODO: define it
    | (Type <<= wfType)
    | (TypeTuple <<= wfType++)
    | (TypeView <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeFunc <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeThrow <<= (Type >>= wfType))
    | (TypeIsect <<= wfType++)
    | (TypeUnion <<= wfType++)
    | TypeVar // TODO:
    | (TypeTrait <<= Classbody)
    | Iso | Imm | Mut
    | (Package <<= (id >>= String | Escaped))
    | (RefType <<= Ident * Typeargs) // TODO: scoped
    | (Var <<= Ident * Type)
    | (Let <<= Ident * Type)
    | (Throw <<= Expr)
    | Expr // TODO: define it
    | (Typeargs <<= wfType++)
    | (Lambda <<= Typeparams * Params * Funcbody)
    | Tuple // TODO: define it
    | (Assign <<= Expr++)
    ;

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
