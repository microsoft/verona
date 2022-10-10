// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace sample
{
  using namespace wf::ops;

  inline constexpr auto wfIdSym = IdSym >>= Ident | Symbol;

  inline constexpr auto wfLiteral =
    Bool | Int | Hex | Bin | Float | HexFloat | Char | Escaped | String;

  // clang-format off
  inline constexpr auto wfParser =
      (Top <<= (Directory | File)++)
    | (Directory <<= (Directory | File)++)
    | (File <<= Group++)
    | (Brace <<= Group++)
    | (Paren <<= Group++)
    | (Square <<= Group++)
    | (List <<= Group++)
    | (Equals <<= Group++)
    | (Group <<=
        (wfLiteral | Brace | Paren | Square | List | Equals | FatArrow | Use |
         Class | TypeAlias | Var | Let | Ref | Throw | Iso | Imm | Mut |
         DontCare | Ident | Ellipsis | Dot | DoubleColon | Symbol | Colon |
         Package)++)
    ;
  // clang-format on

  inline constexpr auto wfModulesTokens = wfLiteral | Brace | Paren | Square |
    List | Equals | FatArrow | Use | Class | TypeAlias | Var | Let | Ref |
    Throw | Iso | Imm | Mut | DontCare | Ident | Ellipsis | Dot | DoubleColon |
    Symbol | Type | Package;

  // clang-format off
  inline constexpr auto wfPassModules =
      (Top <<= Group++)
    | (Brace <<= Group++)
    | (Paren <<= Group++)
    | (Square <<= Group++)
    | (List <<= Group++)
    | (Equals <<= Group++)
    | (Package <<= String | Escaped)
    | (Type <<= wfModulesTokens++)
    | (Group <<= wfModulesTokens++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassStructure =
      (Top <<= Class++)
    | (Class - Ident <<= Ident * TypeParams * Type * ClassBody)
    | (ClassBody <<=
        (Use | Class | TypeAlias | TypeTrait | FieldLet | FieldVar |
         Function)++)
    | (Use <<= Type)
    | (TypeAlias - Ident <<= Ident * TypeParams * (Bounds >>= Type) * Type)
    | (TypeTrait - Ident <<= Ident * ClassBody)
    | (FieldLet - Ident <<= Ident * Type * Expr)
    | (FieldVar - Ident <<= Ident * Type * Expr)
    | (Function - IdSym <<= wfIdSym * TypeParams * Params * Type * FuncBody)
    | (TypeParams <<= TypeParam++)
    | (TypeParam - Ident <<= Ident * (Bounds >>= Type) * Type)
    | (Params <<= Param++)
    | (Param - Ident <<= Ident * Type * Expr)
    | (TypeTuple <<= Type++)
    | (FuncBody <<= (Use | Class | TypeAlias | Expr)++)
    | (Tuple <<= Expr++)
    | (Assign <<= Expr++[2])
    | (TypeArgs <<= Type++)
    | (Lambda <<= TypeParams * Params * FuncBody)
    | (Let - Ident <<= Ident)
    | (Var - Ident <<= Ident)
    | (Throw <<= Expr)
    | (TypeAssert <<= Type * Expr)
    | (Type <<=
        (Type | TypeTuple | TypeVar | TypeArgs | Package | Iso | Imm | Mut |
         DontCare | Ellipsis | Ident | Symbol | Dot | Throw | DoubleColon)++)
    | (Expr <<=
        (Expr | Tuple | Assign | TypeArgs | Lambda | Let | Var | Throw | Ref |
         DontCare | Ellipsis | Dot | Ident | Symbol | DoubleColon | wfLiteral |
         TypeAssert)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassTypeView =
      wfPassStructure

    // Add TypeName, TypeView, TypeList.
    | (TypeName <<= (TypeName >>= (TypeName | TypeUnit)) * Ident * TypeArgs)
    | (TypeView <<= (lhs >>= Type) * (rhs >>= Type))
    | (TypeList <<= Type)

    // Remove DontCare, Ident, TypeArgs, DoubleColon, Dot, Ellipsis from Type.
    | (Type <<=
        (Type | TypeTuple | TypeVar | Package | Iso | Imm | Mut | Symbol |
         Throw | TypeName | TypeView | TypeList)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassTypeFunc =
      wfPassTypeView

    // Add TypeFunc.
    | (TypeFunc <<= (lhs >>= Type) * (rhs >>= Type))
    | (Type <<=
        (Type | TypeTuple | TypeVar | Package | Iso | Imm | Mut | Symbol |
         Throw | TypeName | TypeView | TypeList | TypeFunc)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassTypeThrow =
      wfPassTypeFunc

    // Add TypeThrow.
    | (TypeThrow <<= Type)
    | (Type <<=
        (Type | TypeTuple | TypeVar | Package | Iso | Imm | Mut | Symbol |
         TypeThrow | TypeName | TypeView | TypeList | TypeFunc)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassTypeAlg =
      wfPassTypeThrow

    // Add TypeUnion, TypeIsect.
    | (TypeUnion <<= Type++[2])
    | (TypeIsect <<= Type++[2])

    // Remove Symbol, add TypeUnion and TypeIsect.
    | (Type <<=
        (Type | TypeTuple | TypeVar | Package | Iso | Imm | Mut | TypeThrow |
         TypeName | TypeView | TypeList | TypeFunc | TypeUnion | TypeIsect)++)
    ;
  // clang-format on

  inline constexpr auto wfTypeNoAlg = TypeTuple | TypeVar | Package | Iso |
    Imm | Mut | TypeName | TypeView | TypeList | TypeFunc | TypeUnit;

  inline constexpr auto wfType =
    wfTypeNoAlg | TypeUnion | TypeIsect | TypeThrow;

  // clang-format off
  inline constexpr auto wfPassTypeFlat =
      wfPassTypeAlg

    // No Type nodes inside of type structure.
    | (TypeList <<= wfType)
    | (TypeTuple <<= wfType++[2])
    | (TypeView <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeFunc <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeUnion <<= (wfTypeNoAlg | TypeIsect | TypeThrow)++[2])
    | (TypeThrow <<= wfTypeNoAlg | TypeIsect | TypeUnion)
    | (TypeIsect <<= (wfTypeNoAlg | TypeUnion | TypeThrow)++[2])

    // Types are no longer sequences.
    | (Type <<= wfType)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassTypeDNF =
      wfPassTypeFlat

    // Disjunctive normal form.
    | (TypeUnion <<= (wfTypeNoAlg | TypeIsect | TypeThrow)++[2])
    | (TypeThrow <<= wfTypeNoAlg | TypeIsect)
    | (TypeIsect <<= wfTypeNoAlg++[2])
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassInclude =
      wfPassTypeDNF

    // Replace all Use with Include.
    | (ClassBody <<=
        (Include | Class | TypeAlias | FieldLet | FieldVar | Function)++)
    | (Include <<= TypeName)
    | (FuncBody <<= (Include | Class | TypeAlias | Expr)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassReference =
      wfPassInclude

    // Add Selector, FunctionName.
    | (Selector <<= wfIdSym * TypeArgs)
    | (FunctionName <<= (TypeName >>= (TypeName | TypeUnit)) * Ident * TypeArgs)

    // Remove TypeArgs, Ident, Symbol, DoubleColon.
    // Add RefVar, RefLet, Selector, FunctionName.
    | (Expr <<=
        (Expr | Tuple | Assign | Lambda | Let | Var | Throw | Ref |
         DontCare | Ellipsis | Dot | wfLiteral | TypeAssert |
         RefVar | RefLet | Selector | FunctionName)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassReverseApp =
      wfPassReference

    // Add Call, Args.
    | (Call <<= (Call >>= (Selector | FunctionName)) * Args)
    | (Args <<= Expr++)

    // Remove Dot, Add Call.
    | (Expr <<=
        (Expr | Tuple | Assign | Lambda | Let | Var | Throw | Ref |
         DontCare | Ellipsis | wfLiteral | TypeAssert | RefVar | RefLet |
         Selector | FunctionName | Call)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wfPassApplication =
      wfPassReverseApp

    // TODO: seq of Expr only?
    // Remove Ref.
    | (Expr <<=
        (Expr | Tuple | Assign | Lambda | Let | Var | Throw | DontCare |
         Ellipsis | wfLiteral | TypeAssert | RefVar | RefLet | Selector |
         FunctionName | Call)++)
    ;
  // clang-format on

  // clang-format off
  inline constexpr auto wf =
      Ident
    | (TypeAlias <<= Ident * TypeParams * (Bounds >>= Type) * (Default >>= Type))
    | (Class <<= Ident * TypeParams * Type * ClassBody)
    | (ClassBody <<= (Use | Class | TypeAlias | FieldLet | FieldVar | Function)++)
    | (FieldLet <<= Ident * Type * Expr)
    | (FieldVar <<= Ident * Type * Expr)
    | (Function <<= wfIdSym * TypeParams * Params * (Type >>= wfType) * FuncBody)
    | (Params <<= Param++) | (Param <<= Ident * Type * Expr)
    | FuncBody // TODO: define it
    | (Type <<= wfType)
    | (TypeName <<= (TypeName >>= (TypeName | TypeUnit)) * Ident * TypeArgs)
    | (TypeTuple <<= wfType++)
    | (TypeView <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeFunc <<= (lhs >>= wfType) * (rhs >>= wfType))
    | (TypeThrow <<= (Type >>= wfType))
    | (TypeIsect <<= wfType++)
    | (TypeUnion <<= wfType++)
    | TypeVar // TODO:
    | (TypeTrait <<= ClassBody)
    | Iso | Imm | Mut
    | (Package <<= (id >>= String | Escaped))
    // | (RefType <<= Ident * TypeArgs) // TODO: scoped
    | (Var <<= Ident * Type)
    | (Let <<= Ident * Type)
    | (Throw <<= Expr)
    | Expr // TODO: define it
    | (TypeArgs <<= wfType++)
    | (Lambda <<= TypeParams * Params * FuncBody)
    | Tuple // TODO: define it
    | (Assign <<= Expr++)
    ;
  // clang-format on
}
