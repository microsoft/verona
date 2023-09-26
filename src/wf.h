// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"

namespace verona
{
  using namespace wf::ops;

  inline const auto wfImplicit = Implicit >>= Implicit | Explicit | LambdaFunc;
  inline const auto wfHand = Ref >>= Lhs | Rhs;

  inline const auto wfParserTokens = True | False | Int | Hex | Oct | Bin |
    Float | HexFloat | Char | Escaped | String | LLVM | Iso | Mut | Imm |
    Brace | Paren | Square | List | Equals | Arrow | Use | Class | TypeAlias |
    Where | Var | Let | Ref | Self | If | Else | New | Try | DontCare | Ident |
    Ellipsis | Dot | Colon | DoubleColon | TripleColon | Symbol;

  // clang-format off
  inline const auto wfParser =
      (Top <<= (Directory | File)++)
    | (Directory <<= (Directory | File)++)
    | (File <<= (Group | List | Equals)++)
    | (Brace <<= (Group | List | Equals)++)
    | (Paren <<= (Group | List | Equals)++)
    | (Square <<= (Group | List | Equals)++)
    | (List <<= (Group | Equals)++)
    | (Equals <<= Group++)
    | (Group <<= wfParserTokens++)
    ;
  // clang-format on

  // Remove Colon, Where. Add Type, TypePred, LLVMFuncType.
  inline const auto wfModulesTokens =
    (wfParserTokens - (Colon | Where)) | Type | TypePred | LLVMFuncType;

  // clang-format off
  inline const auto wfPassModules =
      (Top <<= Group++)
    | (Brace <<= (Group | List | Equals)++)
    | (Paren <<= (Group | List | Equals)++)
    | (Square <<= (Group | List | Equals)++)
    | (List <<= (Group | Equals)++)
    | (Equals <<= Group++)
    | (LLVMFuncType <<= (Args >>= LLVMList) * (Return >>= LLVM | Ident))
    | (LLVMList <<= (LLVM | Ident)++)
    | (TypePred <<= Type)
    | (Type <<= (wfModulesTokens | TypeTrue)++)
    | (Group <<= wfModulesTokens++)
    ;
  // clang-format on

  inline const auto wfTypeStructure = Type | TypeTrue | TypeFalse | Iso | Mut |
    Imm | Trait | TypeTuple | TypeVar | TypeArgs | Package | Self | DontCare |
    Ellipsis | Ident | Symbol | Dot | DoubleColon;

  inline const auto wfExprStructure = Expr | ExprSeq | Unit | Tuple | Assign |
    TypeArgs | Self | If | Else | Lambda | Let | Var | New | Try | Ref |
    DontCare | Ellipsis | Dot | Ident | Symbol | DoubleColon | True | False |
    Int | Hex | Oct | Bin | Float | HexFloat | Char | Escaped | String | LLVM |
    TypeAssert;

  inline const auto wfDefault = Default >>= Block | DontCare;

  // clang-format off
  inline const auto wfPassStructure =
      (Top <<= Class++)
    | (Class <<= Ident * TypeParams * Inherit * TypePred * ClassBody)[Ident]
    | (Inherit <<= Type | DontCare)
    | (ClassBody <<=
        (Use | Class | TypeAlias | FieldLet | FieldVar | Function)++)
    | (Use <<= Type)[Include]
    | (TypeAlias <<= Ident * TypeParams * TypePred * Type)[Ident]
    | (Trait <<= Ident * ClassBody)[Ident]
    | (FieldLet <<= wfImplicit * Ident * Type * wfDefault)[Ident]
    | (FieldVar <<= wfImplicit * Ident * Type * wfDefault)[Ident]
    | (Function <<=
        wfImplicit * wfHand * Ident * TypeParams * Params * Type *
        (LLVMFuncType >>= LLVMFuncType | DontCare) * TypePred *
        (Block >>= Block | DontCare))[Ident]
    | (TypeParams <<= TypeParam++)
    | (TypeParam <<= Ident * (Type >>= Type | DontCare))[Ident]
    | (ValueParam <<= Ident * Type * Expr)[Ident]
    | (Params <<= Param++)
    | (Param <<= Ident * Type * wfDefault)[Ident]
    | (TypeTuple <<= Type++)
    | (Block <<= (Use | Class | TypeAlias | Expr)++[1])
    | (ExprSeq <<= Expr++[2])
    | (Tuple <<= Expr++[2])
    | (Assign <<= Expr++[2])
    | (TypeArgs <<= (Type | TypeParamBind)++)
    | (Lambda <<= TypeParams * Params * Type * TypePred * Block)
    | (Let <<= Ident)[Ident]
    | (Var <<= Ident)[Ident]
    | (TypeAssert <<= Expr * Type)
    | (Package <<= (Ident >>= String | Escaped))
    | (LLVMFuncType <<= (Args >>= LLVMList) * (Return >>= LLVM | Ident))
    | (LLVMList <<= (LLVM | Ident)++)
    | (TypePred <<= Type)
    | (Type <<= wfTypeStructure++)
    | (Expr <<= wfExprStructure++[1])
    ;
  // clang-format on

  // Add RefVar, RefLet, Selector, FQFunction.
  inline const auto wfExprReference =
    wfExprStructure | RefVar | RefLet | Selector;

  // clang-format off
  inline const auto wfPassReference =
      wfPassStructure
    | (RefLet <<= Ident)
    | (RefVar <<= Ident)
    | (Selector <<= wfHand * Ident * Int * TypeArgs)
    | (Expr <<= wfExprReference++[1])
    ;
  // clang-format on

  // Remove If, Else. Add Conditional, TypeTest, Cast.
  inline const auto wfExprConditionals =
    (wfExprReference - (If | Else)) | Conditional | TypeTest | Cast;

  // clang-format off
  inline const auto wfPassConditionals =
      wfPassReference
    | (Conditional <<= (If >>= Expr) * Block * Block)
    | (TypeTest <<= Expr * Type)
    | (Cast <<= Expr * Type)
    | (Expr <<= wfExprConditionals++[1])
    ;
  // clang-format on

  // Remove Lambda.
  inline const auto wfExprLambda = wfExprConditionals - Lambda;

  // clang-format off
  inline const auto wfPassLambda =
      wfPassConditionals
    | (Expr <<= wfExprLambda++[1])
    ;
  // clang-format on

  // clang-format off
  inline const auto wfPassDefaultArgs =
      wfPassLambda
    | (FieldLet <<= wfImplicit * Ident * Type)[Ident]
    | (FieldVar <<= wfImplicit * Ident * Type)[Ident]
    | (Param <<= Ident * Type)[Ident]
    ;
  // clang-format on

  // Remove DontCare, Ident. Add FQType.
  inline const auto wfTypeNames =
    (wfTypeStructure - (DontCare | Ident)) | FQType;

  // clang-format off
  inline const auto wfPassTypeNames =
      wfPassDefaultArgs
    | (FQType <<=
        TypePath *
        (Type >>=
          TypeClassName | TypeAliasName | TypeParamName | TypeTraitName))
    | (TypePath <<=
        (TypeClassName | TypeAliasName | TypeParamName | TypeTraitName |
         Selector)++)
    | (TypeClassName <<= Ident * TypeArgs)
    | (TypeAliasName <<= Ident * TypeArgs)
    | (TypeParamName <<= Ident)
    | (TypeTraitName <<= Ident)
    | (Type <<= wfTypeNames++)
    ;
  // clang-format on

  // Remove DoubleColon, Dot, Ellipsis, TypeArgs. Add TypeView, TypeList.
  inline const auto wfTypeView =
    (wfTypeNames - (DoubleColon | Dot | Ellipsis | TypeArgs)) | TypeView |
    TypeList;

  // clang-format off
  inline const auto wfPassTypeView =
      wfPassTypeNames
    | (TypeView <<= Type++[2])
    | (TypeList <<= Type)
    | (Type <<= wfTypeView++)
    ;
  // clang-format on

  // Add TypeUnion, TypeIsect.
  inline const auto wfTypeFunc = wfTypeView | TypeUnion | TypeIsect;

  // clang-format off
  inline const auto wfPassTypeFunc =
      wfPassTypeView
    | (TypeUnion <<= Type++[2])
    | (TypeIsect <<= Type++[2])
    | (Type <<= wfTypeFunc++)
    ;
  // clang-format on

  // Remove Symbol. Add TypeSubtype.
  inline const auto wfTypeAlg = (wfTypeFunc - Symbol) | TypeSubtype;

  // clang-format off
  inline const auto wfPassTypeAlg =
      wfPassTypeFunc
    | (TypeSubtype <<= (Lhs >>= Type) * (Rhs >>= Type))
    | (Type <<= wfTypeAlg++)
    ;
  // clang-format on

  // Remove Type. Types are no longer sequences.
  inline const auto wfType = wfTypeAlg - Type;

  // clang-format off
  inline const auto wfPassTypeFlat =
      wfPassTypeAlg
    | (TypeList <<= wfType)
    | (TypeTuple <<= wfType++[2])
    | (TypeView <<= wfType++[2])
    | (TypeSubtype <<= (Lhs >>= wfType) * (Rhs >>= wfType))
    | (TypeUnion <<= (wfType - TypeUnion)++[2])
    | (TypeIsect <<= (wfType - TypeIsect)++[2])
    | (Type <<= wfType)
    ;
  // clang-format on

  // Remove New, Ident, Symbol, Self, DoubleColon, TypeArgs. Add FQFunction.
  inline const auto wfExprTypeReference =
    (wfExprLambda - (New | Ident | Symbol | Self | DoubleColon | TypeArgs)) |
    FQFunction;

  // clang-format off
  inline const auto wfPassTypeReference =
      wfPassTypeFlat
    | (FQFunction <<= FQType * Selector)
    | (Expr <<= wfExprTypeReference++[1])
    ;

  // Remove Use. Remove implicit marker on fields.
  // clang-format off
  inline const auto wfPassResetImplicit =
      wfPassTypeReference
    | (ClassBody <<= (Class | TypeAlias | FieldLet | FieldVar | Function)++)
    | (Block <<= (Class | TypeAlias | Expr)++[1])
    | (FieldLet <<= Ident * Type)[Ident]
    | (FieldVar <<= Ident * Type)[Ident]
    ;
  // clang-format on

  // Remove Dot. Add Call, NLRCheck.
  inline const auto wfExprReverseApp =
    (wfExprTypeReference - Dot) | Call | NLRCheck;

  // clang-format off
  inline const auto wfPassReverseApp =
      wfPassResetImplicit
    | (Call <<= (Selector >>= (Selector | FQFunction)) * Args)
    | (Args <<= Expr++)
    | (NLRCheck <<= Call)
    | (Expr <<= wfExprReverseApp++[1])
    ;
  // clang-format on

  // Remove Unit, True, False, DontCare, Ellipsis, Selector, FQFunction.
  // Add RefVarLHS.
  inline const auto wfExprApplication =
    (wfExprReverseApp -
     (Unit | True | False | DontCare | Ellipsis | Selector | FQFunction)) |
    RefVarLHS;

  // clang-format off
  inline const auto wfPassApplication =
      wfPassReverseApp
    | (Tuple <<= (Expr | TupleFlatten)++[2])
    | (TupleFlatten <<= Expr)
    | (RefVarLHS <<= Ident)
    | (Expr <<= wfExprApplication++[1])
    ;
  // clang-format on

  // Remove Expr, Try, Ref. Add TupleLHS.
  inline const auto wfExprAssignLHS =
    (wfExprApplication - (Expr | Try | Ref)) | TupleLHS;

  // clang-format off
  inline const auto wfPassAssignLHS =
      wfPassApplication
    | (TupleLHS <<= Expr++[2])

    // Expressions are no longer sequences.
    | (Expr <<= wfExprAssignLHS)
    ;
  // clang-format on

  // Remove Var, RefVar, RefVarLHS.
  inline const auto wfExprAssign = wfExprAssignLHS - (Var | RefVar | RefVarLHS);

  // clang-format off
  inline const auto wfPassLocalVar =
      wfPassAssignLHS
    | (Expr <<= wfExprAssign)
    ;
  // clang-format on

  // Remove Assign, Let, TupleLHS. Add Bind.
  inline const auto wfExprAssignment =
    (wfExprAssign - (Assign | Let | TupleLHS)) | Bind;

  // clang-format off
  inline const auto wfPassAssignment =
      wfPassLocalVar
    | (Bind <<= Ident * Type * Expr)[Ident]
    | (Expr <<= wfExprAssignment)
    ;
  // clang-format on

  // Add FieldRef.
  inline const auto wfExprAutoFields = wfExprAssignment | FieldRef;

  // clang-format off
  inline const auto wfPassAutoFields =
      wfPassAssignment
    | (FieldRef <<= RefLet * Ident)
    | (Expr <<= wfExprAutoFields)
    ;
  // clang-format on

  // Remove NLRCheck.
  inline const auto wfExprNLRCheck = wfExprAutoFields - NLRCheck;

  // clang-format off
  inline const auto wfPassNLRCheck =
      wfPassAutoFields

    // Add Return.
    | (Block <<= (Class | TypeAlias | Expr | Return)++[1])
    | (Return <<= Expr)
    | (Expr <<= wfExprNLRCheck)
    ;
  // clang-format on

  // Remove ExprSeq, TypeAssert, Bind.
  inline const auto wfExprANF = wfExprNLRCheck - (ExprSeq | TypeAssert | Bind);

  // clang-format off
  inline const auto wfPassANF =
      wfPassNLRCheck
    | (Block <<= (Class | TypeAlias | Bind | RefLet | Return | LLVM)++[1])
    | (Return <<= RefLet)
    | (Tuple <<= (RefLet | TupleFlatten)++[2])
    | (TupleFlatten <<= RefLet)
    | (Args <<= RefLet++)
    | (Conditional <<= (If >>= RefLet) * Block * Block)
    | (TypeTest <<= RefLet * Type)
    | (Cast <<= RefLet * Type)
    | (Bind <<= Ident * Type * (Rhs >>= wfExprANF))[Ident]
    ;
  // clang-format on

  // Remove RefLet. Add Copy, Move.
  inline const auto wfExprDrop = (wfExprANF - RefLet) | Copy | Move;

  // clang-format off
  inline const auto wfPassDrop =
      wfPassANF
    | (Copy <<= Ident)
    | (Move <<= Ident)
    | (Drop <<= Ident)
    | (Block <<=
        (Class | TypeAlias | Bind | Return | LLVM | Move | Drop)++[1])
    | (Return <<= (Ref >>= Move))
    | (Tuple <<= (TupleFlatten | Copy | Move)++[2])
    | (TupleFlatten <<= Copy | Move)
    | (Args <<= (Copy | Move)++)
    | (Conditional <<=
        (If >>= Copy | Move) * (True >>= Block) * (False >>= Block))
    | (TypeTest <<= (Ref >>= (Copy | Move)) * Type)
    | (Cast <<= (Ref >>= (Copy | Move)) * Type)
    | (FieldRef <<= (Ref >>= (Copy | Move)) * Ident)
    | (Bind <<= Ident * Type * (Rhs >>= wfExprDrop))[Ident]
    ;
  // clang-format on
}
