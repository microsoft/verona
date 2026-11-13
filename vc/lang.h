#pragma once

#define TRIESTE_EXPOSE_LOG_MACRO
#include <trieste/trieste.h>
#include <vir.h>
#include <virc/lang.h>

namespace vc
{
  using namespace trieste;
  using namespace trieste::wf::ops;
  using namespace vir;
  using namespace virc;

  inline const auto Paren = TokenDef("paren");
  inline const auto Bracket = TokenDef("bracket");
  inline const auto Brace = TokenDef("brace");
  inline const auto List = TokenDef("list");
  inline const auto DoubleColon = TokenDef("doublecolon");
  inline const auto TripleColon = TokenDef("triplecolon");
  inline const auto Dot = TokenDef("dot");
  inline const auto DontCare = TokenDef("dontcare");
  inline const auto SplatLet =
    TokenDef("splatlet", flag::lookup | flag::shadowing);
  inline const auto SplatDontCare = TokenDef("splatdc");
  inline const auto SplatOp = TokenDef("splatop");
  inline const auto ArrayRefFromEnd = TokenDef("arrayrefend");
  inline const auto Ident = TokenDef("ident", flag::print);
  inline const auto Use = TokenDef("use");
  inline const auto Shape = TokenDef("shape");
  inline const auto Hash = TokenDef("hash");

  inline const auto ClassDef = TokenDef(
    "classdef", flag::symtab | flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto TypeAlias = TokenDef(
    "typealias",
    flag::symtab | flag::lookup | flag::lookdown | flag::shadowing);
  inline const auto FieldDef = TokenDef("fielddef");
  inline const auto Function =
    TokenDef("function", flag::symtab | flag::lookup | flag::lookdown);
  inline const auto ParamDef =
    TokenDef("paramdef", flag::lookup | flag::shadowing);

  inline const auto TypeName = TokenDef("typename");
  inline const auto NameElement = TokenDef("nameelement");
  inline const auto TypePath = TokenDef("typepath");
  inline const auto TypeParams = TokenDef("typeparams");
  inline const auto TypeParam =
    TokenDef("typeparam", flag::lookup | flag::shadowing);
  inline const auto TypeArgs = TokenDef("typeargs");
  inline const auto Where = TokenDef("where");
  inline const auto ClassBody = TokenDef("classbody");

  inline const auto Isect = TokenDef("isect");
  inline const auto FuncType = TokenDef("functype");
  inline const auto NoArgType = TokenDef("noargtype");
  inline const auto TypeVar = TokenDef("typevar");
  inline const auto TypeSelf = TokenDef("typeself");

  // Sentinel types for unresolved integer/float literals. Exist only during
  // the infer pass — resolved to U64/F64 before the pass terminates.
  inline const auto DefaultInt = TokenDef("default_int");
  inline const auto DefaultFloat = TokenDef("default_float");

  inline const auto WhereAnd = TokenDef("whereand");
  inline const auto WhereOr = TokenDef("whereor");
  inline const auto WhereNot = TokenDef("wherenot");
  inline const auto SubType = TokenDef("subtype");

  inline const auto Expr = TokenDef("expr");
  inline const auto ExprSeq = TokenDef("exprseq");
  inline const auto Tuple = TokenDef("tuple");
  inline const auto TupleLHS = TokenDef("tuplelhs");
  inline const auto ArrayLit = TokenDef("arraylit");
  inline const auto Lambda = TokenDef("lambda", flag::symtab);
  inline const auto Block = TokenDef("block", flag::symtab);
  inline const auto FuncName = TokenDef("funcname");
  inline const auto MethodName = TokenDef("methodname");
  inline const auto Binop = TokenDef("binop");
  inline const auto Unop = TokenDef("unop");
  inline const auto Nulop = TokenDef("nulop");
  inline const auto NewArgs = TokenDef("newargs");
  inline const auto NewArg = TokenDef("newarg");

  inline const auto Semi = TokenDef("semi");
  inline const auto Let = TokenDef("let", flag::lookup | flag::shadowing);
  inline const auto Var = TokenDef("var", flag::lookup | flag::shadowing);
  inline const auto TypeAssertion = TokenDef("typeassertion");
  inline const auto TypedMove =
    TokenDef("typedmove", flag::lookup | flag::shadowing);
  inline const auto If = TokenDef("if");
  inline const auto Else = TokenDef("else");
  inline const auto While = TokenDef("while");
  inline const auto Break = TokenDef("break");
  inline const auto Continue = TokenDef("continue");
  inline const auto MatchExpr = TokenDef("match");

  const auto ValuePat =
    T(True,
      False,
      Bin,
      Oct,
      Int,
      Hex,
      Float,
      HexFloat,
      String,
      RawString,
      Char,
      Convert,
      Binop,
      Unop,
      Nulop,
      FFI,
      NewArray,
      ArrayRef,
      FieldRef,
      Load,
      New,
      Stack,
      If,
      Else,
      While,
      When,
      Equals,
      LocalId,
      Call,
      CallDyn,
      TryCallDyn,
      Tuple,
      ArrayLit,
      Typetest,
      ExprSeq,
      DontCare);

  inline const auto wfType =
    TypeName | Union | Isect | TupleType | FuncType | TypeVar | TypeSelf;
  inline const auto wfWhere = WhereAnd | WhereOr | WhereNot | SubType;

  inline const auto wfBody = Use | Break | Continue | Return | Raise | Expr;

  inline const auto wfBinop = Add | Sub | Mul | Div | Mod | Pow | And | Or |
    Xor | Shl | Shr | Eq | Ne | Lt | Le | Gt | Ge | Min | Max | LogBase | Atan2;

  inline const auto wfUnop = Neg | Not | Abs | Ceil | Floor | Exp | Log | Sqrt |
    Cbrt | IsInf | IsNaN | Sin | Cos | Tan | Asin | Acos | Atan | Sinh | Cosh |
    Tanh | Asinh | Acosh | Atanh | Bits | Len | MakePtr | Read;

  inline const auto wfNulop = None | Const_E | Const_Pi | Const_Inf |
    Const_NaN | AddExternal | RemoveExternal;

  inline const auto wfExprStructure = ExprSeq | DontCare | TripleColon |
    wfLiteral | Char | String | RawString | Tuple | ArrayLit | Let | Var | New |
    Stack | Lambda | Ref | FuncName | Dot | If | Else | While | When | Equals |
    Hash | FieldRef | GetRaise | SetRaise | MatchExpr | SplatLet |
    SplatDontCare;

  inline const auto wfFuncLhs = Lhs >>= Lhs | Rhs;
  inline const auto wfFuncDefLhs = Lhs >>= Lhs | Rhs | Once;
  inline const auto wfFuncId = Ident >>= Ident | SymbolId;

  // clang-format off
  inline const auto wfPassStructure =
      (Top <<= ClassDef++)
    | (ClassDef <<=
        (Shape >>= Shape | None) * Ident * TypeParams * Where *
        ClassBody)[Ident]
    | (ClassBody <<= (ClassDef | Use | TypeAlias | Lib | FieldDef | Function)++)
    | (Lib <<= String * Symbols)
    | (Symbols <<= (Symbol | Function)++)
    | (Symbol <<=
        SymbolId * (Lhs >>= String) * (Rhs >>= String) *
        (Vararg >>= Vararg | None) * FFIParams * Type)
    | (FFIParams <<= Type++)
    | (Use <<= TypeName)[Include]
    | (TypeAlias <<= Ident * TypeParams * Where * Type)[Ident]
    | (Where <<= ~wfWhere)
    | (FieldDef <<= Ident * Type)
    | (Function <<=
        wfFuncDefLhs * wfFuncId * TypeParams * Params * Type * Where * Body)[Ident]
    | (TypeName <<= NameElement++[1])
    | (NameElement <<= wfFuncId * TypeArgs)
    | (TypeParams <<= TypeParam++)
    | (TypeParam <<= Ident)[Ident]
    | (TypeArgs <<= (Type | Expr)++)
    | (Params <<= (ParamDef | Expr)++)
    | (ParamDef <<= Ident * Type * Body)[Ident]
    | (Type <<= wfType)
    | (Union <<= wfType++[2])
    | (Isect <<= wfType++[2])
    | (TupleType <<= wfType++[2])
    | (FuncType <<= (Lhs >>= wfType | NoArgType) * (Rhs >>= wfType))
    | (WhereAnd <<= wfWhere++[2])
    | (WhereOr <<= wfWhere++[2])
    | (WhereNot <<= wfWhere)
    | (SubType <<= (Lhs >>= Type) * (Rhs >>= Type))
    | (Body <<= wfBody++)
    | (Expr <<= wfExprStructure++)
    | (ExprSeq <<= Expr++)
    | (FieldRef <<= Expr * FieldId)
    | (Load <<= Expr)
    | (Tuple <<= Expr++)
    | (ArrayLit <<= Expr++)
    | (Lambda <<= Params * Type * Body)
    | (Block <<= Body)
    | (MatchExpr <<= Expr * ExprSeq)
    | (FuncName <<= NameElement++[1])
    | (TripleColon <<= NameElement++[1])
    | (Dot <<= wfFuncId * TypeArgs)
    | (If <<= Expr * Block)
    | (Else <<= Expr * Block)
    | (While <<= Expr * Block)
    | (When <<= Expr * Type * Expr)
    | (Equals <<= (Lhs >>= Expr) * (Rhs >>= Expr))
    | (Let <<= Ident * Type)[Ident]
    | (Var <<= Ident * Type)[Ident]
    | (SplatLet <<= Ident)[Ident]
    | (New <<= NewArgs)
    | (Stack <<= Type * NewArgs)
    | (NewArgs <<= NewArg++)
    | (NewArg <<= Ident * Expr)
    | (Break <<= Expr)
    | (Continue <<= Expr)
    | (Return <<= Expr)
    | (Raise <<= Expr)
    | (SetRaise <<= Expr)
    ;
  // clang-format on

  inline const auto wfExprIdent = wfExprStructure | LocalId | MethodName;
  inline const auto wfBodyIdent = wfBody - Use;

  // clang-format off
  inline const auto wfPassIdent =
      wfPassStructure
    | (ClassBody <<= (ClassDef | TypeAlias | Lib | FieldDef | Function)++)
    | (Body <<= wfBodyIdent++)
    | (TypeName <<= NameElement++[1])
    | (FuncName <<= NameElement++[1])
    | (MethodName <<= wfFuncId * TypeArgs)
    | (Expr <<= wfExprIdent++)
    ;
  // clang-format on

  inline const auto wfExprSugar =
    (wfExprIdent | Load | Call | GetRaise | SetRaise | Stack | Typetest | Unop |
     TryCallDyn) -
    Lambda - MatchExpr;
  inline const auto wfBodySugar = wfBodyIdent | TypedMove;

  // clang-format off
  inline const auto wfPassSugar =
      wfPassIdent
    | (Body <<= wfBodySugar++)
    | (When <<= Args * Type * Expr)
    | (ParamDef <<= Ident * Type)[Ident]
    | (Call <<= FuncName * Args)
    | (Args <<= Expr++)
    | (Expr <<= wfExprSugar++)
    | (Typetest <<= Expr * Type)
    | (Unop <<= (MethodId >>= wfUnop) * Args)
    | (Raise <<= Expr * Type)
    | (Ref <<= ~Expr)
    | (TryCallDyn <<= Expr * wfFuncId * TypeArgs * Args)
    | (TypedMove <<= Ident * wfSrc * Type)[Ident]
    ;
  // clang-format on

  inline const auto wfTypeNoFunc = wfType - FuncType - NoArgType;

  // clang-format off
  inline const auto wfPassFuncType =
      wfPassSugar
    | (Type <<= wfTypeNoFunc)
    | (Union <<= wfTypeNoFunc++[2])
    | (Isect <<= wfTypeNoFunc++[2])
    | (TupleType <<= wfTypeNoFunc++[2])
    ;
  // clang-format on

  inline const auto wfExprDot =
    (wfExprSugar | CallDyn | TryCallDyn | Convert | Binop | Nulop | FFI |
     NewArray | ArrayRef | MakeCallback | CodePtrCallback | FreeCallback |
     Freeze | Pin | Unpin | Merge | FFIStruct | FFILoad | FFIStore | ArrayCopy |
     ArrayFill | ArrayCompare) -
    Dot - TripleColon;

  // clang-format off
  inline const auto wfPassDot =
      wfPassFuncType
    | (CallDyn <<= Expr * wfFuncId * TypeArgs * Args)
    | (TryCallDyn <<= Expr * wfFuncId * TypeArgs * Args)
    | (Expr <<= wfExprDot++)
    | (Convert <<= (Type >>= wfPrimitiveType) * Args)
    | (Binop <<= (MethodId >>= wfBinop) * Args)
    | (Nulop <<= (MethodId >>= wfNulop) * Args)
    | (MakeCallback <<= Args)
    | (CodePtrCallback <<= Args)
    | (FreeCallback <<= Args)
    | (Freeze <<= Args)
    | (Pin <<= Args)
    | (Unpin <<= Args)
    | (Merge <<= Args)
    | (FFIStruct <<= Type * Args)
    | (FFILoad <<= Type * Args)
    | (FFIStore <<= Type * Args)
    | (ArrayCopy <<= Args)
    | (ArrayFill <<= Args)
    | (ArrayCompare <<= Args)
    | (FFI <<= SymbolId * Args)
    | (NewArray <<= Type * Args)
    | (ArrayRef <<= Args)
    ;
  // clang-format on

  inline const auto wfExprApplication =
    (wfExprDot | Ref | Hash) - FuncName - MethodName;

  // clang-format off
  inline const auto wfPassApplication =
      wfPassDot
    | (Expr <<= wfExprApplication)
    | (Ref <<= Expr)
    | (Hash <<= Expr)
    ;
  // clang-format on

  inline const auto wfBodyANF = Const | ConstStr | Convert | Copy | Move |
    TypedMove | RegisterRef | FieldRef | ArrayRef | ArrayRefConst | New |
    Stack | NewArray | NewArrayConst | Load | Store | Lookup | Call | CallDyn |
    TryCallDyn | Var | When | wfBinop | wfUnop | wfNulop | FFI | Typetest |
    TypeAssertion | GetRaise | SetRaise | SplatOp | ArrayRefFromEnd |
    MakeCallback | CodePtrCallback | FreeCallback | Freeze | Pin | Unpin |
    Merge | FFIStruct | FFILoad | FFIStore | ArrayCopy | ArrayFill |
    ArrayCompare;

  // clang-format off
  inline const auto wfPassANF =
      wfPassApplication
    | (Function <<=
        wfFuncDefLhs * wfFuncId * TypeParams * Params * Type * Where * Labels)
        [Ident]
    | (Labels <<= Label++)
    | (Label <<= LabelId * Body * (Return >>= wfTerminator))
    | (Body <<= wfBodyANF++)
    | (Const <<= wfDst * (Rhs >>= wfLiteral))
    | (ConstStr <<= wfDst * String)
    | (Convert <<= wfDst * (Type >>= wfPrimitiveType) * wfSrc)
    | (Copy <<= wfDst * wfSrc)
    | (Move <<= wfDst * wfSrc)
    | (TypedMove <<= wfDst * wfSrc * Type)
    | (RegisterRef <<= wfDst * wfSrc)
    | (FieldRef <<= wfDst * Arg * FieldId)
    | (ArrayRef <<= wfDst * Arg * wfSrc)
    | (ArrayRefConst <<= wfDst * Arg * wfLit)
    | (ArrayRefFromEnd <<= wfDst * Arg * wfLit)
    | (SplatOp <<=
        wfDst * Arg * (Lhs >>= wfIntLiteral) * (Rhs >>= wfIntLiteral))
    | (New <<= wfDst * Type * NewArgs)
    | (Stack <<= wfDst * Type * NewArgs)
    | (NewArg <<= Ident * wfSrc)
    | (NewArray <<= wfDst * Type * wfSrc)
    | (NewArrayConst <<= wfDst * Type * wfLit)
    | (Load <<= wfDst * wfSrc)
    | (Store <<= wfDst * wfSrc * Arg)
    | (Lookup <<= wfDst * wfSrc * wfFuncLhs * wfFuncId * TypeArgs * Int)
    | (Call <<= wfDst * wfFuncLhs * FuncName * Args)
    | (CallDyn <<= wfDst * wfSrc * Args)
    | (TryCallDyn <<= wfDst * wfSrc * Args)
    | (Args <<= Arg++)
    | (Arg <<= (Type >>= (ArgMove | ArgCopy)) * wfSrc)
    | (Return <<= LocalId)
    | (Raise <<= LocalId * Type)
    | (Cond <<= LocalId * (Lhs >>= LabelId) * (Rhs >>= LabelId))
    | (GetRaise <<= wfDst)
    | (SetRaise <<= wfDst * wfSrc)
    | (Typetest <<= wfDst * wfSrc * Type)
    | (Jump <<= LabelId)
    | (When <<= wfDst * wfSrc * Args * Type)
    | (Var <<= Ident)[Ident]
    | (TypeAssertion <<= LocalId * Type)
    | (Add <<= wfDst * wfLhs * wfRhs)
    | (Sub <<= wfDst * wfLhs * wfRhs)
    | (Mul <<= wfDst * wfLhs * wfRhs)
    | (Div <<= wfDst * wfLhs * wfRhs)
    | (Mod <<= wfDst * wfLhs * wfRhs)
    | (Pow <<= wfDst * wfLhs * wfRhs)
    | (And <<= wfDst * wfLhs * wfRhs)
    | (Or <<= wfDst * wfLhs * wfRhs)
    | (Xor <<= wfDst * wfLhs * wfRhs)
    | (Shl <<= wfDst * wfLhs * wfRhs)
    | (Shr <<= wfDst * wfLhs * wfRhs)
    | (Eq <<= wfDst * wfLhs * wfRhs)
    | (Ne <<= wfDst * wfLhs * wfRhs)
    | (Lt <<= wfDst * wfLhs * wfRhs)
    | (Le <<= wfDst * wfLhs * wfRhs)
    | (Gt <<= wfDst * wfLhs * wfRhs)
    | (Ge <<= wfDst * wfLhs * wfRhs)
    | (Min <<= wfDst * wfLhs * wfRhs)
    | (Max <<= wfDst * wfLhs * wfRhs)
    | (LogBase <<= wfDst * wfLhs * wfRhs)
    | (Atan2 <<= wfDst * wfLhs * wfRhs)
    | (Neg <<= wfDst * wfSrc)
    | (Not <<= wfDst * wfSrc)
    | (Abs <<= wfDst * wfSrc)
    | (Ceil <<= wfDst * wfSrc)
    | (Floor <<= wfDst * wfSrc)
    | (Exp <<= wfDst * wfSrc)
    | (Log <<= wfDst * wfSrc)
    | (Sqrt <<= wfDst * wfSrc)
    | (Cbrt <<= wfDst * wfSrc)
    | (IsInf <<= wfDst * wfSrc)
    | (IsNaN <<= wfDst * wfSrc)
    | (Sin <<= wfDst * wfSrc)
    | (Cos <<= wfDst * wfSrc)
    | (Tan <<= wfDst * wfSrc)
    | (Asin <<= wfDst * wfSrc)
    | (Acos <<= wfDst * wfSrc)
    | (Atan <<= wfDst * wfSrc)
    | (Sinh <<= wfDst * wfSrc)
    | (Cosh <<= wfDst * wfSrc)
    | (Tanh <<= wfDst * wfSrc)
    | (Asinh <<= wfDst * wfSrc)
    | (Acosh <<= wfDst * wfSrc)
    | (Atanh <<= wfDst * wfSrc)
    | (Const_E <<= wfDst)
    | (Const_Pi <<= wfDst)
    | (Const_Inf <<= wfDst)
    | (Const_NaN <<= wfDst)
    | (AddExternal <<= wfDst)
    | (RemoveExternal <<= wfDst)
    | (Bits <<= wfDst * wfSrc)
    | (Len <<= wfDst * wfSrc)
    | (MakePtr <<= wfDst * wfSrc)
    | (Read <<= wfDst * wfSrc)
    | (FFI <<= wfDst * SymbolId * Args)
    | (ArrayCopy <<= wfDst * Args)
    | (ArrayFill <<= wfDst * Args)
    | (ArrayCompare <<= wfDst * Args)
    | (MakeCallback <<= wfDst * wfSrc)
    | (CodePtrCallback <<= wfDst * wfSrc)
    | (FreeCallback <<= wfDst * wfSrc)
    | (Freeze <<= wfDst * wfSrc)
    | (Pin <<= wfDst * wfSrc)
    | (Unpin <<= wfDst * wfSrc)
    | (Merge <<= wfDst * wfLhs * wfRhs)
    | (FFIStruct <<= wfDst * (Type >>= wfType))
    | (FFILoad <<= wfDst * wfLhs * wfRhs * wfKind * (Type >>= wfType))
    | (FFIStore <<=
        wfDst * wfLhs * wfRhs * wfKind * wfValueSrc * (Type >>= wfType))
    ;
  // clang-format on

  inline const auto wfTypeInfer = wfTypeNoFunc;
  inline const auto wfBodyInfer = wfBodyANF - TypeAssertion;

  // clang-format off
  inline const auto wfPassInfer =
      wfPassANF
    | (Body <<= wfBodyInfer++)
    | (Const <<= wfDst * (Type >>= wfPrimitiveType) * (Rhs >>= wfLiteral))
    | (Type <<= wfTypeInfer)
    | (Union <<= wfTypeInfer++[2])
    | (Isect <<= wfTypeInfer++[2])
    | (TupleType <<= wfTypeInfer++[2])
    ;
  // clang-format on

  inline const auto wfBodyFFIInit = wfBodyInfer | AtTeardown;

  // clang-format off
  inline const auto wfPassFFIInit =
      wfPassInfer
    | (Body <<= wfBodyFFIInit++)
    | (AtTeardown <<= LocalId)
    ;
  // clang-format on

  inline const auto l_local = Location("local");

  Node make_type(NodeRange r = {});
  Node make_typeargs(Node typeparams);
  Nodes scope_path(Node node);
  Node find_def(Node top, const Node& name);
  Node find_func_def(Node top, const Node& funcname, size_t arity, Node hand);
  Node fq_typeparam(const Nodes& path, Node tp);
  Node fq_typeargs(const Nodes& path, Node tps);
  Node make_selftype(Node node, bool fq = false);
  Node type_any();
  Node type_nomatch();
  Node ffi_struct_result_type();
  Node make_nomatch(Node localid);

  // Free type parameter from an enclosing scope.
  struct FreeTP
  {
    std::string name;
    Nodes path; // scope_path of the defining scope
  };

  // Collect free type parameters from enclosing ClassDef/Function scopes.
  std::vector<FreeTP> collect_free_typeparams(Node node);

  // Rewrite FQ TypeName references to free type params so they point at
  // a new class's own type params instead of the enclosing scope's.
  void rewrite_typeparam_refs(
    Node subtree,
    const std::vector<FreeTP>& free_tps,
    const Nodes& cls_path,
    Location new_class_id);

  // A captured field for an anonymous class.
  struct AnonClassField
  {
    Location name; // field / create param name
    Node type; // Type node (cloned)
    Node create_arg; // Expr for the create call argument
    bool is_var = false; // true if captured from a Var (by reference)
  };

  // Result of make_anon_class.
  struct AnonClass
  {
    Node class_def; // ClassDef to Lift into ClassBody
    Node create_call; // Call expression to create an instance
  };

  // Build an anonymous class with fields, a create method, and an apply
  // method. The class mirrors the free type params from enclosing scopes.
  // apply_body must already include field-loading preamble if needed.
  AnonClass make_anon_class(
    Location id,
    Node context_node,
    const std::vector<FreeTP>& free_tps,
    std::vector<AnonClassField>& fields,
    Node apply_params,
    Node apply_ret_type,
    Node apply_body,
    bool is_block = false);

  Location ffi_init_id(const Node& lib);

  Parse parser();
  PassDef structure(const Parse& parse);
  PassDef ident();
  PassDef sugar();
  PassDef functype();
  PassDef dot();
  PassDef application();
  PassDef anf();
  PassDef infer();
  PassDef ffiinit();
  PassDef reify();

  // caller FunctionId -> callee FunctionId -> source call site.
  using CallGraph = std::map<Location, std::map<Location, Node>>;
  using OnceFunctions = std::set<Location>;

  int lower_once(
    Node top, const OnceFunctions& once_funcs, const CallGraph& call_graph);
}
