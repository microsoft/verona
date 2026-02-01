#pragma once

#include <optional>

#include <trieste/trieste.h>

namespace infix {
using namespace trieste;

inline const auto Function =
    TokenDef("function", flag::lookup | flag::lookdown | flag::symtab);
inline const auto Struct =
    TokenDef("struct", flag::symtab | flag::lookup | flag::lookdown);
inline const auto Type = TokenDef("type", flag::lookup | flag::lookdown | flag::symtab);
inline const auto TypeAlias = TokenDef("type_alias");
inline const auto Where = TokenDef("where");
inline const auto Module =
    TokenDef("module", flag::symtab | flag::lookup | flag::lookdown);
inline const auto Eq = TokenDef("eq");
inline const auto Use = TokenDef("use", flag::lookdown);
inline const auto Let = TokenDef("let");
inline const auto Or = TokenDef("or");
inline const auto And = TokenDef("and");

inline const auto Indent = TokenDef("indent");

inline const auto Colon = TokenDef("Colon");
inline const auto DoubleColon = TokenDef("DoubleColon");
inline const auto SemiColon = TokenDef("SemiColon");
inline const auto Comma = TokenDef("Comma");
inline const auto Dot = TokenDef("dot");
inline const auto Paren = TokenDef("paren");
inline const auto Square = TokenDef("square");
inline const auto Hat = TokenDef("hat");
inline const auto Underscore = TokenDef("underscore");
inline const auto Arrow = TokenDef("arrow");
inline const auto LeftArrow = TokenDef("left_arrow");
inline const auto Backtick = TokenDef("Backtick");

inline const auto TypeLookup = TokenDef("type_lookup");
inline const auto TypeReference = TokenDef("type_reference");
inline const auto TypeArgs = TokenDef("type_args");
inline const auto TypeArg = TokenDef("type_arg");

inline const auto Mode = TokenDef("mode");
inline const auto CBN = TokenDef("cbn");
inline const auto CBV = TokenDef("cbv");
inline const auto Partial = TokenDef("partial");
inline const auto Call = TokenDef("call");
inline const auto Create = TokenDef("create");
inline const auto Args = TokenDef("args");

inline const auto Name = TokenDef("name", flag::print);
inline const auto String = TokenDef("string", flag::print);

inline const auto Lhs = TokenDef("lhs");
inline const auto Rhs = TokenDef("rhs");
inline const auto Path = TokenDef("usepath");
inline const auto Parent = TokenDef("../");
inline const auto Current = TokenDef("./");

inline const auto TypeParam = TokenDef("type_param", flag::lookup);
inline const auto TypeParams = TokenDef("type_params");
inline const auto Fields = TokenDef("fields");
inline const auto Field = TokenDef("field", flag::lookup);
inline const auto Param = TokenDef("param", flag::lookup);
inline const auto Body = TokenDef("body");
inline const auto Expr = TokenDef("expr");
inline const auto Lookup = TokenDef("lookup");
inline const auto ExprStack = TokenDef("expr_stack");
inline const auto Assign = TokenDef("assign");
inline const auto OperatorDef =
    TokenDef("operator_def", flag::lookup | flag::lookdown);

using namespace wf::ops;
inline const auto wf_parse_tokens =
    Group | File | Top | Name | Struct | Paren | Square | Function | Type |
    Where | Eq | Let | Or | And | Indent | Colon | Comma | Dot | Paren |
    Square | Underscore | Arrow | Backtick | String | Hat | Use | DoubleColon |
    Path;

inline const auto wf_parser =
    (Top <<= File) | (File <<= Group++) | (Paren <<= wf_parse_tokens++) |
    (Square <<= wf_parse_tokens++) | (Group <<= wf_parse_tokens++) |
    (Indent <<= wf_parse_tokens++);

inline const auto wf_parse_tokens_no_op =
    Group | File | Top | Name | Struct | Paren | Square | Function | Type |
    Where | Eq | Let | Or | And | Indent | Colon | Comma | Dot | Paren |
    Square | Underscore | Arrow | Backtick | String | Hat;

inline const auto wf_operator_defn =
    (Top <<= File) | (File <<= (Group | OperatorDef)++) |
    (Paren <<= wf_parse_tokens_no_op++) | (Square <<= wf_parse_tokens_no_op++) |
    (Group <<= wf_parse_tokens_no_op++) | (Indent <<= wf_parse_tokens++) |
    (OperatorDef <<= Name * Lhs * Rhs)[Name] | (Lhs <<= Underscore++) |
    (Rhs <<= Underscore++);

inline const auto wf_decls = Struct | TypeAlias | Function | Module | Use;

inline const auto wf_term = Paren | Name | Group | Indent | Dot | Arrow |
                            LeftArrow | Colon | Lookup | Eq | SemiColon | Square;

inline const auto wf_function_parse =
    (Top <<= File) | (File <<= wf_decls++) |
    (Struct <<= Name * TypeParams * Fields)[Name] |
    (TypeAlias <<= Name * TypeParams * Type)[Name] |
    (Paren <<= (wf_decls | wf_term)++) | (Indent <<= (wf_decls | wf_term)++) |
    (Group <<= (wf_decls | wf_term)++) |
    (Function <<= TypeParams * Lhs * Name * Rhs * Type * Where * Body)[Name] |
    (Body <<= (ExprStack | wf_term)++) | (Type <<= (Name | Square | Arrow | DoubleColon | TypeLookup)++) |
    (Lhs <<= Param++) | (Rhs <<= Param++) | (Lookup <<= Name * Args) |
    (Args <<= wf_term) | (Param <<= Name * Mode * Type)[Name] |
    (Mode <<= CBN | CBV) | (TypeParams <<= TypeParam++) |
    (TypeParam <<= Name)[Name] | (Square <<= wf_term++) | (Fields <<= Field++) |
    (Field <<= Name * Type)[Name] | (Where <<= wf_term++) |
    (Module <<= Name * TypeParams * Body)[Name] | (Use <<= Type)[Include] | 
    (TypeLookup <<= (Parent | TypeReference)++) | (TypeReference <<= Name * TypeArgs) |
    (TypeArgs <<= Type++);

Parse parser();
std::vector<Pass> passes();

// Utility functions
Nodes lookup_all(Node n);
std::optional<size_t> lookup_levels_up(Node n);

// PassDef factory functions
PassDef get_operator_defn_pass();
PassDef get_function_parse_pass();
PassDef get_parse_types_pass();
PassDef get_resolve_types_pass();
PassDef get_infix_parse_pass();

} // namespace infix