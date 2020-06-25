// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/parser.h"

#include "compiler/ast.h"
#include "compiler/ir/ir.h"
#include "compiler/printing.h"
#include "compiler/typecheck/typecheck.h"

#include <pegmatite.hh>

namespace verona::compiler
{
  using pegmatite::operator""_E;
  using pegmatite::operator""_S;
  using pegmatite::operator""_R;
  using pegmatite::BindAST;
  using pegmatite::BindConstant;
  using pegmatite::ExprPtr;
  using pegmatite::range;
  using pegmatite::Rule;
  using pegmatite::term;
  using pegmatite::trace;

  struct VeronaGrammar
  {
    static ExprPtr sep_by2(const ExprPtr& r, const ExprPtr& sep)
    {
      return (r >> +(sep >> r));
    }
    static ExprPtr sep_by1(const ExprPtr& r, const ExprPtr& sep)
    {
      return (r >> *(sep >> r));
    }
    static ExprPtr sep_by(const ExprPtr& r, const ExprPtr& sep)
    {
      return -(r >> *(sep >> r));
    }
    static ExprPtr sep_by_end(const ExprPtr& r, const ExprPtr& sep)
    {
      return -(r >> *(sep >> r) >> -sep);
    }
    static ExprPtr comma_sep(const ExprPtr& r)
    {
      return sep_by(r, ",");
    }
    static ExprPtr comma_sep1(const ExprPtr& r)
    {
      return sep_by1(r, ",");
    }
    static ExprPtr braces(const ExprPtr& r)
    {
      return "{" >> r >> "}";
    }
    static ExprPtr brackets(const ExprPtr& r)
    {
      return "[" >> r >> "]";
    }
    static ExprPtr parens(const ExprPtr& r)
    {
      return "(" >> r >> ")";
    }

    Rule whitespace = (" \t\r"_S | pegmatite::nl('\n'));
    Rule comment =
      ("/*"_E >>
       (*(!ExprPtr("*/") >> (pegmatite::nl('\n') | pegmatite::any()))) >>
       "*/") |
      "//"_E >> *(!(ExprPtr("\n")) >> pegmatite::any()) >> pegmatite::nl('\n');

    Rule ignored = +(whitespace | comment);

    Rule alpha = range('a', 'z') | range('A', 'Z');
    ExprPtr alphanum = alpha | range('0', '9');

    Rule digit = range('0', '9');
    Rule integer_literal = +digit;

    Rule character = term(
      "\\\""_E | (!ExprPtr('"') >> (pegmatite::nl('\n') | pegmatite::any())));
    Rule string_body = *character;
    Rule string_literal = term('"' >> string_body >> '"');

    Rule keyword = term(
      "while"_E | "if" | "class" | "interface" | "primitive" | "var" | "unit" |
      "match" | "String" | "iso" | "mut" | "imm" | "mut-view" | "in" |
      "static_assert" | "not" | "subtype" | "when" | "from" | "where" | "else" |
      "builtin");

    Rule self = term("self");
    Rule self_def = self;

    /**
     * The set of characters that are allowed in identifiers, after the first
     * character.
     */
    ExprPtr ident_char = (alphanum | "'_"_S);

    /**
     * The expression that defines an identifier.  Rules that match identifiers
     * also specifically exclude valid characters in this.
     */
    ExprPtr base_ident = term((alpha | '_') >> *ident_char);
    /**
     * A reference to an identifier.  Valid identifier strings that are not
     * also keywords.
     */
    Rule ref_ident = !(term(keyword >> (!+ident_char))) >> base_ident;
    /**
     * A definition of a new identifier.  Valid identifier strings that are not
     * also keywords or `self`.
     */
    Rule def_ident = !term((self | keyword) >> (!+ident_char)) >> base_ident;
    Rule local_def = def_ident;

    Rule symbol_expr = ref_ident;
    Rule define_local = trace("var", "var" >> local_def >> -("=" >> expr2));
    Rule assign_local = ref_ident >> "=" >> expr2;

    Rule field_expr = expr4 >> "." >> ref_ident;
    Rule assign_field = expr5 >> "." >> ref_ident >> "=" >> expr2;

    Rule seq = sep_by1(expr1 | empty_expr, ";");

    Rule block = braces(empty_expr) | braces(seq);

    Rule empty_expr = trace("Empty", ""_E);
    Rule while_loop = trace("While", "while" >> expr3 >> block);
    Rule else_expr = "else" >> block;
    Rule if_expr = trace("If", "if" >> expr3 >> block >> -else_expr);
    Rule block_expr = block;

    Rule match_arm = "var" >> local_def >> ":" >> type >> "=>" >>
      (expr1 >> "," | block);
    Rule match_expr = "match" >> expr3 >> braces(*match_arm);

    Rule new_parent = "in" >> ref_ident;
    Rule new_expr = "new" >> ref_ident >> -new_parent;
    Rule mut_view_expr = "mut-view" >> expr5;
    Rule when_clause = "when" >> parens(comma_sep(when_argument)) >> block_expr;

    Rule when_argument = when_argument_as | when_argument_shadow;
    Rule when_argument_as = "var" >> local_def >> "=" >> ExprPtr(expr1);
    Rule when_argument_shadow = local_def;

    Rule argument = ExprPtr(expr1);

    Rule call = expr4 >> "." >> ref_ident >> parens(comma_sep(argument));

    Rule integer_literal_expr = integer_literal;
    Rule string_literal_expr = string_literal;

    Rule operator_add = "+"_E;
    Rule operator_sub = "-"_E;
    Rule operator_mul = "*"_E;
    Rule operator_div = "/"_E;
    Rule operator_mod = "%"_E;
    Rule operator_shl = "<<"_E;
    Rule operator_shr = ">>"_E;
    Rule operator_lt = "<"_E;
    Rule operator_le = "<="_E;
    Rule operator_gt = ">"_E;
    Rule operator_ge = ">="_E;
    Rule operator_eq = "=="_E;
    Rule operator_ne = "!="_E;
    Rule operator_and = "&&"_E;
    Rule operator_or = "||"_E;
    Rule binary_operator = operator_add | operator_sub | operator_mul |
      operator_div | operator_mod | operator_shl | operator_shr | operator_le |
      operator_lt | operator_ge | operator_gt | operator_eq | operator_ne |
      operator_and | operator_or;

    Rule binary_operator_expr = expr3 >> binary_operator >> expr3;

    Rule expr5 =
      symbol_expr | integer_literal_expr | string_literal_expr | parens(expr1);
    Rule expr4 = call | field_expr | expr5;
    Rule expr3 = binary_operator_expr | new_expr | mut_view_expr | expr4;
    Rule expr2 = if_expr | match_expr | when_clause | block_expr | expr3;
    Rule expr1 =
      define_local | assign_local | assign_field | while_loop | expr2;

    Rule isolated = "iso"_E;
    Rule mutable_ = "mut"_E;
    Rule immutable = "imm"_E;

    Rule capability_kind = isolated | mutable_ | immutable;
    Rule capability_type = capability_kind;
    Rule string_type = "String"_E;
    Rule symbol_type = ref_ident >> -brackets(comma_sep(type));
    Rule union_type = sep_by2(type1, "|");
    Rule intersection_type = sep_by2(type1, "&");
    Rule viewpoint_type = type1 >> "->" >> (viewpoint_type | type1);

    Rule type1 = parens(type) | symbol_type | capability_type | string_type;
    Rule type = union_type | intersection_type | viewpoint_type | type1;

    Rule type_param_kind_class = "class"_E;
    Rule type_param_kind = type_param_kind_class;
    Rule type_param_def = -type_param_kind >> def_ident >> -(":" >> type);
    Rule generics = -brackets(comma_sep(type_param_def));

    Rule receiver = self_def >> ":" >> capability_type;
    Rule function_param = local_def >> ":" >> type;
    Rule function_params =
      parens(receiver >> -("," >> comma_sep1(function_param))) |
      parens(comma_sep(function_param));

    Rule fn_signature = generics >> function_params >> -(":" >> type) >>
      -ExprPtr(where_clauses);
    Rule fn_body = block;

    Rule method_builtin = "builtin"_E;
    Rule method = -(method_builtin) >> def_ident >> fn_signature >>
      (fn_body | ";");

    Rule field = def_ident >> ":" >> type >> ";";
    Rule member = trace("member", method | field);

    Rule class_kind = "class"_E;
    Rule interface_kind = "interface"_E;
    Rule primitive_kind = "primitive"_E;
    Rule entity_kind = class_kind | interface_kind | primitive_kind;

    Rule entity =
      trace("entity", entity_kind >> def_ident >> generics >> braces(*member));

    Rule assertion_kind_subtype = "subtype"_E;
    Rule assertion_kind_not_subtype = "not"_E >> "subtype"_E;
    Rule assertion_kind = assertion_kind_subtype | assertion_kind_not_subtype;
    Rule assertion = "static_assert" >> generics >>
      parens(type >> assertion_kind >> type1) >> ";";

    Rule where_clause_return = term("return");
    Rule where_clause_parameter = ref_ident;
    Rule where_clause_term = where_clause_return | where_clause_parameter;
    Rule where_clause_kind_in = term("in");
    Rule where_clause_kind_from = term("from");
    Rule where_clause_kind = where_clause_kind_in | where_clause_kind_from;
    Rule where_clause =
      where_clause_term >> where_clause_kind >> where_clause_parameter;

    Rule where_clauses = term("where") >> comma_sep1(where_clause);

    Rule module = "use" >> string_literal;

    Rule file = *module >> *entity >> *assertion;

    static const VeronaGrammar& get()
    {
      static VeronaGrammar g;
      return g;
    }

  private:
    VeronaGrammar() {}
  };

  struct VeronaParser : public pegmatite::ASTParserDelegate
  {
    const VeronaGrammar& g = VeronaGrammar::get();

    BindAST<pegmatite::ASTInteger> integer_literal = g.integer_literal;
    BindAST<StringLiteral> string_body = g.string_body;

    BindAST<Name> self = g.self;
    BindAST<Name> ref_ident = g.ref_ident;
    BindAST<Name> def_ident = g.def_ident;

    BindAST<LocalDef> self_def = g.self_def;
    BindAST<LocalDef> local_def = g.local_def;

    BindAST<SymbolExpr> symbol_expr = g.symbol_expr;
    BindAST<DefineLocalExpr> define_local = g.define_local;
    BindAST<AssignLocalExpr> assign_local = g.assign_local;

    BindAST<FieldExpr> field_expr = g.field_expr;
    BindAST<AssignFieldExpr> assign_field = g.assign_field;

    BindAST<SeqExpr> seq = g.seq;
    BindAST<WhileExpr> while_loop = g.while_loop;
    BindAST<WhenExpr> when_clause = g.when_clause;
    BindAST<IfExpr> if_expr = g.if_expr;
    BindAST<ElseExpr> else_expr = g.else_expr;
    BindAST<BlockExpr> block_expr = g.block_expr;
    BindAST<EmptyExpr> empty_expr = g.empty_expr;

    BindAST<MatchArm> match_arm = g.match_arm;
    BindAST<MatchExpr> match_expr = g.match_expr;
    BindAST<ViewExpr> mut_view_expr = g.mut_view_expr;

    BindAST<NewParent> new_parent = g.new_parent;
    BindAST<NewExpr> new_expr = g.new_expr;

    BindAST<Argument> argument = g.argument;

    BindAST<WhenArgumentAs> when_argument_as = g.when_argument_as;
    BindAST<WhenArgumentShadow> when_argument_shadow = g.when_argument_shadow;

    BindAST<CallExpr> call = g.call;

    BindAST<IntegerLiteralExpr> integer_literal_expr = g.integer_literal_expr;
    BindAST<StringLiteralExpr> string_literal_expr = g.string_literal_expr;

    BindConstant<BinaryOperator, BinaryOperator::Add> operator_add =
      g.operator_add;
    BindConstant<BinaryOperator, BinaryOperator::Sub> operator_sub =
      g.operator_sub;
    BindConstant<BinaryOperator, BinaryOperator::Mul> operator_mul =
      g.operator_mul;
    BindConstant<BinaryOperator, BinaryOperator::Div> operator_div =
      g.operator_div;
    BindConstant<BinaryOperator, BinaryOperator::Mod> operator_mod =
      g.operator_mod;
    BindConstant<BinaryOperator, BinaryOperator::Shl> operator_shl =
      g.operator_shl;
    BindConstant<BinaryOperator, BinaryOperator::Shr> operator_shr =
      g.operator_shr;
    BindConstant<BinaryOperator, BinaryOperator::Lt> operator_lt =
      g.operator_lt;
    BindConstant<BinaryOperator, BinaryOperator::Le> operator_le =
      g.operator_le;
    BindConstant<BinaryOperator, BinaryOperator::Gt> operator_gt =
      g.operator_gt;
    BindConstant<BinaryOperator, BinaryOperator::Ge> operator_ge =
      g.operator_ge;
    BindConstant<BinaryOperator, BinaryOperator::Eq> operator_eq =
      g.operator_eq;
    BindConstant<BinaryOperator, BinaryOperator::Ne> operator_ne =
      g.operator_ne;
    BindConstant<BinaryOperator, BinaryOperator::And> operator_and =
      g.operator_and;
    BindConstant<BinaryOperator, BinaryOperator::Or> operator_or =
      g.operator_or;
    BindAST<BinaryOperatorExpr> binary_operator_expr = g.binary_operator_expr;

    BindAST<CapabilityTypeExpr> capability_type = g.capability_type;
    BindAST<StringTypeExpr> string_type = g.string_type;
    BindAST<IntersectionTypeExpr> intersection_type = g.intersection_type;
    BindAST<SymbolTypeExpr> symbol_type = g.symbol_type;
    BindAST<UnionTypeExpr> union_type = g.union_type;
    BindAST<ViewpointTypeExpr> viewpoint_type = g.viewpoint_type;

    BindConstant<CapabilityKind, CapabilityKind::Isolated> isolated =
      g.isolated;
    BindConstant<CapabilityKind, CapabilityKind::Immutable> immutable =
      g.immutable;
    BindConstant<CapabilityKind, CapabilityKind::Mutable> mutable_ = g.mutable_;

    BindConstant<TypeParameterDef::Kind, TypeParameterDef::Class>
      type_param_kind_class = g.type_param_kind_class;
    BindAST<TypeParameterDef> type_param_def = g.type_param_def;
    BindAST<Generics> generics = g.generics;
    BindAST<Receiver> receiver = g.receiver;
    BindAST<FnParameter> function_param = g.function_param;

    BindConstant<WhereClause::Kind, WhereClause::In> where_clause_kind_in =
      g.where_clause_kind_in;
    BindConstant<WhereClause::Kind, WhereClause::From> where_clause_kind_from =
      g.where_clause_kind_from;
    BindAST<WhereClauseReturn> where_clause_return = g.where_clause_return;
    BindAST<WhereClauseParameter> where_clause_parameter =
      g.where_clause_parameter;
    BindAST<WhereClause> where_clause = g.where_clause;

    BindAST<FnSignature> fn_signature = g.fn_signature;
    BindAST<FnBody> fn_body = g.fn_body;
    BindConstant<Method::Kind, Method::Builtin> method_builtin =
      g.method_builtin;
    BindAST<Method> method = g.method;

    BindConstant<Entity::Kind, Entity::Class> class_kind = g.class_kind;
    BindConstant<Entity::Kind, Entity::Interface> interface_kind =
      g.interface_kind;
    BindConstant<Entity::Kind, Entity::Primitive> primitive_kind =
      g.primitive_kind;

    BindAST<Field> field = g.field;
    BindAST<Entity> entity = g.entity;

    BindConstant<AssertionKind, AssertionKind::Subtype> assertion_kind_subtype =
      g.assertion_kind_subtype;
    BindConstant<AssertionKind, AssertionKind::NotSubtype>
      assertion_kind_not_subtype = g.assertion_kind_not_subtype;
    BindAST<StaticAssertion> assertion = g.assertion;

    BindAST<File> file = g.file;
  };

  std::unique_ptr<verona::compiler::File>
  parse(Context& context, std::string name, std::istream& input)
  {
    // Pegmatite doesn't let us pass the context, so we use the ThreadContext
    // instead.
    ThreadContext thread_context(context);

    std::unique_ptr<verona::compiler::File> file = nullptr;

    auto stream_input = pegmatite::StreamInput::Create(name, input);

    VeronaParser p;
    p.parse(
      stream_input,
      p.g.file,
      p.g.ignored,
      pegmatite::defaultErrorReporter,
      file);

    return file;
  }
}
