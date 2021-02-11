// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "typed-ast/conversion.h"

using Ast = ast::Ast;
using namespace peg::udl;
using namespace std::literals::string_view_literals;

namespace verona::ast
{
  static SourceLocation getSourceLocation(const Ast& node)
  {
    return SourceLocation(node->path, node->line, node->column);
  }

  /// Convert any kind of expression node into its typed representation.
  /// This dispatches to the various convertXXX methods, depending on the tag of
  /// the node.
  static ExprPtr convertExpression(const Ast& node);

  /// Convert every child of `node` as an expression.
  static std::vector<ExprPtr> convertExpressions(const Ast& node);

  static std::vector<std::unique_ptr<MemberDef>>
  convertTypeBody(const Ast& node);

  static Symbol convertSymbol(const Ast& node)
  {
    assert(
      node->tag == "id"_ || node->tag == "function"_ ||
      node->tag == "localref"_ || node->tag == "lookup"_ ||
      node->tag == "local"_);
    assert(node->nodes.empty());
    return node->token;
  }

  static ExprPtr convertSeq(const Ast& node)
  {
    assert(node->tag == "seq"_);
    SourceLocation location = getSourceLocation(node);
    std::vector<ExprPtr> elements = convertExpressions(node);

    return std::make_unique<SequenceExpr>(location, std::move(elements));
  }

  static ExprPtr convertBlock(const Ast& node)
  {
    assert(node->tag == "block"_);
    assert(node->nodes.size() == 1);

    // Blocks just contain one `seq` node.
    // TODO: For now, we forget the fact that there even was a block, but we'll
    // need to correct this in order to get the correct scoping rules.
    return convertSeq(node->nodes.at(0));
  }

  static ExprPtr convertAssign(const Ast& node)
  {
    assert(node->tag == "assign"_);
    assert(node->nodes.size() == 2);

    SourceLocation location = getSourceLocation(node);
    ExprPtr lhs = convertExpression(node->nodes.at(0));
    ExprPtr rhs = convertExpression(node->nodes.at(1));

    return std::make_unique<AssignmentExpr>(
      location, std::move(lhs), std::move(rhs));
  }

  static ExprPtr convertLet(const Ast& node)
  {
    assert(node->tag == "let"_);
    assert(node->nodes.size() == 2);

    SourceLocation location = getSourceLocation(node);
    Symbol name = convertSymbol(node->nodes.at(0));
    // TODO: nodes.at(1) contains the type of the variable.

    return std::make_unique<LocalDeclExpr>(location, name);
  }

  static ExprPtr convertCall(const Ast& node)
  {
    assert(node->tag == "call"_);
    assert(node->nodes.size() == 4);

    SourceLocation location = getSourceLocation(node);

    Symbol name = convertSymbol(node->nodes.at(0));
    // TODO: nodes.at(1) contains the type arguments
    ExprPtr receiver = convertExpression(node->nodes.at(2));
    std::vector<ExprPtr> arguments = convertExpressions(node->nodes.at(3));

    return std::make_unique<MethodCallExpr>(
      location, name, std::move(receiver), std::move(arguments));
  }

  static ExprPtr convertInvoke(const Ast& node)
  {
    assert(node->tag == "invoke"_);
    assert(node->nodes.size() == 3);
    SourceLocation location = getSourceLocation(node);

    ExprPtr receiver = convertExpression(node->nodes.at(0));
    // TODO: nodes.at(1) contains the type arguments
    std::vector<ExprPtr> arguments = convertExpressions(node->nodes.at(2));

    return std::make_unique<InvokeExpr>(
      location, std::move(receiver), std::move(arguments));
  }

  static ExprPtr convertStaticCall(const Ast& node)
  {
    assert(node->tag == "static-call"_);
    assert(node->nodes.size() == 4);

    SourceLocation location = getSourceLocation(node);

    // TODO: nodes.at(0) contains the receiver type
    Symbol name = convertSymbol(node->nodes.at(1));
    // TODO: nodes.at(2) contains the type arguments
    std::vector<ExprPtr> arguments = convertExpressions(node->nodes.at(3));

    return std::make_unique<StaticCallExpr>(
      location, name, std::move(arguments));
  }

  static ExprPtr convertLocalref(const Ast& node)
  {
    assert(node->tag == "localref"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    Symbol name = convertSymbol(node);
    return std::make_unique<LocalRefExpr>(location, name);
  }

  static ExprPtr convertMember(const Ast& node)
  {
    assert(node->tag == "member"_);
    SourceLocation location = getSourceLocation(node);
    ExprPtr origin = convertExpression(node->nodes.at(0));
    Symbol name = convertSymbol(node->nodes.at(1));
    return std::make_unique<MemberRefExpr>(location, std::move(origin), name);
  }

  /// Convert a `cond` node. The node is just a wrapper around an expression.
  static ExprPtr convertCondition(const Ast& node)
  {
    assert(node->tag == "cond"_);
    assert(node->nodes.size() == 1);
    return convertExpression(node->nodes.at(0));
  }

  static ExprPtr convertWhile(const Ast& node)
  {
    assert(node->tag == "while"_);
    assert(node->nodes.size() == 2);

    SourceLocation location = getSourceLocation(node);
    ExprPtr condition = convertCondition(node->nodes.at(0));
    ExprPtr body = convertBlock(node->nodes.at(1));
    return std::make_unique<WhileExpr>(
      location, std::move(condition), std::move(body));
  }

  static ExprPtr convertContinue(const Ast& node)
  {
    assert(node->tag == "continue"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    return std::make_unique<ContinueExpr>(location);
  }

  static ExprPtr convertBreak(const Ast& node)
  {
    assert(node->tag == "break"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    return std::make_unique<BreakExpr>(location);
  }

  static ExprPtr convertReturn(const Ast& node)
  {
    assert(node->tag == "return"_);
    assert(node->nodes.size() < 2);

    SourceLocation location = getSourceLocation(node);
    ExprPtr value;
    if (node->nodes.size() > 0)
      value = convertExpression(node->nodes.at(0));

    return std::make_unique<ReturnExpr>(location, std::move(value));
  }

  static ExprPtr convertYield(const Ast& node)
  {
    assert(node->tag == "yield"_);
    assert(node->nodes.size() < 2);

    SourceLocation location = getSourceLocation(node);
    ExprPtr value;
    if (node->nodes.size() > 0)
      value = convertExpression(node->nodes.at(0));

    return std::make_unique<YieldExpr>(location, std::move(value));
  }

  static ExprPtr convertInteger(const Ast& node)
  {
    assert(
      node->tag == "int"_ || node->tag == "hex"_ || node->tag == "binary"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    int64_t value = std::stoll(node->token);
    return std::make_unique<IntegerLiteral>(location, value);
  }

  static ExprPtr convertFloat(const Ast& node)
  {
    assert(node->tag == "float"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    double value = std::stod(node->token);
    return std::make_unique<FloatLiteral>(location, value);
  }

  static ExprPtr convertBool(const Ast& node)
  {
    assert(node->tag == "bool"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    bool value = node->token == "true"sv;
    return std::make_unique<BooleanLiteral>(location, value);
  }

  static ExprPtr convertString(const Ast& node)
  {
    assert(node->tag == "string"_);
    assert(node->nodes.empty());

    SourceLocation location = getSourceLocation(node);
    return std::make_unique<StringLiteral>(location, node->token);
  }

  static ExprPtr convertIf(const Ast& node)
  {
    assert(node->tag == "if"_);
    assert(node->nodes.size() == 3);

    SourceLocation location = getSourceLocation(node);
    ExprPtr condition = convertExpression(node->nodes.at(0)->nodes.at(0));
    ExprPtr thenBlock = convertBlock(node->nodes.at(1));
    ExprPtr elseBlock;

    // There is always an else node, though it could be empty.
    if (const Ast& elseNode = node->nodes.at(2); !elseNode->nodes.empty())
    {
      assert(elseNode->nodes.size() == 1);
      assert(node->tag == "block"_ || node->tag == "if"_);
      elseBlock = convertExpression(elseNode->nodes.at(0));
    }

    return std::make_unique<IfExpr>(
      location,
      std::move(condition),
      std::move(thenBlock),
      std::move(elseBlock));
  }

  static ExprPtr convertInterpString(const Ast& node)
  {
    assert(node->tag == "interp_string"_);

    SourceLocation location = getSourceLocation(node);
    std::vector<ExprPtr> elements = convertExpressions(node);
    return std::make_unique<InterpolateExpr>(location, std::move(elements));
  }

  static std::unique_ptr<TupleExpr> convertTuple(const Ast& node)
  {
    assert(node->tag == "tuple"_);
    SourceLocation location = getSourceLocation(node);
    std::vector<ExprPtr> elements = convertExpressions(node);
    return std::make_unique<TupleExpr>(location, std::move(elements));
  }

  static ExprPtr convertWhen(const Ast& node)
  {
    assert(node->tag == "when"_);
    assert(node->nodes.size() == 2);

    SourceLocation location = getSourceLocation(node);
    std::vector<ExprPtr> arguments = convertExpressions(node->nodes.at(0));
    ExprPtr body = convertBlock(node->nodes.at(1));
    return std::make_unique<WhenExpr>(
      location, std::move(arguments), std::move(body));
  }

  static ExprPtr convertNew(const Ast& node)
  {
    assert(node->tag == "new"_);
    assert(node->nodes.size() < 4);

    SourceLocation location = getSourceLocation(node);
    std::vector<std::unique_ptr<MemberDef>> elements;
    std::optional<Symbol> region;

    // The New node is a bit awkward; it has optional elements in its middle.
    // This means the index of the region, for example, depends on whether a
    // type and a body were provided.
    size_t index = 0;
    auto consume = [&](int tag) -> Ast {
      if (index < node->nodes.size() && node->nodes.at(index)->tag == tag)
        return node->nodes.at(index++);
      else
        return nullptr;
    };

    // TODO: use the type
    consume("type"_);

    if (Ast bodyNode = consume("typebody"_))
      elements = convertTypeBody(bodyNode);

    if (Ast regionNode = consume("inregion"_))
    {
      assert(regionNode->nodes.size() == 1);
      region = convertSymbol(regionNode->nodes.at(0));
    }

    assert(index == node->nodes.size());

    return std::make_unique<NewExpr>(location, std::move(elements), region);
  }

  static ExprPtr convertLambda(const Ast& node)
  {
    assert(node->tag == "lambda"_);
    assert(node->nodes.size() == 2);

    SourceLocation location = getSourceLocation(node);
    // TODO: nodes.at(0) contains the lambda's signature
    ExprPtr body = convertBlock(node->nodes.at(1));

    return std::make_unique<LambdaExpr>(location, std::move(body));
  }

  static ExprPtr convertExpression(const Ast& node)
  {
    switch (node->tag)
    {
      case "assign"_:
        return convertAssign(node);
      case "localref"_:
        return convertLocalref(node);
      case "member"_:
        return convertMember(node);
      case "call"_:
        return convertCall(node);
      case "invoke"_:
        return convertInvoke(node);
      case "static-call"_:
        return convertStaticCall(node);
      case "if"_:
        return convertIf(node);
      case "while"_:
        return convertWhile(node);
      case "continue"_:
        return convertContinue(node);
      case "return"_:
        return convertReturn(node);
      case "yield"_:
        return convertYield(node);
      case "break"_:
        return convertBreak(node);
      case "seq"_:
        return convertSeq(node);
      case "let"_:
        return convertLet(node);
      case "block"_:
        return convertBlock(node);
      case "int"_:
      case "hex"_:
      case "binary"_:
        return convertInteger(node);
      case "float"_:
        return convertFloat(node);
      case "bool"_:
        return convertBool(node);
      case "string"_:
        return convertString(node);
      case "interp_string"_:
        return convertInterpString(node);
      case "tuple"_:
        return convertTuple(node);
      case "when"_:
        return convertWhen(node);
      case "new"_:
        return convertNew(node);
      case "lambda"_:
        return convertLambda(node);

      default:
        std::cerr << "Unhandled node " << node->name << std::endl;
        abort();
    }
  }

  static std::vector<ExprPtr> convertExpressions(const Ast& node)
  {
    // Only a handful of nodes are lists of expressions.
    assert(
      node->tag == "seq"_ || node->tag == "tuple"_ || node->tag == "args"_ ||
      node->tag == "whenargs"_ || node->tag == "interp_string"_);

    std::vector<ExprPtr> result;
    for (const Ast& child : node->nodes)
    {
      result.push_back(convertExpression(child));
    }
    return result;
  }

  static std::unique_ptr<MethodDef> convertFunction(const ::ast::Ast& node)
  {
    assert(node->tag == "function"_);
    assert(node->nodes.size() == 3 || node->nodes.size() == 4);

    SourceLocation location = getSourceLocation(node);
    // TODO: nodes.at(0) contains method's qualifiers
    Symbol name = convertSymbol(node->nodes.at(1)->nodes.at(0));
    // TODO: nodes.at(2) contains the signature of the method
    ExprPtr body;
    if (node->nodes.size() == 4)
      body = convertExpression(node->nodes.at(3));

    return std::make_unique<MethodDef>(location, name, std::move(body));
  }

  static std::unique_ptr<EntityDef> convertClassDef(const ::ast::Ast& node)
  {
    assert(node->tag == "classdef"_);
    assert(node->nodes.size() == 5);

    SourceLocation location = getSourceLocation(node);
    Symbol name = convertSymbol(node->nodes.at(0));
    // TODO: nodes.at(1) contains the class' type parameters
    // TODO: nodes.at(2) contains an `oftype` with the class' parents.
    // TODO: nodes.at(3) contains the list of constraints
    std::vector<std::unique_ptr<MemberDef>> elements =
      convertTypeBody(node->nodes.at(4));

    return std::make_unique<EntityDef>(location, name, std::move(elements));
  }

  static std::unique_ptr<FieldDef> convertField(const ::ast::Ast& node)
  {
    assert(node->tag == "field"_);
    assert(node->nodes.size() == 4);

    SourceLocation location = getSourceLocation(node);
    // TODO: nodes.at(0) contains the qualifers
    Symbol name = convertSymbol(node->nodes.at(1));
    // TODO: nodes.at(2) contains the field's type
    ExprPtr initExpr;

    // There is always an initexpr node, although it could be empty
    if (const Ast& initNode = node->nodes.at(3); !initNode->nodes.empty())
    {
      assert(initNode->nodes.size() == 1);
      initExpr = convertExpression(initNode->nodes.at(0));
    }

    return std::make_unique<FieldDef>(location, name, std::move(initExpr));
  }

  static std::unique_ptr<TypeAliasDef> convertTypeDef(const ::ast::Ast& node)
  {
    assert(node->tag == "typedef"_);
    assert(node->nodes.size() == 5);

    SourceLocation location = getSourceLocation(node);
    Symbol name = convertSymbol(node->nodes.at(0));
    // TODO: nodes.at(1) contains the type parameters.
    // TODO: nodes.at(2) contains an `oftype`.
    // TODO: nodes.at(3) contains the constraints.
    // TODO: nodes.at(4) contains the alias' value.
    return std::make_unique<TypeAliasDef>(location, name);
  }

  /// Convert any kind of entity definition. This dispatches to the various
  /// convertXXX methods, depending on the tag of the node.
  static std::unique_ptr<MemberDef> convertMemberDef(const ::ast::Ast& node)
  {
    switch (node->tag)
    {
      case "classdef"_:
        return convertClassDef(node);
      case "function"_:
        return convertFunction(node);
      case "typedef"_:
        return convertTypeDef(node);
      case "field"_:
        return convertField(node);
      default:
        std::cerr << "Unhandled node " << node->name << std::endl;
        abort();
    }
  }

  static std::vector<std::unique_ptr<MemberDef>>
  convertTypeBody(const Ast& node)
  {
    assert(node->tag == "typebody"_);

    std::vector<std::unique_ptr<MemberDef>> result;
    for (const Ast& child : node->nodes)
    {
      result.push_back(convertMemberDef(child));
    }

    return result;
  }

  std::unique_ptr<EntityDef> convertModule(const ::ast::Ast& node)
  {
    // modules are called "classdef" in the untyped AST.
    return convertClassDef(node);
  }
}
