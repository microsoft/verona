// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "typed-ast/node.h"
#include "typed-ast/print.h"

#include <memory>
#include <optional>
#include <vector>

namespace verona::ast
{
  class MemberDef;

  /// This is the base class for expressions, from which all concrete nodes
  /// derive.
  class Expr : public Node
  {
  public:
    enum class Kind
    {
      LocalDecl,
      LocalRef,
      MemberRef,
      Assignment,
      Sequence,
      If,
      While,
      Continue,
      Return,
      Yield,
      Break,
      Invoke,
      MethodCall,
      StaticCall,
      IntegerLiteral,
      FloatLiteral,
      BooleanLiteral,
      StringLiteral,
      Interpolate,
      Tuple,
      When,
      New,
      Lambda,
    };

    Kind getKind() const
    {
      return kind;
    }

  protected:
    explicit Expr(Kind kind, SourceLocation location)
    : Node(location), kind(kind)
    {}
    Kind kind;
  };

  using ExprPtr = std::unique_ptr<Expr>;

  /// This represents a local variable declaration, e.g. `let x`.
  /// TODO: unclear what this "evaluates" to.
  class LocalDeclExpr final : public Expr
  {
    // TODO: add the declaration's optional type.
    Symbol name;

  public:
    explicit LocalDeclExpr(SourceLocation location, Symbol name)
    : Expr(Kind::LocalDecl, location), name(name)
    {}

    Symbol getName() const
    {
      return name;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::LocalDecl;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("decl").field(name).finish();
    }
  };

  /// This represents a local variable, e.g. `x`.
  class LocalRefExpr final : public Expr
  {
    Symbol name;

  public:
    explicit LocalRefExpr(SourceLocation location, Symbol name)
    : Expr(Kind::LocalRef, location), name(name)
    {}

    Symbol getName() const
    {
      return name;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::LocalRef;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("local").field(name).finish();
    }
  };

  /// This represents a reference to an object's member, e.g. `E.f` or `E.m`.
  /// The member could be either a field or method.
  class MemberRefExpr final : public Expr
  {
    ExprPtr origin;
    Symbol name;

  public:
    explicit MemberRefExpr(SourceLocation location, ExprPtr origin, Symbol name)
    : Expr(Kind::MemberRef, location), origin(std::move(origin)), name(name)
    {}

    const Expr* getOrigin() const
    {
      return origin.get();
    }

    Symbol getName() const
    {
      return name;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::MemberRef;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("member").field(origin).field(name).finish();
    }
  };

  /// This represents an assignment, e.g. `E1 = E2`.
  /// The expression evaluates to the lhs's old value.
  class AssignmentExpr final : public Expr
  {
    ExprPtr lhs;
    ExprPtr rhs;

  public:
    explicit AssignmentExpr(SourceLocation location, ExprPtr lhs, ExprPtr rhs)
    : Expr(Kind::Assignment, location), lhs(std::move(lhs)), rhs(std::move(rhs))
    {}

    const Expr* getLHS() const
    {
      return lhs.get();
    };

    const Expr* getRHS() const
    {
      return rhs.get();
    };

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Assignment;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("assign").field(lhs).field(rhs).finish();
    }
  };

  /// This represents a sequence of multiple expressions, e.g. `E1; E2; E3`.
  /// The expression evaluates to the value of the last expression
  class SequenceExpr final : public Expr
  {
    std::vector<ExprPtr> elements;

  public:
    explicit SequenceExpr(
      SourceLocation location, std::vector<ExprPtr> elements)
    : Expr(Kind::Sequence, location), elements(std::move(elements))
    {}

    // TODO: this provides mutable access to individual elements, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getElements() const
    {
      return elements;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Sequence;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("seq").field(elements).finish();
    }
  };

  /// This represents a conditional expression,
  /// e.g. `if (C) { E1 } else { E2 }`.
  ///
  /// The else block is optional. If it is specified, this expression evaluates
  /// to the value of the branch that was taken. Otherwise the expression
  /// evaluates to Unit.
  class IfExpr final : public Expr
  {
    ExprPtr condition;
    ExprPtr then_branch;
    ExprPtr else_branch;

  public:
    explicit IfExpr(
      SourceLocation location,
      ExprPtr condition,
      ExprPtr then_branch,
      ExprPtr else_branch)
    : Expr(Kind::If, location),
      condition(std::move(condition)),
      then_branch(std::move(then_branch)),
      else_branch(std::move(else_branch))
    {}

    const Expr* getCondition() const
    {
      return condition.get();
    }

    const Expr* getThenBranch() const
    {
      return then_branch.get();
    }

    const Expr* getElseBranch() const
    {
      return else_branch.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::If;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("if")
        .field(condition)
        .field(then_branch)
        .optional_field(else_branch)
        .finish();
    }
  };

  /// This represents a while loop expression, e.g. `while(C) { E }`.
  /// The expression evaluates to Unit.
  class WhileExpr final : public Expr
  {
    ExprPtr condition;
    ExprPtr body;

  public:
    explicit WhileExpr(SourceLocation location, ExprPtr condition, ExprPtr body)
    : Expr(Kind::While, location),
      condition(std::move(condition)),
      body(std::move(body))
    {}

    const Expr* getCondition() const
    {
      return condition.get();
    }

    const Expr* getBody() const
    {
      return body.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::While;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("while").field(condition).field(body).finish();
    }
  };

  /// This represents a `continue` statement.
  class ContinueExpr final : public Expr
  {
  public:
    explicit ContinueExpr(SourceLocation location)
    : Expr(Kind::Continue, location)
    {}

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Continue;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("continue").finish();
    }
  };

  /// This represents a `break` statement.
  class BreakExpr final : public Expr
  {
  public:
    explicit BreakExpr(SourceLocation location) : Expr(Kind::Break, location) {}

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Break;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("break").finish();
    }
  };

  /// This represents a `return` statement.
  class ReturnExpr final : public Expr
  {
    ExprPtr value;

  public:
    explicit ReturnExpr(SourceLocation location, ExprPtr value)
    : Expr(Kind::Return, location), value(std::move(value))
    {}

    const Expr* getValue() const
    {
      return value.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Return;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("return").optional_field(value).finish();
    }
  };

  /// This represents a `yield` statement.
  class YieldExpr final : public Expr
  {
    ExprPtr value;

  public:
    explicit YieldExpr(SourceLocation location, ExprPtr value)
    : Expr(Kind::Yield, location), value(std::move(value))
    {}

    const Expr* getValue() const
    {
      return value.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Yield;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("yield").field(value).finish();
    }
  };

  /// This represents a call to an object, e.g. `E1(E2, E3)`.
  ///
  /// In the case of method calls, eg. `E.m()`, the receiver of the call is
  /// `E.m`, a partial application of the method.
  class InvokeExpr final : public Expr
  {
    // TODO: add type arguments.
    ExprPtr receiver;
    std::vector<ExprPtr> arguments;

  public:
    explicit InvokeExpr(
      SourceLocation location, ExprPtr receiver, std::vector<ExprPtr> arguments)
    : Expr(Kind::Invoke, location),
      receiver(std::move(receiver)),
      arguments(std::move(arguments))
    {}

    const Expr* getReceiver() const
    {
      return receiver.get();
    }

    // TODO: this provides mutable access to individual arguments, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getArguments() const
    {
      return arguments;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Invoke;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("invoke").field(receiver).field(arguments).finish();
    }
  };

  /// This represents a static method call, e.g. `T.m(E1, E2)`.
  class StaticCallExpr final : public Expr
  {
    // TODO: add the receiver type.
    // TODO: add type arguments.
    Symbol name;
    std::vector<ExprPtr> arguments;

  public:
    explicit StaticCallExpr(
      SourceLocation location, Symbol name, std::vector<ExprPtr> arguments)
    : Expr(Kind::StaticCall, location),
      name(name),
      arguments(std::move(arguments))
    {}

    Symbol getName() const
    {
      return name;
    }

    // TODO: this provides mutable access to individual arguments, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getArguments() const
    {
      return arguments;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::StaticCall;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("static-call").field(arguments).finish();
    }
  };

  /// This represents a method call.
  ///
  /// MethodCallExpr and InvokeExpr overlap in their functionality, and should
  /// probably be unified. While MethodCallExpr could represent any method call,
  /// in practice it is, for the time being, only used for infix operator calls
  class MethodCallExpr final : public Expr
  {
    // TODO: add type arguments.
    Symbol name;
    ExprPtr receiver;
    std::vector<ExprPtr> arguments;

  public:
    explicit MethodCallExpr(
      SourceLocation location,
      Symbol name,
      ExprPtr receiver,
      std::vector<ExprPtr> arguments)
    : Expr(Kind::MethodCall, location),
      name(name),
      receiver(std::move(receiver)),
      arguments(std::move(arguments))
    {}

    Symbol getName() const
    {
      return name;
    }

    const Expr* getReceiver() const
    {
      return receiver.get();
    }

    // TODO: this provides mutable access to individual arguments, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getArguments() const
    {
      return arguments;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::MethodCall;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("call").field(name).field(receiver).field(arguments).finish();
    }
  };

  /// This represents an integer literal, e.g. 345
  class IntegerLiteral final : public Expr
  {
    // TODO: use something like APInt
    int64_t value;

  public:
    explicit IntegerLiteral(SourceLocation location, int64_t value)
    : Expr(Kind::IntegerLiteral, location), value(value)
    {}

    int64_t getValue() const
    {
      return value;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::IntegerLiteral;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("int-literal").field(value).finish();
    }
  };

  /// This represents an float literal, e.g. 345.67
  class FloatLiteral final : public Expr
  {
    // TODO: use something like APFloat
    double value;

  public:
    explicit FloatLiteral(SourceLocation location, double value)
    : Expr(Kind::FloatLiteral, location), value(value)
    {}

    double getValue() const
    {
      return value;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::FloatLiteral;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("float-literal").field(value).finish();
    }
  };

  /// This represents a boolean literal, i.e. 'true' or 'false'.
  class BooleanLiteral final : public Expr
  {
    bool value;

  public:
    explicit BooleanLiteral(SourceLocation location, bool value)
    : Expr(Kind::BooleanLiteral, location), value(value)
    {}

    bool getValue() const
    {
      return value;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::BooleanLiteral;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("bool-literal").field(value).finish();
    }
  };

  /// This represents a string literal.
  class StringLiteral final : public Expr
  {
    // TODO: std::string may not be the most appropriate way to represent these,
    // if the STL's default encoding doesn't match the source's?
    std::string value;

  public:
    explicit StringLiteral(SourceLocation location, std::string value)
    : Expr(Kind::StringLiteral, location), value(value)
    {}

    const std::string& getValue() const
    {
      return value;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::StringLiteral;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("string-literal").field(value).finish();
    }
  };

  /// This represents a string interpolation. It evaluates to the concatenation
  /// of element.
  class InterpolateExpr final : public Expr
  {
    std::vector<ExprPtr> elements;

  public:
    explicit InterpolateExpr(
      SourceLocation location, std::vector<ExprPtr> elements)
    : Expr(Kind::Interpolate, location), elements(std::move(elements))
    {}

    // TODO: this provides mutable access to individual elements, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getElements() const
    {
      return elements;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Interpolate;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("interpolate").field(elements).finish();
    }
  };

  /// This represents a tuple expression, e.g. `(E1, E2)`.
  ///
  /// While tuple with exactly one elements can be represented in the AST, there
  /// is currently no way to write them in the source language: the "prec" pass
  /// elides them away.
  class TupleExpr final : public Expr
  {
    std::vector<ExprPtr> elements;

  public:
    explicit TupleExpr(SourceLocation location, std::vector<ExprPtr> elements)
    : Expr(Kind::Tuple, location), elements(std::move(elements))
    {}

    // TODO: this provides mutable access to individual elements, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getElements() const
    {
      return elements;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Tuple;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("tuple").field(elements).finish();
    }
  };

  /// This represents a when block, e.g. `when (E1, E2) { E3 }`.
  class WhenExpr final : public Expr
  {
    std::vector<ExprPtr> arguments;
    ExprPtr body;

  public:
    explicit WhenExpr(
      SourceLocation location, std::vector<ExprPtr> arguments, ExprPtr body)
    : Expr(Kind::When, location),
      arguments(std::move(arguments)),
      body(std::move(body))
    {}

    // TODO: this provides mutable access to individual arguments, despite being
    // a `const` method.
    const std::vector<ExprPtr>& getArguments() const
    {
      return arguments;
    }

    const Expr* getBody() const
    {
      return body.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::When;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("when").field(arguments).field(body).finish();
    }
  };

  /// This represents an object allocation, e.g. `new T { ... } in x`.
  class NewExpr final : public Expr
  {
    // TODO: add the optional type.
    std::vector<std::unique_ptr<MemberDef>> elements;
    std::optional<Symbol> region;

  public:
    explicit NewExpr(
      SourceLocation location,
      std::vector<std::unique_ptr<MemberDef>> elements,
      std::optional<Symbol> region)
    : Expr(Kind::New, location), elements(std::move(elements)), region(region)
    {}

    // TODO: this provides mutable access to individual elements, despite being
    // a `const` method.
    const std::vector<std::unique_ptr<MemberDef>>& getElements() const
    {
      return elements;
    }

    const std::optional<Symbol>& getRegion() const
    {
      return region;
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::New;
    }

    // The implementation needs MemberDef to be complete, therefore it is moved
    // to `print.cc`.
    void print(NodePrinter& out) const override;
  };

  /// This represents a lambda expression, e.g. `(x, y) { E }`.
  class LambdaExpr final : public Expr
  {
    // TODO: add the lambda's signature
    ExprPtr body;

  public:
    explicit LambdaExpr(SourceLocation location, ExprPtr body)
    : Expr(Kind::Lambda, location), body(std::move(body))
    {}

    const Expr* getBody() const
    {
      return body.get();
    }

    static bool classof(const Expr* E)
    {
      return E->getKind() == Kind::Lambda;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("lambda").field(body).finish();
    }
  };
}
