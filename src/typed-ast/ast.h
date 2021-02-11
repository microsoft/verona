// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "typed-ast/expr.h"
#include "typed-ast/node.h"

/// This directory contains a typed AST representation of Verona programs.
///
/// "typed" in this context refers to the C++ classes that structure the AST,
/// not about the contents of the AST. This is in constract with peglib's
/// untyped AST, which are effectively S-expressions, with no well-defined
/// structure.
///
/// In the long run, we aim to replace most/all uses of the untyped AST with the
/// typed one.
///
/// The AST supports [LLVM-style RTTI]. Nodes that have multiple variants (eg.
/// expressions) declare a `Kind` enum, with a value per concrete node. The kind
/// is exposed through a `getKind` method and a static `classof` method. These
/// are generally not used directly, but through LLVM's `isa<>`, `dyn_cast<>`,
/// `TypeSwitch`, ... utilities.
///
/// The AST currently uses one allocation per node, and its lifetime is managed
/// automatically using `std::unique_ptr` on every edge of the tree. In the
/// future, we should consider using an arena to allocate all the nodes, since
/// we are unlikely to ever deallocate individual nodes.
///
/// TODO: the AST definition does not make it obvious which fields are optional
/// and which are not.
///
/// [LLVM-style RTTI]: https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html
namespace verona::ast
{
  /// This is the base class for entity members (i.e. fields, methods and child
  /// entities), from which all concrete nodes derive.
  class MemberDef : public Node
  {
  public:
    enum class Kind
    {
      Entity,
      Field,
      Method,
      TypeAlias,
    };

    Kind getKind() const
    {
      return kind;
    }

    const Symbol& getName() const
    {
      return name;
    }

  protected:
    MemberDef(Kind kind, SourceLocation location, Symbol name)
    : Node(location), kind(kind), name(name)
    {}

  private:
    Kind kind;
    Symbol name;
  };

  /// This represents an entity definition. Entities can be either a class,
  /// module or interface.
  class EntityDef : public MemberDef
  {
    std::vector<std::unique_ptr<MemberDef>> elements;

  public:
    EntityDef(
      SourceLocation location,
      Symbol name,
      std::vector<std::unique_ptr<MemberDef>> elements)
    : MemberDef(Kind::Entity, location, name), elements(std::move(elements))
    {}

    // TODO: this provides mutable access to individual elements, despite being
    // a `const` method.
    const std::vector<std::unique_ptr<MemberDef>>& getElements() const
    {
      return elements;
    }

    static bool classof(const MemberDef* M)
    {
      return M->getKind() == Kind::Entity;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("entity").field(getName()).field(elements).finish();
    }
  };

  /// This represents a field definition.
  class FieldDef : public MemberDef
  {
    // TODO: add field qualifiers (eg. static, private)
    // TODO: add the field's type
    ExprPtr initializer;

  public:
    FieldDef(SourceLocation location, Symbol name, ExprPtr initializer)
    : MemberDef(Kind::Field, location, name),
      initializer(std::move(initializer))
    {}

    const Expr* getInitializer() const
    {
      return initializer.get();
    }

    static bool classof(const MemberDef* M)
    {
      return M->getKind() == Kind::Field;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("field").field(getName()).optional_field(initializer).finish();
    }
  };

  /// This represents a method definition.
  class MethodDef : public MemberDef
  {
    // TODO: add method qualifiers (eg. static, private)
    // TODO: add the method's signature
    ExprPtr body;

  public:
    MethodDef(SourceLocation location, Symbol name, ExprPtr body)
    : MemberDef(Kind::Method, location, name), body(std::move(body))
    {}

    const Expr* getBody() const
    {
      return body.get();
    }

    static bool classof(const MemberDef* M)
    {
      return M->getKind() == Kind::Method;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("function").field(getName()).optional_field(body).finish();
    }
  };

  /// This represents a type alias definition, of the form `type Foo = T`.
  class TypeAliasDef : public MemberDef
  {
  public:
    TypeAliasDef(SourceLocation location, Symbol name)
    : MemberDef(Kind::TypeAlias, location, name)
    {}

    static bool classof(const MemberDef* M)
    {
      return M->getKind() == Kind::TypeAlias;
    }

    void print(NodePrinter& out) const override
    {
      out.begin("typedef").field(getName()).finish();
    }
  };
}
