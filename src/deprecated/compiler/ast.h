// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/context.h"
#include "compiler/instantiation.h"
#include "compiler/local_id.h"
#include "compiler/pegmatite-extra.h"
#include "compiler/type.h"

#include <map>
#include <optional>
#include <variant>

using pegmatite::ASTChild;
using pegmatite::ASTConstant;
using pegmatite::ASTContainer;
using pegmatite::ASTList;
using pegmatite::ASTPtr;

namespace verona::compiler
{
  struct Expression;
  struct Type;
  struct Name;

  struct TypeExpression;
  struct StaticAssertion;

  struct LocalDef;
  struct Entity;
  struct TypeParameterDef;

  // Dummy symbol type used to propagate errors.
  struct ErrorSymbol
  {};

  /**
   * A symbol is a reference to any named element of a program, i.e. a local
   * variable, an entity or a type parameter.
   *
   * Symbol has an additional error state, used during resolution if a name
   * cannot be found. If resolution succeeds without any error the AST will be
   * free of any symbol in this state.
   */
  struct Symbol : public std::variant<
                    const LocalDef*,
                    const Entity*,
                    const TypeParameterDef*,
                    ErrorSymbol>
  {
    using std::variant<
      const LocalDef*,
      const Entity*,
      const TypeParameterDef*,
      ErrorSymbol>::variant;

    /**
     * Get the Name from the symbol's definition. The name will have the
     * appropriate source location.
     */
    const Name& name() const;
  };

  struct StringLiteral : public pegmatite::ASTString
  {
    bool construct(
      const pegmatite::InputRange& r,
      pegmatite::ASTStack& st,
      const pegmatite::ErrorReporter& er) override
    {
      pegmatite::ASTString::construct(r, st, er);
      size_type newline;
      while ((newline = find("\\n")) != npos)
      {
        replace(newline, 2, "\n");
      }

      return true;
    }
  };

  /**
   * Class that encapsulates behaviour related to source locations.  This is
   * composed with other AST elements that may be looked up by source location
   * later.
   */
  template<typename AST>
  struct SourceLocatableMixin : public AST
  {
    /**
     * The location of this AST element.
     */
    SourceManager::SourceRange source_range;

    /**
     * Copy constructor.  When copying AST nodes, preserves the source location.
     */
    SourceLocatableMixin(const SourceLocatableMixin& other)
    {
      source_range = other.source_range;
    }

    /**
     * Default constructor.
     */
    SourceLocatableMixin() = default;

    /**
     * Construct method.  Sets the source location.
     */
    bool construct(
      const pegmatite::InputRange& r,
      pegmatite::ASTStack& st,
      const pegmatite::ErrorReporter& err)
    {
      source_range = ThreadContext::get().source_range_from_input_range(r);
      return AST::construct(r, st, err);
    }

    operator SourceManager::SourceRange() const
    {
      return source_range;
    }
  };

  /**
   * AST Node for identifiers. Has an associated source location.
   */
  struct Name : public SourceLocatableMixin<pegmatite::ASTString>
  {};

  /**
   * AST Node used by all expressions that introduce a new local variable.
   *
   * Pointers to LocalDef are used throughout the compiler to uniquely identify
   * a given local variable, possibly through the opaque LocalID wrapper.
   */
  struct LocalDef : public pegmatite::ASTContainer
  {
    ASTChild<Name> name;

    friend std::ostream& operator<<(std::ostream& s, const LocalDef& self)
    {
      return s << self.name;
    }
  };

  /**
   * AST Node for a type parameter definition.
   *
   * Every type parameter has an index corresponding to its depth.
   * For example, in the following code, W, X, Y and Z respectively have indices
   * 0, 1, 2 and 2.
   *
   * class C[W, X]
   *   fun m1[Y]() { }
   *   fun m2[Z]() { }
   *
   * The index matches the offset into the Instantiation when replacing type
   * parameters.
   *
   * A type parameter can be declared as a `class` type parameter, in which case
   * every use is implicitely wrapped in an entity-of.
   *
   * class C[class T]
   *   fun m(x: T) {}
   *
   * This is equivalent to:
   *
   * class C[T]
   *   fun m(x: entity-of(T)) {}
   *
   */
  struct TypeParameterDef : public ASTContainer
  {
    enum class Kind
    {
      Any,
      Class,
    };
    static constexpr Kind Any = Kind::Any;
    static constexpr Kind Class = Kind::Class;

    ASTPtr<ASTConstant<Kind>, /* optional */ true> kind_;
    ASTChild<Name> name;
    ASTPtr<TypeExpression, /* optional */ true> bound_expression;

    Kind kind() const
    {
      if (kind_)
        return kind_->value();
      else
        return Kind::Any;
    }

    // Added during resolution
    size_t index = SIZE_MAX;
    TypePtr bound;
  };

  struct Generics : public ASTContainer
  {
    ASTList<TypeParameterDef> types;
  };

  struct FnParameter : public ASTContainer
  {
    ASTPtr<LocalDef> local;
    ASTPtr<TypeExpression> type_expression;
  };

  struct Receiver : public ASTContainer
  {
    ASTPtr<LocalDef> local;
    ASTPtr<TypeExpression> type_expression;
  };

  struct WhereClauseTerm : public SourceLocatableMixin<ASTContainer>
  {};
  struct WhereClauseReturn : public WhereClauseTerm
  {};
  struct WhereClauseParameter : public WhereClauseTerm
  {
    ASTChild<Name> name;

    // Added during resolution
    LocalID local = nullptr;
  };

  struct WhereClause : public SourceLocatableMixin<ASTContainer>
  {
    enum class Kind
    {
      In,
      Under,
      From,
    };
    static constexpr Kind In = Kind::In;
    static constexpr Kind Under = Kind::Under;
    static constexpr Kind From = Kind::From;

    ASTPtr<WhereClauseTerm> left;
    ASTPtr<ASTConstant<Kind>> kind;
    ASTPtr<WhereClauseParameter> right;
  };

  struct FnSignature : public ASTContainer
  {
    ASTPtr<Generics> generics;
    ASTPtr<Receiver, /* Optional */ true> receiver;
    ASTList<FnParameter> parameters;
    ASTPtr<TypeExpression, /* optional */ true> return_type_expression;
    ASTList<WhereClause> where_clauses;

    // Added during resolution
    TypeSignature types;
  };

  struct FnBody : public ASTContainer
  {
    ASTPtr<Expression> expression;
  };

  struct Member : public SourceLocatableMixin<ASTContainer>
  {
    /**
     * Pointer back to the Entity which contains this member.
     * Added during resolution
     */
    const Entity* parent = nullptr;

    virtual const Name& get_name() const = 0;
    std::string path() const;
    virtual std::string
    instantiated_path(const Instantiation& instantiation) const = 0;
  };

  struct Entity : public ASTContainer
  {
    enum class Kind
    {
      Class,
      Interface,
      Primitive,
    };
    static constexpr Kind Class = Kind::Class;
    static constexpr Kind Interface = Kind::Interface;
    static constexpr Kind Primitive = Kind::Primitive;

    ASTPtr<ASTConstant<Kind>> kind;
    ASTChild<Name> name;
    ASTPtr<Generics> generics;
    ASTList<Member> members;

    // Added during resolution
    std::unordered_map<std::string, Member*> members_table;

    std::string path() const;
    std::string instantiated_path(const Instantiation& instantiation) const;
  };

  struct Method : public Member
  {
    enum class Kind
    {
      Regular,
      Builtin,
    };
    static constexpr Kind Regular = Kind::Regular;
    static constexpr Kind Builtin = Kind::Builtin;

    ASTPtr<ASTConstant<Kind>, /* optional */ true> kind_;
    ASTChild<Name> name;
    ASTPtr<FnSignature> signature;

    // Body is required in classes. This is enforced during resolution.
    ASTPtr<FnBody, /* optional */ true> body;

    std::string
    instantiated_path(const Instantiation& instantiation) const final;

    const Name& get_name() const final
    {
      return name;
    }

    Kind kind() const
    {
      if (kind_)
        return kind_->value();
      else
        return Kind::Regular;
    }

    bool is_finaliser() const
    {
      return name == "final";
    }
  };

  struct Field : public Member
  {
    ASTChild<Name> name;
    ASTPtr<TypeExpression> type_expression;

    // Added during resolution
    TypePtr type;

    const Name& get_name() const final
    {
      return name;
    }
    std::string
    instantiated_path(const Instantiation& instantiation) const final;
  };

  struct File : public ASTContainer
  {
    ASTList<StringLiteral> modules;
    ASTList<Entity> entities;
    ASTList<StaticAssertion> assertions;
  };

  struct Program
  {
    std::vector<std::unique_ptr<File>> files;

    // Added during resolution
    std::unordered_map<std::string, const Entity*> entities_table;

    const Entity* find_entity(const std::string& name) const
    {
      auto it = entities_table.find(name);
      if (it != entities_table.end())
        return it->second;
      else
        return nullptr;
    };
  };

  struct Expression : public SourceLocatableMixin<ASTContainer>
  {};

  struct DefineLocalExpr : public Expression
  {
    ASTPtr<LocalDef> local;
    ASTPtr<Expression, /* optional */ true> right;
  };

  struct SymbolExpr : public Expression
  {
    ASTChild<Name> name;

    // Added during resolution
    Symbol symbol = ErrorSymbol();
  };

  struct AssignLocalExpr : public Expression
  {
    ASTChild<Name> name;
    ASTPtr<Expression> right;

    // Added during resolution
    LocalID local = nullptr;
  };

  struct FieldExpr : public Expression
  {
    ASTPtr<Expression> expr;
    ASTChild<Name> field_name;
  };

  struct AssignFieldExpr : public Expression
  {
    ASTPtr<Expression> expr;
    ASTChild<Name> field_name;
    ASTPtr<Expression> right;
  };

  struct SeqExpr : public Expression
  {
    ASTList<Expression> elements;
    ASTPtr<Expression> last;
  };

  struct NewParent : public SourceLocatableMixin<ASTContainer>
  {
    ASTChild<Name> name;

    // Added during resolution
    LocalID local = nullptr;
  };

  struct NewExpr : public Expression
  {
    ASTChild<Name> class_name;
    ASTPtr<NewParent, /* optional */ true> parent;

    // Added during resolution. Must point to a Class entity.
    const Entity* definition = nullptr;
  };

  struct Argument : public ASTContainer
  {
    ASTPtr<Expression> inner;
  };

  struct CallExpr : public Expression
  {
    ASTPtr<Expression> receiver;
    ASTChild<Name> method_name;
    ASTList<Argument> arguments;
  };

  struct WhileExpr : public Expression
  {
    ASTPtr<Expression> condition;
    ASTPtr<Expression> body;
  };

  struct ElseExpr : public Expression
  {
    ASTPtr<Expression> body;
  };

  struct IfExpr : public Expression
  {
    ASTPtr<Expression> condition;
    ASTPtr<Expression> then_block;
    ASTPtr<ElseExpr, /* optional */ true> else_block;
  };

  struct BlockExpr : public Expression
  {
    ASTPtr<Expression> inner;
  };

  struct WhenArgument : public SourceLocatableMixin<ASTContainer>
  {
    virtual const LocalDef* get_binder() = 0;
    virtual ~WhenArgument() = default;
  };
  struct WhenArgumentShadow : public WhenArgument
  {
    ASTPtr<LocalDef> binder;
    // Added during resolution
    LocalID local = nullptr;

    const LocalDef* get_binder() final
    {
      return binder.get();
    }
  };
  struct WhenArgumentAs : public WhenArgument
  {
    ASTPtr<LocalDef> binder;
    ASTPtr<Expression> inner;

    const LocalDef* get_binder() final
    {
      return binder.get();
    }
  };

  struct WhenExpr : public Expression
  {
    ASTList<WhenArgument> cowns;
    ASTPtr<BlockExpr> body;

    // Added during resolution
    std::vector<LocalID> captures;
  };

  struct EmptyExpr : public Expression
  {};

  struct MatchArm : public SourceLocatableMixin<ASTContainer>
  {
    ASTPtr<LocalDef> local;
    ASTPtr<TypeExpression> type_expression;
    ASTPtr<Expression> expr;

    // Added during resolution
    TypePtr type;
  };

  struct MatchExpr : public Expression
  {
    ASTPtr<Expression> expr;
    ASTList<MatchArm> arms;
  };

  struct IntegerLiteralExpr : public Expression
  {
    ASTChild<pegmatite::ASTInteger> value;
  };

  struct StringLiteralExpr : public Expression
  {
    ASTChild<StringLiteral> value;
  };

  struct ViewExpr : public Expression
  {
    ASTPtr<Expression> expr;
  };

  enum class BinaryOperator
  {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
  };
  std::string_view binary_operator_method_name(BinaryOperator op);

  struct BinaryOperatorExpr : public Expression
  {
    ASTPtr<Expression> left;
    ASTPtr<ASTConstant<BinaryOperator>> kind;
    ASTPtr<Expression> right;
  };

  /**
   * AST Node for types as it appears in the input.
   *
   * TypeExpression nodes are interpreted into their semantic counterpart,
   * Type, during name resolution. Most of the compiler works on these semantic
   * types, not on type expressions.
   */
  struct TypeExpression : public SourceLocatableMixin<ASTContainer>
  {};

  struct SymbolTypeExpr final : public TypeExpression
  {
    ASTChild<Name> name;
    ASTList<TypeExpression> arguments;

    // Added during resolution
    Symbol symbol = ErrorSymbol();
  };

  struct UnionTypeExpr final : public TypeExpression
  {
    ASTList<TypeExpression> elements;
  };

  struct IntersectionTypeExpr final : public TypeExpression
  {
    ASTList<TypeExpression> elements;
  };

  struct ViewpointTypeExpr final : public TypeExpression
  {
    ASTPtr<TypeExpression> left;
    ASTPtr<TypeExpression> right;
  };

  struct CapabilityTypeExpr final : public TypeExpression
  {
    ASTPtr<ASTConstant<CapabilityKind>> kind;
  };

  enum class AssertionKind
  {
    // Check that the left type is a subtype of the right
    Subtype,

    // Check that the left type is *not* a subtype of the right
    NotSubtype,
  };

  struct StaticAssertion : public SourceLocatableMixin<ASTContainer>
  {
    ASTPtr<Generics> generics;
    ASTPtr<TypeExpression> left_expression;
    ASTPtr<ASTConstant<AssertionKind>> kind;
    ASTPtr<TypeExpression> right_expression;

    // Added during resolution
    TypePtr left_type;
    TypePtr right_type;

    // Unique per-assertion index. Used to choose filenames when printing dumps
    // about the assertion. Added during resolution.
    size_t index = SIZE_MAX;
  };
}
