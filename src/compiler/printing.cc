// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/printing.h"

#include "compiler/format.h"
#include "compiler/ir/ir.h"
#include "compiler/typecheck/constraint.h"
#include "compiler/visitor.h"
#include "ds/helpers.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  using format::comma_sep;
  using format::defaulted;
  using format::optional;
  using format::optional_list;
  using format::prefixed;
  using format::separated_by;
  using format::sorted;

  class PrintVisitor : public ExprVisitor<>,
                       public TypeVisitor<>,
                       public TypeExpressionVisitor<>,
                       public MemberVisitor<>
  {
  public:
    PrintVisitor(std::ostream& out) : out_(out) {}

    template<typename... Args>
    void print(std::string_view fmt, Args&&... args)
    {
      fmt::print(out_, fmt, std::forward<Args>(args)...);
    }

    void visit_method(Method* m) final
    {
      print(
        "(method {} {}{})",
        m->name,
        *m->signature,
        optional(prefixed(" ", m->body)));
    }

    void visit_field(Field* fld) final
    {
      print("(field {} {})", fld->name, *fld->type_expression);
    }

    void visit_symbol(SymbolExpr& e) final
    {
      print("(symbol {})", e.name);
    }

    void visit_field(FieldExpr& e) final
    {
      print("(field {} {})", *e.expr, e.field_name);
    }

    void visit_assign_local(AssignLocalExpr& e) final
    {
      print("(assign-local {} {})", e.name, *e.right);
    }

    void visit_assign_field(AssignFieldExpr& e) final
    {
      print("(assign-field {} {} {})", *e.expr, e.field_name, *e.right);
    }

    void visit_seq(SeqExpr& e) final
    {
      if (e.elements.empty())
        print("(seq [{}])", *e.last);
      else
        print("(seq [{}, {}])", comma_sep(e.elements), *e.last);
    }

    void visit_call(CallExpr& e) final
    {
      print(
        "(call {} {} [{}])",
        *e.receiver,
        e.method_name,
        comma_sep(e.arguments));
    }

    void visit_when(WhenExpr& e) final
    {
      if (e.captures.empty())
        print("(when ({}) {})", comma_sep(e.cowns), *e.body);
      else
        print(
          "(when ({}) [{}] {})",
          comma_sep(e.cowns),
          comma_sep(e.captures),
          *e.body);
    }

    void visit_while(WhileExpr& e) final
    {
      print("(while {} {})", *e.condition, *e.body);
    }

    void visit_if(IfExpr& e) final
    {
      if (e.else_block)
      {
        print("(if {} {} {}", *e.condition, *e.then_block, *e.else_block->body);
      }
      else
      {
        print("(if {} {})", *e.condition, *e.then_block);
      }
    }

    void visit_block(BlockExpr& e) final
    {
      print("(block {})", *e.inner);
    }

    void visit_empty(EmptyExpr& e) final
    {
      print("(empty)");
    }

    void visit_define_local(DefineLocalExpr& e) final
    {
      print("(define-local {}{})", *e.local, optional(prefixed(" ", e.right)));
    }

    void visit_match_expr(MatchExpr& e) final
    {
      print("(match {} [{}])", *e.expr, comma_sep(e.arms));
    }

    void visit_new_expr(NewExpr& e) final
    {
      print("(new {}{})", e.class_name, optional(prefixed(" ", e.parent)));
    }

    void visit_integer_literal_expr(IntegerLiteralExpr& e) final
    {
      print("(integer {})", e.value.value);
    }

    void visit_string_literal_expr(StringLiteralExpr& e) final
    {
      // TODO: we should be escaping e.value
      print("(string \"{}\")", e.value);
    }

    void visit_binary_operator_expr(BinaryOperatorExpr& e) final
    {
      print("({} {} {})", e.kind->value(), *e.left, *e.right);
    }

    void visit_view_expr(ViewExpr& e) final
    {
      print("(mut-view {})", *e.expr);
    }

    void visit_entity_type(const EntityTypePtr& ty) final
    {
      print("{}{}", ty->definition->name, optional_list(ty->arguments));
    }

    void visit_static_type(const StaticTypePtr& ty) final
    {
      print(
        "(static {}{})", ty->definition->name, optional_list(ty->arguments));
    }

    void visit_type_parameter(const TypeParameterPtr& ty) final
    {
      print("{}", ty->definition->name);
    }
    void visit_capability(const CapabilityTypePtr& ty) final
    {
      print("{}{}", ty->kind, ty->region);
    }
    void visit_apply_region(const ApplyRegionTypePtr& ty) final
    {
      print("(apply-{} {} {})", ty->mode, ty->region, *ty->type);
    }
    void visit_unapply_region(const UnapplyRegionTypePtr& ty) final
    {
      print("(unapply {})", *ty->type);
    }
    void visit_union(const UnionTypePtr& ty) final
    {
      if (ty->elements.empty())
        print("(bottom)");
      else
        print("({})", separated_by(sorted(ty->elements), " | "));
    }
    void visit_intersection(const IntersectionTypePtr& ty) final
    {
      if (ty->elements.empty())
        print("(top)");
      else
        print("({})", separated_by(sorted(ty->elements), " & "));
    }
    void visit_infer(const InferTypePtr& ty) final
    {
      print(
        "{}{}{}",
        ty->polarity,
        ty->index,
        optional(prefixed(".", ty->subindex)));
    }
    void visit_range_type(const RangeTypePtr& ty) final
    {
      print("({} ... {})", *ty->lower, *ty->upper);
    }

    void visit_viewpoint_type(const ViewpointTypePtr& ty) final
    {
      print(
        "(viewpoint {} [{}] {})",
        format::defaulted(ty->capability, "()"),
        comma_sep(sorted(ty->variables)),
        *ty->right);
    }
    void visit_unit_type(const UnitTypePtr& ty) final
    {
      print("unit");
    }
    void visit_has_field_type(const HasFieldTypePtr& ty) final
    {
      print(
        "(has-field {} '{}' {} {})",
        *ty->view,
        ty->name,
        *ty->read_type,
        *ty->write_type);
    }
    void visit_delayed_field_view_type(const DelayedFieldViewTypePtr& ty) final
    {
      print("(delayed-field-view '{}' {})", ty->name, *ty->type);
    }
    void visit_has_method_type(const HasMethodTypePtr& ty) final
    {
      print("(has-method '{}' {})", ty->name, ty->signature);
    }
    void visit_has_applied_method_type(const HasAppliedMethodTypePtr& ty) final
    {
      print(
        "(has-applied-method '{}' {} {})",
        ty->name,
        ty->application,
        ty->signature);
    }
    void visit_is_entity_type(const IsEntityTypePtr& ty) final
    {
      print("(is-entity)");
    }
    void visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      print("(fixpoint {})", *ty->inner);
    }
    void visit_fixpoint_variable_type(const FixpointVariableTypePtr& ty) final
    {
      print("(fixpoint-var {})", ty->depth);
    }
    void visit_entity_of_type(const EntityOfTypePtr& ty) final
    {
      print("(entity-of {})", *ty->inner);
    }
    void visit_variable_renaming_type(const VariableRenamingTypePtr& ty) final
    {
      print("(rename {} {})", ty->renaming, *ty->type);
    }
    void visit_path_compression_type(const PathCompressionTypePtr& ty) final
    {
      auto format_entry = [](const auto& entry) {
        return fmt::format("{}: {}", entry.first, *entry.second);
      };
      print(
        "(compress [{}] {})",
        comma_sep(ty->compression, format_entry),
        *ty->type);
    }
    void visit_indirect_type(const IndirectTypePtr& ty) final
    {
      print("(indirect {}:{})", *ty->block, ty->variable);
    }
    void visit_not_child_of_type(const NotChildOfTypePtr& ty) final
    {
      print("(not-child-of {})", ty->region);
    }

    void visit_region(const RegionHole& r)
    {
      print("(•)");
    }

    void visit_region(const RegionNone& r)
    {
      print("(none)");
    }

    void visit_region(const RegionReceiver& r)
    {
      print("(self)");
    }

    void visit_region(const RegionParameter& r)
    {
      print("(param {})", r.index);
    }

    void visit_region(const RegionExternal& r)
    {
      print("(external {})", r.index);
    }

    void visit_region(const RegionVariable& r)
    {
      print("(variable {})", r.variable);
    }

    void visit_type_sequence(const UnboundedTypeSequence& seq)
    {
      print("(unbounded-sequence {})", seq.index);
    }

    void visit_type_sequence(const BoundedTypeSequence& seq)
    {
      print("[{}]", comma_sep(seq.types));
    }

    void visit_symbol_type_expr(SymbolTypeExpr& te)
    {
      print("{}{}", te.name, optional_list(te.arguments));
    }

    void visit_capability_type_expr(CapabilityTypeExpr& te)
    {
      print("{}", te.kind->value());
    }

    void visit_union_type_expr(UnionTypeExpr& te)
    {
      // elements is a list, not a set. Does not need to be sorted.
      print("({})", separated_by(te.elements, " | "));
    }

    void visit_intersection_type_expr(IntersectionTypeExpr& te)
    {
      // elements is a list, not a set. Does not need to be sorted.
      print("({})", separated_by(te.elements, " & "));
    }

    void visit_viewpoint_type_expr(ViewpointTypeExpr& te)
    {
      print("(viewpoint {} {})", *te.left, *te.right);
    }

  private:
    std::ostream& out_;
  };

  std::ostream& operator<<(std::ostream& out, const LocalDef* p)
  {
    fmt::print(out, "{}", p->name);
    return out;
  };

  std::ostream& operator<<(std::ostream& out, const Argument& p)
  {
    fmt::print(out, "{}", *p.inner);
    return out;
  };

  std::ostream& operator<<(std::ostream& out, const WhenArgument& p)
  {
    if (auto argument = dynamic_cast<const WhenArgumentAs*>(&p))
    {
      fmt::print(out, "var {} = {}", *argument->binder, *argument->inner);
    }
    else if (auto argument = dynamic_cast<const WhenArgumentShadow*>(&p))
    {
      fmt::print(out, "{}", *argument->binder);
    }
    else
    {
      abort();
    }
    return out;
  };

  std::ostream& operator<<(std::ostream& out, const MatchArm& a)
  {
    fmt::print(out, "(arm {} {} {})", *a.local, *a.type_expression, *a.expr);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const BinaryOperator& op)
  {
    switch (op)
    {
      case BinaryOperator::Add:
        fmt::print(out, "+");
        break;
      case BinaryOperator::Sub:
        fmt::print(out, "-");
        break;
      case BinaryOperator::Mul:
        fmt::print(out, "*");
        break;
      case BinaryOperator::Div:
        fmt::print(out, "/");
        break;
      case BinaryOperator::Mod:
        fmt::print(out, "%");
        break;
      case BinaryOperator::Shl:
        fmt::print(out, "<<");
        break;
      case BinaryOperator::Shr:
        fmt::print(out, ">>");
        break;
      case BinaryOperator::Lt:
        fmt::print(out, "<");
        break;
      case BinaryOperator::Le:
        fmt::print(out, "<=");
        break;
      case BinaryOperator::Gt:
        fmt::print(out, ">");
        break;
      case BinaryOperator::Ge:
        fmt::print(out, ">=");
        break;
      case BinaryOperator::Eq:
        fmt::print(out, "==");
        break;
      case BinaryOperator::Ne:
        fmt::print(out, "!=");
        break;
      case BinaryOperator::And:
        fmt::print(out, "&&");
        break;
      case BinaryOperator::Or:
        fmt::print(out, "||");
        break;

        EXHAUSTIVE_SWITCH;
    }
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Polarity& p)
  {
    switch (p)
    {
      case Polarity::Positive:
        return out << "+";
      case Polarity::Negative:
        return out << "-";

        EXHAUSTIVE_SWITCH;
    }
  }

  std::ostream& operator<<(std::ostream& out, const CapabilityKind& c)
  {
    switch (c)
    {
      case CapabilityKind::Isolated:
        return out << "iso";
      case CapabilityKind::Immutable:
        return out << "imm";
      case CapabilityKind::Mutable:
        return out << "mut";
      case CapabilityKind::Subregion:
        return out << "sub";

        EXHAUSTIVE_SWITCH;
    }
  }

  std::ostream& operator<<(std::ostream& out, const ApplyRegionType::Mode& m)
  {
    switch (m)
    {
      case ApplyRegionType::Mode::Adapt:
        return out << "adapt";
      case ApplyRegionType::Mode::Under:
        return out << "under";
      case ApplyRegionType::Mode::Extract:
        return out << "extract";

        EXHAUSTIVE_SWITCH;
    }
  }

  std::ostream& operator<<(std::ostream& out, const Entity::Kind& k)
  {
    switch (k)
    {
      case Entity::Class:
        return out << "class";
      case Entity::Interface:
        return out << "interface";
      case Entity::Primitive:
        return out << "primitive";

        EXHAUSTIVE_SWITCH;
    }
  }

  std::ostream& operator<<(std::ostream& out, const Generics& g)
  {
    fmt::print(out, "(generics [{}])", comma_sep(g.types));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const TypeParameterDef& param)
  {
    fmt::print(out, "(typeparamdef {})", param.name);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const FnSignature& sig)
  {
    fmt::print(
      out,
      "{} [{}] {} [{}]",
      defaulted(sig.receiver, "static"),
      comma_sep(sig.parameters),
      defaulted(sig.return_type_expression, "()"),
      comma_sep(sig.where_clauses));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const FnBody& body)
  {
    return out << *body.expression;
  }

  std::ostream& operator<<(std::ostream& out, const Receiver& r)
  {
    fmt::print(out, "(receiver {} {})", *r.local, *r.type_expression);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const FnParameter& p)
  {
    fmt::print(out, "(param {} {})", *p.local, *p.type_expression);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const File& f)
  {
    fmt::print(
      out,
      "(program [{}] [{}])",
      comma_sep(f.entities),
      comma_sep(f.assertions));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Program& p)
  {
    fmt::print(out, "(program [])", comma_sep(p.files));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Entity& entity)
  {
    fmt::print(
      out,
      "({} {} {} [{}])",
      entity.kind->value(),
      entity.name,
      *entity.generics,
      comma_sep(entity.members));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const WhereClause& clause)
  {
    fmt::print(
      out,
      "(where {} {} {})",
      *clause.left,
      clause.kind->value(),
      *clause.right);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const WhereClause::Kind& k)
  {
    switch (k)
    {
      case WhereClause::In:
        fmt::print(out, "in");
        break;

      case WhereClause::Under:
        fmt::print(out, "under");
        break;

      case WhereClause::From:
        fmt::print(out, "from");
        break;

        EXHAUSTIVE_SWITCH
    }

    return out;
  }

  std::ostream& operator<<(std::ostream& out, const WhereClauseTerm& t)
  {
    if (auto ret = dynamic_cast<const WhereClauseReturn*>(&t))
    {
      fmt::print(out, "return");
    }
    else if (auto param = dynamic_cast<const WhereClauseParameter*>(&t))
    {
      fmt::print(out, "(param {})", param->name);
    }
    else
    {
      abort();
    }

    return out;
  }

  std::ostream& operator<<(std::ostream& out, const AssertionKind& kind)
  {
    switch (kind)
    {
      case AssertionKind::Subtype:
        return out << "subtype";

      case AssertionKind::NotSubtype:
        return out << "not-subtype";

        EXHAUSTIVE_SWITCH;
    }
  }

  std::ostream& operator<<(std::ostream& out, const StaticAssertion& assertion)
  {
    fmt::print(
      out,
      "(static-assert {} {} {} {})",
      *assertion.generics,
      *assertion.left_expression,
      assertion.kind->value(),
      *assertion.right_expression);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Constraint& c)
  {
    fmt::print(out, "{} <: {}   --- {}", *c.left, *c.right, c.reason);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const TypeSignature& signature)
  {
    fmt::print(
      out,
      "(signature {} [{}] {})",
      *signature.receiver,
      comma_sep(signature.arguments),
      *signature.return_type);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const LocalID& l)
  {
    return out << *l.definition_;
  }

  std::ostream& operator<<(std::ostream& out, const NewParent& p)
  {
    return out << p.name;
  }

  std::ostream& operator<<(std::ostream& out, const Expression& e)
  {
    PrintVisitor v(out);
    v.visit_expr(const_cast<Expression&>(e));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Type& ty)
  {
    PrintVisitor v(out);
    v.visit_type(ty.shared_from_this());
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Region& r)
  {
    PrintVisitor v(out);
    std::visit([&](const auto& inner) { v.visit_region(inner); }, r);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const Member& member)
  {
    PrintVisitor v(out);
    v.visit_member(const_cast<Member*>(&member));
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const InferableTypeSequence& seq)
  {
    PrintVisitor v(out);
    std::visit([&](const auto& inner) { v.visit_type_sequence(inner); }, seq);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const TypeExpression& te)
  {
    PrintVisitor v(out);
    v.visit_type_expression(const_cast<TypeExpression&>(te));
    return out;
  }
}
