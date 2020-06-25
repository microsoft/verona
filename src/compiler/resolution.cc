// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/resolution.h"

#include "compiler/ast.h"
#include "compiler/instantiation.h"
#include "compiler/mapper.h"
#include "compiler/printing.h"
#include "compiler/recursive_visitor.h"
#include "compiler/substitution.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

#include <cassert>

namespace verona::compiler
{
  class ResolutionVisitor : private MemberVisitor<>,
                            private RecursiveExprVisitor<>,
                            private TypeExpressionVisitor<TypePtr>
  {
  public:
    ResolutionVisitor(Context& context) : context_(context)
    {
      scopes_.push_back(Scope(0));
    }

    void visit_program(Program* program)
    {
      assert(scopes_.size() == 1);
      push_scope([&]() {
        for (const auto& file : program->files)
        {
          for (const auto& entity : file->entities)
          {
            add_symbol(entity.get());
            program->entities_table.insert({entity->name, entity.get()});
          }
        }
        for (const auto& file : program->files)
        {
          for (const auto& entity : file->entities)
          {
            visit_entity(entity.get());
          }
          for (const auto& assertion : file->assertions)
          {
            visit_assertion(assertion.get());
          }
        }
      });
      assert(scopes_.size() == 1);
    }

  private:
    void push_scope(std::function<void()> f)
    {
      assert(!scopes_.empty());
      scopes_.push_back(Scope(scopes_.back().next_type_param));
      f();
      scopes_.pop_back();
    }

    void push_closure_scope(std::function<void()> f)
    {
      assert(!scopes_.empty());
      scopes_.push_back(Scope(scopes_.back().next_type_param, true));
      f();
      scopes_.pop_back();
    }

    void add_symbol(Symbol symbol)
    {
      const Name& name = symbol.name();
      assert(!scopes_.empty());
      auto [it, inserted] = scopes_.back().symbols.insert({name, symbol});
      if (!inserted)
      {
        report(
          context_,
          name,
          DiagnosticKind::Error,
          Diagnostic::SymbolAlreadyExists,
          name);
        report(
          context_,
          it->second.name(),
          DiagnosticKind::Note,
          Diagnostic::PreviousDefinitionHere,
          name);
      }
    }

    Symbol resolve(const Name& name)
    {
      bool captured = false;
      for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); scope++)
      {
        if (scope->is_closure_scope)
          captured = true;

        auto it = scope->symbols.find(name);
        if (it != scope->symbols.end())
        {
          Symbol result = it->second;
          if (captured)
          {
            if (auto l = std::get_if<const LocalDef*>(&result))
            {
              // Skip current scope as this has defined these variables.
              scope--;
              // Walk back adding symbol to captures
              for (; scope != scopes_.rbegin(); scope--)
              {
                if (scope->is_closure_scope)
                {
                  scope->captures_.push_back(*l);
                  // Add so we only capture it once.
                  scope->symbols.insert({name, result});
                }
              }
            }
          }
          return result;
        }
      }

      report(
        context_,
        name.source_range,
        DiagnosticKind::Error,
        Diagnostic::UndefinedSymbol,
        name);
      return ErrorSymbol();
    }

    std::optional<LocalID> resolve_local(const Name& name)
    {
      Symbol symbol = resolve(name);
      if (auto local = std::get_if<const LocalDef*>(&symbol))
      {
        return *local;
      }
      else if (!std::holds_alternative<ErrorSymbol>(symbol))
      {
        report(
          context_,
          name.source_range,
          DiagnosticKind::Error,
          Diagnostic::SymbolNotLocal,
          name);
      }

      return std::nullopt;
    }

    void add_generics(Generics* generics)
    {
      for (auto& param : generics->types)
      {
        param->index = scopes_.back().next_type_param++;
        add_symbol(param.get());
      }
    }

    void resolve_generics(Generics* generics)
    {
      for (auto& param : generics->types)
      {
        if (param->bound_expression != nullptr)
          param->bound = visit_type_expression(*param->bound_expression);
        else
          param->bound = context_.mk_top();
      }
    }

    void visit_entity(Entity* entity)
    {
      for (const auto& member : entity->members)
      {
        member->parent = entity;

        const Name& name = member->get_name();
        auto [it, inserted] =
          entity->members_table.insert({name, member.get()});
        if (!inserted)
        {
          report(
            context_,
            name,
            DiagnosticKind::Error,
            Diagnostic::MemberAlreadyExists,
            entity->name,
            name);
          report(
            context_,
            it->second->get_name(),
            DiagnosticKind::Note,
            Diagnostic::PreviousDefinitionHere,
            name);
        }
      }

      push_scope([&]() {
        add_generics(entity->generics.get());
        resolve_generics(entity->generics.get());
        visit_members(entity->members);
      });
    }

    void visit_assertion(StaticAssertion* assertion)
    {
      assertion->index = next_assertion_index_++;
      push_scope([&]() {
        add_generics(assertion->generics.get());
        resolve_generics(assertion->generics.get());
        assertion->left_type =
          visit_type_expression(*assertion->left_expression);
        assertion->right_type =
          visit_type_expression(*assertion->right_expression);
      });
    }

    void add_signature(FnSignature* signature)
    {
      add_generics(signature->generics.get());
      if (const auto& receiver = signature->receiver)
      {
        add_symbol(receiver->local.get());
      }
      for (const auto& param : signature->parameters)
      {
        add_symbol(param->local.get());
      }
    }

    void resolve_signature(FnSignature* signature)
    {
      resolve_generics(signature->generics.get());

      if (const auto& receiver = signature->receiver)
        signature->types.receiver =
          visit_type_expression(*receiver->type_expression);
      else
        signature->types.receiver = context_.mk_top();

      for (const auto& param : signature->parameters)
      {
        signature->types.arguments.push_back(
          visit_type_expression(*param->type_expression));
      }

      if (signature->return_type_expression)
        signature->types.return_type =
          visit_type_expression(*signature->return_type_expression);
      else
        signature->types.return_type = context_.mk_unit();

      for (const auto& clause : signature->where_clauses)
      {
        visit_where_clause(clause.get());
      }
    }

    void visit_where_clause_term(WhereClauseTerm* term)
    {
      if (auto param = dynamic_cast<WhereClauseParameter*>(term))
      {
        if (auto local = resolve_local(param->name))
        {
          param->local = *local;
        }
      }
      else if (auto ret = dynamic_cast<WhereClauseReturn*>(term))
      {}
      else
      {
        abort();
      }
    }

    void visit_where_clause(WhereClause* clause)
    {
      visit_where_clause_term(clause->left.get());
      visit_where_clause_term(clause->right.get());
    }

    void visit_method(Method* method) final
    {
      push_scope([&]() {
        add_signature(method->signature.get());
        resolve_signature(method->signature.get());

        switch (method->kind())
        {
          case Method::Regular:
            if (!method->body && method->parent->kind->value() == Entity::Class)
            {
              report(
                context_,
                method->name,
                DiagnosticKind::Error,
                Diagnostic::MissingMethodBodyInClass,
                method->name,
                method->parent->name);
            }

            if (
              !method->body &&
              method->parent->kind->value() == Entity::Primitive)
            {
              report(
                context_,
                method->name,
                DiagnosticKind::Error,
                Diagnostic::MissingMethodBodyInPrimitive,
                method->name,
                method->parent->name);
            }
            break;

          case Method::Builtin:
            if (method->body)
            {
              report(
                context_,
                method->name,
                DiagnosticKind::Error,
                Diagnostic::BuiltinMethodHasBody,
                method->name,
                method->parent->name);
            }
            break;
        }

        if (method->body)
          visit_expr(*method->body->expression);
      });
    }

    void visit_field(Field* fld) final
    {
      if (fld->parent->kind->value() == Entity::Primitive)
      {
        report(
          context_, *fld, DiagnosticKind::Error, Diagnostic::FieldInPrimitive);
      }

      fld->type = visit_type_expression(*fld->type_expression);
    }

    void visit_define_local(DefineLocalExpr& e) final
    {
      if (e.right)
        visit_expr(*e.right);

      add_symbol(e.local.get());
    }

    void visit_new_expr(NewExpr& e) final
    {
      Symbol class_symbol = resolve(e.class_name);
      if (auto definition = std::get_if<const Entity*>(&class_symbol);
          definition && (*definition)->kind->value() == Entity::Class)
      {
        e.definition = *definition;
      }
      else if (!std::holds_alternative<ErrorSymbol>(class_symbol))
      {
        report(
          context_,
          e.class_name,
          DiagnosticKind::Error,
          Diagnostic::SymbolNotClass,
          e.class_name);
      }

      if (e.parent)
      {
        if (auto local = resolve_local(e.parent->name))
        {
          e.parent->local = *local;
        }
      }

      RecursiveExprVisitor<>::visit_new_expr(e);
    }

    void visit_symbol(SymbolExpr& e) final
    {
      e.symbol = resolve(e.name);
    }

    void visit_assign_local(AssignLocalExpr& e) final
    {
      if (auto local = resolve_local(e.name))
      {
        e.local = *local;
      }

      visit_expr(*e.right);
    }

    void visit_match_expr(MatchExpr& e) final
    {
      visit_expr(*e.expr);

      for (auto& arm : e.arms)
      {
        arm->type = visit_type_expression(*arm->type_expression);
        push_scope([&]() {
          add_symbol(arm->local.get());
          visit_expr(*arm->expr);
        });
      }
    }

    void visit_when_argument(WhenArgument& argument)
    {
      if (auto as = dynamic_cast<WhenArgumentAs*>(&argument))
      {
        visit_expr(*as->inner);
      }
      else if (auto shadow = dynamic_cast<WhenArgumentShadow*>(&argument))
      {
        if (auto local = resolve_local(shadow->binder->name))
        {
          shadow->local = *local;
        }
      }
      else
      {
        abort();
      }
    }

    void visit_when(WhenExpr& expr) final
    {
      for (auto& cown : expr.cowns)
      {
        visit_when_argument(*cown);
      }
      push_closure_scope([&]() {
        for (auto& rb : expr.cowns)
        {
          add_symbol(rb->get_binder());
        }
        visit_expr(*expr.body);
        expr.captures = std::move(scopes_.back().captures_);
      });
    }

    void visit_if(IfExpr& expr) final
    {
      visit_expr(*expr.condition);
      push_scope([&]() { visit_expr(*expr.then_block); });
      if (expr.else_block)
        push_scope([&]() { visit_expr(*expr.else_block->body); });
    }

    void visit_while(WhileExpr& expr) final
    {
      visit_expr(*expr.condition);
      push_scope([&]() { visit_expr(*expr.body); });
    }

    void visit_block(BlockExpr& expr) final
    {
      push_scope([&]() { visit_expr(*expr.inner); });
    }

    TypePtr visit_string_type_expr(StringTypeExpr& te)
    {
      return context_.mk_string_type();
    }

    TypePtr visit_symbol_type_expr(SymbolTypeExpr& te)
    {
      te.symbol = resolve(te.name);

      if (std::holds_alternative<ErrorSymbol>(te.symbol))
      {
        return context_.mk_top();
      }
      else if (auto definition = std::get_if<const Entity*>(&te.symbol))
      {
        if (te.arguments.size() != (*definition)->generics->types.size())
        {
          report(
            context_,
            te,
            DiagnosticKind::Error,
            Diagnostic::IncorrectNumberOfTypeArguments,
            te.name,
            (*definition)->generics->types.size(),
            te.arguments.size());
          return context_.mk_top();
        }

        TypeList arguments;
        for (const auto& argument : te.arguments)
        {
          arguments.push_back(visit_type_expression(*argument));
        }

        return context_.mk_entity_type(*definition, arguments);
      }
      else if (
        auto definition = std::get_if<const TypeParameterDef*>(&te.symbol))
      {
        if (!te.arguments.empty())
        {
          report(
            context_,
            te,
            DiagnosticKind::Error,
            Diagnostic::CannotApplyTypeParametersToTypeVariable,
            te.name);
        }

        TypePtr ty =
          context_.mk_type_parameter(*definition, TypeParameter::Expanded::No);

        switch ((*definition)->kind())
        {
          case TypeParameterDef::Any:
            return ty;
          case TypeParameterDef::Class:
            return context_.mk_entity_of(ty);
            EXHAUSTIVE_SWITCH
        }
      }
      else
      {
        report(
          context_,
          te,
          DiagnosticKind::Error,
          Diagnostic::SymbolNotType,
          te.name);
        return context_.mk_top();
      }
    }

    TypePtr visit_capability_type_expr(CapabilityTypeExpr& te)
    {
      switch (te.kind->value())
      {
        case CapabilityKind::Mutable:
          return context_.mk_mutable(RegionHole());
        case CapabilityKind::Immutable:
          return context_.mk_immutable();
        case CapabilityKind::Isolated:
          return context_.mk_isolated(RegionHole());
        case CapabilityKind::Subregion:
          // sub does not exist in the source syntax.
          abort();

          EXHAUSTIVE_SWITCH
      }
    }

    TypePtr visit_union_type_expr(UnionTypeExpr& te)
    {
      TypeSet elements;
      for (const auto& element : te.elements)
      {
        elements.insert(visit_type_expression(*element));
      }
      return context_.mk_union(elements);
    }

    TypePtr visit_intersection_type_expr(IntersectionTypeExpr& te)
    {
      TypeSet elements;
      for (const auto& element : te.elements)
      {
        elements.insert(visit_type_expression(*element));
      }
      return context_.mk_intersection(elements);
    }

    TypePtr visit_viewpoint_type_expr(ViewpointTypeExpr& te)
    {
      return context_.mk_viewpoint(
        visit_type_expression(*te.left), visit_type_expression(*te.right));
    }

    struct Scope
    {
      explicit Scope(size_t next_type_param, bool is_closure_scope = false)
      : next_type_param(next_type_param), is_closure_scope(is_closure_scope)
      {}

      std::unordered_map<std::string, Symbol> symbols;

      bool is_closure_scope;
      std::vector<LocalID> captures_;

      // Used to assign indices to each type parameter
      size_t next_type_param;
    };
    std::vector<Scope> scopes_;

    // Used to assign indices to each assertion
    size_t next_assertion_index_ = 0;

    Context& context_;
  };

  bool name_resolution(Context& context, Program* program)
  {
    ResolutionVisitor visitor(context);
    visitor.visit_program(program);
    return !context.have_errors_occurred();
  }

  TypePtr lookup_field_type(
    Context& context, const TypePtr& ty, const std::string& name)
  {
    if (auto entity = ty->dyncast<EntityType>())
    {
      if (const Field* field = lookup_member<Field>(entity->definition, name))
        return Instantiation(entity->arguments).apply(context, field->type);
    }

    return nullptr;
  }

  TypeSet lookup_field_types(
    Context& context, const TypeSet& types, const std::string& name)
  {
    TypeSet result;
    for (const auto& elem : types)
    {
      if (TypePtr field = lookup_field_type(context, elem, name))
      {
        result.insert(field);
      }
    }
    return result;
  }

  std::optional<std::pair<TypeList, const FnSignature*>>
  lookup_method_signature(const TypePtr& ty, const std::string& name)
  {
    if (auto entity = ty->dyncast<EntityType>())
    {
      if (const auto* method = lookup_member<Method>(entity->definition, name))
      {
        return std::make_pair(entity->arguments, method->signature.get());
      }
    }
    else if (auto static_ = ty->dyncast<StaticType>())
    {
      if (const auto* method = lookup_member<Method>(static_->definition, name))
      {
        // We cannot lookup non-static methods on static types
        if (method->signature->receiver != nullptr)
          return std::nullopt;

        return std::make_pair(static_->arguments, method->signature.get());
      }
    }

    return std::nullopt;
  }
}
