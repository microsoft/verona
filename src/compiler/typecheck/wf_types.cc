// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/wf_types.h"

#include "compiler/instantiation.h"
#include "compiler/recursive_visitor.h"
#include "compiler/typecheck/constraint.h"
#include "compiler/typecheck/solver.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  /**
   * The WF Types pass checks that all type arguments match their bound.
   *
   * This has to be a separate pass from resolution and elaboration because
   * subtype checking requires a fully formed AST.
   */
  class WfTypesVisitor
  : private RecursiveTypeVisitor<SourceManager::SourceRange, std::ostream&>,
    private MemberVisitor<>,
    private RecursiveExprVisitor<std::ostream&>
  {
  public:
    WfTypesVisitor(Context& context) : context_(context) {}

    void visit_program(Program* program)
    {
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
    }

  private:
    /**
     * This is the core of the WF pass. Everything else is just about finding
     * types in the AST.
     *
     * We use the solver in "Verify" mode to check that every type is a subtype
     * of its bound.
     *
     * Types don't currently carry any source information, so we use the
     * location of the top-level type expression, passed as an argument to the
     * visitor. This is not as precise as we'd want it to be.
     *
     * For example for a field declaration:
     *
     *   foo: Cell[A] & mut;
     *
     * If A doesn't satify the bound, the entire type expression,
     * `Cell[A] & mut` is used to report the error.
     */
    void visit_entity_type(
      const EntityTypePtr& ty,
      SourceManager::SourceRange source_range,
      std::ostream& solver_output) final
    {
      RecursiveTypeVisitor::visit_entity_type(ty, source_range, solver_output);

      const Entity* definition = ty->definition;

      Instantiation instantiation(ty->arguments);

      for (const auto& [generic, argument] :
           safe_zip(definition->generics->types, ty->arguments))
      {
        // We need to substitute the type arguments in the bounds themselves
        // (using the Instantiation) in order to support F-bounded polymorphism.
        TypePtr bound = instantiation.apply(context_, generic->bound);

        Constraint constraint(argument, bound, 0, context_);

        Solver solver(context_, solver_output);
        Solver::SolutionSet solutions =
          solver.solve_one(constraint, SolverMode::Verify);
        solver.print_stats(solutions);

        if (solutions.empty())
        {
          report(
            context_,
            source_range,
            DiagnosticKind::Error,
            Diagnostic::TypeArgumentDoesNotSatisfyBound,
            *argument,
            definition->name,
            *generic->bound);
        }
      }
    }

    void visit_entity(Entity* entity)
    {
      auto output = context_.dump(entity->path(), "wf-types");
      visit_generics(entity->generics.get(), *output);
      visit_members(entity->members);
    }

    void visit_assertion(StaticAssertion* assertion)
    {
      auto output = context_.dump("assertion", assertion->index, "wf-types");

      visit_generics(assertion->generics.get(), *output);

      visit_type(
        assertion->left_type,
        assertion->left_expression->source_range,
        *output);

      visit_type(
        assertion->right_type,
        assertion->right_expression->source_range,
        *output);
    }

    void visit_field(Field* fld) final
    {
      auto output = context_.dump(fld->path(), "wf-types");
      visit_type(fld->type, fld->type_expression->source_range, *output);
    }

    void visit_signature(FnSignature* signature, std::ostream& output)
    {
      visit_generics(signature->generics.get(), output);

      if (signature->receiver)
        visit_type(
          signature->types.receiver,
          signature->receiver->type_expression->source_range,
          output);

      for (const auto& [ty, param] :
           safe_zip(signature->types.arguments, signature->parameters))
      {
        visit_type(ty, param->type_expression->source_range, output);
      }

      if (signature->return_type_expression)
        visit_type(
          signature->types.return_type,
          signature->return_type_expression->source_range,
          output);
    }

    void visit_method(Method* method) final
    {
      auto output = context_.dump(method->path(), "wf-types");

      visit_signature(method->signature.get(), *output);
      if (method->body)
        visit_expr(*method->body->expression, *output);
    }

    void visit_generics(Generics* generics, std::ostream& output)
    {
      for (auto& param : generics->types)
      {
        if (param->bound_expression)
          visit_type(
            param->bound, param->bound_expression->source_range, output);
      }
    }

    void visit_match_expr(MatchExpr& e, std::ostream& output) final
    {
      RecursiveExprVisitor::visit_match_expr(e, output);
      for (auto& arm : e.arms)
      {
        visit_type(arm->type, arm->type_expression->source_range, output);
      }
    }

    Context& context_;
  };

  bool check_wf_types(Context& context, Program* program)
  {
    WfTypesVisitor visitor(context);
    visitor.visit_program(program);
    return !context.have_errors_occurred();
  }
}
