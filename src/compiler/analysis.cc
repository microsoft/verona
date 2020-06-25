// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/analysis.h"

#include "compiler/dataflow/liveness.h"
#include "compiler/ir/print.h"
#include "compiler/regionck/check_regions.h"
#include "compiler/source_manager.h"
#include "compiler/typecheck/assertion.h"
#include "compiler/typecheck/permission_check.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  class AnalysisVisitor : private MemberVisitor<>
  {
  public:
    AnalysisVisitor(
      Context& context, const Program& program, AnalysisResults* results)
    : context_(context), program_(program), results_(results)
    {}

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
    void visit_entity(Entity* entity)
    {
      visit_members(entity->members);
    }

    void visit_assertion(StaticAssertion* assertion)
    {
      if (!check_static_assertion(context_, *assertion))
      {
        results_->ok = false;
      }
    }

    void visit_field(Field* fld) final {}

    /**
     * Check basic properties of special methods.
     *
     * Currently checks finalisers have the right signature.
     *
     * Returns false, if the method is incorrect.
     **/
    bool check_special_methods(Method* method)
    {
      if (method->is_finaliser())
      {
        // TODO: Check receiver is READONLY

        if (method->signature->parameters.size() != 0)
        {
          report(
            context_,
            *method,
            SourceManager::DiagnosticKind::Error,
            SourceManager::Diagnostic::FinaliserHasNoParameters,
            method->parent->name);

          return false;
        }
        if (method->signature->generics->types.size() != 0)
        {
          report(
            context_,
            *method,
            SourceManager::DiagnosticKind::Error,
            SourceManager::Diagnostic::FinaliserNotGeneric,
            method->parent->name);
          return false;
        }
      }
      return true;
    }

    void visit_method(Method* method) final
    {
      if (!method->body)
        return;

      if (!check_special_methods(method))
      {
        results_->ok = false;
        return;
      }

      std::string path = method->path();

      FnAnalysis& analysis = results_->functions[method];

      analysis.ir = IRBuilder::build(*method->signature, *method->body);
      IRPrinter(*context_.dump(path, "ir")).print("IR", *method, *analysis.ir);

      analysis.liveness = compute_liveness(*analysis.ir);
      IRPrinter(*context_.dump(path, "liveness"))
        .with_liveness(*analysis.liveness)
        .print("Liveness Analysis", *method, *analysis.ir);

      analysis.inference =
        infer(context_, program_, *method, *analysis.ir, *analysis.liveness);

      analysis.typecheck = typecheck(context_, method, *analysis.inference);
      if (!analysis.typecheck)
      {
        report(
          context_,
          *method,
          SourceManager::DiagnosticKind::Error,
          SourceManager::Diagnostic::InferenceFailedForMethod,
          method->name);

        results_->ok = false;
        return;
      }
      IRPrinter(*context_.dump(path, "typed-ir"))
        .with_types(*analysis.typecheck)
        .print("Typed IR", *method, *analysis.ir);

      if (!check_permissions(context_, *analysis.ir, *analysis.typecheck))
      {
        results_->ok = false;
      }

      analysis.region_graphs =
        make_region_graphs(context_, *method, *analysis.typecheck);

      CheckRegions(context_, *analysis.typecheck, *analysis.region_graphs)
        .process(*analysis.ir);
    }

    Context& context_;
    const Program& program_;
    AnalysisResults* results_;
  };

  /**
   * This visitor may be used on pre-resolution AST, which doesn't have
   * back pointers yet. We pass the parent as a visitor argument instead, and
   * don't use Member::path.
   */
  class DumpAST : private MemberVisitor<void, const Entity*>
  {
  public:
    DumpAST(Context& context, const std::string& name)
    : context_(context), name_(name)
    {}

    void visit_program(Program* program)
    {
      auto out = context_.dump(name_);
      *out << "Program AST:" << std::endl;
      *out << " " << *program << std::endl << std::endl;

      for (const auto& file : program->files)
      {
        for (const auto& entity : file->entities)
        {
          visit_entity(entity.get());
        }
      }
    }

  private:
    void visit_entity(Entity* entity)
    {
      dump_definition(*entity, entity->name);
      visit_members(entity->members, entity);
    }

    void visit_method(Method* method, const Entity* parent) final
    {
      dump_definition(*method, parent->name + "." + method->name);
    }

    void visit_field(Field* fld, const Entity* parent) final {}

    template<typename Ast>
    void dump_definition(const Ast& ast, const std::string& path)
    {
      auto out = context_.dump(path, name_);
      fmt::print(*out, "AST for {}:\n {}\n\n", path, ast);
    }

    Context& context_;
    const std::string& name_;
  };

  std::unique_ptr<AnalysisResults> analyse(Context& context, Program* program)
  {
    auto results = std::make_unique<AnalysisResults>();
    results->ok = true;

    AnalysisVisitor visitor(context, *program, results.get());
    visitor.visit_program(program);

    return results;
  }

  void dump_ast(Context& context, Program* program, const std::string& name)
  {
    DumpAST visitor(context, name);
    visitor.visit_program(program);
  }
}
