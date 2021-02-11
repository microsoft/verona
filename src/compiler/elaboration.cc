// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/elaboration.h"

#include "compiler/visitor.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  class ElaborationVisitor : public MemberVisitor<>
  {
  public:
    ElaborationVisitor(Context& context) : context_(context) {}

    void visit_program(Program* program)
    {
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
      visit_members(entity->members);
    }

    /**
     * Creates a mapping from local identifiers to Region. Both the receiver and
     * the other arguments are added to that map. The former maps to
     * RegionReceiver() where as the arguments map to RegionParameter(0),
     * RegionParameter(1), ...
     *
     * This is used to give a meaning to `x` in `where ... in x` clauses.
     */
    typedef std::unordered_map<LocalID, Region> RegionMap;
    RegionMap build_region_map(FnSignature* signature)
    {
      std::unordered_map<LocalID, Region> regions;

      if (signature->receiver)
        regions.insert({signature->receiver->local, RegionReceiver()});

      uint64_t index = 0;
      for (auto& param : signature->parameters)
      {
        regions.insert({param->local, RegionParameter(index++)});
      }

      return regions;
    }

    /**
     * An easier to work with representation of a function's where clauses.
     */
    struct ClauseMap
    {
      // The map contains (x, y) if `where x in y`.
      // TODO: This should be acyclic, but we don't enforce that yet.
      std::unordered_map<LocalID, const WhereClause&> parameters;

      // Equal to (Kind, x) if `where return Kind y`, where Kind is either `in`,
      // `under` or `from`.
      std::optional<std::reference_wrapper<const WhereClause>> return_;
    };

    void add_return_clause(ClauseMap* result, const WhereClause& clause)
    {
      if (result->return_.has_value())
      {
        report(
          context_,
          clause,
          DiagnosticKind::Error,
          Diagnostic::MultipleWhereClauses,
          "return");
        report(
          context_,
          result->return_->get(),
          DiagnosticKind::Note,
          Diagnostic::PreviousWhereClauseHere,
          "return");
      }
      else
      {
        result->return_ = clause;
      }
    }

    void add_parameter_clause(
      ClauseMap* result, const WhereClause& clause, LocalID local)
    {
      switch (clause.kind->value())
      {
        case WhereClause::In:
        {
          auto [previous, inserted] =
            result->parameters.insert({local, clause});
          if (!inserted)
          {
            report(
              context_,
              clause,
              DiagnosticKind::Error,
              Diagnostic::MultipleWhereClauses,
              local);
            report(
              context_,
              previous->second,
              DiagnosticKind::Note,
              Diagnostic::PreviousWhereClauseHere,
              local);
          }
          break;
        }

        case WhereClause::Under:
          report(
            context_,
            clause,
            DiagnosticKind::Error,
            Diagnostic::ParameterCannotBeWhereUnder);
          break;

        case WhereClause::From:
          report(
            context_,
            clause,
            DiagnosticKind::Error,
            Diagnostic::ParameterCannotBeWhereFrom);
          break;

          EXHAUSTIVE_SWITCH
      }
    }

    ClauseMap build_clause_map(const ASTList<WhereClause>& where_clauses)
    {
      ClauseMap result;

      for (const auto& clause : where_clauses)
      {
        if (
          dynamic_cast<const WhereClauseReturn*>(clause->left.get()) != nullptr)
        {
          add_return_clause(&result, *clause);
        }
        else if (
          auto param =
            dynamic_cast<const WhereClauseParameter*>(clause->left.get()))
        {
          add_parameter_clause(&result, *clause, param->local);
        }
        else
        {
          abort();
        }
      }

      return result;
    }

    void apply_parameter(
      LocalID param,
      TypePtr* type,
      const ClauseMap& clauses,
      const RegionMap& regions,
      uint64_t& next_fresh_external)
    {
      Region region;

      auto it = clauses.parameters.find(param);
      if (it != clauses.parameters.end())
        region = regions.at(it->second.right->local);
      else
        region = RegionExternal(next_fresh_external++);

      ApplyRegionType::Mode mode = ApplyRegionType::Mode::Extract;
      *type = context_.mk_apply_region(mode, region, *type);
    }

    void apply_return(
      TypePtr* type,
      const ClauseMap& clauses,
      const RegionMap& regions,
      uint64_t& next_fresh_external)
    {
      ApplyRegionType::Mode mode;
      Region region;

      if (clauses.return_)
      {
        const WhereClause& clause = clauses.return_->get();
        LocalID parent = clause.right->local;
        region = regions.at(parent);
        switch (clause.kind->value())
        {
          case WhereClause::In:
            mode = ApplyRegionType::Mode::Adapt;
            break;
          case WhereClause::Under:
            mode = ApplyRegionType::Mode::Under;
            break;
          case WhereClause::From:
            mode = ApplyRegionType::Mode::Extract;
            break;

            EXHAUSTIVE_SWITCH
        }
      }
      else
      {
        mode = ApplyRegionType::Mode::Extract;
        region = RegionExternal(next_fresh_external++);
      }

      *type = context_.mk_apply_region(mode, region, *type);
    }

    void visit_signature(FnSignature* signature)
    {
      ClauseMap clauses = build_clause_map(signature->where_clauses);
      RegionMap regions = build_region_map(signature);

      uint64_t next_fresh_external = 0;

      if (signature->receiver)
      {
        apply_parameter(
          signature->receiver->local,
          &signature->types.receiver,
          clauses,
          regions,
          next_fresh_external);
      }

      for (const auto& [param, type] :
           safe_zip(signature->parameters, signature->types.arguments))
      {
        apply_parameter(
          param->local, &type, clauses, regions, next_fresh_external);
      }

      apply_return(
        &signature->types.return_type, clauses, regions, next_fresh_external);
    }

    void visit_method(Method* method) final
    {
      visit_signature(method->signature.get());
    }

    void visit_field(Field* fld) final {}

  private:
    Context& context_;
  };

  bool elaborate(Context& context, Program* program)
  {
    ElaborationVisitor visitor(context);
    visitor.visit_program(program);

    return !context.have_errors_occurred();
  }
}
