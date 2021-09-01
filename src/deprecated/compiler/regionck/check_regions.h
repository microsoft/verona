// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/dataflow/liveness_reason.h"
#include "compiler/typecheck/capability_predicate.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  class CheckRegions
  {
  public:
    CheckRegions(
      Context& context,
      const TypecheckResults& typecheck,
      const RegionGraphs& region_graphs)
    : context_(context), typecheck_(typecheck), region_graphs_(region_graphs)
    {}

    void process(const MethodIR& mir)
    {
      auto liveness = ComputeLivenessReason::run(mir);

      for (auto& ir : mir.function_irs)
      {
        IRTraversal traversal(*ir);
        while (BasicBlock* bb = traversal.next())
        {
          const TypeAssignment& assignment = typecheck_.types.at(bb);
          const RegionGraph& graph = region_graphs_.at(bb);
          std::vector<LivenessReasonState> live_outs =
            liveness->statements_live_out(bb);

          for (const auto& [stmt, live_out] :
               safe_zip(bb->statements, live_outs))
          {
            std::visit(
              [&, &live_out = live_out](const auto& s) {
                visit_stmt(assignment, graph, live_out, s);
              },
              stmt);
          }

          // We don't need to visit the terminator because only ReturnTerminator
          // consumes a variable, but there can never be anything live out of
          // it.
        }
      }
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const CallStmt& stmt)
    {
      // TODO:
      // This is not catching things like m(x, x), m(x, x.f) or m(x.f, x),
      // because it only considers live_out, which doesn't cover the other uses
      // within the same statement.
      //
      // Additionally, if an argument/receiver is mutable, it also needs to
      // invalidate sibling / child regions (like WriteFieldStmt).
      consume_variable(assignment, graph, live_out, stmt.receiver);
      consume_variables(assignment, graph, live_out, stmt.arguments);
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const StaticTypeStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const NewStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const MatchBindStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const ReadFieldStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const WriteFieldStmt& stmt)
    {
      // TODO:
      // This is not catching things like x.f = x, or (x.f).g = x.
      // We would need to treat stmt.base as live, in addition to live_out.
      //
      // It also needs to invalidate any children / sibling regions of
      // stmt.base.
      consume_variable(assignment, graph, live_out, stmt.right);
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const ViewStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const WhenStmt& stmt)
    {
      consume_variables(assignment, graph, live_out, stmt.cowns);
      consume_variables(assignment, graph, live_out, stmt.captures);
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const CopyStmt& stmt)
    {
      consume_variable(assignment, graph, live_out, stmt.input);
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const IntegerLiteralStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const StringLiteralStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const UnitStmt& stmt)
    {}

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const EndScopeStmt& stmt)
    {
      for (Variable dead_variable : stmt.dead_variables)
      {
        // TODO:
        // stmt.source_range refers to the entire scope's range, which produces
        // confusing error messages. We should be pointing at just the last
        // line, or closing brace.
        kill_variable(
          assignment,
          graph,
          live_out,
          dead_variable,
          stmt.source_range,
          Diagnostic::ParentWasOverwrittenHere,
          dead_variable);
      }
    }

    void visit_stmt(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const OverwriteStmt& stmt)
    {
      kill_variable(
        assignment,
        graph,
        live_out,
        stmt.dead_variable,
        stmt.source_range,
        Diagnostic::ParentWasOverwrittenHere,
        stmt.dead_variable);
    }

    template<typename... Args>
    void kill_variable(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      Variable dead_variable,
      SourceManager::SourceRange reason_range,
      Diagnostic reason_diagnostic,
      Args&&... reason_args)
    {
      TypePtr ty = assignment.at(dead_variable);
      PredicateSet predicates = predicates_for_type(ty);

      // If the type is non-linear, we can do path-compression and keep using
      // the children.
      if (predicates.contains(CapabilityPredicate::NonLinear))
        return;

      check_children_not_live(
        graph,
        live_out,
        dead_variable,
        reason_range,
        reason_diagnostic,
        std::forward<Args>(reason_args)...);
    }

    void consume_variable(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      IRInput input)
    {
      TypePtr ty = assignment.at(input);
      PredicateSet predicates = predicates_for_type(ty);
      if (predicates.contains(CapabilityPredicate::NonLinear))
        return;

      check_not_live(
        live_out,
        input.variable,
        input.source_range,
        Diagnostic::WasPreviouslyConsumedHere,
        input.variable);

      check_children_not_live(
        graph,
        live_out,
        input.variable,
        input.source_range,
        Diagnostic::ParentWasConsumedHere,
        input.variable);
    }

    void consume_variables(
      const TypeAssignment& assignment,
      const RegionGraph& graph,
      const LivenessReasonState& live_out,
      const std::vector<IRInput>& inputs)
    {
      for (const auto& input : inputs)
      {
        consume_variable(assignment, graph, live_out, input);
      }
    }

    template<typename... Args>
    void check_not_live(
      const LivenessReasonState& live,
      Variable variable,
      SourceManager::SourceRange reason_range,
      Diagnostic reason_diagnostic,
      Args&&... reason_args)
    {
      if (auto it = live.live_variables.find(variable);
          it != live.live_variables.end())
      {
        for (const auto& source_range : it->second)
        {
          report(
            context_,
            source_range,
            DiagnosticKind::Error,
            Diagnostic::CannotUseVariable,
            variable);
          report(
            context_,
            reason_range,
            DiagnosticKind::Note,
            reason_diagnostic,
            std::forward<Args>(reason_args)...);
        }
      }
    }

    template<typename... Args>
    void check_children_not_live(
      const RegionGraph& graph,
      const LivenessReasonState& live,
      Variable root,
      SourceManager::SourceRange reason_range,
      Diagnostic reason_diagnostic,
      Args&&... reason_args)
    {
      std::unordered_set<Variable> visited;
      std::vector<Variable> todo;

      /**
       * We walk the region graph to find all transitive direct and indirect
       * children, and call check_not_live on them.
       */

      auto push_children = [&](Variable v) {
        if (auto it = graph.incoming_edges.find(v);
            it != graph.incoming_edges.end())
        {
          for (auto [child, _] : it->second)
          {
            if (visited.find(child) == visited.end())
              todo.push_back(child);
          }
        }
      };

      push_children(root);
      while (!todo.empty())
      {
        Variable current = todo.back();
        todo.pop_back();

        if (auto [_, inserted] = visited.insert(current); inserted)
        {
          check_not_live(
            live, current, reason_range, reason_diagnostic, reason_args...);
          push_children(current);
        }
      }
    }

  private:
    Context& context_;
    const TypecheckResults& typecheck_;
    const RegionGraphs& region_graphs_;
  };
}
