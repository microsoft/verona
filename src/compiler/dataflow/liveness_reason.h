// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/backwards_analysis.h"
#include "compiler/dataflow/use_def.h"
#include "compiler/dataflow/variable_set.h"
#include "compiler/ir/ir.h"

/**
 * Compute liveness information for the IR, with additional information about
 * why a variable is live.
 *
 * TODO: There's a lot of duplication here with the "standard" liveness analysis
 * (in particular the incremental liveness bit) that could be factored out.
 *
 * We might not even want to keep both analyses, and instead use the more
 * powerful LivenessReason everywhere. Or we could only compute the standard
 * liveness until we notice an error, then compute the LivenessReason in order
 * to get a useful message.
 */
namespace verona::compiler
{
  struct LivenessReasonState
  {
    /**
     * For each live variable, we track the set of source ranges that are
     * causing it to be live.
     */
    std::unordered_map<Variable, std::set<SourceManager::SourceRange>>
      live_variables;

    size_t size() const
    {
      size_t total = 0;
      for (const auto& [_, ranges] : live_variables)
      {
        total += ranges.size();
      }
      return total;
    }

    void kill_variable(Variable variable)
    {
      // LivenessReason doesn't track zombie variables
    }

    void use_variable(IRInput input)
    {
      live_variables[input.variable].insert(input.source_range);
    }

    void define_variable(Variable variable)
    {
      live_variables.erase(variable);
    }

    // We handle Phi nodes through renaming in `propagate`.
    void phi_inputs(const std::vector<Variable>& vs) {}
    void phi_outputs(const std::vector<Variable>& vs) {}

    static void propagate(
      const BasicBlock* successor,
      const LivenessReasonState& state_in,
      const BasicBlock* current,
      LivenessReasonState& state_out)
    {
      auto renaming = VariableRenaming::backwards(current, successor);

      for (const auto& [variable, ranges] : state_in.live_variables)
      {
        auto [it, inserted] =
          state_out.live_variables.insert({renaming.apply(variable), ranges});
        if (!inserted)
        {
          it->second.insert(ranges.begin(), ranges.end());
        }
      }
    }
  };

  struct LivenessReasonAnalysis : public DataflowResults<LivenessReasonState>
  {
    const LivenessReasonState& live_in(const BasicBlock* bb) const
    {
      return state_in(bb);
    }

    LivenessReasonState live_out(const BasicBlock* bb) const
    {
      LivenessReasonState state;
      bb->visit_successors([&](const BasicBlock* successor) {
        LivenessReasonState::propagate(
          successor, live_in(successor), bb, state);
      });
      return state;
    }

    /**
     * Get the live-out of each statement.
     */
    std::vector<LivenessReasonState>
    statements_live_out(const BasicBlock* bb) const
    {
      LivenessReasonState state = live_out(bb);
      std::vector<LivenessReasonState> result;
      for (auto it = bb->statements.rbegin(); it != bb->statements.rend(); it++)
      {
        result.push_back(state);
        UseDefVisitor<LivenessReasonState&>(state).visit_stmt(*it);
      }
      std::reverse(result.begin(), result.end());
      return result;
    }
  };

  struct ComputeLivenessReason : public DataflowAnalysis<
                                   ComputeLivenessReason,
                                   LivenessReasonState,
                                   LivenessReasonAnalysis>
  {
    bool has_changed(
      const LivenessReasonState& old_state,
      const LivenessReasonState& new_state)
    {
      assert(new_state.size() >= old_state.size());
      return new_state.size() > old_state.size();
    }

    void propagate(
      const BasicBlock* successor,
      const LivenessReasonState& state_in,
      const BasicBlock* current,
      LivenessReasonState& state_out)
    {
      LivenessReasonState::propagate(successor, state_in, current, state_out);
    }

    void compute(const BasicBlock* bb, LivenessReasonState& state)
    {
      UseDefVisitor<LivenessReasonState&>(state).visit_basic_block(bb);
    }
  };
}
