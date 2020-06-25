// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/dataflow/liveness.h"

#include "compiler/dataflow/use_def.h"
#include "compiler/dataflow/work_set.h"

namespace verona::compiler
{
  std::vector<Liveness>
  LivenessAnalysis::statements_out(const BasicBlock* block) const
  {
    std::vector<Liveness> result;
    Liveness current = state_out(block);

    // Liveness requires us to process the terminator first, followed by
    // statements in reverse order. We reverse the resulting vector at the end
    // to put it back in normal order.
    current.visit_term(*block->terminator);
    for (auto it = block->statements.rbegin(); it != block->statements.rend();
         it++)
    {
      result.push_back(current);
      current.visit_stmt(*it);
    }
    std::reverse(result.begin(), result.end());

    return result;
  }

  std::vector<Liveness>
  LivenessAnalysis::statements_in(const BasicBlock* block) const
  {
    std::vector<Liveness> result;
    Liveness current = state_out(block);

    // Liveness requires us to process the terminator first, followed by
    // statements in reverse order. We reverse the resulting vector at the end
    // to put it back in normal order.
    current.visit_term(*block->terminator);
    for (auto it = block->statements.rbegin(); it != block->statements.rend();
         it++)
    {
      current.visit_stmt(*it);
      result.push_back(current);
    }
    std::reverse(result.begin(), result.end());

    return result;
  }

  Liveness LivenessAnalysis::state_out(const BasicBlock* block) const
  {
    Liveness result;
    block->visit_successors([&](const BasicBlock* successor) {
      Liveness::propagate(successor, state_in(successor), block, result);
    });
    return result;
  }

  Liveness LivenessAnalysis::terminator_in(const BasicBlock* block) const
  {
    Liveness result = state_out(block);
    result.visit_term(*block->terminator);
    return result;
  }

  class ComputeLiveness
  : public DataflowAnalysis<ComputeLiveness, Liveness, LivenessAnalysis>
  {
  public:
    void propagate(
      const BasicBlock* successor,
      const Liveness& state_in,
      const BasicBlock* current,
      Liveness& state_out)
    {
      Liveness::propagate(successor, state_in, current, state_out);
    }

    void compute(const BasicBlock* bb, Liveness& state)
    {
      const UseDefAnalysis& use_def = get_use_def(bb);

      state.live_variables.remove_all(use_def.defs);
      state.zombie_variables.remove_all(use_def.defs);
      state.zombie_variables.remove_all(use_def.uses);

      state.live_variables.insert_all(use_def.uses);
      state.zombie_variables.insert_all(use_def.kills);
    }

    bool has_changed(const Liveness& old_value, const Liveness& new_value)
    {
      assert(new_value.size() >= old_value.size());
      return new_value.size() > old_value.size();
    }

  private:
    /**
     * Get the use-def information for this basic block.
     *
     * The result is cached such that it is only computed on the first
     * iteration.
     */
    const UseDefAnalysis& get_use_def(const BasicBlock* bb)
    {
      auto [it, inserted] = use_def_.insert({bb, UseDefAnalysis()});
      if (inserted)
      {
        it->second = UseDefAnalysis::compute(bb);
      }
      return it->second;
    }

    std::unordered_map<const BasicBlock*, UseDefAnalysis> use_def_;
  };

  std::unique_ptr<LivenessAnalysis> compute_liveness(const MethodIR& ir)
  {
    return ComputeLiveness::run(ir);
  }
}
