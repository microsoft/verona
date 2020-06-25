// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/backwards_analysis.h"
#include "compiler/dataflow/use_def.h"
#include "compiler/dataflow/variable_set.h"
#include "compiler/ir/ir.h"

/**
 * Compute liveness information for the IR.
 *
 * In addition to live variables, this analysis computes zombie variables. A
 * variable is zombie between its last use and the point where it is killed by
 * an out-of-scope or overwrite statements.
 */
namespace verona::compiler
{
  class ComputeLiveness;

  /**
   * Set of live and zombie variables at a given program point.
   *
   * The visit_stmt and visit_term allow incrementally compute the liveness at
   * different points, by moving backwards within a basic block.
   */
  struct Liveness
  {
    // TODO: since a variable can never be both live and zombie, it may be
    // nicer to make this a map. Alternatively we could have a live and a
    // not-dead (aka live or zombie) sets, which might be a better fit, given
    // how these are generated and used.
    VariableSet live_variables;
    VariableSet zombie_variables;

    /**
     * Provide a metric over the liveness state, which can only increase from
     * one iteration to the next.
     */
    std::pair<size_t, size_t> size() const
    {
      /**
       * The order here matters. From one iteration to the next, a variable can
       * go from being zombie to being live. It cannot go the other way.
       *
       * Therefore we want { live(), zombie(x) } < { live(x), zombie() }.
       */
      return {live_variables.size(), zombie_variables.size()};
    }

    void visit_stmt(const Statement& stmt)
    {
      UseDefVisitor<Liveness&>(*this).visit_stmt(stmt);
    }

    void visit_term(const Terminator& term)
    {
      UseDefVisitor<Liveness&>(*this).visit_term(term);
    }

    static void propagate(
      const BasicBlock* successor,
      const Liveness& state_in,
      const BasicBlock* current,
      Liveness& state_out)
    {
      auto renaming = VariableRenaming::backwards(current, successor);

      for (Variable variable : state_in.zombie_variables)
      {
        Variable renamed = renaming.apply(variable);

        // Another successor block may have marked this variable as live
        // already. We always prefer prefer live over zombie, so do nothing if
        // it is live.
        if (!state_out.live_variables.contains(renamed))
        {
          state_out.zombie_variables.insert(renamed);
        }
      }

      for (Variable variable : state_in.live_variables)
      {
        Variable renamed = renaming.apply(variable);

        state_out.live_variables.insert(renamed);

        // Similarily, if another successor has already made this variable a
        // zombie, we're now overriding it to be live. Because a variable can
        // never be both zombie and live, we clear its zombie state (which is a
        // no-op if the variable wasn't actually zombie).
        state_out.zombie_variables.remove(renamed);
      }
    }

  private:
    friend UseDefVisitor<Liveness&>;

    // These functions are used by UseDefVisitor.

    void kill_variable(Variable v)
    {
      zombie_variables.insert(v);
    }

    void use_variable(const IRInput& input)
    {
      zombie_variables.remove(input.variable);
      live_variables.insert(input.variable);
    }

    void define_variable(Variable v)
    {
      zombie_variables.remove(v);
      live_variables.remove(v);
    }

    // We handle Phi nodes through renaming in `propagate`.
    void phi_inputs(const std::vector<Variable>& vs) {}
    void phi_outputs(const std::vector<Variable>& vs) {}
  };

  /**
   * Liveness information for a function.
   *
   * Only per-basic block input liveness is stored. The output liveness and
   * per-statement can be recomputed.
   */
  struct LivenessAnalysis : public DataflowResults<Liveness>
  {
    /**
     * Get the state after each statement of the basic block.
     *
     * This method creates a new Liveness for each statement. When possible,
     * you should start with the `state_out` and walk backwards incrementally
     * instead by calling `visit_stmt`.
     */
    std::vector<Liveness> statements_out(const BasicBlock* block) const;
    std::vector<Liveness> statements_in(const BasicBlock* block) const;

    Liveness terminator_in(const BasicBlock* block) const;
    Liveness state_out(const BasicBlock* block) const;
  };

  std::unique_ptr<LivenessAnalysis> compute_liveness(const MethodIR& ir);
}
