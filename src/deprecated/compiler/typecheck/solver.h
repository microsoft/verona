// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/typecheck/constraint.h"

namespace verona::compiler
{
  struct SolverState;
  class Solver
  {
  public:
    struct Solution
    {
      Substitution substitution;

      Solution extend(Context& context, const Solution& other) const;

      bool operator<(const Solution& other) const
      {
        return substitution < other.substitution;
      }
    };
    typedef std::set<Solution> SolutionSet;

    Solver(Context& context, std::ostream& output)
    : context_(context), output_(output)
    {}

    SolutionSet solve_all(Constraints constraints, SolverMode mode);
    SolutionSet solve_one(Constraint constraints, SolverMode mode);
    void print_stats(const SolutionSet& solutions);

  private:
    void apply_solution(
      const Constraint& constraint,
      const Constraint::Trivial& solution,
      std::vector<SolverState>* state_stack);
    void apply_solution(
      const Constraint& constraint,
      const Substitution& substitution,
      std::vector<SolverState>* state_stack);
    void apply_solution(
      const Constraint& constraint,
      const Constraint::Compound& solution,
      std::vector<SolverState>* state_stack);
    void apply_solution(
      const Constraint& constraint,
      const Constraint::Backtracking& solution,
      std::vector<SolverState>* state_stack);

    template<typename... Args>
    void trace(const SolverState& state, const Args&... args);

    uint64_t total_steps_ = 0;
    Context& context_;
    std::ostream& output_;
  };
}
