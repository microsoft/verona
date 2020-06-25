// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/solver.h"

#include "compiler/printing.h"
#include "compiler/recursive_visitor.h"

#include <ctime>
#include <fmt/ostream.h>
#include <fstream>
#include <iomanip>

namespace verona::compiler
{
  struct SolverState
  {
    Constraints constraints;

    Substitution substitution;
    Assumptions assumptions;

    uint64_t steps = 0;
    uint64_t depth = 0;

    explicit SolverState(Constraints constraints) : constraints(constraints) {}

    void apply_substitution(Context& context, const Substitution& s)
    {
      constraints = s.apply(context, constraints);
      assumptions = s.apply(context, assumptions);
      s.apply_to(context, &substitution);
    }

    void add_constraints(Constraints cs)
    {
      constraints.insert(constraints.end(), cs.begin(), cs.end());
    }

    void add_constraint(Constraint c)
    {
      constraints.push_back(c);
    }

    bool done()
    {
      return constraints.empty();
    }

    Constraint pop_constraint()
    {
      Constraint c = constraints.back();
      constraints.pop_back();
      depth = c.depth;
      return c;
    }
  };

  Solver::Solution
  Solver::Solution::extend(Context& context, const Solution& other) const
  {
    Substitution substitution = this->substitution;
    other.substitution.apply_to(context, &substitution);

    return Solution{substitution};
  }

  Solver::SolutionSet
  Solver::solve_all(Constraints constraints, SolverMode mode)
  {
    SolutionSet solutions;
    solutions.insert(Solution());

    output_ << "------------" << std::endl;

    for (const Constraint& c : constraints)
    {
      output_ << "solutions found: " << solutions.size() << std::endl;
      output_ << "------------" << std::endl;

      SolutionSet next_solutions;
      for (const Solution& current : solutions)
      {
        if (!current.substitution.is_trivial())
        {
          output_ << "Current substitution:" << std::endl;
          current.substitution.print(output_);
          output_ << "------------" << std::endl;
        }

        Constraint to_solve = current.substitution.apply(context_, c);
        output_ << "Solving " << to_solve << std::endl;
        SolutionSet results = solve_one(to_solve, mode);
        output_ << "------------" << std::endl;

        bool found_trivial = false;
        for (const Solution& result : results)
        {
          if (result.substitution.is_trivial())
          {
            next_solutions.insert(current.extend(context_, result));
            found_trivial = true;
            break;
          }
        }

        if (!found_trivial)
        {
          for (const Solution& result : results)
          {
            next_solutions.insert(current.extend(context_, result));
          }
        }
      }
      solutions = next_solutions;

      if (solutions.empty())
        break;
    }

    return solutions;
  }

  void Solver::print_stats(const SolutionSet& solutions)
  {
    fmt::print(output_, "Done in {} steps.\n", total_steps_);
    fmt::print(output_, "Found {} solutions.\n", solutions.size());
  }

  Solver::SolutionSet Solver::solve_one(Constraint initial, SolverMode mode)
  {
    SolutionSet solutions;

    std::vector<SolverState> state_stack;
    state_stack.emplace_back(SolverState({initial}));

    while (!state_stack.empty())
    {
      SolverState& state = state_stack.back();

      if (state.done())
      {
        trace(state, "  done");
        solutions.insert(Solution{state.substitution});
        state_stack.pop_back();
        continue;
      }

      Constraint c = state.pop_constraint();
      total_steps_++;
      state.steps++;

      trace(state, c);

      if (state.assumptions.find(c) != state.assumptions.end())
      {
        // Do nothing
        trace(state, "  assumed");
        continue;
      }

      auto solution = Constraint::solve(c, mode, context_);
      if (!solution)
      {
        trace(state, "  Cannot solve constraint ", c);
        state_stack.pop_back();
        if (!state_stack.empty())
        {
          trace(state_stack.back(), " backtracking...");
        }
        continue;
      }

      // This call can invalidate `state`, by modifying state_stack
      std::visit(
        [&](const auto& inner) { apply_solution(c, inner, &state_stack); },
        solution.value());
    }

    return solutions;
  }

  void Solver::apply_solution(
    const Constraint& constraint,
    const Constraint::Trivial& solution,
    std::vector<SolverState>* state_stack)
  {}

  void Solver::apply_solution(
    const Constraint& constraint,
    const Substitution& substitution,
    std::vector<SolverState>* state_stack)
  {
    assert(!state_stack->empty());

    SolverState& state = state_stack->back();
    state.assumptions.insert(constraint);
    for (auto it : substitution.types())
    {
      trace(state, "  ", *it.first, " --> ", *it.second);
    }
    for (auto it : substitution.sequences())
    {
      trace(state, "  ", it.first, " --> ", it.second);
    }
    state.apply_substitution(context_, substitution);
  }

  void Solver::apply_solution(
    const Constraint& constraint,
    const Constraint::Compound& solution,
    std::vector<SolverState>* state_stack)
  {
    assert(!state_stack->empty());
    SolverState& state = state_stack->back();

    state.add_constraints(solution.subconstraints);
    state.assumptions.insert(
      solution.assumptions.begin(), solution.assumptions.end());

    for (auto it : solution.substitution.types())
    {
      trace(state, "  ", *it.first, " --> ", *it.second);
    }
    for (auto it : solution.substitution.sequences())
    {
      trace(state, "  ", it.first, " --> ", it.second);
    }
    state.apply_substitution(context_, solution.substitution);
  }

  void Solver::apply_solution(
    const Constraint& constraint,
    const Constraint::Backtracking& solution,
    std::vector<SolverState>* state_stack)
  {
    assert(!state_stack->empty());

    SolverState state = state_stack->back();
    state_stack->pop_back();

    trace(state, " with backtracking ", solution.subconstraints.size());
    for (const Constraint& subconstraint : solution.subconstraints)
    {
      SolverState substate = state;
      substate.add_constraint(subconstraint);
      state_stack->push_back(substate);
    }
  }

  template<typename... Args>
  void Solver::trace(const SolverState& state, const Args&... args)
  {
    if (output_.good())
    {
      output_ << " " << std::setw(3) << state.steps << ": "
              << std::string(2 * state.depth, ' ');
      (output_ << ... << args) << std::endl;
    }
  }
}
