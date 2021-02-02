// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/assertion.h"

#include "compiler/context.h"
#include "compiler/typecheck/solver.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  namespace
  {
    /**
     * Returns whether the assertion holds given the resulting solution set.
     */
    bool check_assertion_result(
      const StaticAssertion& assertion, const Solver::SolutionSet& solutions)
    {
      switch (assertion.kind->value())
      {
        case AssertionKind::Subtype:
          return !solutions.empty();
        case AssertionKind::NotSubtype:
          return solutions.empty();

          EXHAUSTIVE_SWITCH;
      }
    }

    /**
     * Given a failing assertion, report the appropriate error in the context.
     */
    void
    report_assertion_failure(Context& context, const StaticAssertion& assertion)
    {
      switch (assertion.kind->value())
      {
        case AssertionKind::Subtype:
          report(
            context,
            assertion.source_range,
            DiagnosticKind::Error,
            Diagnostic::SubtypeAssertionFailed,
            *assertion.left_type,
            *assertion.right_type);
          break;

        case AssertionKind::NotSubtype:
          report(
            context,
            assertion.source_range,
            DiagnosticKind::Error,
            Diagnostic::NotSubtypeAssertionFailed,
            *assertion.left_type,
            *assertion.right_type);
          break;

          EXHAUSTIVE_SWITCH;
      }
    }
  }

  bool
  check_static_assertion(Context& context, const StaticAssertion& assertion)
  {
    Constraint constraint(
      assertion.left_type, assertion.right_type, 0, context);

    auto output = context.dump("assertion", assertion.index, "solver");

    auto loc = context.expand_source_location(assertion.source_range.first);
    fmt::print(
      *output,
      "Checking assertion '{}' at {}:{}\n",
      constraint,
      loc.filename,
      loc.line);

    Solver solver(context, *output);
    Solver::SolutionSet solutions =
      solver.solve_one(constraint, SolverMode::Verify);
    solver.print_stats(solutions);

    bool ok = check_assertion_result(assertion, solutions);
    if (ok)
    {
      fmt::print(*output, "Assertion succeeded\n\n");
    }
    else
    {
      fmt::print(*output, "Assertion failed\n\n");
      report_assertion_failure(context, assertion);
    }

    return ok;
  }
}
