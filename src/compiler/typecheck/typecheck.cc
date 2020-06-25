// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/typecheck.h"

#include "compiler/format.h"
#include "compiler/polarize.h"
#include "compiler/typecheck/indirect_type.h"
#include "compiler/typecheck/solver.h"
#include "compiler/zip.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  void dump_solutions(
    Context& context,
    const Method* method,
    const Solver::SolutionSet& solutions)
  {
    std::string path = method->path();
    auto output = context.dump(path, "substitution");

    int i = 0;
    for (const auto& solution : solutions)
    {
      fmt::print(*output, "Substitution {} for {}:\n", i, path);
      for (auto it : solution.substitution.types())
      {
        fmt::print(
          *output,
          " {} --> {}\n",
          *it.first,
          *Flatten::apply(context, it.second));
      }
      for (auto it : solution.substitution.sequences())
      {
        fmt::print(
          *output,
          " {} --> {}\n",
          it.first,
          Flatten::apply(context, it.second));
      }
      *output << "\n";
      i++;
    }
  }

  Solver::SolutionSet find_solutions(
    Context& context, const Method* method, const InferResults& inference)
  {
    std::string path = method->path();
    auto output = context.dump(path, "solver");
    *output << "Solver trace for " << path << ":" << std::endl;

    Solver solver(context, *output);
    Solver::SolutionSet solutions =
      solver.solve_all(inference.constraints, SolverMode::Infer);
    solver.print_stats(solutions);

    *output << "\n";

    dump_solutions(context, method, solutions);

    return solutions;
  }

  std::unique_ptr<TypecheckResults> typecheck(
    Context& context, const Method* method, const InferResults& inference)
  {
    auto solutions = find_solutions(context, method, inference);
    if (solutions.empty())
      return nullptr;

    const Solver::Solution& solution = *solutions.begin();

    auto results = std::make_unique<TypecheckResults>();
    auto apply_solution = [&](const auto& v) {
      auto subst = solution.substitution.apply(context, v);
      auto flattened = Flatten::apply(context, subst);
      return context.polarizer().apply(flattened, Polarity::Positive);
    };

    results->types = apply_solution(inference.types);

    for (auto& [bb, assignment] : results->types)
    {
      for (auto& [var, type] : assignment)
      {
        type = SimplifyIndirectTypes(context, results->types).expand(bb, var);
      }
    }

    SimplifyIndirectTypes simplify(context, results->types);
    for (const auto& [id, arguments] : inference.type_arguments)
    {
      InferableTypeSequence applied_arguments =
        simplify.apply(apply_solution(arguments));

      if (
        BoundedTypeSequence* bounded =
          std::get_if<BoundedTypeSequence>(&applied_arguments))
      {
        results->type_arguments.insert({id, bounded->types});
      }
      else
      {
        std::cerr << "Did not infer type arguments in " << method->name
                  << std::endl;
        abort();
      }
    }

    dump_types(context, *method, "types", "Types", results->types);

    return results;
  }
}
