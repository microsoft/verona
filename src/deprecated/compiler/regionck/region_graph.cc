// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/regionck/region_graph.h"

#include "compiler/format.h"
#include "compiler/typecheck/solver.h"
#include "compiler/typecheck/typecheck.h"
#include "compiler/visitor.h"

namespace verona::compiler
{
  void dump_region_graphs(
    Context& context, const Method& method, const RegionGraphs& graphs)
  {
    auto out = context.dump(method.path(), "region-graph");
    fmt::print(*out, "Region Graphs for {}\n", method.path());

    for (const auto& [bb, graph] : graphs)
    {
      fmt::print(*out, "  {}:\n", *bb);

      size_t padding = 0;
      for (const auto& [from, _] : graph.outgoing_edges)
      {
        padding = std::max(padding, fmt::formatted_size("{}", from));
      }

      for (const auto& [from, edges] : graph.outgoing_edges)
      {
        bool first = true;
        for (const auto& [to, kind] : edges)
        {
          char kind_char;
          switch (kind)
          {
            case RegionGraph::Edge::Direct:
              kind_char = ' ';
              break;
            case RegionGraph::Edge::Indirect:
              kind_char = '*';
              break;
            case RegionGraph::Edge::Maybe:
              kind_char = '?';
              break;
          }
          if (first)
            fmt::print(
              *out, "   {:>{}} ->{} {}\n", from, padding, kind_char, to);
          else
            fmt::print(*out, "   {:>{}} ->{} {}\n", "", padding, kind_char, to);
          first = false;
        }
      }
    }
  }

  std::optional<RegionGraph::Edge> compute_edge(
    Context& context,
    Solver& solver,
    Variable variable,
    TypePtr type,
    Region region)
  {
    auto is_subtype = [&](TypePtr rhs) {
      Constraint constraint(type, rhs, 0, context);
      return !solver.solve_all({constraint}, SolverMode::MakeRegionGraph)
                .empty();
    };

    if (is_subtype(context.mk_not_child_of(region)))
    {
      return std::nullopt;
    }
    if (is_subtype(context.mk_mutable(region)))
    {
      return RegionGraph::Edge::Direct;
    }
    else if (is_subtype(context.mk_subregion(region)))
    {
      return RegionGraph::Edge::Indirect;
    }
    else
    {
      return RegionGraph::Edge::Maybe;
    }

    return std::nullopt;
  }

  std::unique_ptr<RegionGraphs> make_region_graphs(
    Context& context, const Method& method, const TypecheckResults& typecheck)
  {
    auto output = context.dump(method.path(), "make-region-graph");
    Solver solver(context, *output);

    std::unique_ptr<RegionGraphs> result = std::make_unique<RegionGraphs>();
    for (const auto& [bb, assignment] : typecheck.types)
    {
      RegionGraph& graph = (*result)[bb];
      for (const auto& [variable, ty] : assignment)
      {
        for (const auto& [parent, _] : assignment)
        {
          Region parent_region = RegionVariable{parent};
          RegionGraph& graph = (*result)[bb];
          if (
            auto edge =
              compute_edge(context, solver, variable, ty, parent_region))
          {
            graph.outgoing_edges[variable][parent] = *edge;
            graph.incoming_edges[parent][variable] = *edge;
          }
        }
      }
    }

    dump_region_graphs(context, method, *result);
    return result;
  }
}
