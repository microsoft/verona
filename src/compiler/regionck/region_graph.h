// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "compiler/ir/ir.h"

#include <map>
#include <unordered_map>

namespace verona::compiler
{
  /**
   * The region graph is a directed graph that tracks the relationship between
   * regions inside a function body. It is used during region checking to
   * determine what needs to be invalidated, and (TODO) during region subtyping.
   *
   * A node may have multiple outgoing edges, which together form a disjunction.
   * For instance if x has Direct edges to both y and z, then x is either in the
   * region of y or it is in the region of z (i.e. it has type mut(y) | mut(z)).
   *
   * Conjunctions aren't really handled correctly at the moment, which is mostly
   * fine because it is hard / impossible to get the IR to produce intersections
   * to different regions. Would be nice to be handling this properly though.
   *
   * The graph should be acyclic, but this is not yet the case because path
   * compression isn't properly implemented, which can lead to variables that
   * mention themselves in their own type.
   *
   * Currently only edges between RegionVariable regions are tracked. This is
   * fine because the graph is only being used to find children of a region.
   * When we start looking for sibling regions, we'll need to add edges to
   * RegionExternal as well.
   */
  struct RegionGraph
  {
    enum class Edge
    {
      /**
       * A Maybe edge from x to y means x may be in the same region or a child
       * region of y.
       */
      Maybe,

      /**
       * A Direct edge from x to y means that x is in the same region as y (i.e.
       * it has type mut(x)).
       */
      Direct,

      /**
       * A Indirect edge from x to y means that x is in the same or a child
       * region of y (i.e. it has type iso(x) | mut(x)).
       */
      Indirect,
    };

    std::map<Variable, std::map<Variable, Edge>> outgoing_edges;
    std::map<Variable, std::map<Variable, Edge>> incoming_edges;

    void add_edge(Variable from, Variable to, Edge edge);
  };
  typedef std::unordered_map<const BasicBlock*, RegionGraph> RegionGraphs;

  /**
   * Returns Indirect is either argument is Indirect.
   */
  RegionGraph::Edge operator|(RegionGraph::Edge e1, RegionGraph::Edge e2);
  RegionGraph::Edge& operator|=(RegionGraph::Edge& e1, RegionGraph::Edge e2);

  std::unique_ptr<RegionGraphs> make_region_graphs(
    Context& context, const Method& method, const TypecheckResults& typecheck);
}
