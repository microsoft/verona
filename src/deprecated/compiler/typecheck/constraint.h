// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/substitution.h"

namespace verona::compiler
{
  enum class SolverMode
  {
    // Ignore regions, generate substitutions when encountering inference
    // variables.
    Infer,

    // Check that regions are identical, don't support inference variables.
    // This mode should never generate a non-trivial substitution.
    // TODO: Allow equivalent regions and path compression.
    Verify,

    MakeRegionGraph,
  };

  /*
   * Subtyping constraint, of the form `left <: right`.
   *
   * Contraints are generated from the AST and provided as the input to the
   * solver.
   */
  struct Constraint
  {
    struct Trivial;
    struct Compound;
    struct Backtracking;
    using Solution =
      std::variant<Trivial, Compound, Backtracking, Substitution>;

    TypePtr left;
    TypePtr right;
    std::string reason;

    /**
     * The depth at which the constraint was produced.
     *
     * Constraints produced directly from the AST have a depth of 0.
     * The depth is incremented when generating subconstraints used to solve
     * existing ones.
     *
     * This is only for debugging / tracing purposes. The solver does not
     * depend on this. It is ignored during comparison of constraints.
     */
    uint64_t depth;

    Constraint(
      TypePtr left,
      TypePtr right,
      uint64_t depth,
      Context& context,
      std::string reason = std::string());

    Constraint apply_mapper(TypeMapper<>& mapper) const;
    bool operator<(const Constraint& other) const;

    static std::optional<Solution>
    solve(const Constraint& c, SolverMode mode, Context& context);
  };

  typedef std::set<Constraint> Assumptions;
  typedef std::vector<Constraint> Constraints;

  struct Constraint::Trivial
  {};

  struct Constraint::Compound
  {
    std::vector<Constraint> assumptions;
    std::vector<Constraint> subconstraints;
    Substitution substitution;

    Compound(const Constraint& parent, Context& context)
    : depth_(parent.depth + 1), context_(&context)
    {}

    Compound& add(TypePtr left, TypePtr right)
    {
      subconstraints.emplace_back(left, right, depth_, *context_);
      return *this;
    }

    // Elements of `left` are subtypes of `right`
    Compound& add(TypeSet left, TypePtr right)
    {
      for (auto left_elem : left)
      {
        add(left_elem, right);
      }
      return *this;
    }

    // `left` is a subtype of elements of `right`
    Compound& add(TypePtr left, TypeSet right)
    {
      for (auto right_elem : right)
      {
        add(left, right_elem);
      }
      return *this;
    }

    // Pairwise subtyping of `left` and `right`.
    // The two lists must have the same length.
    Compound& add_pairwise(TypeList left, TypeList right)
    {
      assert(left.size() == right.size());

      for (auto left_it = left.begin(), right_it = right.begin();
           left_it != left.end() && right_it != right.end();
           left_it++, right_it++)
      {
        add(*left_it, *right_it);
      }

      return *this;
    }

    Compound& assume(Constraint c)
    {
      assumptions.push_back(c);
      return *this;
    }

    template<typename T, typename U>
    void substitute(T from, U to)
    {
      substitution.insert(from, to);
    }

  private:
    uint64_t depth_;
    Context* context_;
  };

  struct Constraint::Backtracking
  {
    std::vector<Constraint> subconstraints;

    Backtracking(const Constraint& parent, Context& context)
    : depth_(parent.depth + 1), context_(&context)
    {}

    Backtracking& add(TypePtr left, TypePtr right)
    {
      subconstraints.emplace_back(left, right, depth_, *context_);
      return *this;
    }

    // Elements of `left` are subtypes of `right`
    Backtracking& add(TypeSet left, TypePtr right)
    {
      for (auto left_elem : left)
      {
        add(left_elem, right);
      }
      return *this;
    }

    // `left` is a subtype of elements of `right`
    Backtracking& add(TypePtr left, TypeSet right)
    {
      for (auto right_elem : right)
      {
        add(left, right_elem);
      }
      return *this;
    }

  private:
    uint64_t depth_;
    Context* context_;
  };
}
