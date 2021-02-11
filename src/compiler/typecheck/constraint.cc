// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/constraint.h"

#include "compiler/fixpoint.h"
#include "compiler/instantiation.h"
#include "compiler/polarize.h"
#include "compiler/resolution.h"
#include "compiler/typecheck/structural.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  Constraint::Constraint(
    TypePtr left,
    TypePtr right,
    uint64_t depth,
    Context& context,
    std::string reason)
  : depth(depth)
  {
    assert(context.free_variables(left).is_fixpoint_closed());
    assert(context.free_variables(right).is_fixpoint_closed());

    this->left = context.polarizer().apply(left, Polarity::Positive);
    this->right = context.polarizer().apply(right, Polarity::Negative);
    this->reason = reason;
  }

  bool Constraint::operator<(const Constraint& other) const
  {
    return std::tie(left, right) < std::tie(other.left, other.right);
  }

  Constraint Constraint::apply_mapper(TypeMapper<>& mapper) const
  {
    return Constraint(
      mapper.apply(left), mapper.apply(right), depth, mapper.context());
  }

  /* static */
  std::optional<Constraint::Solution>
  Constraint::solve(const Constraint& c, SolverMode mode, Context& context)
  {
    if (c.left == c.right)
      return Trivial();
    if (auto left = c.left->dyncast<IntersectionType>();
        left && left->elements.find(c.right) != left->elements.end())
      return Trivial();
    if (auto right = c.right->dyncast<UnionType>();
        right && right->elements.find(c.left) != right->elements.end())
      return Trivial();

    /*
     * ------------
     *   α+ <: α-
     */
    if (InferTypePtr infer_right = c.right->dyncast<InferType>(),
        infer_left = c.left->dyncast<InferType>();
        infer_left && infer_right && infer_left->index == infer_right->index &&
        infer_left->subindex == infer_right->subindex)
    {
      assert(infer_left->polarity == Polarity::Positive);
      assert(infer_right->polarity == Polarity::Negative);
      return Trivial();
    }

    /*
     * -----------------------
     *   α+ <: entity-of(α-)
     */
    if (auto right = c.right->dyncast<EntityOfType>())
    {
      if (auto infer_right = right->inner->dyncast<InferType>())
      {
        if (auto left = c.left->dyncast<InferType>())
        {
          if (
            left->index == infer_right->index &&
            left->subindex == infer_right->subindex)
          {
            return Trivial();
          }
        }
      }
    }

    /*
     *     T1 <: T2
     *     T1 <: T3
     * -----------------
     *  T1 <: (T2 & T3)
     */
    if (auto right = c.right->dyncast<IntersectionType>())
    {
      return Compound(c, context).add(c.left, right->elements);
    }

    /*
     *   α+ ∈ FV(T)
     * -------------- => [ μβ. (α+ | T [ β / α+ ]) / α+ ]
     *    T <: α-
     *
     *
     *   α+ ∉ FV(T)
     * -------------- => [ (α+ | T) / α+ ]
     *    T <: α-
     *
     */
    if (auto infer = c.right->dyncast<InferType>())
    {
      assert(infer->polarity == Polarity::Negative);
      InferTypePtr var = reverse_polarity(infer, context);

      if (context.free_variables(c.left).contains(var))
      {
        if (auto union_left = c.left->dyncast<UnionType>())
        {
          // If there is a union type we might be able to avoid the fixpoint
          bool opt = true;
          for (auto x : union_left->elements)
          {
            if (x != var && context.free_variables(x).contains(var))
            {
              opt = false;
              break;
            }
          }
          if (opt)
          {
            return Substitution(var, context.mk_union(var, c.left));
          }
        }
        return Substitution(
          var,
          context.mk_fixpoint(
            context.mk_union(var, close_fixpoint(context, var, c.left))));
      }
      else
      {
        return Substitution(var, context.mk_union(var, c.left));
      }
    }

    /*
     *     T1 <: T3
     *     T2 <: T3
     * -----------------
     *  (T1 | T2) <: T3
     */
    if (auto left = c.left->dyncast<UnionType>())
    {
      return Compound(c, context).add(left->elements, c.right);
    }

    /*
     *   α- ∈ FV(T)
     * -------------- => [ μβ. (α- & T [ β / α- ]) / α- ]
     *    α+ <: T
     *
     *
     *   α- ∉ FV(T)
     * -------------- => [ (α- & T) / α- ]
     *    α+ <: T
     *
     */
    if (auto infer = c.left->dyncast<InferType>())
    {
      assert(infer->polarity == Polarity::Positive);
      InferTypePtr var = reverse_polarity(infer, context);

      if (context.free_variables(c.right).contains(var))
      {
        return Substitution(
          var,
          context.mk_fixpoint(context.mk_intersection(
            var, close_fixpoint(context, var, c.right))));
      }
      else
      {
        return Substitution(var, context.mk_intersection(var, c.right));
      }
    }

    /*
     * In infer mode, regions are ignored:
     *
     * ----------------
     *   K(_) <: K(_)
     *
     * In verify mode, we require regions to be identical:
     *
     * ----------------
     *   K(r) <: K(r)
     *
     * TODO: Allow equivalent regions and path compression.
     */
    if (auto left = c.left->dyncast<CapabilityType>())
    {
      if (auto right = c.right->dyncast<CapabilityType>())
      {
        switch (mode)
        {
          case SolverMode::Infer:
            if (left->kind == right->kind)
              return Trivial();
            break;

          case SolverMode::Verify:
            if (left->kind == right->kind && left->region == right->region)
              return Trivial();
            break;

          case SolverMode::MakeRegionGraph:
            break;

            EXHAUSTIVE_SWITCH;
        }
      }
    }

    /*
     * In infer mode, the regions are ignored:
     *
     *       T1 <: T2
     * --------------------
     *  x -> T1 <: y -> T2
     *
     * In verify mode, we require the regions to be identical:
     *
     *       T1 <: T2
     * --------------------
     *  x -> T1 <: x -> T2
     *
     * TODO: Allow equivalent regions and path compression.
     */
    if (auto left = c.left->dyncast<ApplyRegionType>())
    {
      if (auto right = c.right->dyncast<ApplyRegionType>())
      {
        switch (mode)
        {
          case SolverMode::Infer:
            // TODO: does the mode (adapt vs extract) matter?
            return Compound(c, context).add(left->type, right->type);

          case SolverMode::Verify:
            if (left->mode == right->mode && left->region == right->region)
              return Compound(c, context).add(left->type, right->type);
            break;

          case SolverMode::MakeRegionGraph:
            break;

            EXHAUSTIVE_SWITCH;
        }
      }
    }

    /*
     * ----------
     *   X <: X
     */
    if (auto left = c.left->dyncast<TypeParameter>())
    {
      if (auto right = c.right->dyncast<TypeParameter>())
      {
        if (left->definition == right->definition)
        {
          return Trivial();
        }
      }
    }

    /*
     *
     *   T1 [ (μβ. T1) / β ] <: T2
     * -----------------------------
     *        (μβ. T1) <: T2
     *
     */
    if (auto left = c.left->dyncast<FixpointType>())
    {
      return Compound(c, context)
        .add(unfold_fixpoint(context, left), c.right)
        .assume(c);
    }

    /*
     *
     *   T1 <: T2 [ (μβ. T2) / β ]
     * -----------------------------
     *        T1 <: (μβ. T2)
     *
     */
    if (auto right = c.right->dyncast<FixpointType>())
    {
      return Compound(c, context)
        .add(c.left, unfold_fixpoint(context, right))
        .assume(c);
    }

    /*
     *
     *      T1 <: T2
     * ------------------
     *   K->T1 <: K->T2
     *
     *
     *      T1 <: T2
     * ------------------
     *   X->T1 <: X->T2
     *
     */
    if (auto left = c.left->dyncast<ViewpointType>())
    {
      if (auto right = c.right->dyncast<ViewpointType>())
      {
        if (left->capability == right->capability)
        {
          // Variables on the left have the `Expanded::Yes` flag set, which
          // prevents direct comparison. We recreate the set without that flag.
          TypeParameterSet left_variables;
          std::transform(
            left->variables.begin(),
            left->variables.end(),
            std::inserter(left_variables, left_variables.end()),
            [&](auto& var) {
              return context.mk_type_parameter(
                var->definition, TypeParameter::Expanded::No);
            });

          if (left_variables == right->variables)
          {
            return Compound(c, context).add(left->right, right->right);
          }
        }
      }
    }

    /*
     * ----------------
     *   unit <: unit
     */
    if (auto left = c.left->dyncast<UnitType>())
    {
      if (auto right = c.right->dyncast<UnitType>())
      {
        return Trivial();
      }
    }

    /*
     * We define rules on intersections rather than rely on backtracking,
     * because we want a single solution that includes all elements of the
     * conjunction, rather than one solution per conjunction.
     *
     *        Tview -> (T1.f & T2.f) <: Tread
     *        Twrite <: Tview -> (T1.f | T2.f)
     * ----------------------------------------------
     *  (T1 & T2) <: { Tview -> f: (Tread, Twrite) }
     *
     */
    if (auto right = c.right->dyncast<HasFieldType>())
    {
      if (c.left->dyncast<EntityType>() || c.left->dyncast<IntersectionType>())
      {
        Compound solution(c, context);
        if (solve_has_field(context, &solution, c.left, right))
          return solution;
      }
    }

    /*
     *  class-of(T1) <: { T1 -> f: (T2, bottom) }
     * -------------------------------------------
     *       T1 <: { delayed-view f: T2 }
     */
    if (auto right = c.right->dyncast<DelayedFieldViewType>())
    {
      // Delayed views need at least an entity and a capability component.
      //
      // It can't possibly work on anything other than an intersection (or a
      // union, but that's handled by the basic union rule).
      if (c.left->dyncast<IntersectionType>())
      {
        return Compound(c, context)
          .add(
            c.left,
            context.mk_has_field(
              c.left, right->name, right->type, context.mk_bottom()));
      }
    }

    /*
     * TODO: Handle intersections by looking at every element simultaneaously,
     * similar to how HasField is implemented, rather than with backtracking.
     */
    if (auto right = c.right->dyncast<HasMethodType>())
    {
      if (c.left->dyncast<EntityType>() || c.left->dyncast<StaticType>())
      {
        Compound solution(c, context);
        if (solve_has_method(context, &solution, c.left, right))
          return solution;
      }
    }

    if (auto right = c.right->dyncast<HasAppliedMethodType>())
    {
      if (c.left->dyncast<EntityType>() || c.left->dyncast<StaticType>())
      {
        Compound solution(c, context);
        if (solve_has_applied_method(context, &solution, c.left, right))
          return solution;
      }
    }

    /*
     *  T1 <: σ⁻¹(T2)
     * --------------
     *   σ(T1) <: T2
     */
    if (auto left = c.left->dyncast<VariableRenamingType>())
    {
      return Compound(c, context)
        .add(
          left->type,
          context.mk_variable_renaming(left->renaming.invert(), c.right));
    }

    /**
     *       T1 <: C
     * --------------------
     *  entity-of(T1) <: C
     *
     *  This works for any entity-like types on the RHS.
     *
     *  TODO: We probably want to support more complex things like
     *    entity-of(α) <: (C1 | C2)
     *  without resorting to backtracking.
     */
    if (auto left = c.left->dyncast<EntityOfType>())
    {
      if (
        c.right->dyncast<EntityType>() || c.right->dyncast<HasFieldType>() ||
        c.right->dyncast<HasMethodType>() || c.right->dyncast<EntityOfType>() ||
        c.right->dyncast<HasAppliedMethodType>())
      {
        return Compound(c, context).add(left->inner, c.right);
      }
    }

    /**
     *
     * ---------------------- subst [ (α+ | C) / α+ ]
     *   C <: entity-of(-α)
     *
     * TODO: Is the following sound?
     *
     * ---------------------- subst [ (α+ | X) / α+ ]
     *   X <: entity-of(-α)
     *
     * TODO: Could we generalise this to
     *
     * ---------------------- subst [ (α+ | entity-of(T)) / α+]
     *   T <: entity-of(-α)
     *
     */
    if (auto right = c.right->dyncast<EntityOfType>())
    {
      if (auto infer = right->inner->dyncast<InferType>())
      {
        if (c.left->dyncast<EntityType>() || c.left->dyncast<TypeParameter>())
        {
          InferTypePtr var = reverse_polarity(infer, context);

          if (context.free_variables(c.left).contains(var))
          {
            return Substitution(
              var,
              context.mk_fixpoint(
                context.mk_union(var, close_fixpoint(context, var, c.left))));
          }
          else
          {
            return Substitution(var, context.mk_union(var, c.left));
          }
        }
      }
    }

    /*
     * ---------------------
     *   X <: entity-of(X)
     */
    if (auto right = c.right->dyncast<EntityOfType>())
    {
      if (auto right_param = right->inner->dyncast<TypeParameter>())
      {
        if (auto left = c.left->dyncast<TypeParameter>())
        {
          if (left->definition == right_param->definition)
          {
            return Trivial();
          }
        }
      }
    }

    /**
     * The is-entity constraint is used as a requirement to satisfy any
     * interface. This prevents eg. capabilities from being subtypes of the
     * empty interface.
     *
     * "Should this type be a subtype of the empty interface" is a pretty
     * reliable way of deciding whether to add something here.
     *
     * -------------------
     *  C[T] <: is-entity
     *
     */
    if (c.right->dyncast<IsEntityType>() && c.left->dyncast<EntityType>())
    {
      return Trivial();
    }

    /**
     * Structural subtyping. An interface on the right will be replaced by a
     * combination of HasField and HasMethod constraints.
     *
     * We don't make any special requirements on the left, allowing eg. two
     * components of an intersection to contribute members towards satisfying
     * the interface.
     */
    if (auto right = c.right->dyncast<EntityType>();
        right && right->definition->kind->value() == Entity::Interface)
    {
      Compound solution(c, context);
      add_structural_constraints(context, &solution, c.left, right);
      return solution.assume(c);
    }

    /*
     * Entity types have invariant type arguments.
     *
     *     T1 <: T2
     *     T2 <: T1
     * ----------------
     *  C[T1] <: C[T2]
     *
     * Note that this must come after the structural subtyping rule, to allow
     * for variance between two identical interfaces.
     */
    if (auto left = c.left->dyncast<EntityType>())
    {
      if (auto right = c.right->dyncast<EntityType>())
      {
        if (left->definition == right->definition)
        {
          assert(left->arguments.size() == right->arguments.size());
          return Compound(c, context)
            .add_pairwise(left->arguments, right->arguments)
            .add_pairwise(right->arguments, left->arguments);
        }
      }
    }

    if (mode == SolverMode::Infer)
    {
      /**
       *     T <: U
       *    x ∉ FV(U)
       * --------------
       *   ∃x. T <: U
       *
       * TODO: this is probably OK in Infer mode, because we ignore region
       * values anyway. In region subtyping mode we'll need to do stuff to
       * actually use the replacement value of `x`.
       */
      if (auto left = c.left->dyncast<PathCompressionType>())
      {
        const FreeVariables& freevars = context.free_variables(c.right);
        auto isnt_free = [&](const std::pair<Variable, TypePtr>& entry) {
          return !freevars.contains_region(entry.first);
        };

        // It seems that `x ∉ FV(U)` always hold, but we protect against it just
        // in case. If these assert ever fail, they can be fixed by turning them
        // into an if block around the return statement.
        assert(!freevars.has_indirect_type);
        assert(!freevars.has_inferable_regions);
        assert(std::all_of(
          left->compression.begin(), left->compression.end(), isnt_free));

        return Compound(c, context).add(left->type, c.right);
      }

      /*
       *   unapply(T) <: α-
       * --------------------
       *    T <: (x -> α-)
       *
       */
      if (auto right = c.right->dyncast<ApplyRegionType>())
      {
        if (auto infer = right->type->dyncast<InferType>())
        {
          if (!context.free_variables(c.left).has_inferable_regions)
          {
            assert(infer->polarity == Polarity::Negative);
            return Compound(c, context)
              .add(context.mk_unapply_region(c.left), infer);
          }
        }
      }

      /*
       *   α+ <: unapply(T)
       * --------------------
       *    (x -> α+) <: T
       *
       */
      if (auto left = c.left->dyncast<ApplyRegionType>())
      {
        if (auto infer = left->type->dyncast<InferType>())
        {
          if (!context.free_variables(c.right).has_inferable_regions)
          {
            assert(infer->polarity == Polarity::Positive);
            return Compound(c, context)
              .add(infer, context.mk_unapply_region(c.right));
          }
        }
      }
    }

    if (mode == SolverMode::MakeRegionGraph)
    {
      /*
       *      K <: K'
       * -----------------
       *   K(x) <: K'(x)
       *
       */
      if (auto right_cap = c.right->dyncast<CapabilityType>())
      {
        if (auto left_cap = c.left->dyncast<CapabilityType>())
        {
          if (
            (left_cap->region == right_cap->region) &&
            capability_subkind(left_cap->kind, right_cap->kind))
          {
            return Trivial();
          }
        }
      }

      /*
       *          T <: K(y)
       * ---------------------------
       *   ∃(x: T). mut(x) <: K(y)
       */
      if (auto right_cap = c.right->dyncast<CapabilityType>())
      {
        if (auto left_compression = c.left->dyncast<PathCompressionType>())
        {
          if (auto left_cap = left_compression->type->dyncast<CapabilityType>();
              left_cap && left_cap->kind == CapabilityKind::Mutable)
          {
            const auto& region = std::get<RegionVariable>(left_cap->region);
            return Compound(c, context)
              .add(left_compression->compression.at(region.variable), c.right);
          }
        }
      }

      /*
       *          T <: sub(y)
       * ---------------------------
       *   ∃(x: T). K(x) <: sub(y)
       */
      if (auto right_cap = c.right->dyncast<CapabilityType>();
          right_cap && right_cap->kind == CapabilityKind::Subregion)
      {
        if (auto left_compression = c.left->dyncast<PathCompressionType>())
        {
          if (auto left_cap = left_compression->type->dyncast<CapabilityType>())
          {
            const auto& region = std::get<RegionVariable>(left_cap->region);
            return Compound(c, context)
              .add(left_compression->compression.at(region.variable), c.right);
          }
        }
      }

      /**
       *           x != y
       * ---------------------------
       *     mut(x) <: !child-of(y)
       *     sub(x) <: !child-of(y)
       *     iso(x) <: !child-of(y)
       *      imm() <: !child-of(y)
       *      iso() <: !child-of(y)
       *       x->T <: !child-of(y)
       *        U64 <: !child-of(y)
       *   static E <: !child-of(y)
       *     String <: !child-of(y)
       *       unit <: !child-of(y)
       */
      if (auto not_child_of = c.right->dyncast<NotChildOfType>())
      {
        // These are types that are used without any capability.
        if (c.left->dyncast<StaticType>() || c.left->dyncast<UnitType>())
          return Trivial();

        if (auto left_cap = c.left->dyncast<CapabilityType>())
        {
          switch (left_cap->kind)
          {
            case CapabilityKind::Immutable:
              assert(std::holds_alternative<RegionNone>(left_cap->region));
              return Trivial();

            case CapabilityKind::Subregion:
            case CapabilityKind::Mutable:
            case CapabilityKind::Isolated:
              if (left_cap->region != not_child_of->region)
                return Trivial();
              break;
          }
        }

        if (auto apply_region = c.left->dyncast<ApplyRegionType>())
        {
          if (apply_region->region != not_child_of->region)
            return Trivial();
        }
      }

      /**
       *        T <: !child-of(z)
       * ---------------------------------
       *   ∃(x: T). K(x) <: !child-of(z)
       *
       * TODO: Is this right?
       */
      if (auto not_child_of = c.right->dyncast<NotChildOfType>())
      {
        if (auto left_compression = c.left->dyncast<PathCompressionType>())
        {
          if (auto left_cap = left_compression->type->dyncast<CapabilityType>())
          {
            // left_cap->region must be a RegionVariable, and be bound by the
            // path compression, otherwise we would have normalized the
            // compression away already.
            const auto& region = std::get<RegionVariable>(left_cap->region);

            PathCompressionMap compression = left_compression->compression;
            auto it = compression.find(region.variable);
            assert(it != compression.end());
            TypePtr replacement = it->second;

            // Path compressions cannot be cyclic.
            compression.erase(it);

            // The replacement itself could be being compressed, hence we still
            // need to apply the compression constructor.
            return Compound(c, context)
              .add(
                context.mk_path_compression(compression, replacement), c.right);
          }
        }
      }

      /*
       *
       *   ∃x. (T1 [ (μβ. T1) / β] ) <: T2
       * -----------------------------------
       *        ∃x. (μβ. T1) <: T2
       *
       * This is the same rule as regular fixpoint unfolding, just under an path
       * compression constructor.
       */
      if (auto left_compression = c.left->dyncast<PathCompressionType>())
      {
        if (
          auto left_fixpoint = left_compression->type->dyncast<FixpointType>())
        {
          TypePtr left_unfolded = context.mk_path_compression(
            left_compression->compression,
            unfold_fixpoint(context, left_fixpoint));

          return Compound(c, context).add(left_unfolded, c.right).assume(c);
        }
      }
    }

    /*
     *     T1 <: T2
     * -----------------
     *  T1 <: (T2 | T3)
     *
     *     T1 <: T3
     * -----------------
     *  T1 <: (T2 | T3)
     */
    if (auto right = c.right->dyncast<UnionType>())
    {
      return Backtracking(c, context).add(c.left, right->elements);
    }

    /*
     *     T2 <: T3
     * -----------------
     *  (T1 & T2) <: T3
     *
     *     T1 <: T3
     * -----------------
     *  (T1 & T2) <: T3
     */
    if (auto left = c.left->dyncast<IntersectionType>())
    {
      return Backtracking(c, context).add(left->elements, c.right);
    }

    if (mode == SolverMode::Infer)
    {
      /*
       *   α+ <: unapply(T)
       * --------------------
       *    (x -> α+) <: T
       *
       */
      if (auto left = c.left->dyncast<ApplyRegionType>())
      {
        if (auto infer = left->type->dyncast<InferType>())
        {
          assert(infer->polarity == Polarity::Positive);
          return Compound(c, context)
            .add(infer, context.mk_unapply_region(c.right));
        }
      }
    }

    return std::nullopt;
  }
}
