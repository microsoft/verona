// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "subtype.h"

#include <cassert>

namespace verona
{
  void merge(Bounds& lhs, Bounds& rhs)
  {
    for (auto& [k, v] : rhs)
    {
      auto it = lhs.find(k);

      if (it == lhs.end())
      {
        lhs[k] = v;
      }
      else
      {
        // TODO: subsume bounds?
        it->second.lower.insert(
          it->second.lower.end(), v.lower.begin(), v.lower.end());
        it->second.upper.insert(
          it->second.upper.end(), v.upper.begin(), v.upper.end());
      }
    }
  }

  struct Assume
  {
    Btype sub;
    Btype sup;

    Assume(Btype sub_, Btype sup_) : sub(sub_), sup(sup_)
    {
      assert(sub->in({Class, Trait}));
      assert(sup == Trait);
    }
  };

  struct Sequent
  {
    Btypes lhs_pending;
    Btypes rhs_pending;
    Btypes lhs_atomic;
    Btypes rhs_atomic;
    Btypes self;
    Btypes predicates;
    std::vector<Assume> assumptions;
    Bounds bounds;

    Sequent() = default;

    Sequent(Sequent& rhs)
    : lhs_pending(rhs.lhs_pending),
      rhs_pending(rhs.rhs_pending),
      lhs_atomic(rhs.lhs_atomic),
      rhs_atomic(rhs.rhs_atomic),
      self(rhs.self),
      predicates(rhs.predicates),
      assumptions(rhs.assumptions),
      bounds(rhs.bounds)
    {}

    void push_assume(Btype sub, Btype sup)
    {
      assumptions.emplace_back(sub, sup);
    }

    void pop_assume()
    {
      assumptions.pop_back();
    }

    void push_self(Btype s)
    {
      assert(s == Class);
      self.push_back(s);
    }

    void pop_self()
    {
      self.pop_back();
    }

    bool reduce(Btype l, Btype r)
    {
      // Start a fresh reduction, keeping the existing Self binding, predicates,
      // and assumptions.
      Sequent seq;
      seq.lhs_pending.push_back(l);
      seq.rhs_pending.push_back(r);
      seq.self = self;
      seq.predicates = predicates;
      seq.assumptions = assumptions;

      if (!seq.reduce())
        return false;

      merge(bounds, seq.bounds);
      return true;
    }

    bool lhs_reduce(Btype t)
    {
      Sequent seq(*this);
      seq.lhs_pending.push_back(t);

      if (!seq.reduce())
        return false;

      merge(bounds, seq.bounds);
      return true;
    }

    bool rhs_reduce(Btype t)
    {
      Sequent seq(*this);
      seq.rhs_pending.push_back(t);

      if (!seq.reduce())
        return false;

      merge(bounds, seq.bounds);
      return true;
    }

    bool reduce()
    {
      while (!rhs_pending.empty())
      {
        auto r = rhs_pending.back();
        rhs_pending.pop_back();

        if (r == TypeUnion)
        {
          // Π ⊩ Γ ⊢ Δ, A, B
          // ---
          // Π ⊩ Γ ⊢ Δ, (A | B)

          // RHS union becomes RHS formulae.
          for (auto& t : *r->node)
            rhs_pending.push_back(r->make(t));
        }
        else if (r == TypeIsect)
        {
          // Π ⊩ Γ ⊢ Δ, A
          // Π ⊩ Γ ⊢ Δ, B
          // ---
          // Π ⊩ Γ ⊢ Δ, (A & B)

          // RHS isect is a sequent split.
          for (auto& t : *r->node)
          {
            if (!rhs_reduce(r->make(t)))
              return false;
          }

          return true;
        }
        else if (r == TypeAlias)
        {
          // Demand that we satisfy the type predicate, which is a split.
          if (!rhs_reduce(r / TypePred))
            return false;

          // Try both the typealias and the underlying type.
          rhs_pending.push_back(r / Type);
          rhs_atomic.push_back(r);
        }
        else if (r == TypeView)
        {
          auto [rr, done] = reduce_view(r);

          if (done)
            rhs_atomic.push_back(rr);
          else
            rhs_pending.push_back(rr);
        }
        else if (r == Self)
        {
          // Try both Self and the current self type.
          rhs_atomic.push_back(r);

          if (!self.empty())
            rhs_atomic.push_back(self.back());
        }
        else
        {
          rhs_atomic.push_back(r);
        }
      }

      while (!lhs_pending.empty())
      {
        auto l = lhs_pending.back();
        lhs_pending.pop_back();

        if (l == TypeSubtype)
        {
          // Π, A < B ⊩ Γ ⊢ Δ, A
          // Π, A < B ⊩ Γ, B ⊢ Δ
          // ---
          // Π ⊩ Γ, A < B ⊢ Δ
          predicates.push_back(l);

          if (!rhs_reduce(l / Lhs))
            return false;

          lhs_pending.push_back(l / Rhs);
        }
        else if (l == TypeIsect)
        {
          // Γ, A, B ⊢ Δ
          // ---
          // Γ, (A & B) ⊢ Δ

          // LHS isect becomes LHS formulae.
          for (auto& t : *l->node)
            lhs_pending.push_back(l->make(t));
        }
        else if (l == TypeUnion)
        {
          // Γ, A ⊢ Δ
          // Γ, B ⊢ Δ
          // ---
          // Γ, (A | B) ⊢ Δ

          // LHS union is a sequent split.
          for (auto& t : *l->node)
          {
            if (!lhs_reduce(l->make(t)))
              return false;
          }

          return true;
        }
        else if (l == TypeAlias)
        {
          // Assume that we've satisfied the type predicate.
          lhs_pending.push_back(l / TypePred);

          // Try both the typealias and the underlying type.
          lhs_pending.push_back(l / Type);
          lhs_atomic.push_back(l);
        }
        else if (l == TypeView)
        {
          auto [ll, done] = reduce_view(l);

          if (done)
            rhs_atomic.push_back(ll);
          else
            rhs_pending.push_back(ll);
        }
        else if (l == Self)
        {
          // Try both Self and the current self type.
          lhs_atomic.push_back(l);

          if (!self.empty())
            lhs_atomic.push_back(self.back());
        }
        else
        {
          lhs_atomic.push_back(l);
        }
      }

      // If either side is empty, the sequent is trivially false.
      if (lhs_atomic.empty() || rhs_atomic.empty())
        return false;

      // First try without checking any TypeVars.
      // G, A |- D, A
      if (std::any_of(lhs_atomic.begin(), lhs_atomic.end(), [&](Btype& l) {
            return std::any_of(
              rhs_atomic.begin(), rhs_atomic.end(), [&](Btype& r) {
                return subtype_one(l, r);
              });
          }))
      {
        return true;
      }

      // TODO: accumulate bounds on TypeVars. This isn't right, yet.
      return std::any_of(lhs_atomic.begin(), lhs_atomic.end(), [&](Btype& l) {
        return std::any_of(rhs_atomic.begin(), rhs_atomic.end(), [&](Btype& r) {
          return typevar_bounds(l, r);
        });
      });
    }

    bool subtype_one(Btype& l, Btype& r)
    {
      // TypeFalse is a subtype of everything.
      if (l == TypeFalse)
        return true;

      // Everything is a subtype of TypeTrue.
      if (r == TypeTrue)
        return true;

      // Skip TypeVar on either side.
      if ((l == TypeVar) || (r == TypeVar))
        return false;

      // These must be the same type.
      // TODO: region tracking
      if (r->in({Iso, Mut, Imm, Self}))
        return l->type() == r->type();

      // Tuples must be the same arity and each element must be a subtype.
      // TODO: remove TypeTuple from the language, use a trait
      if (r == TypeTuple)
      {
        return (l == TypeTuple) &&
          std::equal(
                 l->node->begin(),
                 l->node->end(),
                 r->node->begin(),
                 r->node->end(),
                 [&](auto& t, auto& u) {
                   return reduce(l->make(t), r->make(u));
                 });
      }

      // Nothing is a subtype of a TypeList. Two TypeLists may have
      // different instantiated arity, even if they have the same bounds.
      // Use a TypeParam with a TypeList upper bounds to get subtyping.
      if (r == TypeList)
        return false;

      // Check for the same definition site.
      if (r == TypeParam)
        return same_def_site(l, r);

      // Check for the same definition site with invariant typeargs.
      if (r->in({TypeAlias, Class}))
        return same_def_site(l, r) && invariant_typeargs(l, r);

      // A package resolves to a class. Once we have package resolution,
      // compare the classes, as different strings could resolve to the
      // same package.
      if (r == Package)
      {
        return (l == Package) &&
          ((l->node / Ident)->location() == (r->node / Ident)->location());
      }

      // Check predicate subtyping.
      if (r == TypeSubtype)
      {
        // ⊩ Π, A ⊢ B
        // ---
        // Π ⊩ Γ ⊢ Δ, A < B
        Sequent seq;
        seq.lhs_pending = predicates;
        seq.lhs_pending.push_back(r / Lhs);
        seq.rhs_pending.push_back(r / Rhs);
        seq.bounds = bounds;

        if (!seq.reduce())
          return false;

        merge(bounds, seq.bounds);
        return true;
      }

      // Check structural subtyping.
      if (r == Trait)
      {
        if (!l->in({Class, Trait}))
          return false;

        // If any assumption is true, the trait is satisfied.
        if (std::any_of(
              assumptions.begin(), assumptions.end(), [&](auto& assume) {
                // Effectively: (l < assume.sub) && (assume.sup < r)
                return same_def_site(r, assume.sup) &&
                  same_def_site(l, assume.sub) &&
                  invariant_typeargs(r, assume.sup) &&
                  invariant_typeargs(l, assume.sub);
              }))
        {
          return true;
        }

        push_assume(l, r);

        if (l == Class)
          push_self(l);

        bool ok = true;
        auto rbody = r->node / ClassBody;

        for (auto rf : *rbody)
        {
          if (rf != Function)
            continue;

          // At this point, traits have been decomposed into intersections of
          // single-function traits.
          auto id = (rf / Ident)->location();
          auto arity = (rf / Params)->size();
          auto lfs = l->node->lookdown(id);
          auto it = std::find_if(lfs.begin(), lfs.end(), [&](auto& lf) {
            return (lf == Function) && ((lf / Params)->size() == arity);
          });

          if (it == lfs.end())
          {
            ok = false;
            break;
          }

          auto lf = *it;

          // The functions must take the same number of type parameters.
          if ((lf / TypeParams)->size() != (rf / TypeParams)->size())
          {
            ok = false;
            break;
          }

          // Contravariant predicates: rf.predicates < lf.predicates
          if (!reduce(r->make(rf / TypePred), l->make(lf / TypePred)))
          {
            ok = false;
            break;
          }

          // Contravariant parameters: rf.params < lf.params
          auto rparams = rf / Params;
          auto lparams = lf / Params;

          if (!std::equal(
                rparams->begin(),
                rparams->end(),
                lparams->begin(),
                lparams->end(),
                [&](auto& rparam, auto& lparam) {
                  return reduce(r->make(rparam / Type), l->make(lparam / Type));
                }))
          {
            ok = false;
            break;
          }

          // Covariant result: lmember.result < rmember.result
          if (!reduce(l->make(lf / Type), r->make(rf / Type)))
          {
            ok = false;
            break;
          }
        }

        // TODO: If the check succeeded, memoize it.
        pop_assume();

        if (l == Class)
          pop_self();

        return ok;
      }

      // TODO: handle viewpoint adaptation
      if (r == TypeView)
      {
        // TODO: the end of a TypeView can be a TypeParam. If it is, we need to
        // be able to use that to fulfill Class / Trait / etc if the TypeView is
        // on the LHS, or to demand it if the TypeView is on the RHS.
        return false;
      }

      // Shouldn't get here in non-testing code.
      return false;
    }

    bool typevar_bounds(Btype& l, Btype& r)
    {
      bool ok = false;

      if (l == TypeVar)
      {
        // TODO: l.upper += r
        bounds[l->node->location()].upper.push_back(r);
        ok = true;
      }

      if (r == TypeVar)
      {
        // TODO: r.lower += l
        bounds[r->node->location()].lower.push_back(l);
        ok = true;
      }

      return ok;
    }

    bool same_def_site(Btype& l, Btype& r)
    {
      // The types must have the same definition site.
      return (l->node == r->node);
    }

    bool invariant_typeargs(Btype& l, Btype& r)
    {
      // Check for invariant type arguments in all enclosing scopes.
      auto node = r->node;

      while (node)
      {
        if (node->in({Class, TypeAlias, Function}))
        {
          for (auto& tp : *(node / TypeParams))
          {
            auto la = l->make(tp);
            auto ra = r->make(tp);

            if (!reduce(la, ra) || !reduce(ra, la))
              return false;
          }
        }

        node = node->parent({Class, TypeAlias, Function});
      }

      return true;
    }

    std::pair<Btype, bool> reduce_view(Btype& t)
    {
      assert(t == TypeView);
      auto start = t->node->begin();
      auto end = t->node->end();

      for (auto it = start; it != end; ++it)
      {
        auto lhs = NodeRange{start, it};
        auto rhs = NodeRange{it + 1, end};
        auto r = t->make(*it);

        if (r->in({Package, Class, Trait, TypeTuple, TypeTrue, TypeFalse}))
        {
          // The viewpoint path can be discarded.
          if (*it == t->node->back())
            return {r, false};

          // There is no view through this type, so treat it as true, i.e. top.
          return {t->make(TypeTrue), false};
        }
        else if (r == TypeList)
        {
          // A.(B...) = (A.B)...
          if (*it == t->node->back())
            return {
              r->make(TypeList << (TypeView << -lhs << -(r->node / Type))),
              false};

          // There is no view through this type, so treat it as true, i.e. top.
          return {t->make(TypeTrue), false};
        }
        else if (r->in({TypeUnion, TypeIsect}))
        {
          // A.(B | C).D = A.B.D | A.C.D
          // A.(B & C).D = A.B.D & A.C.D
          Node node = r->type();

          for (auto& rr : *r->node)
            node << (TypeView << -lhs << -rr << -rhs);

          return {r->make(node), false};
        }
        else if (r == TypeAlias)
        {
          return {
            (r / Type)->make(TypeView << -lhs << -r->node << -rhs), false};
        }
        else if (r == TypeView)
        {
          // A.(B.C).D = A.B.C.D
          auto node = TypeView << -lhs;

          for (auto& rr : *r->node)
            node << -rr;

          node << -rhs;
          return {r->make(node), false};
        }
      }

      // The TypeView contains only TypeParams and capabilities.
      auto t_imm = t->make(Imm);

      for (auto it = start; it != end; ++it)
      {
        auto r = t->make(*it);

        // If any step in the view is Imm, the whole view is Imm.
        // TODO: if r is a TypeVar, this will bind it to `imm` and succeed.
        if (reduce(r, t_imm))
        {
          if (*it == t->node->back())
            return {r, false};

          return {t->make(TypeIsect << Imm << -t->node->back()), false};
        }
      }

      // Indicate the TypeView needs no further reduction.
      return {t, true};
    }
  };

  bool subtype(Btypes& predicates, Btype sub, Btype sup)
  {
    Sequent seq;
    seq.lhs_pending.insert(
      seq.lhs_pending.end(), predicates.begin(), predicates.end());
    seq.lhs_pending.push_back(sub);
    seq.rhs_pending.push_back(sup);
    return seq.reduce();
  }

  bool subtype(Btypes& predicates, Btype sub, Btype sup, Bounds& bounds)
  {
    Sequent seq;
    seq.lhs_pending.insert(
      seq.lhs_pending.end(), predicates.begin(), predicates.end());
    seq.lhs_pending.push_back(sub);
    seq.rhs_pending.push_back(sup);

    if (!seq.reduce())
      return false;

    merge(bounds, seq.bounds);
    return true;
  }
}
