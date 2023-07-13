// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "subtype.h"

#include "btype.h"

#include <cassert>

namespace verona
{
  struct Assume
  {
    Btype sub;
    Btype sup;

    Assume(Btype sub, Btype sup) : sub(sub), sup(sup)
    {
      assert(sub->type().in({Class, TypeTrait}));
      assert(sup->type() == TypeTrait);
    }
  };

  struct Sequent
  {
    std::vector<Btype> lhs_pending;
    std::vector<Btype> rhs_pending;
    std::vector<Btype> lhs_atomic;
    std::vector<Btype> rhs_atomic;
    std::vector<Btype> self;
    std::vector<Btype> predicates;
    std::vector<Assume> assumptions;

    Sequent() = default;

    Sequent(Sequent& rhs)
    : lhs_pending(rhs.lhs_pending),
      rhs_pending(rhs.rhs_pending),
      lhs_atomic(rhs.lhs_atomic),
      rhs_atomic(rhs.rhs_atomic),
      self(rhs.self),
      predicates(rhs.predicates),
      assumptions(rhs.assumptions)
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
      assert(s->type() == Class);
      self.push_back(s);
    }

    void pop_self()
    {
      self.pop_back();
    }

    void add_predicates(Btype& t)
    {
      auto p = t->node;

      while (p)
      {
        if (p->type().in({Function, Class, TypeAlias}))
          predicates.push_back(t->make(p / TypePred));

        p = p->parent({Function, Class, TypeAlias});
      }
    }

    bool reduce(Btype l, Btype r)
    {
      Sequent seq;
      seq.lhs_pending.push_back(l);
      seq.rhs_pending.push_back(r);
      seq.self = self;
      seq.predicates = predicates;
      seq.assumptions = assumptions;
      seq.add_predicates(l);
      seq.add_predicates(r);
      return seq.reduce();
    }

    bool reduce()
    {
      while (!rhs_pending.empty())
      {
        auto r = rhs_pending.back();
        rhs_pending.pop_back();

        if (r->type() == TypeUnion)
        {
          // Π ⊩ Γ ⊢ Δ, A, B
          // ---
          // Π ⊩ Γ ⊢ Δ, (A | B)

          // RHS union becomes RHS formulae.
          for (auto& t : *r->node)
            rhs_pending.push_back(r->make(t));
        }
        else if (r->type() == TypeIsect)
        {
          // Π ⊩ Γ ⊢ Δ, A
          // Π ⊩ Γ ⊢ Δ, B
          // ---
          // Π ⊩ Γ ⊢ Δ, (A & B)

          // RHS isect is a sequent split.
          for (auto& t : *r->node)
          {
            Sequent seq(*this);
            seq.rhs_pending.push_back(r->make(t));

            if (!seq.reduce())
              return false;
          }

          return true;
        }
        else if (r->type() == TypeAlias)
        {
          // Demand that we satisfy the type predicate, which is a split.
          Sequent seq(*this);
          seq.rhs_pending.push_back(r->field(TypePred));

          if (!seq.reduce())
            return false;

          // Try both the typealias and the underlying type.
          rhs_pending.push_back(r->field(Type));
          rhs_atomic.push_back(r);
        }
        else if (r->type() == TypeView)
        {
          auto [rr, done] = reduce_view(r);

          if (done)
            rhs_atomic.push_back(rr);
          else
            rhs_pending.push_back(rr);
        }
        else if (r->type() == Self)
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

        if (l->type() == TypeSubtype)
        {
          // Π, A < B ⊩ Γ ⊢ Δ, A
          // Π, A < B ⊩ Γ, B ⊢ Δ
          // ---
          // Π ⊩ Γ, A < B ⊢ Δ
          predicates.push_back(l);

          Sequent seq(*this);
          seq.rhs_pending.push_back(l->field(Lhs));

          if (!seq.reduce())
            return false;

          lhs_pending.push_back(l->field(Rhs));
        }
        else if (l->type() == TypeIsect)
        {
          // Γ, A, B ⊢ Δ
          // ---
          // Γ, (A & B) ⊢ Δ

          // LHS isect becomes LHS formulae.
          for (auto& t : *l->node)
            lhs_pending.push_back(l->make(t));
        }
        else if (l->type() == TypeUnion)
        {
          // Γ, A ⊢ Δ
          // Γ, B ⊢ Δ
          // ---
          // Γ, (A | B) ⊢ Δ

          // LHS union is a sequent split.
          for (auto& t : *l->node)
          {
            Sequent seq(*this);
            seq.lhs_pending.push_back(l->make(t));

            if (!seq.reduce())
              return false;
          }

          return true;
        }
        else if (l->type() == TypeAlias)
        {
          // Assume that we've satisfied the type predicate.
          lhs_pending.push_back(l->field(TypePred));

          // Try both the typealias and the underlying type.
          lhs_pending.push_back(l->field(Type));
          lhs_atomic.push_back(l);
        }
        else if (l->type() == TypeView)
        {
          auto [ll, done] = reduce_view(l);

          if (done)
            rhs_atomic.push_back(ll);
          else
            rhs_pending.push_back(ll);
        }
        else if (l->type() == Self)
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
      if (l->type() == TypeFalse)
        return true;

      // Everything is a subtype of TypeTrue.
      if (r->type() == TypeTrue)
        return true;

      // Skip TypeVar on either side.
      if ((l->type() == TypeVar) || (r->type() == TypeVar))
        return false;

      // These must be the same type.
      // TODO: region tracking
      if (r->type().in({Iso, Mut, Imm, Self}))
        return l->type() == r->type();

      // Tuples must be the same arity and each element must be a subtype.
      // TODO: remove TypeTuple from the language, use a trait
      if (r->type() == TypeTuple)
      {
        return (l->type() == TypeTuple) &&
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
      if (r->type() == TypeList)
        return false;

      // Check for the same definition site.
      if (r->type() == TypeParam)
        return same_def_site(l, r);

      // Check for the same definition site with invariant typeargs.
      if (r->type().in({TypeAlias, Class}))
        return same_def_site(l, r) && invariant_typeargs(l, r);

      // A package resolves to a class. Once we have package resolution,
      // compare the classes, as different strings could resolve to the
      // same package.
      if (r->type() == Package)
      {
        return (l->type() == Package) &&
          ((l->node / Id)->location() == (r->node / Id)->location());
      }

      // Check predicate subtyping.
      if (r->type() == TypeSubtype)
      {
        // ⊩ Π, A ⊢ B
        // ---
        // Π ⊩ Γ ⊢ Δ, A < B
        Sequent seq;
        seq.lhs_pending = predicates;
        seq.lhs_pending.push_back(r->field(Lhs));
        seq.rhs_pending.push_back(r->field(Rhs));
        return seq.reduce();
      }

      // Check structural subtyping.
      if (r->type() == TypeTrait)
      {
        if (!l->type().in({Class, TypeTrait}))
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

        if (l->type() == Class)
          push_self(l);

        bool ok = true;
        auto rbody = r->node / ClassBody;

        for (auto rf : *rbody)
        {
          if (rf->type() != Function)
            continue;

          // At this point, traits have been decomposed into intersections of
          // single-function traits.
          auto id = (rf / Ident)->location();
          auto lfs = l->node->lookdown(id);

          // Function names are distinguished by arity at this point.
          if ((lfs.size() != 1) || (lfs.front()->type() != Function))
          {
            ok = false;
            break;
          }

          // The functions must take the same number of type parameters.
          auto lf = lfs.front();

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

        if (l->type() == Class)
          pop_self();

        return ok;
      }

      // TODO: handle viewpoint adaptation
      if (r->type() == TypeView)
      {
        // TODO: the ned of a TypeView can be a TypeParam. If it is, we need to
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

      if (l->type() == TypeVar)
      {
        // TODO: l.upper += r
        ok = true;
      }

      if (r->type() == TypeVar)
      {
        // TODO: r.lower += l
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
        if (node->type().in({Class, TypeAlias, Function}))
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
      assert(t->type() == TypeView);
      auto start = t->node->begin();
      auto end = t->node->end();

      for (auto it = start; it != end; ++it)
      {
        auto lhs = NodeRange{start, it};
        auto rhs = NodeRange{it + 1, end};
        auto r = t->make(*it);

        if (r->type().in(
              {Package, Class, TypeTrait, TypeTuple, TypeTrue, TypeFalse}))
        {
          // The viewpoint path can be discarded.
          if (*it == t->node->back())
            return {r, false};

          // There is no view through this type, so treat it as true, i.e. top.
          return {t->make(TypeTrue), false};
        }
        else if (r->type() == TypeList)
        {
          // A.(B...) = (A.B)...
          if (*it == t->node->back())
            return {
              r->make(TypeList << (TypeView << -lhs << -(r->node / Type))),
              false};

          // There is no view through this type, so treat it as true, i.e. top.
          return {t->make(TypeTrue), false};
        }
        else if (r->type().in({TypeUnion, TypeIsect}))
        {
          // A.(B | C).D = A.B.D | A.C.D
          // A.(B & C).D = A.B.D & A.C.D
          Node node = r->type();

          for (auto& rr : *r->node)
            node << (TypeView << -lhs << -rr << -rhs);

          return {r->make(node), false};
        }
        else if (r->type() == TypeAlias)
        {
          return {
            r->field(Type)->make(TypeView << -lhs << -r->node << -rhs), false};
        }
        else if (r->type() == TypeView)
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

  bool subtype(Node sub, Node sup)
  {
    Sequent seq;
    return seq.reduce(make_btype(sub), make_btype(sup));
  }

  bool valid_typeargs(Node tn)
  {
    // TODO: handle FunctionName
    if (!tn->type().in({TypeClassName, TypeAliasName}))
      return true;

    // This should only fail in testing code.
    auto bt = make_btype(tn);

    if (!bt->type().in({Class, TypeAlias}))
      return true;

    Sequent seq;
    seq.lhs_pending.push_back(make_btype(TypeTrue));
    seq.rhs_pending.push_back(bt->field(TypePred));
    return seq.reduce();
  }
}
