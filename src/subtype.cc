// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "subtype.h"

#include <cassert>

namespace verona
{
  using Constraint = std::vector<Btypes>;
  using Constraints = std::map<Location, Constraint>;

  template<class InputIt, class UnaryPredicate>
  InputIt one_of(InputIt begin, InputIt end, UnaryPredicate p)
  {
    auto it = std::find_if(begin, end, p);

    if ((it == end) || (std::find_if(it + 1, end, p) != end))
      return end;

    return it;
  }

  template<class InputIt, class UnaryFunction>
  bool if_one_typevar(InputIt begin, InputIt end, UnaryFunction f)
  {
    auto it = one_of(begin, end, [&](auto& t) { return t == TypeVar; });

    if (it == end)
      return false;

    return f(*it);
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
    SequentPtrs* delayed = nullptr;
    Constraints* constraints = nullptr;

    void str(std::ostream& out, size_t level) const
    {
      out << indent(level) << "sequent: {" << std::endl
          << indent(level + 1) << "lhs: {" << std::endl;

      for (auto& l : lhs_atomic)
        l->str(out, level + 2);

      out << indent(level + 1) << "}," << std::endl
          << indent(level + 1) << "rhs: {" << std::endl;

      for (auto& r : rhs_atomic)
        r->str(out, level + 2);

      out << indent(level + 1) << "}," << std::endl
          << indent(level + 1) << "preds: {" << std::endl;

      for (auto& p : predicates)
        p->str(out, level + 2);

      out << indent(level + 1) << "}," << std::endl
          << indent(level) << "}" << std::endl;
    }

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
      seq.delayed = delayed;
      seq.constraints = constraints;
      return seq.reduce();
    }

    bool reduce(Constraints& c, SequentPtrs* d = nullptr)
    {
      // Try again, using a constraint map for TypeVars.
      lhs_pending = std::move(lhs_atomic);
      rhs_pending = std::move(rhs_atomic);
      auto prev_delayed = delayed;
      delayed = d;
      constraints = &c;
      auto ok = reduce();
      delayed = prev_delayed;
      constraints = nullptr;
      return ok;
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
            Sequent seq(*this);
            seq.rhs_pending.push_back(r->make(t));

            if (!seq.reduce())
              return false;
          }

          return true;
        }
        else if (r == TypeAlias)
        {
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
        else if (r == TypeFalse)
        {
          // Only bother with TypeFalse if we have nothing else.
          if (rhs_atomic.empty())
            rhs_atomic.push_back(r);
        }
        else
        {
          // Don't do TypeVar reduction on the RHS.
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

          Sequent seq(*this);
          seq.rhs_pending.push_back(l / Lhs);

          if (!seq.reduce())
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
            Sequent seq(*this);
            seq.lhs_pending.push_back(l->make(t));

            if (!seq.reduce())
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
        else if (constraints && (l == TypeVar))
        {
          // TypeVar reduction.
          auto it = constraints->find(l->node->location());

          if (it == constraints->end())
          {
            // No constraints.
            lhs_atomic.push_back(l);
          }
          else
          {
            // Constraints are a disjunction of conjunctions. Each conjunction
            // is a sequent split.
            for (auto& isect : it->second)
            {
              Sequent seq(*this);
              seq.lhs_pending.insert(
                seq.lhs_pending.end(), isect.begin(), isect.end());

              if (!seq.reduce())
                return false;
            }

            return true;
          }
        }
        else if (l == TypeTrue)
        {
          // Only bother with TypeTrue if we have nothing else.
          if (lhs_atomic.empty())
            lhs_atomic.push_back(l);
        }
        else
        {
          lhs_atomic.push_back(l);
        }
      }

      // If either side is empty, the sequent is trivially false.
      if (lhs_atomic.empty() || rhs_atomic.empty())
        return false;

      // If anything on the LHS proves anything on the RHS, we're done.
      if (std::any_of(lhs_atomic.begin(), lhs_atomic.end(), [&](Btype& l) {
            return std::any_of(
              rhs_atomic.begin(), rhs_atomic.end(), [&](Btype& r) {
                return subtype_one(l, r);
              });
          }))
      {
        return true;
      }

      // If there are any TypeVars, then lhs may eventually be a subtype of rhs.
      auto is_typevar = [](Btype& t) { return t == TypeVar; };

      if (
        std::any_of(lhs_atomic.begin(), lhs_atomic.end(), is_typevar) ||
        std::any_of(rhs_atomic.begin(), rhs_atomic.end(), is_typevar))
      {
        if (delayed)
          delayed->push_back(std::make_shared<Sequent>(std::move(*this)));

        return true;
      }

      return false;
    }

    bool subtype_one(Btype& l, Btype& r)
    {
      // TypeFalse is a subtype of everything.
      if (l == TypeFalse)
        return true;

      // Everything is a subtype of TypeTrue.
      if (r == TypeTrue)
        return true;

      // TypeVars must be an exact match.
      if ((l == TypeVar) || (r == TypeVar))
      {
        return (l->type() == r->type()) &&
          (l->node->location() == r->node->location());
      }

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

        // A < B ⊩ Π, A ⊢ B
        // ---
        // Π ⊩ Γ ⊢ Δ, A < B
        Sequent seq;
        seq.predicates.push_back(r);
        seq.lhs_pending = predicates;
        seq.lhs_pending.push_back(r / Lhs);
        seq.rhs_pending.push_back(r / Rhs);
        seq.delayed = delayed;
        return seq.reduce();
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
          auto hand = (rf / Ref)->type();
          auto id = (rf / Ident)->location();
          auto arity = (rf / Params)->size();
          auto lfs = l->node->lookdown(id);
          auto it = std::find_if(lfs.begin(), lfs.end(), [&](auto& lf) {
            return (lf == Function) && ((lf / Ref) == hand) &&
              ((lf / Params)->size() == arity);
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

      if (r == InferSelector)
      {
        if (!l->in({Class, Trait}))
          return false;

        if (l == Class)
          push_self(l);

        auto hand = (r->node / Selector / Ref)->type();
        auto id = (r->node / Selector / Ident)->location();
        auto arity = (r->node / Params)->size();
        auto lfs = l->node->lookdown(id);
        auto it = std::find_if(lfs.begin(), lfs.end(), [&](auto& lf) {
          return (lf == Function) && ((lf / Ref) == hand) &&
            ((lf / Params)->size() == arity);
        });

        if (it == lfs.end())
          return false;

        auto lf = *it;

        // Prove the predicate.
        if (!reduce(r->make(TypeTrue), l->make(lf / TypePred)))
          return false;

        // Contravariant parameters.
        auto rparams = r->node / Params;
        auto lparams = lf / Params;

        if (!std::equal(
              rparams->begin(),
              rparams->end(),
              lparams->begin(),
              lparams->end(),
              [&](auto& rparam, auto& lparam) {
                return reduce(r->make(rparam), l->make(lparam / Type));
              }))
        {
          return false;
        }

        // Covariant result.
        if (!reduce(l->make(lf / Type), r->make(r->node / Type)))
          return false;

        if (l == Class)
          pop_self();

        return true;
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

  bool subtype(Btypes& assume, Btype prove)
  {
    Sequent seq;
    seq.lhs_pending = assume;
    seq.rhs_pending.push_back(prove);
    return seq.reduce();
  }

  bool subtype(Btypes& assume, Btype prove, SequentPtrs& delayed)
  {
    Sequent seq;
    seq.lhs_pending = assume;
    seq.rhs_pending.push_back(prove);
    seq.delayed = &delayed;
    return seq.reduce();
  }

  bool subtype(Btype sub, Btype sup)
  {
    Sequent seq;
    seq.lhs_pending.push_back(sub);
    seq.rhs_pending.push_back(sup);
    return seq.reduce();
  }

  bool subtype(Btypes& assume, Btype sub, Btype sup, SequentPtrs& delayed)
  {
    Sequent seq;
    seq.lhs_pending = assume;
    seq.lhs_pending.push_back(sub);
    seq.rhs_pending.push_back(sup);
    seq.delayed = &delayed;
    return seq.reduce();
  }

  template<class InputIt, class BinaryPredicate>
  constexpr bool
  any_of_compare(InputIt first, InputIt last, InputIt ref, BinaryPredicate p)
  {
    if (ref == last)
      return false;

    for (; first != last; ++first)
    {
      if (ref == first)
        continue;

      if (p(*ref, *first))
        return true;
    }

    return false;
  }

  static bool simplify_constraint(Constraint& constraint)
  {
    // Discard uninhabited intersection types.
    auto end =
      std::remove_if(constraint.begin(), constraint.end(), [&](Btypes& isect) {
        // If there's a TypeFalse, the conjunction is uninhabited.
        if (std::find(isect.begin(), isect.end(), TypeFalse) != isect.end())
          return true;

        // If there's a class, classes and traits can't conflict.
        auto cls = std::find(isect.begin(), isect.end(), Class);

        if (any_of_compare(
              isect.begin(), isect.end(), cls, [&](Btype& cls, Btype& other) {
                return other->in({Class, Trait}) && !subtype(cls, other);
              }))
          return true;

        // If there's a cap, caps can't conflict.
        auto cap = std::find_if(isect.begin(), isect.end(), [&](Btype& t) {
          return t->in({Iso, Mut, Imm});
        });

        if (any_of_compare(
              isect.begin(), isect.end(), cap, [&](Btype& cap, Btype& other) {
                return other->in({Iso, Mut, Imm}) && !subtype(cap, other);
              }))
          return true;

        if (cls != isect.end())
        {
          // Discard traits and other classes if there's a class.
          auto bcls = *cls;
          auto end = std::remove_if(isect.begin(), isect.end(), [&](Btype& t) {
            return (t != bcls) && t->in({Class, Trait});
          });

          isect.erase(end, isect.end());
        }

        if (cap != isect.end())
        {
          // Discard other caps if there's a cap.
          auto bcap = *cap;
          auto end = std::remove_if(isect.begin(), isect.end(), [&](Btype& t) {
            return (t != bcap) && t->in({Iso, Mut, Imm});
          });

          isect.erase(end, isect.end());
        }

        // Discard TypeTrue.
        isect.erase(
          std::remove(isect.begin(), isect.end(), TypeTrue), isect.end());

        // TODO:
        // check trait subtyping, discard if subsumed

        // Discard empty conjunctions.
        return isect.empty();
      });

    constraint.erase(end, constraint.end());

    // TODO: simplify union types
    // discard conjunctions contained in other conjunctions

    // Return failure if the constraint is empty.
    return !constraint.empty();
  }

  static bool simplify_constraints(Constraints& constraints)
  {
    // Simplify each constraint.
    bool ok = true;

    std::for_each(
      constraints.begin(), constraints.end(), [&](auto& constraint) {
        if (!simplify_constraint(constraint.second))
        {
          // TODO: An empty constraint is uninhabitable, so it's an error.
          ok = false;
          std::cout << constraint.first.view() << " is uninhabitable"
                    << std::endl;
        }
      });

    return ok;
  }

  static std::vector<Btypes> constrain_union(std::vector<Btypes>& a, Btypes& b)
  {
    std::vector<Btypes> result;

    if (a.empty())
    {
      // Change to DNF.
      for (auto& bb : b)
        result.push_back({bb});

      return result;
    }

    if (b.empty())
      return a;

    for (auto& aa : a)
    {
      for (auto& bb : b)
      {
        // Add each element as a conjunction to each disjunction.
        Btypes seq;
        seq.insert(seq.end(), aa.begin(), aa.end());
        seq.push_back(bb);
        result.push_back(seq);
      }
    }

    return result;
  }

  void one_rhs_typevar(Constraints& constraints, SequentPtrs& seqs)
  {
    // Find one TypeVar on the RHS. LHS is an isect constraint.
    auto end = std::remove_if(seqs.begin(), seqs.end(), [&](auto& seq) {
      return if_one_typevar(
        seq->rhs_atomic.begin(), seq->rhs_atomic.end(), [&](auto& typevar) {
          constraints[typevar->node->location()].push_back(seq->lhs_atomic);
          return true;
        });
    });

    seqs.erase(end, seqs.end());
  }

  void one_lhs_typevar_constrained(Constraints& constraints, SequentPtrs& seqs)
  {
    // Try to solve remaining Sequents if (a) they are one-LHS and (b) the LHS
    // typevar has a constraint.
    SequentPtrs accum;

    while (true)
    {
      auto end = std::remove_if(seqs.begin(), seqs.end(), [&](auto& seq) {
        return if_one_typevar(
          seq->lhs_atomic.begin(), seq->lhs_atomic.end(), [&](auto& typevar) {
            if (
              constraints.find(typevar->node->location()) == constraints.end())
              return false;

            return seq->reduce(constraints, &accum);
          });
      });

      seqs.erase(end, seqs.end());

      if (accum.empty())
        return;

      one_rhs_typevar(constraints, accum);
      seqs.insert(seqs.end(), accum.begin(), accum.end());
      accum.clear();
    }
  }

  void
  one_lhs_typevar_unconstrained(Constraints& constraints, SequentPtrs& seqs)
  {
    // Combine remaining one-LHS Sequents into a single constraint.
    auto end = std::remove_if(seqs.begin(), seqs.end(), [&](auto& seq) {
      return if_one_typevar(
        seq->lhs_atomic.begin(), seq->lhs_atomic.end(), [&](auto& typevar) {
          constraints[typevar->node->location()] = constrain_union(
            constraints[typevar->node->location()], seq->rhs_atomic);
          return true;
        });
    });

    seqs.erase(end, seqs.end());
  }

  void print(const Constraints& constraints);

  bool infer_types(SequentPtrs& delayed)
  {
    Constraints constraints;

    one_rhs_typevar(constraints, delayed);

    // TODO: remove this
    print(constraints);
    return true;

    one_lhs_typevar_constrained(constraints, delayed);
    one_lhs_typevar_unconstrained(constraints, delayed);

    if (!simplify_constraints(constraints))
      return false;

    // Prove remaining Sequents.
    auto end = std::remove_if(delayed.begin(), delayed.end(), [&](auto& seq) {
      return seq->reduce(constraints);
    });

    delayed.erase(end, delayed.end());

    // TODO: shouldn't have any Sequents left. Anything left is an unprovable
    // error. Emit errors.
    if (!delayed.empty())
      std::cout << "unprovable sequents: " << delayed.size() << std::endl;

    // TODO: print constraints, remove this
    std::for_each(
      constraints.begin(), constraints.end(), [&](auto& constraint) {
        std::cout << constraint.first.view() << " = {" << std::endl;
        std::for_each(
          constraint.second.begin(), constraint.second.end(), [&](auto& isect) {
            std::cout << "{";
            std::for_each(
              isect.begin(), isect.end(), [&](auto& t) { std::cout << t; });
            std::cout << "}" << std::endl;
          });
        std::cout << "}" << std::endl;
      });

    // TODO: feed constraints back into the AST
    return true;
  }

  inline std::ostream& operator<<(std::ostream& out, const Sequent& seq)
  {
    seq.str(out, 0);
    return out;
  }

  [[gnu::used]] inline void print(const Sequent& seq)
  {
    std::cout << seq;
  }

  [[gnu::used]] inline void print(const SequentPtr& seq)
  {
    std::cout << *seq;
  }

  [[gnu::used]] inline void print(const SequentPtrs& seqs)
  {
    std::for_each(seqs.begin(), seqs.end(), [](auto& seq) { print(seq); });
  }

  [[gnu::used]] inline void print(const Constraint& c)
  {
    std::cout << "constraint: {" << std::endl;
    std::for_each(c.begin(), c.end(), [](auto& isect) { print(isect); });
    std::cout << "}" << std::endl;
  }

  [[gnu::used]] inline void print(const Constraints& constraints)
  {
    std::for_each(constraints.begin(), constraints.end(), [](auto& constraint) {
      std::cout << constraint.first.view() << " = {" << std::endl;
      std::for_each(
        constraint.second.begin(), constraint.second.end(), [](auto& isect) {
          std::cout << "{";
          std::for_each(
            isect.begin(), isect.end(), [](auto& t) { std::cout << t; });
          std::cout << "}" << std::endl;
        });
      std::cout << "}" << std::endl;
    });
  }

  [[gnu::used]] inline Constraint
  get(const Constraints& constraints, const Btype& t)
  {
    auto it = constraints.find(t->node->location());

    if (it != constraints.end())
      return it->second;

    return {};
  }
}
