// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  auto on_lhs(auto pattern)
  {
    return (In(Assign) * (pattern * ++T(Expr))) / (In(TupleLHS) * pattern);
  }

  PassDef assignlhs()
  {
    return {
      "assignlhs",
      wfPassAssignLHS,
      dir::topdown,
      {
        // Turn a Tuple on the LHS of an assignment into a TupleLHS.
        on_lhs(T(Expr) << T(Tuple)[Lhs]) >>
          [](Match& _) { return Expr << (TupleLHS << *_[Lhs]); },

        on_lhs(T(Expr) << (T(TypeAssert) << (T(Tuple)[Lhs] * T(Type)[Type]))) >>
          [](Match& _) {
            return Expr << (TypeAssert << (TupleLHS << *_[Lhs]) << _(Type));
          },

        // Rewrite the selector for a Call on the LHS of an assignment to be an
        // LHS selector.
        on_lhs(T(Expr) << RhsCall[Call]) >>
          [](Match& _) { return Expr << call_lhs(_(Call)); },

        on_lhs(T(Expr) << (T(TypeAssert) << (RhsCall[Call] * T(Type)[Type]))) >>
          [](Match& _) {
            return Expr << (TypeAssert << call_lhs(_(Call)) << _(Type));
          },

        on_lhs(T(Expr) << (T(NLRCheck) << RhsCall[Call])) >>
          [](Match& _) { return Expr << (NLRCheck << call_lhs(_(Call))); },

        on_lhs(
          T(Expr)
          << (T(TypeAssert)
              << ((T(NLRCheck) << RhsCall[Call]) * T(Type)[Type]))) >>
          [](Match& _) {
            return Expr
              << (TypeAssert << (NLRCheck << call_lhs(_(Call))) << _(Type));
          },

        // Turn a RefVar on the LHS of an assignment into a RefVarLHS.
        on_lhs(T(Expr) << T(RefVar)[Lhs]) >>
          [](Match& _) { return Expr << (RefVarLHS << *_[Lhs]); },

        on_lhs(
          T(Expr) << (T(TypeAssert) << (T(RefVar)[Lhs] * T(Type)[Type]))) >>
          [](Match& _) {
            return Expr << (TypeAssert << (RefVarLHS << *_[Lhs]) << _(Type));
          },

        In(Expr) * T(Ref)[Ref] >>
          [](Match& _) {
            return err(_(Ref), "`ref` must be in front of a variable or call");
          },

        In(Expr) * T(Try)[Try] >>
          [](Match& _) {
            return err(_(Try), "`try` must be in front of a call or lambda");
          },

        T(Expr)[Expr] << (Any * Any * Any++) >>
          [](Match& _) {
            return err(
              _(Expr), "Adjacency on this expression isn't meaningful");
          },

        In(TupleLHS) * T(TupleFlatten) >>
          [](Match& _) {
            return err(
              _(TupleFlatten),
              "Can't flatten a tuple on the left-hand side of an assignment");
          },

        In(Expr) * T(Expr)[Expr] >>
          [](Match& _) {
            return err(
              _(Expr),
              "Well-formedness allows this but it can't occur on written code");
          },
      }};
  }
}
