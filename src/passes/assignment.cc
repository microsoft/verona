// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  const auto l_store = Location("store");

  PassDef assignment()
  {
    return {
      "assignment",
      wfPassAssignment,
      dir::topdown,
      {
        // Let binding.
        In(Assign) *
            (T(Expr)
             << ((T(Let) << T(Ident)[Ident]) /
                 (T(TypeAssert) << (T(Expr) << (T(Let) << T(Ident)[Ident])) *
                    T(Type)[Type]))) *
            T(Expr)[Rhs] * End >>
          [](Match& _) {
            return Expr << (Bind << _(Ident) << typevar(_, Type) << _(Rhs));
          },

        // Destructuring assignment.
        In(Assign) *
            (T(Expr)
             << (T(TupleLHS)[Lhs] /
                 (T(TypeAssert)
                  << ((T(Expr) << T(TupleLHS)[Lhs]) * T(Type)[Type])))) *
            T(Expr)[Rhs] * End >>
          [](Match& _) {
            // let $rhs_id = Rhs
            auto rhs_id = _.fresh();
            auto rhs_e = Expr
              << (Bind << (Ident ^ rhs_id) << typevar(_) << _(Rhs));
            Node seq = ExprSeq;

            Node lhs_tuple = Tuple;
            Node rhs_tuple = Tuple;
            auto ty = _(Type);
            size_t index = 0;

            for (auto lhs_child : *_(Lhs))
            {
              auto lhs_e = lhs_child->front();

              if (lhs_e == Let)
              {
                // lhs_child is already a Let.
                lhs_tuple << (Expr << (RefLet << clone(lhs_e / Ident)));
              }
              else
              {
                // let $lhs_id = lhs_child
                auto lhs_id = _.fresh();
                seq
                  << (Expr
                      << (Bind << (Ident ^ lhs_id) << typevar(_) << lhs_child));
                lhs_child = Expr << (RefLet << (Ident ^ lhs_id));
                lhs_tuple << clone(lhs_child);
              }

              // $lhs_id = $rhs_id._index
              rhs_tuple
                << (Expr
                    << (Assign
                        << lhs_child
                        << (Expr << call(
                              selector(Location("_" + std::to_string(index++))),
                              RefLet << (Ident ^ rhs_id)))));
            }

            // TypeAssert comes after the let bindings for the LHS.
            if (ty)
              seq << (Expr << (TypeAssert << (Expr << lhs_tuple) << ty));

            // The RHS tuple is the last expression in the sequence.
            return Expr << (seq << rhs_e << (Expr << rhs_tuple));
          },

        // Assignment to anything else.
        In(Assign) * T(Expr)[Lhs] * T(Expr)[Rhs] * End >>
          [](Match& _) {
            return Expr << call(selector(l_store), _(Lhs), _(Rhs));
          },

        // Compact assigns after they're reduced.
        T(Assign) << ((T(Expr) << Any[Lhs]) * End) >>
          [](Match& _) { return _(Lhs); },

        // An assign with an error can't be compacted, so it's an error.
        T(Assign)[Assign] << (T(Expr)++ * T(Error)) >>
          [](Match& _) { return err(_(Assign), "Error inside an assignment"); },

        T(Expr)[Expr] << T(Let)[Let] >>
          [](Match& _) { return err(_(Expr), "`let` must have a value"); },

        // Well-formedness allows this but it can't occur on written code.
        T(Expr)[Expr] << T(TupleLHS)[TupleLHS] >>
          [](Match& _) { return Expr << (Tuple << *_[TupleLHS]); },
      }};
  }
}
