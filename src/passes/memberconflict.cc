// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"

namespace verona
{
  Token handed(Node& node)
  {
    assert(node->type().in({FieldLet, FieldVar, Function}));

    // Return Op to mean both.
    if (node->type() == FieldVar)
      return Op;
    else if (node->type() == FieldLet)
      return Lhs;
    else
      return (node / Ref)->type();
  }

  std::pair<size_t, size_t> arity(Node& node)
  {
    assert(node->type().in({FieldLet, FieldVar, Function}));

    if (node->type() != Function)
      return {1, 1};

    auto params = node / Params;
    auto arity_hi = params->size();
    auto arity_lo = arity_hi;

    for (auto& param : *params)
    {
      if ((param / Default)->type() != DontCare)
        arity_lo--;
    }

    return {arity_lo, arity_hi};
  }

  bool conflict(Node& a, Node& b)
  {
    // Check for handedness overlap.
    auto a_hand = handed(a);
    auto b_hand = handed(b);

    if ((a_hand != b_hand) && (a_hand != Op) && (b_hand != Op))
      return false;

    // Check for arity overlap.
    auto [a_lo, a_hi] = arity(a);
    auto [b_lo, b_hi] = arity(b);
    return (b_hi >= a_lo) && (a_hi >= b_lo);
  }

  PassDef memberconflict()
  {
    return {
      dir::topdown | dir::once,
      {
        (T(FieldLet) / T(FieldVar) / T(Function))[Op] >> ([](Match& _) -> Node {
          auto f = _(Op);
          bool implicit = (f / Implicit)->type() == Implicit;
          auto defs = f->scope()->lookdown((f / Ident)->location());
          Nodes conflicts;

          for (auto& def : defs)
          {
            if (!conflict(f, def))
              continue;

            if (implicit == ((def / Implicit)->type() == Implicit))
            {
              // If both are implicit or both are explicit, it's an error.
              if (def->precedes(f))
                conflicts.push_back(def);
            }
            else if (implicit)
            {
              // Discard the implicit definition.
              return {};
            }
          }

          if (!conflicts.empty())
          {
            auto e = err(f, "this member conflicts with other members");

            for (auto& def : conflicts)
              e << (ErrorAst ^ (def / Ident));

            return e;
          }

          return NoChange;
        }),
      }};
  }
}
