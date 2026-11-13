#pragma once

#include "sequent.h"

#include <vir.h>

namespace virc
{
  using namespace trieste;
  using namespace vir;

  inline Node IRResolveAlias(const SequentCtx& ctx, const Node& t)
  {
    if (!ctx.scope || (t != TypeId))
      return t;

    auto defs = ctx.scope->look(t->location());

    for (auto& def : defs)
    {
      if (def == Type)
        return def / Type;
    }

    return t;
  }

  inline const Axiom IRAxiomEq(const SequentCalculus& c)
  {
    return [&](const SequentCtx& ctx, Node& l, Node& r) {
      if (l == TypeId)
      {
        auto resolved = IRResolveAlias(ctx, l);

        if (resolved.get() != l.get())
          return c(ctx, resolved, r);
      }

      return l->equals(r);
    };
  }

  inline const Axiom IRAxiomInvariant(const SequentCalculus& c)
  {
    return [&](const SequentCtx& ctx, Node& l, Node& r) {
      if (l == TypeId)
      {
        auto resolved = IRResolveAlias(ctx, l);

        if (resolved.get() != l.get())
          return c(ctx, resolved, r);
      }

      return (l->type() == r->type()) && c.invariant(ctx, l / Type, r / Type);
    };
  }

  inline const SequentCalculus IRSubtype = {
    {},
    {Union},
    {},
    {},
    {},
    {
      Dyn >> AxiomTrue,
      None >> IRAxiomEq(IRSubtype),
      Bool >> IRAxiomEq(IRSubtype),
      I8 >> IRAxiomEq(IRSubtype),
      I16 >> IRAxiomEq(IRSubtype),
      I32 >> IRAxiomEq(IRSubtype),
      I64 >> IRAxiomEq(IRSubtype),
      U8 >> IRAxiomEq(IRSubtype),
      U16 >> IRAxiomEq(IRSubtype),
      U32 >> IRAxiomEq(IRSubtype),
      U64 >> IRAxiomEq(IRSubtype),
      ISize >> IRAxiomEq(IRSubtype),
      USize >> IRAxiomEq(IRSubtype),
      ILong >> IRAxiomEq(IRSubtype),
      ULong >> IRAxiomEq(IRSubtype),
      F32 >> IRAxiomEq(IRSubtype),
      F64 >> IRAxiomEq(IRSubtype),
      Ptr >> IRAxiomEq(IRSubtype),
      ClassId >> IRAxiomEq(IRSubtype),
      Array >> IRAxiomInvariant(IRSubtype),
      Ref >> IRAxiomInvariant(IRSubtype),
      Cown >> IRAxiomInvariant(IRSubtype),

      TupleType >>
        [](const SequentCtx& ctx, Node& l, Node& r) {
          if (l == TypeId)
          {
            auto resolved = IRResolveAlias(ctx, l);

            if (resolved.get() != l.get())
              return IRSubtype(ctx, resolved, r);
          }

          return (l == TupleType) && (r == TupleType) &&
            std::equal(
                   l->begin(),
                   l->end(),
                   r->begin(),
                   r->end(),
                   [&](auto& t, auto& u) { return IRSubtype(ctx, t, u); });
        },

      TypeId >>
        [](const SequentCtx& ctx, Node& l, Node& r) {
          auto l_resolved = IRResolveAlias(ctx, l);
          auto r_resolved = IRResolveAlias(ctx, r);

          if ((l_resolved.get() != l.get()) || (r_resolved.get() != r.get()))
            return IRSubtype(ctx, l_resolved, r_resolved);

          return l->equals(r);
        },
    },
    {}};
}
