// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/type.h"

#include "compiler/context.h"
#include "compiler/intern.h"
#include "ds/helpers.h"

namespace verona::compiler
{
  Polarity reverse_polarity(Polarity polarity)
  {
    switch (polarity)
    {
      case Polarity::Positive:
        return Polarity::Negative;
      case Polarity::Negative:
        return Polarity::Positive;

        EXHAUSTIVE_SWITCH;
    }
  }

  InferTypePtr reverse_polarity(const InferTypePtr& ty, Context& context)
  {
    return context.mk_infer(
      ty->index, ty->subindex, reverse_polarity(ty->polarity));
  }
}
