// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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
//polarity class oluşturulup yeni bir polarity nesnesi 
//oluşturmuş ve onu switch sorgusana sokmuş
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
