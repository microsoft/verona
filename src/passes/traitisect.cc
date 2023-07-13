// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef traitisect()
  {
    // Turn all traits into intersections of single-function traits. Do this
    // late so that fields have already been turned into accessor functions and
    // partial application functions have already been generated.
    return {
      dir::once | dir::topdown,
      {
        T(TypeTrait)[TypeTrait] << (T(Ident)[Id] * T(ClassBody)[ClassBody]) >>
          [](Match& _) {
            // If we're inside a TypeIsect, put the new traits inside it.
            // Otherwise, create a new TypeIsect.
            Node r =
              (_(TypeTrait)->parent()->type() == TypeIsect) ? Seq : TypeIsect;

            Node base = ClassBody;
            r << (TypeTrait << _(Id) << base);

            for (auto& member : *_(ClassBody))
            {
              if (member->type() == Function)
              {
                // Strip any default implementation.
                (member / Block) = DontCare;
                r
                  << (TypeTrait << (Ident ^ _.fresh(l_trait))
                                << (ClassBody << member));
              }
              else
              {
                base << member;
              }
            }

            if (r->size() == 1)
              return r->front();

            return r;
          },
      }};
  }
}
