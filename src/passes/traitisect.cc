// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  PassDef traitisect()
  {
    // Turn all traits into intersections of single-function traits. Do this
    // late so that fields have already been turned into accessor functions and
    // partial application functions have already been generated.
    return {
      "traitisect",
      wfPassAutoFields,
      dir::bottomup | dir::once,
      {
        T(Trait)[Trait] << (T(Ident)[Ident] * T(ClassBody)[ClassBody]) >>
          [](Match& _) {
            // If we're inside a TypeIsect, put the new traits inside it.
            // Otherwise, create a new TypeIsect.
            Node r = (_(Trait)->parent() == TypeIsect) ? Seq : TypeIsect;

            Node base = ClassBody;
            r << (Trait << _(Ident) << base);

            for (auto& member : *_(ClassBody))
            {
              if (member == Function)
              {
                // Strip any default implementation.
                (member / Block) = DontCare;
                r
                  << (Trait << (Ident ^ _.fresh(l_trait))
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
