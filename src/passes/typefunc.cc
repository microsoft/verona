// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef typefunc()
  {
    return {
      TypeStruct * (TypeElem[Lhs] * T(Symbol, "->")) *
          (TypeElem * T(Symbol, "->"))++[Op] * TypeElem[Rhs] >>
        [](Match& _) {
          // T1...->T2 =
          //   ({ (Self & mut, T1...): T2 } & mut)
          // | ({ (Self & imm, T1...): T2 } & imm)
          Node r = TypeUnion;
          std::initializer_list<Token> caps = {Mut, Imm};

          for (auto& cap : caps)
          {
            auto params = Params
              << (Param << (Ident ^ _.fresh(l_param))
                        << (Type
                            << (TypeIsect << (Type << Self) << (Type << cap)))
                        << DontCare)
              << (Param << (Ident ^ _.fresh(l_param)) << (Type << clone(_(Lhs)))
                        << DontCare);

            auto it = _[Op].first;
            auto end = _[Op].second;

            while (it != end)
            {
              params
                << (Param << (Ident ^ _.fresh(l_param)) << (Type << clone(*it))
                          << DontCare);
              it = it + 2;
            }

            r
              << (Type
                  << (TypeIsect
                      << (Type
                          << (TypeTrait
                              << (Ident ^ _.fresh(l_trait))
                              << (ClassBody
                                  << (Function << DontCare << apply_id()
                                               << TypeParams << params
                                               << (Type << clone(_(Rhs)))
                                               << DontCare << typepred()
                                               << (Block << (Expr << Unit))))))
                      << (Type << cap)));
          }

          return r;
        },

      TypeStruct * T(Symbol, "->")[Symbol] >>
        [](Match& _) { return err(_[Symbol], "misplaced function type"); },
    };
  }
}
