// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "btype.h"
#include "lang.h"
#include "lookup.h"

namespace verona
{
  PassDef typevalid()
  {
    return {
      dir::once | dir::topdown,
      {
        T(TypeAlias)[TypeAlias] >> ([](Match& _) -> Node {
          if (recursive_typealias(_(TypeAlias)))
            return err(_[TypeAlias], "recursive type alias");

          return NoChange;
        }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) *
            T(TypeAliasName)[TypeAliasName] >>
          ([](Match& _) -> Node {
            if (!make_btype(_(TypeAliasName))->valid_predicate())
              return err(
                _[Type], "this type alias isn't a valid type predicate");

            return NoChange;
          }),

        In(TypePred)++ * --(In(TypeSubtype, TypeArgs)++) *
            (TypeCaps / T(TypeClassName) / T(TypeParamName) / T(TypeTraitName) /
             T(TypeTrait) / T(TypeTuple) / T(Self) / T(TypeList) / T(TypeView) /
             T(TypeVar) / T(Package))[Type] >>
          [](Match& _) {
            return err(_[Type], "can't put this in a type predicate");
          },

        In(Inherit)++ * --(In(TypeArgs)++) * T(TypeAliasName)[TypeAliasName] >>
          ([](Match& _) -> Node {
            if (!make_btype(_(TypeAliasName))->valid_inherit())
              return err(
                _[Type], "this type alias isn't valid for inheritance");

            return NoChange;
          }),

        In(Inherit)++ * --(In(TypeArgs)++) *
            (TypeCaps / T(TypeParamName) / T(TypeTuple) / T(Self) /
             T(TypeList) / T(TypeView) / T(TypeUnion) / T(TypeVar) /
             T(Package) / T(TypeSubtype) / T(TypeTrue) / T(TypeFalse))[Type] >>
          [](Match& _) { return err(_[Type], "can't inherit from this type"); },
      }};
  }
}
