// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  PassDef typereference()
  {
    return {
      "typereference",
      wfPassTypeReference,
      dir::topdown,
      {
        // Unscoped reference.
        In(Expr) * T(Ident, Symbol, Self)[Ident] * ~T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            auto id = _(Ident);
            auto ta = _(TypeArgs);
            auto defs = lookup(id, ta);

            if (defs.size() == 1)
            {
              auto def = *defs.begin();

              if (def.too_many_typeargs)
              {
                return Error << (ErrorMsg ^ "too many type arguments")
                             << ((ErrorAst ^ id) << id << ta);
              }

              auto fq = make_fq(def);

              if (fq == FQType)
                return fq;
            }

            if (defs.size() > 1)
            {
              // If there are multiple definitions, it's an ambiguous reference.
              auto err = Error << (ErrorMsg ^ "ambiguous reference")
                               << ((ErrorAst ^ id) << id << ta);

              for (auto& other : defs)
                err << (ErrorAst ^ (other.def / Ident));

              return err;
            }

            // If there isn't a single type definition, treat it as a selector.
            return selector(id, ta);
          },

        // Scoped reference.
        In(Expr) * T(FQType)[Lhs] * T(DoubleColon) * T(Selector)[Selector] >>
          [](Match& _) {
            auto sel = _(Selector);
            auto id = sel / Ident;
            auto ta = sel / TypeArgs;
            auto def = resolve_fq(_(Lhs));
            auto defs = lookdown(def, id, ta);

            if (defs.size() == 0)
              return Error << (ErrorMsg ^ "unknown reference")
                           << ((ErrorAst ^ id) << id << ta);

            if (defs.size() == 1)
            {
              auto ldef = *defs.begin();

              if (ldef.too_many_typeargs)
              {
                return Error << (ErrorMsg ^ "too many type arguments")
                             << ((ErrorAst ^ id) << id << ta);
              }

              return make_fq(ldef);
            }

            if (std::any_of(defs.begin(), defs.end(), [](auto& d) {
                  return d.def != Function;
                }))
            {
              // If there are multiple definitions, and at least one of them is
              // not a function, then we have an ambiguous reference.
              auto err = Error << (ErrorMsg ^ "ambiguous reference")
                               << ((ErrorAst ^ id) << id << ta);

              for (auto& other : defs)
                err << (ErrorAst ^ (other.def / Ident));

              return err;
            }

            // Select the smallest arity function.
            auto l =
              *std::min_element(defs.begin(), defs.end(), [](auto& a, auto& b) {
                return (a.def / Params)->size() < (b.def / Params)->size();
              });

            return make_fq(l);
          },

        // Error out on invalid scoped references.
        In(Expr) * T(DoubleColon)[DoubleColon] >>
          [](Match& _) {
            return err(_(DoubleColon), "Expected a scoped reference");
          },

        // Lone TypeArgs are typeargs on apply.
        In(Expr) * T(TypeArgs)[TypeArgs] >>
          [](Match& _) { return Seq << Dot << selector(l_apply, _(TypeArgs)); },

        // Create sugar, with no arguments.
        In(Expr) * T(FQType)[FQType] * ~T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            return append_fq(_(FQType), selector(l_create, _(TypeArgs)));
          },

        // New sugar.
        In(Expr) * T(New)[New] >>
          [](Match& _) {
            return append_fq(
              local_fq(_(New)->parent({Class, Trait})), selector(l_new));
          },
      }};
  }
}
