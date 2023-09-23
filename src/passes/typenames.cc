// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  Node check_type(Lookups& defs, Node id, Node ta)
  {
    if (defs.size() == 0)
      return Error << (ErrorMsg ^ "unknown type name")
                   << ((ErrorAst ^ id) << id << ta);

    if (defs.size() > 1)
    {
      auto err = Error << (ErrorMsg ^ "ambiguous type name")
                       << ((ErrorAst ^ id) << id << ta);

      for (auto& def : defs)
        err << (ErrorAst ^ (def.def / Ident));

      return err;
    }

    auto def = *defs.begin();

    if (def.too_many_typeargs)
    {
      return Error << (ErrorMsg ^ "too many type arguments")
                   << ((ErrorAst ^ id) << id << ta);
    }

    auto fq = make_fq(def);

    if (fq != FQType)
      return Error << (ErrorMsg ^ "type name is not a type")
                   << ((ErrorAst ^ id) << id << ta);

    return fq;
  }

  PassDef typenames()
  {
    return {
      "typenames",
      wfPassTypeNames,
      dir::topdown,
      {
        TypeStruct * T(DontCare)[DontCare] >>
          [](Match& _) { return typevar(_); },

        // Names on their own must be types.
        TypeStruct * T(Ident)[Ident] * ~T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            auto id = _(Ident);
            auto ta = _(TypeArgs);
            auto defs = lookup(id, ta);
            return check_type(defs, id, ta);
          },

        // Scoping binds most tightly.
        TypeStruct * T(FQType)[FQType] * T(DoubleColon) * T(Ident)[Ident] *
            ~T(TypeArgs)[TypeArgs] >>
          [](Match& _) {
            auto id = _(Ident);
            auto ta = _(TypeArgs);
            auto l = resolve_fq(_(FQType));
            auto defs = lookdown(l, id, ta);
            return check_type(defs, id, ta);
          },
      }};
  }
}
