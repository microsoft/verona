// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

#include "lookup.h"

namespace verona::parser
{
  struct Resolve : Pass<Resolve>
  {
    AST_PASS;

    void post(TypeRef& tr)
    {
      auto paths = look_up(stack, tr.typenames);

      if (paths.empty())
      {
        error() << tr.location << "Couldn't find a definition of this type."
                << text(tr.location);
        return;
      }

      if (paths.size() > 1)
      {
        auto& out = error()
          << tr.location << "Found multiple definitions of this type."
          << text(tr.location);

        for (auto& path : paths)
        {
          auto& loc = path.back()->location;
          out << loc << "Found a definition here." << text(loc);
        }
        return;
      }

      auto& def = paths.front().back();

      if (!is_kind(
            def->kind(),
            {Kind::Class, Kind::Interface, Kind::TypeAlias, Kind::TypeParam}))
      {
        error() << tr.location << "Expected a type, but got a "
                << kindname(def->kind()) << text(tr.location) << def->location
                << "Definition is here" << text(def->location);
      }
    }

    void post(StaticRef& sr)
    {
      auto paths = look_up(stack, sr.typenames);

      if (paths.empty())
      {
        error() << sr.location
                << "Couldn't find a definition for this reference."
                << text(sr.location);
        return;
      }

      if (paths.size() > 1)
      {
        auto& out = error()
          << sr.location << "Found multiple definitions of this reference."
          << text(sr.location);

        for (auto& path : paths)
        {
          auto& loc = path.back()->location;
          out << loc << "Found a definition here." << text(loc);
        }
        return;
      }

      auto& def = paths.front().back();

      if (is_kind(def->kind(), {Kind::Class, Kind::Interface, Kind::TypeAlias}))
      {
        // TODO: it's a type
      }
      else if (is_kind(def->kind(), {Kind::Function}))
      {
        // TODO: it's a static function
      }
      else
      {
        error() << sr.location
                << "Expected a type or a static function, but got a "
                << kindname(def->kind()) << text(sr.location) << def->location
                << "Definition is here" << text(def->location);
      }
    }
  };

  bool resolve_pass(Ast& ast)
  {
    return Resolve() << ast;
  }
}
