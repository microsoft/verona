// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

namespace verona::parser
{
  void Resolve::post(TypeRef& tr)
  {
    auto& tn = tr.typenames.front();
    auto def = get_sym(stack, tn->value.location);

    if (!def)
    {
      error() << tn->location << "Couldn't resolve type \""
              << tn->value.location.view() << "\"" << text(tn->location);
      return;
    }

    if (!is_type(def, tn))
      return;

    for (size_t i = 1; i < tr.typenames.size(); i++)
    {
      tn = tr.typenames[i];
      auto sub = def->get_sym(tn->value.location);

      if (!sub)
      {
        error() << tn->location << "Couldn't resolve type \""
                << tn->value.location.view() << "\"" << text(tn->location);
        return;
      }

      if (!is_type(sub, tn))
        return;

      def = sub;
    }
  }

  bool Resolve::is_type(Node<NodeDef>& def, Node<NodeDef> ref)
  {
    switch (def->kind())
    {
      case Kind::Class:
      case Kind::Interface:
      case Kind::TypeAlias:
        return true;

      default:
      {
        error() << ref->location << "Expected a type, but got a "
                << kindname(def->kind()) << text(ref->location)
                << def->location << "Definition is here" << text(def->location);
        return false;
      }
    }
  }
}
