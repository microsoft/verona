// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

namespace verona::parser
{
  void Resolve::post(TypeRef& tr)
  {
    Node<NodeDef> def;

    for (auto& name : tr.typenames)
    {
      if (!def)
        def = get_sym(stack, name->value.location);
      else
        def = def->get_sym(name->value.location);

      if (!is_type(def, name))
        return;
    }
  }

  void Resolve::post(StaticRef& sr)
  {
    Node<NodeDef> def;

    for (size_t i = 0; i < sr.typenames.size() - 1; i++)
    {
      auto& tn = sr.typenames[i];

      if (!def)
        def = get_sym(stack, tn->value.location);
      else
        def = def->get_sym(tn->value.location);

      if (!is_type(def, tn))
        return;
    }

    auto& tn = sr.typenames.back();

    if (!def)
      def = get_sym(stack, tn->value.location);
    else
      def = def->get_sym(tn->value.location);

    if (!def)
    {
      error() << tn->location << "Couldn't resolve static reference"
              << text(tn->location);
      return;
    }

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
      error() << tn->location
              << "Expected a type or a static function, but got a "
              << kindname(def->kind()) << text(tn->location) << def->location
              << "Definition is here" << text(def->location);
    }
  }

  bool Resolve::is_type(Node<NodeDef>& def, Node<NodeDef> ref)
  {
    if (!def)
    {
      error() << ref->location << "Couldn't resolve type"
              << text(ref->location);
      return false;
    }

    if (is_kind(def->kind(), {Kind::Class, Kind::Interface, Kind::TypeAlias}))
      return true;

    error() << ref->location << "Expected a type, but got a "
            << kindname(def->kind()) << text(ref->location) << def->location
            << "Definition is here" << text(def->location);
    return false;
  }
}
