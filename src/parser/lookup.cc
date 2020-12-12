// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

namespace verona::parser
{
  AstPaths look_in(AstPath& path, List<Type>& types, Location& name);
  AstPaths look_down(AstPaths& paths, Location& name);

  AstPaths look_in(AstPath& path, Node<Type>& type, Location& name)
  {
    // We have a type which is the definition of a type alias or the upper
    // bounds of a type parameter. We want to look inside that type for a
    // definition of `name`. That type is defined in the context of `path`.
    switch (type->kind())
    {
      case Kind::ExtractType:
      case Kind::ViewType:
      {
        // Lookup through the right-hand side of the type pair.
        return look_in(path, type->as<TypePair>().right, name);
      }

      case Kind::TypeRef:
      {
        // Look up this type and look down from there.
        auto paths = look_up(path, type->as<TypeRef>().typenames);
        return look_down(paths, name);
      }

      case Kind::IsectType:
      {
        // Look in all conjunctions.
        auto& isect = type->as<IsectType>();
        return look_in(path, isect.types, name);
      }

      default:
        return {};
    }
  }

  AstPaths look_in(AstPath& path, List<Type>& types, Location& name)
  {
    // Find `name` by looking in every type in `types`.
    // This will yield some number of new paths.
    AstPaths results;

    for (auto& type : types)
    {
      auto find = look_in(path, type, name);
      results.insert(results.end(), find.begin(), find.end());
    }

    return results;
  }

  AstPaths look_down(AstPath& path, Location& name)
  {
    // This looks for `name` in the last element of `path`.
    if (path.empty())
      return {};

    auto& def = path.back();

    switch (def->kind())
    {
      case Kind::Class:
      case Kind::Interface:
      {
        // If we are looking up in an entity, expect to find the name in the
        // entity's symbol table.
        auto sub = def->get_sym(name);

        if (!sub)
          return {};

        AstPath r{path.begin(), path.end()};
        r.push_back(sub);
        return {r};
      }

      case Kind::TypeAlias:
      {
        auto& type = def->as<TypeAlias>().type;
        return look_in(path, type, name);
      }

      case Kind::TypeParam:
      {
        auto& type = def->as<TypeParam>().type;
        return look_in(path, type, name);
      }

      default:
        return {};
    }
  }

  AstPaths look_down(AstPaths& paths, Location& name)
  {
    // Find `name` by looking down from every path in `paths`.
    // This will yield some number of new paths.
    AstPaths results;

    for (auto& path : paths)
    {
      auto find = look_down(path, name);
      results.insert(results.end(), find.begin(), find.end());
    }

    return results;
  }

  AstPaths look_up(Find mode, AstPath& path, Location& name)
  {
    // Find visible definitions of `name`, sorted from closest to furthest.
    AstPaths results;
    auto begin = path.begin();

    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
      auto& node = *it;
      auto def = node->get_sym(name);

      if (def)
      {
        AstPath r{begin, it.base()};
        r.push_back(def);
        results.push_back(r);

        if (mode == Find::First)
          return results;
      }
    }

    return results;
  }

  AstPaths look_up(AstPath& path, List<TypeName>& names)
  {
    if (names.empty())
      return {};

    auto& name = names.front()->location;

    // If `names` has a single element, find all visible definitions of that
    // element.
    if (names.size() == 1)
      return look_up(Find::All, path, name);

    // Find the closest definition of the first element.
    auto paths = look_up(Find::First, path, name);

    if (paths.empty())
      return {};

    // For each element, find all possible definitions and look inside them.
    for (size_t i = 1; i < names.size(); i++)
    {
      paths = look_down(paths, names[i]->location);

      if (paths.empty())
        return {};
    }

    return paths;
  }
}
