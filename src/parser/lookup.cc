// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lookup.h"

#include <iostream>
#include <unordered_set>

namespace verona::parser
{
  AstPaths
  look_down_all(AstPaths& paths, const Location& name, bool from_using);

  AstPaths
  look_in_with_using(AstPath& path, const Location& name, bool from_using);

  void add(AstPaths& rs, AstPath& r)
  {
    if (r.empty())
      return;

    auto& node = r.back();

    for (auto& path : rs)
    {
      if (path.back() == node)
        return;
    }

    rs.push_back(r);
  }

  void add(AstPaths& rs1, AstPaths& rs2)
  {
    if (rs2.empty())
      return;

    for (auto& r : rs2)
      add(rs1, r);
  }

  AstPaths look_in_definition(
    AstPath& path, Node<Type>& type, const Location& name, bool from_using)
  {
    // We have a type which is the definition of a type alias or the upper
    // bounds of a type parameter. We want to look inside that type for a
    // definition of `name`. That type is defined in the context of `path`.
    if (!type)
      return {};

    switch (type->kind())
    {
      case Kind::ExtractType:
      case Kind::ViewType:
      {
        // Lookup through the right-hand side of the type pair.
        return look_in_definition(
          path, type->as<TypePair>().right, name, from_using);
      }

      case Kind::TypeRef:
      {
        // Look up this type and look down from there.
        auto paths = look_up(path, type->as<TypeRef>().typenames, from_using);
        return look_down_all(paths, name, from_using);
      }

      case Kind::IsectType:
      {
        // Look in all conjunctions.
        auto& isect = type->as<IsectType>();
        AstPaths rs;

        for (auto& type : isect.types)
        {
          auto find = look_in_definition(path, type, name, from_using);
          add(rs, find);
        }

        return rs;
      }

      default:
        return {};
    }
  }

  AstPaths look_down(AstPath& path, const Location& name, bool from_using)
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
        return look_in_with_using(path, name, from_using);
      }

      case Kind::TypeAlias:
      {
        auto& type = def->as<TypeAlias>().type;
        return look_in_definition(path, type, name, from_using);
      }

      case Kind::TypeParam:
      {
        auto& type = def->as<TypeParam>().type;
        return look_in_definition(path, type, name, from_using);
      }

      default:
        return {};
    }
  }

  AstPaths look_down_all(AstPaths& paths, const Location& name, bool from_using)
  {
    // Find `name` by looking down from every path in `paths`.
    // This will yield some number of new paths.
    AstPaths rs;

    for (auto& path : paths)
    {
      auto rs2 = look_down(path, name, from_using);
      add(rs, rs2);
    }

    return rs;
  }

  Ast look_in(Ast& ast, const Location& name)
  {
    auto st = ast->symbol_table();

    if (!st)
      return {};

    auto find = st->map.find(name);

    if (find == st->map.end())
      return {};

    return find->second;
  }

  AstPaths
  look_in_with_using(AstPath& path, const Location& name, bool from_using)
  {
    if (path.empty())
      return {};

    Ast ast = path.back();
    auto st = ast->symbol_table();

    if (!st)
      return {};

    // Look in this node's symbol table.
    AstPaths rs;
    auto find = st->map.find(name);

    if (find != st->map.end())
    {
      AstPath r{path.begin(), path.end()};
      r.push_back(find->second);
      add(rs, r);
    }

    if (from_using)
      return rs;

    for (auto it = st->use.rbegin(); it != st->use.rend(); ++it)
    {
      auto use = *it;

      if (!is_kind(ast, {Kind::Class, Kind::Interface}))
      {
        // Only accept `using` statements in the same file.
        if (use->location.source->origin != name.source->origin)
          continue;

        // Only accept `using` statements that are earlier in scope.
        if (use->location.start > name.start)
          continue;
      }

      // Look in the type we are `using`. Accept all answers from that.
      // Note that we don't follow `using` once we are following a `using`.
      // A `using` statement doesn't export the names being used, it only
      // imports them for use locally.
      auto rs2 = look_up(path, use->type->as<TypeRef>().typenames, true);
      rs2 = look_down_all(rs2, name, true);
      add(rs, rs2);
    }

    return rs;
  }

  Ast look_up_local(AstPath& path, const Location& name)
  {
    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
      auto& node = *it;
      auto def = look_in(node, name);

      if (def)
      {
        if (is_kind(def, {Kind::Param, Kind::Let, Kind::Var}))
          return def;
        else
          return {};
      }
    }

    return {};
  }

  AstPaths look_up(AstPath& path, const Location& name, bool from_using)
  {
    if (path.empty())
      return {};

    AstPaths rs;
    auto begin = path.begin();

    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
      AstPath r{begin, it.base()};
      auto rs2 = look_in_with_using(r, name, from_using);
      add(rs, rs2);
    }

    return rs;
  }

  AstPaths look_up(AstPath& path, const Location& name)
  {
    return look_up(path, name, false);
  }

  AstPaths look_up(AstPath& path, List<TypeName>& names, bool from_using)
  {
    if (path.empty() || names.empty())
      return {};

    // Find all visible definitions of the first element.
    auto rs = look_up(path, names.front()->location, from_using);

    // For each following element, find all possible definitions in the paths
    // we have so far.
    for (size_t i = 1; i < names.size(); i++)
      rs = look_down_all(rs, names[i]->location, from_using);

    return rs;
  }
}
