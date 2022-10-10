// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "wf.h"

namespace sample
{
  void typeargs(Found& found, Node ta)
  {
    // TODO: what if def is a TypeParam?
    // use the bounds somehow?
    if (!found.def || !ta)
      return;

    // TODO: error node if it's something that doesn't take typeargs?
    if (!found.def->type().in({Class, Function, TypeAlias}))
      return;

    auto tp = found.def->at(
      wf / Class / TypeParams,
      wf / Function / TypeParams,
      wf / TypeAlias / TypeParams);

    // TODO: error node if there are too many typeargs?
    Nodes args{ta->begin(), ta->end()};
    args.resize(tp->size());

    std::transform(
      tp->begin(),
      tp->end(),
      args.begin(),
      std::inserter(found.map, found.map.end()),
      [](auto param, auto arg) { return std::make_pair(param, arg); });
  }

  Found resolve(Node typeName)
  {
    Found found;
    assert(typeName->type() == TypeName);
    auto ctx = typeName->at(wf / TypeName / TypeName);
    auto id = typeName->at(wf / TypeName / Ident);
    auto ta = typeName->at(wf / TypeName / TypeArgs);

    // A[T1, T2]::B[T3]::C[T4]
    // ctx = A::B
    // ctx = A
    // found = A, {A_0 -> T1, A_1 -> T2}
    // type A[A_0, A_1] = D[A_0]
    //   found = scope_A::D, {D_0 -> A_0}
    // found = scope_A::D::B, {A_0 -> T1, A_1 -> T2, B_0 -> T3}
    // found = A::B::C, {A_0 -> T1, A_1 -> T2, B_0 -> T3, C_0 -> T4}

    if (ctx->type() == TypeUnit)
    {
      found.def = id->lookup_first();
    }
    else if (ctx->type() == TypeName)
    {
      found = resolve(ctx);
      if (!found.def)
        return found;

      found.def = lookdown(found, id);
    }
    else
    {
      assert(false && "unexpected type for TypeName context");
    }

    typeargs(found, ta);
    return found;
  }

  Node lookdown(Found& found, Node id)
  {
    NodeSet visited;

    while (true)
    {
      if (!found.def)
        return {};

      // Check if we've visited this node before. If so, we've found a cycle.
      auto [it, inserted] = visited.insert(found.def);
      if (!inserted)
        return {};

      if (found.def->type().in({Class, TypeTrait}))
      {
        return *found.def->lookdown(id).first;
      }
      else if (found.def->type() == TypeParam)
      {
        auto it = found.map.find(found.def);
        if ((it != found.map.end()) && it->second)
        {
          found.def = it->second;
          continue;
        }

        auto bounds = found.def->at(wf / TypeParam / Bounds);
        if (!bounds->empty())
        {
          found.def = bounds;
          continue;
        }
      }
      else if (found.def->type() == TypeAlias)
      {
        found.def = found.def->at(wf / TypeAlias / Type);
        continue;
      }
      else if (found.def->type() == Type)
      {
        found.def = found.def->at(wf / Type / Type);
        continue;
      }
      else if (found.def->type() == TypeView)
      {
        found.def = found.def->at(wf / TypeView / rhs);
        continue;
      }
      else if (found.def->type() == TypeThrow)
      {
        found.def = found.def->at(wf / TypeThrow / Type);
        continue;
      }
      else if (found.def->type() == TypeName)
      {
        found |= resolve(found.def);
        continue;
      }
      else if (found.def->type() == TypeIsect)
      {
        // TODO:
      }
      // TODO: typeisect, typeunion

      // Other nodes don't have children to look down.
      break;
    }

    return {};
  }
}
