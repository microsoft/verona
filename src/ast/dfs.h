// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace dfs
{
  enum Color
  {
    white,
    grey,
    black
  };

  template<typename Node, typename Func, typename... Args>
  bool pre(Color expect, Node& node, Func f, Args&... args)
  {
    assert(expect != grey);
    assert(node->color == expect);
    node->color = grey;
    auto ok = f(node, args...);

    if (ok)
    {
      for (auto it = node->edges.begin(); it != node->edges.end();)
      {
        if ((*it)->color == expect)
          ok &= pre(expect, *it, f, args...);

        ++it;
      }
    }

    if (expect == white)
      node->color = black;
    else
      node->color = white;

    return ok;
  }

  template<typename Node, typename Func, typename... Args>
  bool pre(Node& node, Func f, Args&... args)
  {
    return pre(node->color, node, f, args...);
  }

  template<typename Node, typename Func, typename... Args>
  bool post(Color expect, Node& node, Func f, Args&... args)
  {
    assert(expect != grey);
    assert(node->color == expect);
    node->color = grey;
    auto ok = true;

    for (auto it = node->edges.begin(); it != node->edges.end();)
    {
      if ((*it)->color == expect)
        ok &= post(expect, *it, f, args...);

      ++it;
    }

    if (ok)
      ok = f(node, args...);

    if (expect == white)
      node->color = black;
    else
      node->color = white;

    return ok;
  }

  template<typename Node, typename Func, typename... Args>
  bool post(Node& node, Func f, Args&... args)
  {
    return post(node->color, node, f, args...);
  }

  template<typename Node, typename Func, typename... Args>
  bool cycles(Color expect, Node& node, Func f, Args&... args)
  {
    assert(expect != grey);
    assert(node->color == expect);
    node->color = grey;
    auto ok = true;

    for (auto it = node->edges.begin(); it != node->edges.end();)
    {
      if ((*it)->color == expect)
      {
        ok &= cycles(expect, *it, f, args...);
      }
      else if ((*it)->color == grey)
      {
        // A cycle has been detected. Always break the cycle.
        ok &= f(node, *it, args...);
        it = node->edges.erase(it);
        continue;
      }

      ++it;
    }

    if (expect == white)
      node->color = black;
    else
      node->color = white;

    return ok;
  }

  template<typename Node, typename Func, typename... Args>
  bool cycles(Node& node, Func f, Args&... args)
  {
    return cycles(node->color, node, f, args...);
  }
}
