// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace dfs
{
  // This implements depth-first search over an arbitrary graph. The `Node` type
  // is expected to have an iterable field `edges` which is a container of
  // Nodes, and to have a `dfs::Color color` as a field. The `pre` function
  // calls the supplied `f` in preorder (call on the way down), and `post` calls
  // in postorder (call on the way up). The `cycles` function breaks cycles
  // (erasing edges), creating a DAG, and calls `f` on each cyclic pair it
  // finds. The `color` field should be all black or all white for a graph when
  // a dfs function is called (the colors are interchangeable in every way), and
  // the entire graph will be the other color when the function completes.

  enum Color
  {
    white,
    grey,
    black
  };

  namespace detail
  {
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
  }

  template<typename Node, typename Func, typename... Args>
  bool pre(Node& node, Func f, Args&... args)
  {
    return detail::pre(node->color, node, f, args...);
  }

  template<typename Node, typename Func, typename... Args>
  bool post(Node& node, Func f, Args&... args)
  {
    return detail::post(node->color, node, f, args...);
  }

  template<typename Node, typename Func, typename... Args>
  bool cycles(Node& node, Func f, Args&... args)
  {
    return detail::cycles(node->color, node, f, args...);
  }
}
