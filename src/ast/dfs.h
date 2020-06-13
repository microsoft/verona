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

  template<typename Node>
  struct Default
  {
    using NodePtr = std::shared_ptr<Node>;

    inline bool pre(NodePtr& node)
    {
      return true;
    }

    inline bool post(NodePtr& node)
    {
      return true;
    }

    inline bool fail(NodePtr& parent, NodePtr& child)
    {
      return false;
    }
  };

  template<typename Node, typename Action>
  bool dfs(Color expect, std::shared_ptr<Node>& node, Action& action)
  {
    assert(expect != grey);
    assert(node->color == expect);
    node->color = grey;
    auto ok = action.pre(node);

    if (ok)
    {
      for (auto it = node->edges.begin(); it != node->edges.end(); ++it)
      {
        if ((*it)->color == expect)
        {
          ok &= dfs(expect, *it, action);
        }
        else if ((*it)->color == grey)
        {
          // A cycle has been detected. Always break the cycle.
          ok &= action.fail(node, *it);
          node->edges.erase(it);
        }
      }

      if (ok)
        ok = action.post(node);
    }

    if (expect == white)
      node->color = black;
    else
      node->color = white;

    return ok;
  }

  template<typename Node, typename Action>
  bool dfs(std::shared_ptr<Node>& node, Action& action)
  {
    return dfs(node->color, node, action);
  }

  template<typename Node>
  using CyclicPairs =
    std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>;

  template<typename Node>
  bool detect_cycles(
    std::shared_ptr<Node>& node,
    CyclicPairs<Node>& pairs)
  {
    using NodePtr = std::shared_ptr<Node>;

    struct DetectCycles : public Default<Node>
    {
      CyclicPairs<Node>& pairs;
      DetectCycles(CyclicPairs<Node>& pairs) : pairs(pairs) {}

      bool fail(NodePtr& parent, NodePtr& child)
      {
        pairs.emplace_back(parent, child);
        return false;
      }
    };

    DetectCycles dc(pairs);
    return dfs(node, dc);
  }
}
