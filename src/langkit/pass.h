#pragma once

#include "rewrite.h"

namespace langkit
{
  enum class dir
  {
    bottomup,
    topdown,
  };

  class PassDef;
  using Pass = std::shared_ptr<PassDef>;
  using Callback = std::function<void()>;

  class PassDef
  {
  private:
    Callback pre_;
    Callback post_;
    dir direction_;
    std::vector<detail::PatternEffect<Node>> rules_;

  public:
    PassDef(dir direction = dir::topdown) : direction_(direction) {}

    PassDef(const std::initializer_list<detail::PatternEffect<Node>>& r)
    : direction_(dir::topdown), rules_(r)
    {}

    PassDef(
      dir direction,
      const std::initializer_list<detail::PatternEffect<Node>>& r)
    : direction_(direction), rules_(r)
    {}

    operator Pass() const
    {
      return std::make_shared<PassDef>(*this);
    }

    void pre(Callback f)
    {
      pre_ = f;
    }

    void post(Callback f)
    {
      post_ = f;
    }

    template<typename... Ts>
    void rules(Ts... r)
    {
      std::vector<detail::PatternEffect<Node>> rules = {r...};
      rules_.insert(rules_.end(), rules.begin(), rules.end());
    }

    void rules(const std::initializer_list<detail::PatternEffect<Node>>& r)
    {
      rules_.insert(rules_.end(), r.begin(), r.end());
    }

    std::tuple<Node, size_t, size_t> run(Node node)
    {
      size_t changes = 0;
      size_t changes_sum = 0;
      size_t count = 0;

      if (pre_)
        pre_();

      // Because apply runs over child nodes, the top node is never visited.
      // Use a synthetic top node.
      auto top = NodeDef::create(Top);
      top->push_back(node);

      do
      {
        changes = apply(top);

        auto lifted = lift(top);
        if (!lifted.empty())
          throw std::runtime_error("lifted nodes with no destination");

        changes_sum += changes;
        count++;
      } while (changes > 0);

      if (post_)
        post_();

      return {top->front(), count, changes_sum};
    }

  private:
    size_t apply(Node node)
    {
      Match match;
      auto it = node->begin();
      size_t changes = 0;

      while (it != node->end())
      {
        if (direction_ == dir::bottomup)
          changes += apply(*it);

        bool replaced = false;

        for (auto& rule : rules_)
        {
          auto start = it;
          match.clear();

          if (rule.first.match(it, node->end(), match))
          {
            // Replace [start, it) with whatever the rule builds.
            auto replace = rule.second(match);
            it = node->erase(start, it);

            if (replace && replace->type() == Seq)
              it = node->insert(it, replace->begin(), replace->end());
            else
              it = node->insert(it, replace);

            match.bind();
            replaced = true;
            changes++;
            break;
          }
        }

        if (!replaced)
        {
          if (direction_ == dir::topdown)
            changes += apply(*it);

          ++it;
        }
      }

      return changes;
    }

    Nodes lift(Node node)
    {
      Nodes uplift;
      auto it = node->begin();

      while (it != node->end())
      {
        auto lifted = lift(*it);
        bool removed = false;

        if ((*it)->type() == Lift)
        {
          lifted.push_back(*it);
          it = node->erase(it, it + 1);
          removed = true;
        }

        for (auto& lnode : lifted)
        {
          if (lnode->front()->type() == node->type())
            it = node->insert(it, lnode->begin() + 1, lnode->end()) + 1;
          else
            uplift.push_back(lnode);
        }

        if (!removed)
          ++it;
      }

      return uplift;
    }
  };
}
