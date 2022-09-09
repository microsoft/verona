#pragma once

#include "rewrite.h"

namespace langkit
{
  template<typename T>
  class LookupDef
  {
  private:
    Callback pre_;
    Callback post_;
    std::vector<detail::PatternEffect<T>> rules_;

  public:
    LookupDef() {}

    void pre(Callback f)
    {
      pre_ = f;
    }

    void post(Callback f)
    {
      post_ = f;
    }

    void rules(const std::initializer_list<detail::PatternEffect<T>> r)
    {
      rules_.insert(rules_.end(), r.begin(), r.end());
    }

    void rule(const detail::PatternEffect<T>& r)
    {
      rules_.push_back(r);
    }

    template<typename... Ts>
    T at(Ts... rest) const
    {
      Nodes nodes = {rest...};
      if (std::find(nodes.begin(), nodes.end(), nullptr) != nodes.end())
        return {};

      return at({nodes.begin(), nodes.end()});
    }

    T at(NodeRange range) const
    {
      if (pre_)
        pre_();

      auto r = run(range);

      if (post_)
        post_();

      return r;
    }

  private:
    T run(NodeRange range) const
    {
      Match match;

      for (auto& rule : rules_)
      {
        auto it = range.first;
        match.clear();

        if (rule.first.match(it, range.second, match))
          return rule.second(match);
      }

      return {};
    }
  };

  template<typename T>
  using Lookup = std::shared_ptr<LookupDef<T>>;
}
