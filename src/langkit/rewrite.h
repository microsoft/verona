// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>
#include <optional>
#include <regex>

namespace langkit
{
  class Lookup;

  namespace detail
  {
    class ILookup;

    struct BindLookup
    {
      const ILookup& lookup;
      Token type;
    };

    class ILookup
    {
    public:
      virtual Node run(NodeRange range) const
      {
        return {};
      }
    };

    class Capture
    {
      friend class langkit::Lookup;

    private:
      std::map<Token, NodeRange> captures;
      std::map<Token, Location> defaults;
      std::map<Location, Node> bindings;
      std::vector<std::pair<Node, Node>> includes;
      const ILookup* lookup_ = nullptr;

    public:
      Capture() = default;

      Capture(const Capture& that)
      {
        lookup_ = that.lookup_;
      }

      NodeRange& operator[](const Token& token)
      {
        return captures[token];
      }

      void def(const Token& token, Location loc)
      {
        defaults[token] = loc;
      }

      Node operator()(const Token& token)
      {
        return *captures[token].first;
      }

      Node operator()(Binding binding)
      {
        auto loc = bind_location(binding.first);
        binding.second->location = loc;
        bindings[loc] = binding.second;
        return binding.second;
      }

      Node include(Binding binding)
      {
        auto loc = bind_location(binding.first);
        Node node = binding.first;
        node->location = loc;
        includes.emplace_back(node, binding.second);
        return node;
      }

      Location bind_location(const Token& token)
      {
        auto range = captures[token];

        if (range.first == range.second)
          return defaults[token];
        else if ((range.first + 1) != range.second)
          throw std::runtime_error("Can only bind to a single node");
        else
          return (*range.first)->location;
      }

      Node find(const Token& token)
      {
        if (!lookup_)
          return {};

        return lookup_->run(captures[token]);
      }

      Node find(Node node)
      {
        if (!lookup_)
          return {};

        auto v = std::vector<Node>{node};
        return lookup_->run({v.begin(), v.end()});
      }

      void operator+=(const Capture& that)
      {
        captures.insert(that.captures.begin(), that.captures.end());
        bindings.insert(that.bindings.begin(), that.bindings.end());
        defaults.insert(that.defaults.begin(), that.defaults.end());
        lookup_ = that.lookup_;
      }

      void clear()
      {
        captures.clear();
        bindings.clear();
        defaults.clear();
      }

      void bind()
      {
        for (auto& [loc, node] : bindings)
          node->bind(loc, node);

        for (auto& [node, include] : includes)
          node->include(include);
      }
    };

    class PatternDef
    {
    public:
      virtual ~PatternDef() = default;

      virtual bool match(NodeIt& it, NodeIt end, Capture& captures) const
      {
        return false;
      }
    };

    using PatternPtr = std::shared_ptr<PatternDef>;

    class Cap : public PatternDef
    {
    private:
      Token name;
      PatternPtr pattern;

    public:
      Cap(const Token& name, PatternPtr pattern) : name(name), pattern(pattern)
      {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto begin = it;
        auto captures2 = captures;

        if (!pattern->match(it, end, captures2))
          return false;

        captures += captures2;
        captures[name] = {begin, it};
        return true;
      }
    };

    class Anything : public PatternDef
    {
    public:
      Anything() {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if (it == end)
          return false;

        ++it;
        return true;
      }
    };

    class TokenMatch : public PatternDef
    {
    private:
      Token type;

    public:
      TokenMatch(const Token& type) : type(type) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if ((it == end) || ((*it)->type != type))
          return false;

        ++it;
        return true;
      }
    };

    class LookupMatch : public PatternDef
    {
    private:
      PatternPtr pattern;
      Token lookup_type;
      const ILookup& lookup;

    public:
      LookupMatch(
        PatternPtr pattern, const Token& lookup_type, const ILookup& lookup)
      : pattern(pattern), lookup_type(lookup_type), lookup(lookup)
      {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto begin = it;
        auto captures2 = captures;

        if (!pattern->match(it, end, captures2))
          return false;

        auto find = lookup.run({begin, it});
        if (!find || (find->type != lookup_type))
        {
          it = begin;
          return false;
        }

        captures += captures2;
        return true;
      }
    };

    class RegexMatch : public PatternDef
    {
    private:
      Token type;
      std::regex regex;

    public:
      RegexMatch(const Token& type, const std::string& r) : type(type)
      {
        regex = std::regex("^" + r + "$", std::regex_constants::optimize);
      }

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if ((it == end) || ((*it)->type != type))
          return false;

        auto s = (*it)->location.view();
        if (!std::regex_match(s.begin(), s.end(), regex))
          return false;

        ++it;
        return true;
      }
    };

    class Opt : public PatternDef
    {
    private:
      PatternPtr pattern;

    public:
      Opt(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto captures2 = captures;

        if (pattern->match(it, end, captures2))
          captures += captures2;

        return true;
      }
    };

    class Rep : public PatternDef
    {
    private:
      PatternPtr pattern;

    public:
      Rep(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        while ((it != end) && pattern->match(it, end, captures))
          ;
        return true;
      }
    };

    class Not : public PatternDef
    {
    private:
      PatternPtr pattern;

    public:
      Not(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if (it == end)
          return false;

        auto captures2 = captures;
        auto begin = it;

        if (pattern->match(it, end, captures2))
        {
          it = begin;
          return false;
        }

        it = begin + 1;
        return true;
      }
    };

    class Seq : public PatternDef
    {
    private:
      PatternPtr first;
      PatternPtr second;

    public:
      Seq(PatternPtr first, PatternPtr second) : first(first), second(second) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto captures2 = captures;
        auto begin = it;

        if (!first->match(it, end, captures2))
          return false;

        if (!second->match(it, end, captures2))
        {
          it = begin;
          return false;
        }

        captures += captures2;
        return true;
      }
    };

    class Choice : public PatternDef
    {
    private:
      PatternPtr first;
      PatternPtr second;

    public:
      Choice(PatternPtr first, PatternPtr second) : first(first), second(second)
      {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto captures2 = captures;

        if (first->match(it, end, captures2))
        {
          captures += captures2;
          return true;
        }

        auto captures3 = captures;

        if (second->match(it, end, captures3))
        {
          captures += captures3;
          return true;
        }

        return false;
      }
    };

    class Inside : public PatternDef
    {
    private:
      Token type;

    public:
      Inside(const Token& type) : type(type) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if (it == end)
          return false;

        if (!(*it)->parent)
          return false;

        auto parent = (*it)->parent->acquire();
        return parent->type == type;
      }
    };

    class First : public PatternDef
    {
    public:
      First() {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        if (it == end)
          return false;

        if (!(*it)->parent)
          return false;

        auto parent = (*it)->parent->acquire();
        return it == parent->children.begin();
      }
    };

    class Last : public PatternDef
    {
    public:
      Last() {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        return it == end;
      }
    };

    class Children : public PatternDef
    {
    private:
      PatternPtr pattern;
      PatternPtr children;

    public:
      Children(PatternPtr pattern, PatternPtr children)
      : pattern(pattern), children(children)
      {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto captures2 = captures;
        auto begin = it;

        if (!pattern->match(it, end, captures2))
          return false;

        auto it2 = (*begin)->children.begin();
        auto end2 = (*begin)->children.end();

        if (!children->match(it2, end2, captures2))
        {
          it = begin;
          return false;
        }

        captures += captures2;
        return true;
      }
    };

    class Pred : public PatternDef
    {
    private:
      PatternPtr pattern;

    public:
      Pred(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto begin = it;
        auto captures2 = captures;
        bool ok = pattern->match(it, end, captures2);
        it = begin;
        return ok;
      }
    };

    class NegPred : public PatternDef
    {
    private:
      PatternPtr pattern;

    public:
      NegPred(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const override
      {
        auto begin = it;
        auto captures2 = captures;
        bool ok = pattern->match(it, end, captures2);
        it = begin;
        return !ok;
      }
    };

    class Pattern
    {
    private:
      PatternPtr pattern;

    public:
      Pattern(PatternPtr pattern) : pattern(pattern) {}

      bool match(NodeIt& it, NodeIt end, Capture& captures) const
      {
        return pattern->match(it, end, captures);
      }

      Pattern operator[](const Token& name) const
      {
        return {std::make_shared<Cap>(name, pattern)};
      }

      Pattern operator~() const
      {
        return {std::make_shared<Opt>(pattern)};
      }

      Pattern operator-() const
      {
        return {std::make_shared<Pred>(pattern)};
      }

      Pattern operator--() const
      {
        return {std::make_shared<NegPred>(pattern)};
      }

      Pattern operator++(int) const
      {
        return {std::make_shared<Rep>(pattern)};
      }

      Pattern operator!() const
      {
        return {std::make_shared<Not>(pattern)};
      }

      Pattern operator*(Pattern rhs) const
      {
        return {std::make_shared<Seq>(pattern, rhs.pattern)};
      }

      Pattern operator/(Pattern rhs) const
      {
        return {std::make_shared<Choice>(pattern, rhs.pattern)};
      }

      Pattern operator<<(Pattern rhs) const
      {
        return {std::make_shared<Children>(pattern, rhs.pattern)};
      }

      Pattern operator()(BindLookup bind) const
      {
        return {std::make_shared<LookupMatch>(pattern, bind.type, bind.lookup)};
      }
    };

    using Effect = std::function<Node(Capture&)>;
    using PatternEffect = std::pair<Pattern, Effect>;

    inline PatternEffect operator>>(Pattern pattern, Effect effect)
    {
      return {pattern, effect};
    }

    struct RangeContents
    {
      NodeRange range;
    };

    struct RangeOr
    {
      NodeRange range;
      Node node;
    };
  }

  const auto Any = detail::Pattern(std::make_shared<detail::Anything>());
  const auto Start = detail::Pattern(std::make_shared<detail::First>());
  const auto End = detail::Pattern(std::make_shared<detail::Last>());

  inline detail::Pattern T(const Token& type)
  {
    return detail::Pattern(std::make_shared<detail::TokenMatch>(type));
  }

  inline detail::Pattern T(const Token& type, const std::string& r)
  {
    return detail::Pattern(std::make_shared<detail::RegexMatch>(type, r));
  }

  inline detail::Pattern In(const Token& type)
  {
    return detail::Pattern(std::make_shared<detail::Inside>(type));
  }

  inline detail::RangeContents operator*(NodeRange range)
  {
    return {range};
  }

  inline detail::RangeOr operator|(NodeRange range, Node node)
  {
    return {range, node};
  }

  inline detail::RangeOr operator|(NodeRange range, const Token& token)
  {
    return {range, NodeDef::create(token)};
  }

  inline Node operator<<(Node node1, Node node2)
  {
    node1->push_back(node2);
    return node1;
  }

  inline Node operator<<(Node node, NodeRange range)
  {
    node->push_back(range);
    return node;
  }

  inline Node operator<<(Node node, detail::RangeContents range_contents)
  {
    for (auto it = range_contents.range.first;
         it != range_contents.range.second;
         ++it)
    {
      node->push_back({(*it)->children.begin(), (*it)->children.end()});
      (*it)->children.clear();
    }

    return node;
  }

  inline Node operator<<(Node node, detail::RangeOr range_or)
  {
    if (range_or.range.first != range_or.range.second)
      node->push_back(range_or.range);
    else
      node->push_back(range_or.node);

    return node;
  }

  inline Node operator<<(Node node, const Token& type)
  {
    node->push_back(NodeDef::create(type));
    return node;
  }

  inline Node operator<<(const Token& type, Node node)
  {
    auto node1 = NodeDef::create(type);
    return node1 << node;
  }

  inline Node operator<<(const Token& type, NodeRange range)
  {
    auto node = NodeDef::create(type);
    return node << range;
  }

  inline Node
  operator<<(const Token& type, detail::RangeContents range_contents)
  {
    auto node = NodeDef::create(type);
    return node << range_contents;
  }

  inline Node operator<<(const Token& type, detail::RangeOr range_or)
  {
    auto node = NodeDef::create(type);
    return node << range_or;
  }

  inline Node operator<<(const Token& type1, const Token& type2)
  {
    auto node = NodeDef::create(type1);
    return node << type2;
  }

  inline Node operator^(Node node1, Node node2)
  {
    node1->location = node2->location;
    return node1;
  }

  enum class dir
  {
    bottomup,
    topdown,
  };

  class Pass
  {
  private:
    std::vector<detail::PatternEffect> rules;
    dir direction = dir::topdown;

  public:
    Pass() {}
    Pass(const std::initializer_list<detail::PatternEffect>& r) : rules(r) {}

    Pass& operator()(dir d)
    {
      direction = d;
      return *this;
    }

    Pass& operator()(const std::initializer_list<detail::PatternEffect>& r)
    {
      rules.insert(rules.end(), r.begin(), r.end());
      return *this;
    }

    std::pair<Node, size_t> run(Node node)
    {
      // Because apply runs over child nodes, the top node is never visited.
      // Use a synthetic top node.
      Node top = Group;
      top->push_back(node);
      size_t changes;

      switch (direction)
      {
        case dir::bottomup:
        {
          changes = bottom_up(top);
          break;
        }

        case dir::topdown:
        {
          changes = top_down(top);
          break;
        }
      }

      return {top->children.front(), changes};
    }

    std::tuple<Node, size_t, size_t> repeat(Node node)
    {
      size_t changes = 0;
      size_t changes_sum = 0;
      size_t count = 0;

      do
      {
        std::tie(node, changes) = run(node);
        changes_sum += changes;
        count++;
      } while (changes > 0);

      return {node, count, changes_sum};
    }

  private:
    size_t bottom_up(Node node)
    {
      size_t changes = 0;

      for (auto& child : node->children)
        changes += bottom_up(child);

      changes += apply(node);
      return changes;
    }

    size_t top_down(Node node)
    {
      size_t changes = apply(node);

      for (auto& child : node->children)
        changes += top_down(child);

      return changes;
    }

    size_t apply(Node node)
    {
      detail::Capture captures;
      auto it = node->children.begin();
      size_t changes = 0;

      while (it != node->children.end())
      {
        bool replaced = false;

        for (auto& rule : rules)
        {
          auto start = it;
          captures.clear();

          if (rule.first.match(it, node->children.end(), captures))
          {
            // Replace [start, it) with whatever the rule builds.
            auto replace = rule.second(captures);
            replace->parent = node.get();
            it = node->children.erase(start, it);
            it = node->children.insert(it, replace);
            captures.bind();
            replaced = true;
            changes++;
            break;
          }
        }

        if (!replaced)
          ++it;
      }

      return changes;
    }
  };

  class Lookup : public detail::ILookup
  {
  private:
    std::vector<detail::PatternEffect> rules;

  public:
    Lookup(const std::initializer_list<detail::PatternEffect>& r) : rules(r) {}

    Node operator()(NodeRange range) const
    {
      return run(range);
    }

    Node run(NodeRange range) const override
    {
      detail::Capture captures;
      captures.lookup_ = this;

      for (auto& rule : rules)
      {
        auto it = range.first;
        captures.clear();

        if (rule.first.match(it, range.second, captures))
          return rule.second(captures);
      }

      return {};
    }

    detail::BindLookup operator=(const Token& type) const
    {
      return {*this, type};
    }
  };

  namespace detail
  {
    struct WFShape
    {
      Token type;
      Pattern pattern;

      WFShape(const Token& type, const Pattern& pattern)
      : type(type), pattern(pattern)
      {}

      bool match(Node node) const
      {
        detail::Capture captures;
        auto it = node->children.begin();
        return pattern.match(it, node->children.end(), captures);
      }

      bool operator<(const WFShape& that) const
      {
        return type < that.type;
      }
    };
  }

  class WellFormed
  {
  private:
    std::vector<detail::WFShape> shapes;

  public:
    WellFormed(const std::initializer_list<detail::WFShape>& r) : shapes(r)
    {
      std::sort(shapes.begin(), shapes.end());
    }

    bool check(Node node) const
    {
      if (!node)
        return false;

      auto shape = std::lower_bound(
        shapes.begin(),
        shapes.end(),
        node->type,
        [](const auto& a, const auto& b) { return a.type < b; });

      if ((shape == shapes.end()) || (shape->type != node->type))
        return false;

      if (!shape->match(node))
        return false;

      for (auto& node : node->children)
      {
        if (!check(node))
          return false;
      }

      return true;
    }
  };
}
