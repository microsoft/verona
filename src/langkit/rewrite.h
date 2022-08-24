// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <cassert>
#include <optional>
#include <regex>

namespace langkit
{
  class PassDef;

  namespace detail
  {
    class Capture
    {
      friend class langkit::PassDef;

    private:
      std::map<Token, NodeRange> captures;
      std::map<Token, Location> defaults;
      std::map<Location, Node> bindings;
      std::vector<std::pair<Node, Node>> includes;

    public:
      Capture() = default;

      NodeRange& operator[](const Token& token)
      {
        return captures[token];
      }

      void def(const Token& token, const Location& loc)
      {
        defaults[token] = loc;
      }

      Node operator()(const Token& token)
      {
        auto it = captures.find(token);
        if (it == captures.end())
          return {};

        return *it->second.first;
      }

      Node operator()(LocBinding binding)
      {
        bindings[binding.first] = binding.second;
        return binding.second;
      }

      Node operator()(Binding binding)
      {
        bindings[bind_location(binding.first)] = binding.second;
        return binding.second;
      }

      Node include(Node site, Node target)
      {
        includes.emplace_back(site, target);
        return site;
      }

      void operator+=(const Capture& that)
      {
        captures.insert(that.captures.begin(), that.captures.end());
        bindings.insert(that.bindings.begin(), that.bindings.end());
        defaults.insert(that.defaults.begin(), that.defaults.end());
      }

      void clear()
      {
        captures.clear();
        bindings.clear();
        defaults.clear();
      }

    private:
      const Location& bind_location(const Token& token)
      {
        auto range = captures[token];

        if (range.first == range.second)
          return defaults[token];
        else if ((range.first + 1) != range.second)
          throw std::runtime_error("Can only bind to a single node");
        else
          return (*range.first)->location();
      }

      void bind()
      {
        for (auto& [loc, node] : bindings)
          node->bind(loc);

        for (auto& [site, target] : includes)
          site->include(target);
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
        if ((it == end) || ((*it)->type() != type))
          return false;

        ++it;
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
        if ((it == end) || ((*it)->type() != type))
          return false;

        auto s = (*it)->location().view();
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

        auto p = (*it)->parent();
        return p && (p->type() == type);
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

        auto p = (*it)->parent();
        return p && (it == p->begin());
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

        auto it2 = (*begin)->begin();
        auto end2 = (*begin)->end();

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

    class Pattern;

    template<typename T>
    using Effect = std::function<T(Capture&)>;

    template<typename T>
    using PatternEffect = std::pair<Pattern, Effect<T>>;

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

      Pattern operator++() const
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
    };

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

  template<typename F>
  inline auto operator>>(detail::Pattern pattern, F effect)
    -> detail::PatternEffect<decltype(effect(std::declval<detail::Capture&>()))>
  {
    return {pattern, effect};
  }

  inline const auto Any = detail::Pattern(std::make_shared<detail::Anything>());
  inline const auto Start = detail::Pattern(std::make_shared<detail::First>());
  inline const auto End = detail::Pattern(std::make_shared<detail::Last>());

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
      node->move_children(*it);
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

  inline Node operator<<(Node node, Nodes range)
  {
    node->push_back({range.begin(), range.end()});
    return node;
  }

  inline Node operator^(const Token& type, Node node)
  {
    return NodeDef::create(type, node->location());
  }

  inline Node operator^(const Token& type, Location loc)
  {
    return NodeDef::create(type, loc);
  }

  inline Node clone(Node node)
  {
    if (node)
      return node->clone();
    else
      return {};
  }

  inline Nodes clone(NodeRange range)
  {
    Nodes nodes;
    nodes.reserve(std::distance(range.first, range.second));

    for (auto it = range.first; it != range.second; ++it)
      nodes.push_back((*it)->clone());

    return nodes;
  }
}
