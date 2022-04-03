// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "token.h"

#include <map>
#include <sstream>
#include <vector>

namespace langkit
{
  struct indent
  {
    size_t level;

    indent(size_t level) : level(level) {}
  };

  inline std::ostream& operator<<(std::ostream& out, const indent in)
  {
    for (size_t i = 0; i < in.level; i++)
      out << "  ";

    return out;
  }

  class NodeDef;
  using Node = std::shared_ptr<NodeDef>;
  using NodeIt = std::vector<Node>::iterator;
  using NodeRange = std::pair<NodeIt, NodeIt>;

  class SymtabDef
  {
    friend class NodeDef;

  private:
    std::map<Location, std::vector<Node>> symbols;
    std::vector<std::pair<Location, Node>> includes;
    size_t next_id = 0;

  public:
    SymtabDef() = default;

    Location fresh()
    {
      return Location("$" + std::to_string(next_id++));
    }

    std::string str(size_t level);
  };

  using Symtab = std::shared_ptr<SymtabDef>;

  struct Index
  {
    Token type;
    size_t index;

    constexpr Index() : type(Invalid), index(std::numeric_limits<size_t>::max())
    {}
    constexpr Index(const Token& type, size_t index) : type(type), index(index)
    {}
  };

  class NodeDef : public std::enable_shared_from_this<NodeDef>
  {
  private:
    Token type_;
    Location location_;
    Symtab symtab_;
    NodeDef* parent_;
    std::vector<Node> children;

    NodeDef(const Token& type, Location location)
    : type_(type), location_(location), parent_(nullptr)
    {
      if (type_ & flag::symtab)
        symtab_ = std::make_shared<SymtabDef>();
    }

  public:
    ~NodeDef() {}

    static Node create(const Token& type)
    {
      return std::shared_ptr<NodeDef>(new NodeDef(type, {nullptr, 0, 0}));
    }

    static Node create(const Token& type, Location location)
    {
      return std::shared_ptr<NodeDef>(new NodeDef(type, location));
    }

    static Node create(const Token& type, NodeRange range)
    {
      if (range.first == range.second)
        return create(type);

      return std::shared_ptr<NodeDef>(new NodeDef(
        type, (*range.first)->location_ * (*(range.second - 1))->location_));
    }

    const Token& type()
    {
      return type_;
    }

    const Location& location()
    {
      return location_;
    }

    NodeDef* parent()
    {
      return parent_;
    }

    void extend(const Location& loc)
    {
      location_ *= loc;
    }

    NodeIt begin()
    {
      return children.begin();
    }

    NodeIt end()
    {
      return children.end();
    }

    NodeRange range()
    {
      return {begin(), end()};
    }

    bool empty()
    {
      return children.empty();
    }

    size_t size()
    {
      return children.size();
    }

    template<typename... Ts>
    Node at(const Index& index, const Ts&... indices)
    {
      if (index.type != type_)
      {
        if constexpr (sizeof...(Ts) > 0)
          return at(indices...);

        throw std::runtime_error("invalid index");
      }

      assert(index.index < children.size());
      return children.at(index.index);
    }

    Node at(size_t index)
    {
      assert(index < children.size());
      return children.at(index);
    }

    Node front()
    {
      if (children.empty())
        return {};

      return children.front();
    }

    Node back()
    {
      if (children.empty())
        return {};

      return children.back();
    }

    void push_back(Node node)
    {
      if (!node)
        return;

      children.push_back(node);
      node->parent_ = this;
    }

    void push_back(NodeIt it)
    {
      push_back(*it);
    }

    void push_back(NodeRange range)
    {
      for (auto it = range.first; it != range.second; ++it)
        push_back(it);
    }

    void move_children(Node from)
    {
      if (!from)
        return;

      push_back({from->begin(), from->end()});
      from->children.clear();
    }

    Node pop_back()
    {
      if (children.empty())
        return {};

      auto node = children.back();
      children.pop_back();
      node->parent_ = nullptr;
      return node;
    }

    NodeIt erase(NodeIt first, NodeIt last)
    {
      for (auto it = first; it != last; ++it)
      {
        // Only clear the parent if the node is not shared.
        if ((*it)->parent_ == this)
          (*it)->parent_ = nullptr;
      }

      return children.erase(first, last);
    }

    NodeIt insert(NodeIt pos, Node node)
    {
      if (!node)
        return pos;

      node->parent_ = this;
      return children.insert(pos, node);
    }

    Node scope()
    {
      auto p = parent_;

      while (p)
      {
        auto node = p->shared_from_this();

        if (node->symtab_)
          return node;

        p = node->parent_;
      }

      return {};
    }

    std::vector<Node> lookup_all(const Location& loc)
    {
      // Find all bindings for this location by looking upwards in the symbol
      // table chain.
      std::vector<Node> r;
      auto st = scope();

      while (st)
      {
        auto def_ok = !(st->type_ & flag::defbeforeuse);
        auto it = st->symtab_->symbols.find(loc);

        if (it != st->symtab_->symbols.end())
        {
          if (def_ok)
          {
            r.insert(r.end(), it->second.begin(), it->second.end());
          }
          else
          {
            for (auto& node : it->second)
            {
              if (node->location_.before(loc))
                r.push_back(node);
            }
          }
        }

        for (auto& [def, node] : st->symtab_->includes)
        {
          if (def_ok || def.before(loc))
          {
            auto find = node->symtab_->symbols.find(loc);

            if (find != node->symtab_->symbols.end())
              r.insert(r.end(), find->second.begin(), find->second.end());
          }
        }

        st = st->scope();
      }

      return r;
    }

    Node lookup_first()
    {
      // Find ourself in the enclosing symbol table chain.
      return lookup_first(location_);
    }

    Node lookup_first(const Location& loc)
    {
      // Find this location in the enclosing symbol table chain.
      auto st = scope();

      while (st)
      {
        auto def_ok = !(st->type_ & flag::defbeforeuse);
        auto it = st->symtab_->symbols.find(loc);
        Node r;

        if (it != st->symtab_->symbols.end())
        {
          // Select the last definition. In a defbeforeuse context, select the
          // last definition before the use site.
          for (auto& node : it->second)
          {
            if (
              (def_ok || node->location_.before(loc)) &&
              (!r || r->location_.before(node->location_)))
            {
              r = node;
            }
          }
        }

        Location pdef;

        if (r)
          pdef = r->location_;

        for (auto& [def, node] : st->symtab_->includes)
        {
          // An include before the selected definition is ignored.
          if ((def_ok || def.before(loc)) && (!r || pdef.before(def)))
          {
            r = node->lookdown_first(loc);
            pdef = def;
          }
        }

        if (r)
          return r;

        st = st->scope();
      }

      return {};
    }

    Node lookdown_first(Node that)
    {
      // Find the location of this node in our symbol table.
      return lookdown_first(that->location_);
    }

    Node lookdown_first(const Location& loc)
    {
      // Find this location in our symbol table.
      if (!symtab_)
        return {};

      auto find = symtab_->symbols.find(loc);
      if (find == symtab_->symbols.end())
        return {};

      return find->second.front();
    }

    void bind(const Location& loc)
    {
      // Change the location of the node, find the enclosing scope, and bind the
      // new location to this node in the symbol table.
      location_ = loc;
      auto st = scope();

      if (!st)
        throw std::runtime_error("No symbol table");

      st->symtab_->symbols[loc].push_back(shared_from_this());
    }

    void include(Node target)
    {
      auto st = scope();

      if (!st)
        throw std::runtime_error("No symbol table");

      st->symtab_->includes.emplace_back(location_, target);
    }

    std::string str(size_t level = 0)
    {
      std::stringstream ss;
      ss << indent(level) << "(" << type_.str();

      if (type_ & flag::print)
        ss << " " << location_.view();

      if (symtab_)
        ss << std::endl << symtab_->str(level + 1);

      for (auto child : children)
        ss << std::endl << child->str(level + 1);

      ss << ")";
      return ss.str();
    }
  };

  inline Token::operator Node() const
  {
    return NodeDef::create(*this);
  }

  inline Node Token::operator()(Location loc) const
  {
    return NodeDef::create(*this, loc);
  }

  inline std::string SymtabDef::str(size_t level)
  {
    std::stringstream ss;
    ss << indent(level) << "{";

    for (auto& [loc, sym] : symbols)
    {
      ss << std::endl << indent(level + 1) << loc.view() << " =";

      if (sym.size() == 1)
      {
        ss << " " << sym.back()->type().str();
      }
      else
      {
        for (auto& node : sym)
          ss << std::endl << indent(level + 2) << node->type().str();
      }
    }

    for (auto& [loc, node] : includes)
    {
      ss << std::endl
         << indent(level + 1) << "include " << node->location().view();
    }

    ss << "}";
    return ss.str();
  }
}
