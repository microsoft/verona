// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "token.h"

#include <limits>
#include <map>
#include <set>
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

  using Nodes = std::vector<Node>;
  using NodeIt = Nodes::iterator;
  using NodeRange = std::pair<NodeIt, NodeIt>;
  using NodeSet = std::set<Node, std::owner_less<>>;

  template<typename T>
  using NodeMap = std::map<Node, T, std::owner_less<>>;

  class SymtabDef
  {
    friend class NodeDef;

  private:
    // The location in `symbols` is used as an identifier.
    // The location in `includes` is used for its position in the file.
    std::map<Location, Nodes> symbols;
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
    Nodes children;

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

    Node parent(const Token& type)
    {
      auto p = parent_;

      while (p)
      {
        if (p->type_ == type)
          return p->shared_from_this();

        p = p->parent_;
      }

      return {};
    }

    void set_location(const Location& loc)
    {
      if (!location_.source)
        location_ = loc;
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

    NodeIt insert(NodeIt pos, NodeIt first, NodeIt last)
    {
      if (first == last)
        return pos;

      for (auto it = first; it != last; ++it)
        (*it)->parent_ = this;

      return children.insert(pos, first, last);
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

    Nodes lookup_all(const Location& loc)
    {
      // Find all bindings for this identifier by looking upwards in the symbol
      // table chain.
      Nodes r;
      auto st = scope();

      while (st)
      {
        auto unordered = !(st->type_ & flag::defbeforeuse);
        auto it = st->symtab_->symbols.find(loc);

        if (it != st->symtab_->symbols.end())
        {
          // If we aren't unordered, use only bindings that occur before the
          // identifier's file location.
          for (auto& def : it->second)
          {
            if (unordered || def->location_.before(loc))
              r.push_back(def);
          }
        }

        for (auto& [def, target] : st->symtab_->includes)
        {
          // If we aren't unordered, use only includes that occur before the
          // identifier's file location.
          if (unordered || def.before(loc))
          {
            // Use all bindings, as order is meaningless through an include.
            auto find = target->symtab_->symbols.find(loc);
            if (find != target->symtab_->symbols.end())
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
      Node r;
      auto st = scope();

      while (st)
      {
        auto unordered = !(st->type_ & flag::defbeforeuse);
        auto it = st->symtab_->symbols.find(loc);

        if (it != st->symtab_->symbols.end())
        {
          // Select the last definition. In a defbeforeuse context, select the
          // last definition before the use site.
          for (auto& def : it->second)
          {
            if (
              (unordered || def->location_.before(loc)) &&
              (!r || r->location_.before(def->location_)))
            {
              r = def;
            }
          }
        }

        for (auto& [def, target] : st->symtab_->includes)
        {
          // An include before the use site or current result is ignored.
          if (
            (unordered || def.before(loc)) && (!r || r->location().before(def)))
          {
            r = *target->lookdown(loc).first;
          }
        }

        if (r)
          break;

        st = st->scope();
      }

      return r;
    }

    NodeRange lookdown(Node that)
    {
      // Find the location of this node in our symbol table.
      return lookdown(that->location_);
    }

    NodeRange lookdown(const Location& loc)
    {
      // This is used for scoped resolution, where we're looking in this symbol
      // table specifically. Don't use includes, as those are for lookup only.
      if (!symtab_)
        return {};

      auto find = symtab_->symbols.find(loc);
      if (find == symtab_->symbols.end())
        return {};

      return {find->second.begin(), find->second.end()};
    }

    void bind(const Location& loc)
    {
      // Find the enclosing scope and bind the new location to this node in the
      // symbol table.
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

    Location fresh()
    {
      auto st = scope();

      if (st)
        return st->symtab_->fresh();

      if (!symtab_)
        throw std::runtime_error("No symbol table");

      return symtab_->fresh();
    }

    Location unique()
    {
      auto p = this;

      while (p->parent_)
        p = p->parent_;

      assert(p->type_ == Top);
      return p->symtab_->fresh();
    }

    Node clone()
    {
      // This doesn't preserve the symbol table.
      auto node = create(type_, location_);

      for (auto& child : children)
        node->push_back(child->clone());

      return node;
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

  inline TokenDef::operator Node() const
  {
    return NodeDef::create(Token(*this));
  }

  inline Token::operator Node() const
  {
    return NodeDef::create(*this);
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

  inline std::ostream& operator<<(std::ostream& os, const Node& node)
  {
    if (node)
      os << node->str() << std::endl;
    return os;
  }

  inline std::ostream& operator<<(std::ostream& os, const NodeRange& range)
  {
    for (auto it = range.first; it != range.second; ++it)
      os << (*it)->str();
    return os;
  }
}
