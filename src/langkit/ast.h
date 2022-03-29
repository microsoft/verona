// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

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

  class Token;
  class NodeDef;
  using Node = std::shared_ptr<NodeDef>;
  using NodeIt = std::vector<Node>::iterator;
  using NodeRange = std::pair<NodeIt, NodeIt>;
  using Binding = std::pair<Token, Node>;

  class Token
  {
  public:
    using flag = uint64_t;

  private:
    const char* name;
    flag fl;

  public:
    constexpr Token(const char* name, flag fl = 0) : name(name), fl(fl) {}

    operator Node() const;

    bool operator&(flag f) const
    {
      return (fl & f) != 0;
    }

    Node operator()(Location loc) const;

    Binding operator=(Node n) const
    {
      return {*this, n};
    }

    bool operator==(const Token& that) const
    {
      return name == that.name;
    }

    bool operator!=(const Token& that) const
    {
      return name != that.name;
    }

    bool operator<(const Token& that) const
    {
      return name < that.name;
    }

    bool operator>(const Token& that) const
    {
      return name > that.name;
    }

    bool operator<=(const Token& that) const
    {
      return name <= that.name;
    }

    bool operator>=(const Token& that) const
    {
      return name >= that.name;
    }

    const char* str() const
    {
      return name;
    }
  };

  namespace flag
  {
    constexpr Token::flag none = 0;
    constexpr Token::flag print = 1 << 0;
    constexpr Token::flag symtab = 1 << 1;
    constexpr Token::flag defbeforeuse = 1 << 2;
  }

  constexpr auto Invalid = Token("invalid");
  constexpr auto Unclosed = Token("unclosed");
  constexpr auto Group = Token("group");
  constexpr auto File = Token("file");
  constexpr auto Directory = Token("directory");

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

  class NodeDef : public std::enable_shared_from_this<NodeDef>
  {
  public:
    Token type;
    Location location;
    Symtab symtab;
    NodeDef* parent;
    std::vector<Node> children;

  private:
    NodeDef(const Token& type, Location location)
    : type(type), location(location), parent(nullptr)
    {
      if (type & flag::symtab)
        symtab = std::make_shared<SymtabDef>();
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

      Location loc = (*range.first)->location;
      loc.extend((*(range.second - 1))->location);

      return std::shared_ptr<NodeDef>(new NodeDef(type, loc));
    }

    Node acquire()
    {
      return shared_from_this();
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
      node->parent = this;
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

    Node pop_back()
    {
      if (children.empty())
        return {};

      auto node = children.back();
      children.pop_back();
      node->parent = nullptr;
      return node;
    }

    Node scope()
    {
      auto p = parent;

      while (p)
      {
        auto node = p->acquire();

        if (node->symtab)
          return node;

        p = node->parent;
      }

      return {};
    }

    std::vector<Node> find_all(const Location& loc)
    {
      std::vector<Node> r;
      auto st = scope();

      while (st)
      {
        auto def_ok = !(st->type & flag::defbeforeuse);
        auto it = st->symtab->symbols.find(loc);

        if (it != st->symtab->symbols.end())
        {
          if (def_ok)
          {
            r.insert(r.end(), it->second.begin(), it->second.end());
          }
          else
          {
            for (auto& node : it->second)
            {
              if (node->location.before(loc))
                r.push_back(node);
            }
          }
        }

        for (auto& [def, node] : st->symtab->includes)
        {
          if (def_ok || def.before(loc))
          {
            auto find = node->symtab->symbols.find(loc);

            if (find != node->symtab->symbols.end())
              r.insert(r.end(), find->second.begin(), find->second.end());
          }
        }

        st = st->scope();
      }

      return r;
    }

    Node find_first()
    {
      return find_first(location);
    }

    Node find_first(const Location& loc)
    {
      auto st = scope();

      while (st)
      {
        auto def_ok = !(st->type & flag::defbeforeuse);
        auto it = st->symtab->symbols.find(loc);
        Node r;

        if (it != st->symtab->symbols.end())
        {
          // Select the last definition. In a defbeforeuse context, select the
          // last definition before the use site.
          for (auto& node : it->second)
          {
            if (
              (def_ok || node->location.before(loc)) &&
              (!r || r->location.before(node->location)))
            {
              r = node;
            }
          }
        }

        Location pdef;

        if (r)
          pdef = r->location;

        for (auto& [def, node] : st->symtab->includes)
        {
          // An include before the selected definition is ignored.
          if ((def_ok || def.before(loc)) && (!r || pdef.before(def)))
          {
            r = node->at(loc);
            pdef = def;
          }
        }

        if (r)
          return r;

        st = st->scope();
      }

      return {};
    }

    Node at(Node that)
    {
      return at(that->location);
    }

    Node at(const Location& loc)
    {
      if (!symtab)
        return {};

      auto find = symtab->symbols.find(loc);
      if (find == symtab->symbols.end())
        return {};

      return find->second.front();
    }

    void bind(const Location& loc, Node node)
    {
      auto st = scope();

      if (!st)
        throw std::runtime_error("No symbol table");

      st->symtab->symbols[loc].push_back(node);
    }

    void include(Node node)
    {
      auto st = scope();

      if (!st)
        throw std::runtime_error("No symbol table");

      st->symtab->includes.emplace_back(this->location, node);
    }

    std::string str(size_t level = 0)
    {
      std::stringstream ss;
      ss << indent(level) << "(" << type.str();

      if (type & flag::print)
        ss << " " << location.view();

      if (symtab)
        ss << std::endl << symtab->str(level + 1);

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
        ss << " " << sym.back()->type.str();
      }
      else
      {
        for (auto& node : sym)
          ss << std::endl << indent(level + 2) << node->type.str();
      }
    }

    for (auto& [loc, node] : includes)
    {
      ss << std::endl
         << indent(level + 1) << "include " << node->location.view();
    }

    ss << "}";
    return ss.str();
  }
}
