// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <optional>
#include <path/path.h>
#include <regex>

namespace langkit
{
  class Parse
  {
  public:
    class Make
    {
      friend class Parse;

    private:
      Node top;
      Node node;
      Location location;
      std::cmatch match_;
      std::optional<std::string> mode_;

    public:
      Make(Source& source)
      {
        node = NodeDef::create(File, {source, 0, 0});
        top = node;
      }

      size_t& pos()
      {
        return location.pos;
      }

      size_t& len()
      {
        return location.len;
      }

      std::pair<size_t, size_t> linecol() const
      {
        return location.linecol();
      }

      const std::cmatch& match() const
      {
        return match_;
      }

      void mode(const std::string& next)
      {
        mode_ = next;
      }

      bool in(Token type)
      {
        return node->type == type;
      }

      bool previous(Token type)
      {
        if (!in(Group))
          return false;

        if (node->children.empty())
          return false;

        return node->children.back()->type == type;
      }

      void add(Token type)
      {
        if ((type != Group) && !in(Group))
          push(Group);

        auto n = NodeDef::create(type, location);
        node->push_back(n);
      }

      void seq(Token type, std::initializer_list<Token> skip = {})
      {
        if (in(Group))
        {
          while (std::find(skip.begin(), skip.end(), node->parent->type) !=
                 skip.end())
          {
            node = node->parent->acquire();
          }

          auto p = node->parent;

          if (p->type == type)
          {
            node = p->acquire();
          }
          else
          {
            auto seq = NodeDef::create(type, location);
            auto group = p->pop_back();
            p->push_back(seq);
            seq->push_back(group);
            node = seq;
          }
        }
        else
        {
          invalid();
        }
      }

      void push(Token type)
      {
        add(type);
        node = node->back();
      }

      void pop(Token type)
      {
        if (in(type))
        {
          extend();
          node = node->parent->acquire();
        }
        else
        {
          invalid();
        }
      }

      void term(std::initializer_list<Token> end = {})
      {
        try_pop(Group);

        for (auto& t : end)
          try_pop(t);
      }

      void extend()
      {
        node->location = node->location.extend(location);
      }

      void invalid()
      {
        if (node->type == Invalid)
          extend();
        else
          add(Invalid);
      }

    private:
      bool try_pop(Token type)
      {
        if (in(type))
        {
          extend();
          node = node->parent->acquire();
          return true;
        }

        return false;
      }

      Node done()
      {
        term();

        while (node->parent)
        {
          add(Unclosed);
          term();
          node = node->parent->acquire();
          term();
        }

        if (node != top)
          throw std::runtime_error("malformed AST");

        return top;
      }
    };

    class Rule
    {
      friend class Parse;

    private:
      std::regex regex;
      std::function<void(Make&)> effect;

    public:
      Rule(const std::string& r, std::function<void(Make&)> effect)
      : effect(effect)
      {
        regex = std::regex("^" + r, std::regex_constants::optimize);
      }
    };

    enum class depth
    {
      file,
      directory,
      subdirectories
    };

    using PostF = std::function<void(Parse&, Node)>;

  private:
    std::string ext;
    depth depth_;

    PostF postfile_;
    PostF postdir_;
    PostF postparse_;
    std::map<const std::string, std::vector<Rule>> rules;
    bool in_post = false;

  public:
    Parse(const std::string& ext, depth depth_) : ext(ext), depth_(depth_) {}

    Parse&
    operator()(const std::string& mode, const std::initializer_list<Rule> r)
    {
      rules[mode].insert(rules[mode].end(), r.begin(), r.end());
      return *this;
    }

    void postfile(PostF f)
    {
      postfile_ = f;
    }

    void postdir(PostF f)
    {
      postdir_ = f;
    }

    void postparse(PostF f)
    {
      postparse_ = f;
    }

    Node parse(const std::string& path)
    {
      Node ast;

      switch (path::type(path))
      {
        case path::Type::File:
        {
          ast = parse_file(path);
          break;
        }

        case path::Type::Directory:
        {
          if (depth_ == depth::file)
            return {};

          ast = parse_directory(path);
          break;
        }

        default:
          return {};
      }

      if (!in_post && postparse_)
      {
        in_post = true;
        postparse_(*this, ast);
        in_post = false;
      }

      return ast;
    }

  private:
    Node parse_file(const std::string& file)
    {
      if (path::extension(file) != ext)
        return {};

      auto filename = path::canonical(file);
      auto source = SourceDef::load(filename);

      if (!source)
        return {};

      auto make = Make(source);
      auto it = source->view().cbegin();
      auto st = it;
      auto end = source->view().cend();

      // Find the start rules.
      auto find = rules.find("start");
      if (find == rules.end())
        throw std::runtime_error("unknown mode: start");

      size_t pos = 0;

      while (it != end)
      {
        bool matched = false;

        for (auto& rule : find->second)
        {
          matched = std::regex_search(it, end, make.match_, rule.regex);

          if (matched)
          {
            pos += make.match_.position();
            size_t len = make.match_.length();
            make.location = {source, pos, len};
            rule.effect(make);

            pos += len;
            it = st + pos;

            if (make.mode_)
            {
              find = rules.find(*make.mode_);
              if (find == rules.end())
                throw std::runtime_error("unknown mode: " + *make.mode_);

              make.mode_ = std::nullopt;
            }
            break;
          }
        }

        if (!matched)
        {
          make.invalid();
          it++;
        }
      }

      auto ast = make.done();

      if (postfile_)
        postfile_(*this, ast);

      return ast;
    }

    Node parse_directory(const std::string& path)
    {
      auto dir = path::to_directory(path::canonical(path));
      Node top = NodeDef::create(Directory, {SourceDef::directory(dir), 0, 0});

      auto files = path::files(dir);

      for (auto& file : files)
      {
        auto filename = path::join(dir, file);
        auto ast = parse_file(filename);

        if (ast)
          top->push_back(ast);
      }

      if (depth_ == depth::subdirectories)
      {
        auto dirs = path::directories(dir);

        for (auto& subdir : dirs)
        {
          auto filename = path::join(dir, subdir);
          auto ast = parse_directory(filename);

          if (ast)
            top->push_back(ast);
        }
      }

      if (top->children.empty())
        return {};

      if (postdir_)
        postdir_(*this, top);

      return top;
    }
  };

  inline Parse::Rule
  operator>>(const std::string& r, std::function<void(Parse::Make&)> make)
  {
    return {r, make};
  }
}
