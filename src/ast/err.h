// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <sstream>
#include <vector>

namespace err
{
  struct endtoken
  {};

  struct Error
  {
    ast::Ast ast;
    std::string msg;
  };

  struct Errors
  {
    std::vector<std::vector<Error>> list;
    std::ostringstream ss;

    bool empty() const;

    Errors& operator<<(ast::Ast& ast);
    Errors& operator<<(const std::string& s);
    Errors& operator<<(size_t s);
    Errors& operator<<(std::ostream& (*f)(std::ostream&));
    Errors& operator<<(const Errors& err);
    Errors& operator<<(const endtoken&);

    std::string to_s() const;

    friend std::ostream& operator<<(std::ostream& out, const Errors& err)
    {
      if (err.empty())
        return out;

      bool first_group = true;

      for (auto& group : err.list)
      {
        if (first_group)
        {
          first_group = false;
        }
        else if (group.size() > 0)
        {
          out << "---\n";
        }

        for (auto& e : group)
        {
          if (e.ast)
          {
            out << e.ast->path << ":" << e.ast->line << ":" << e.ast->column
                << ": ";
          }

          out << e.msg;

          if (e.msg.empty() || (e.msg.back() != '\n'))
            out << "\n";
        }
      }

      return out;
    }
  };

  constexpr endtoken end;
}
