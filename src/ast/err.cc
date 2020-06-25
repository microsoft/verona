// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "err.h"

namespace err
{
  bool Errors::empty() const
  {
    return list.size() == 0;
  }

  Errors& Errors::operator<<(ast::Ast& ast)
  {
    if (list.size() == 0)
    {
      // Create a group if none exist.
      list.push_back({});
    }
    else if (list.back().size() > 0)
    {
      // Finish the previous error in this group.
      list.back().back().msg = ss.str();
      ss.str(std::string());
    }

    list.back().push_back({ast, {}});
    return *this;
  }

  Errors& Errors::operator<<(const std::string& s)
  {
    ss << s;
    return *this;
  }

  Errors& Errors::operator<<(size_t s)
  {
    ss << s;
    return *this;
  }

  Errors& Errors::operator<<(std::ostream& (*f)(std::ostream&))
  {
    ss << f;
    return *this;
  }

  Errors& Errors::operator<<(const Errors& err)
  {
    list.insert(list.end(), err.list.begin(), err.list.end());
    return *this;
  }

  Errors& Errors::operator<<(const endtoken&)
  {
    auto msg = ss.str();
    ss.str(std::string());

    if ((list.size() > 0) && (list.back().size() > 0))
    {
      // Finish the last error in this group and start a new one.
      list.back().back().msg = msg;
      list.push_back({});
    }
    else if (!msg.empty())
    {
      list.push_back({});
      list.back().push_back({{}, msg});
    }

    return *this;
  }

  std::string Errors::to_s() const
  {
    std::ostringstream ss;
    ss << *this;
    return ss.str();
  }
}
