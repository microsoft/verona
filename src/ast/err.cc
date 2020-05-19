#include "err.h"

namespace err
{
  bool Errors::empty()
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

    list.back().push_back({ast, std::string()});
    return *this;
  }

  Errors& Errors::operator<<(const std::string& s)
  {
    ss << s;
    return *this;
  }

  Errors& Errors::operator<<(std::ostream& (*f)(std::ostream&))
  {
    ss << f;
    return *this;
  }

  Errors& Errors::operator<<(const err::endtoken&)
  {
    // Finish the last error in this group and start a new one.
    if ((list.size() > 0) && (list.back().size() > 0))
    {
      list.back().back().msg = ss.str();
      ss.str(std::string());
      list.push_back({});
    }

    return *this;
  }

  std::string Errors::to_s() const
  {
    std::ostringstream ss;
    bool first_group = true;

    for (auto& group : list)
    {
      if (first_group)
      {
        first_group = false;
      }
      else if (group.size() > 0)
      {
        ss << "---" << std::endl;
      }

      for (auto& err : group)
      {
        ss << err.ast->path << ":" << err.ast->line << ":" << err.ast->column
           << ": " << err.msg << std::endl;
      }
    }

    return ss.str();
  }
}
