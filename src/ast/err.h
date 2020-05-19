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

    bool empty();

    Errors& operator<<(ast::Ast& ast);
    Errors& operator<<(const std::string& s);
    Errors& operator<<(std::ostream& (*f)(std::ostream&));
    Errors& operator<<(const err::endtoken&);

    std::string to_s() const;
  };

  constexpr endtoken end;
}
