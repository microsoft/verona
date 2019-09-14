// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "interpreter/bytecode.h"

#include "ds/helpers.h"

#include <fmt/ostream.h>

namespace verona::bytecode
{
  std::ostream& operator<<(std::ostream& out, const Register& self)
  {
    fmt::print(out, "r{:d}", self.index);
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const BinaryOperator& self)
  {
    switch (self)
    {
      case BinaryOperator::Add:
        fmt::print(out, "ADD");
        break;
      case BinaryOperator::Sub:
        fmt::print(out, "SUB");
        break;
      case BinaryOperator::Lt:
        fmt::print(out, "LT");
        break;
      case BinaryOperator::Le:
        fmt::print(out, "LE");
        break;
      case BinaryOperator::Gt:
        fmt::print(out, "GT");
        break;
      case BinaryOperator::Ge:
        fmt::print(out, "GE");
        break;
      case BinaryOperator::Eq:
        fmt::print(out, "EQ");
        break;
      case BinaryOperator::Ne:
        fmt::print(out, "NE");
        break;
      case BinaryOperator::And:
        fmt::print(out, "AND");
        break;
      case BinaryOperator::Or:
        fmt::print(out, "OR");
        break;

        EXHAUSTIVE_SWITCH;
    }
    return out;
  }
}
