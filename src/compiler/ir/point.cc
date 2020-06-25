// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/point.h"

#include "compiler/ir/ir.h"
#include "ds/helpers.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  std::ostream& operator<<(std::ostream& out, const IRPoint& point)
  {
    match(
      point.offset,
      [&](const IRPoint::Entry&) {
        fmt::print(out, "{}:entry", *point.basic_block);
      },
      [&](const IRPoint::Statement& stmt) {
        fmt::print(out, "{}:statement({})", *point.basic_block, stmt.index);
      },
      [&](const IRPoint::Terminator&) {
        fmt::print(out, "{}:terminator", *point.basic_block);
      });
    return out;
  }
}
