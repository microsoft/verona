// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace verona::ast
{
  /// This represents identifiers that appear in the source code.
  typedef std::string Symbol;

  /// This represents a location in the source.
  /// TODO: reuse the code from `compiler/source_manager.h` for a more efficient
  /// representation.
  struct SourceLocation
  {
    // TODO: Using an std::string here is incredibly wasteful, but is the
    // easiest solution until we the memory management of the AST settles down.
    std::string file;
    size_t line;
    size_t column;

    explicit SourceLocation(std::string file, size_t line, size_t column)
    : file(file), line(line), column(column)
    {}
  };

  /// Base class from which all AST nodes derive from.
  class Node
  {
    SourceLocation location;

  public:
    /// Get the SourceLocation corresponding to the beginning of the node.
    /// TODO: We should have a `getEndLocation` as well, but peglib does not
    /// provide us with this information.
    SourceLocation getBeginLocation()
    {
      return location;
    }

  protected:
    explicit Node(SourceLocation location) : location(location) {}
  };
}
