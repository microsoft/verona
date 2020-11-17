// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "typed-ast/node.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace verona::ast
{
  /// Print an AST node to the specified stream.
  ///
  /// The node is formatted to fit in the given width.
  void print(
    std::ostream& out, const Node& node, size_t indent = 0, size_t width = 80);

  /// An instance of this class is passed to each Node's print method.
  ///
  /// Nodes should call the `begin` method once, the `field` methods zero or
  /// more times, and finally end with the `finish` method.
  ///
  /// This allows the node implementations to be agnostic to the details of the
  /// format, and ensures all nodes are printed consistently.
  class NodePrinter
  {
  public:
    NodePrinter(std::ostream& out, bool compact, size_t indent, size_t width)
    : out(out), compact(compact), indent(indent), width(width)
    {}

    NodePrinter& begin(std::string_view name);
    void finish();

    NodePrinter& field(const Node& node);
    NodePrinter& field(std::string_view value);
    NodePrinter& field(int64_t value);
    NodePrinter& field(double value);
    NodePrinter& field(bool value);

    template<
      typename T,
      typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
    NodePrinter& field(const std::vector<std::unique_ptr<T>>& elements);

    template<
      typename T,
      typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
    NodePrinter& field(const std::unique_ptr<T>& node)
    {
      assert(node);
      return field(*node);
    }

    template<
      typename T,
      typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
    NodePrinter& optional_field(const std::unique_ptr<T>& node)
    {
      if (node)
        return field(*node);
      else
        return empty_field();
    }

    template<typename T>
    NodePrinter& optional_field(const std::optional<T>& value)
    {
      if (value.has_value())
        return field(*value);
      else
        return empty_field();
    }

    NodePrinter& empty_field();

  private:
    /// Prepare the output to receive the contents of a new field.
    void next_field();

    std::ostream& out;
    bool compact;
    size_t indent;
    size_t width;

    /// This is the depth of each indentation, in spaces.
    static constexpr size_t INDENT_STEP = 2;
  };
}
