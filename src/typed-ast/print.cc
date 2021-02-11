// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "typed-ast/print.h"

#include "typed-ast/ast.h"

#include <iomanip>
#include <vector>

namespace verona::ast
{
  namespace
  {
    /// Custom ostreambuf which counts the number of outputted characters.
    /// The actual contents printed to the stream are discarded.
    struct length_ostreambuf : public std::streambuf
    {
    public:
      /// Get the number of characters which have been written to this streambuf
      /// since its creation.
      size_t size() const
      {
        return size_;
      }

    protected:
      int overflow(int c) override
      {
        // We don't care about the actual value of the character (other than
        // EOF), only that we're writing one more character.
        if (c != EOF)
          size_ += 1;
        return c;
      }

      // The first argument is the actual data, which we don't care about.
      // We're only interested in the length, which is in the second argument.
      std::streamsize xsputn(const char*, std::streamsize n) override
      {
        size_ += n;
        return n;
      };

    private:
      size_t size_ = 0;
    };

    /// Print a quoted string to `out`, escaping any special characters.
    // TODO: This is a very primitive implementation, which almost certainly
    // doesn't handle non-ASCII character correctly. It's good enough for
    // printing AST dumps.
    void print_string(std::ostream& out, std::string_view s)
    {
      out << '"';
      for (char c : s)
      {
        if (c == '\\' || c == '"')
          out << '\\' << c;
        else if (c == '\a')
          out << "\\a";
        else if (c == '\b')
          out << "\\b";
        else if (c == 0x1B) // \e is a GCC extension
          out << "\\e";
        else if (c == '\f')
          out << "\\f";
        else if (c == '\n')
          out << "\\n";
        else if (c == '\r')
          out << "\\r";
        else if (c == '\t')
          out << "\\t";
        else if (c == '\v')
          out << "\\v";
        else if (c == '\0')
          out << "\\0";
        else if (c >= 0x20 && c < 0x7f)
          out << c;
        else
        {
          unsigned int code =
            static_cast<unsigned int>(static_cast<unsigned char>(c));
          std::ios_base::fmtflags f(out.flags());
          out << std::hex << std::setw(2) << std::setfill('0');
          out << "\\x" << code;
          out.flags(f);
        }
      }
      out << '"';
    }

    void print_compact(std::ostream& out, const Node& node)
    {
      NodePrinter printer(out, true, 0, 0);
      node.print(printer);
    }

    void print_extended(
      std::ostream& out, const Node& node, size_t indent, size_t width)
    {
      NodePrinter printer(out, false, indent, width);
      node.print(printer);
    }

    /// Compute the length that would be occupied if the node was printed in its
    /// compact form.
    ///
    /// We do this by running the actual printing code, but on a special ostream
    /// the counts the number of printed characters.
    size_t compact_length(const Node& node)
    {
      length_ostreambuf buf;
      std::ostream out(&buf);
      print_compact(out, node);
      return buf.size();
    }
  }

  void print(std::ostream& out, const Node& node, size_t indent, size_t width)
  {
    // We first try to print the node in its compact form. If this takes more
    // space than what is available, we fall back to the extended form.
    //
    // TODO: there is a possible quadratic blow up involved here: we first
    // compute the length of the top-level node, which involves traversing each
    // child node. Assuming we pick the extended format, we'll be calling
    // `print` on every direct sub-node, which will again traverse all of their
    // children to compute the length. Some memoization, using a `Node* ->
    // size_t` map tracking the size of every subnode, would help if this ever
    // becomes a bottleneck.
    size_t len = compact_length(node);
    if (indent + len < width)
    {
      out << std::string(indent, ' ');
      print_compact(out, node);
    }
    else
    {
      print_extended(out, node, indent, width);
    }
  }

  NodePrinter& NodePrinter::begin(std::string_view name)
  {
    if (!compact)
    {
      out << std::string(indent, ' ');
      indent += INDENT_STEP;
    }
    out << "(" << name;
    return *this;
  }

  void NodePrinter::finish()
  {
    out << ")";
    if (!compact)
      indent -= INDENT_STEP;
  }

  void NodePrinter::next_field()
  {
    if (compact)
      out << " ";
    else
      out << "\n";
  }

  NodePrinter& NodePrinter::field(const Node& node)
  {
    next_field();
    if (compact)
    {
      print_compact(out, node);
    }
    else
    {
      // Even if the current node is formatted in its extended form, nested
      // nodes could be formatted in the compact form, if they fit. We use the
      // top-level `print` method to make that decision again.
      print(out, node, indent, width);
    }
    return *this;
  }

  NodePrinter& NodePrinter::field(std::string_view value)
  {
    next_field();
    out << std::string(indent, ' ');
    print_string(out, value);
    return *this;
  }

  NodePrinter& NodePrinter::field(int64_t value)
  {
    next_field();
    out << std::string(indent, ' ') << value;
    return *this;
  }

  NodePrinter& NodePrinter::field(double value)
  {
    next_field();
    out << std::string(indent, ' ') << value;
    return *this;
  }

  NodePrinter& NodePrinter::field(bool value)
  {
    next_field();
    out << std::string(indent, ' ') << value;
    return *this;
  }

  template<typename T, typename>
  NodePrinter&
  NodePrinter::field(const std::vector<std::unique_ptr<T>>& elements)
  {
    next_field();
    if (compact)
    {
      out << "[";
      bool first = true;
      for (auto& elem : elements)
      {
        if (!first)
          out << " ";
        else
          first = false;

        print_compact(out, *elem);
      }
      out << "]";
    }
    else
    {
      out << std::string(indent, ' ') << "[\n";
      indent += INDENT_STEP;

      for (auto& elem : elements)
      {
        print(out, *elem, indent, width);
        out << "\n";
      }

      indent -= INDENT_STEP;
      out << std::string(indent, ' ') << "])";
    }
    return *this;
  }

  NodePrinter& NodePrinter::empty_field()
  {
    next_field();
    out << std::string(indent, ' ') << "()";
    return *this;
  }

  template NodePrinter&
  NodePrinter::field(const std::vector<std::unique_ptr<Node>>& elements);
  template NodePrinter&
  NodePrinter::field(const std::vector<std::unique_ptr<Expr>>& elements);
  template NodePrinter&
  NodePrinter::field(const std::vector<std::unique_ptr<MemberDef>>& elements);

  void NewExpr::print(NodePrinter& out) const
  {
    out.begin("new").field(elements).optional_field(region).finish();
  }
}
