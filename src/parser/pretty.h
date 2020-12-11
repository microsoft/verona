// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <memory>
#include <ostream>

namespace verona::parser
{
  struct separator
  {};
  /// Print this to the `PrettyStream` to allow a line break, if line break is
  /// not required, will print a space.
  constexpr separator sep;

  /// Print this to the `PrettyStream` to start a bracketed expression.
  struct start
  {
    std::string_view label;

    char bracket;

    /// Takes a label, `l`, and a bracket type, `b` to use.
    constexpr start(const char* l, char b = '(') : label(l), bracket(b) {}
  };

  struct quote
  {};

  /// Print this to the `PrettyStream` to print a quote.
  constexpr quote q;

  /// Print this to the `PrettyStream` to complete a bracketed expression.
  struct endtoken
  {
    char bracket;
    constexpr endtoken(char b = ')') : bracket(b) {}
  };

  /// Print this to the `PrettyStream` to complete a bracketed expression with a
  /// `)`.
  constexpr endtoken end{};

  /// Private Implementation Details
  class PrettyStreamImpl;

  /**
   * Pretty printing stream that prints bracketed expressions, and breaks line-
   * s nicely.
   */
  class PrettyStream
  {
    /**
     * Private Implementation Details
     *
     * Note using pointer for encapsulation in C++, otherwise leaks too much
     * implementation details into the header. Someone should write a
     * better programming language for this sort of thing.
     */
    std::unique_ptr<PrettyStreamImpl> impl;

  public:
    /// out - is the underlying output stream to pretty print to
    /// width - is the maximum output width.
    PrettyStream(std::ostream& out, uint16_t width);

    ~PrettyStream();

    /// Start a bracketed expression
    PrettyStream& operator<<(const start& st);

    /// Output a quote.
    PrettyStream& operator<<(const quote& q);

    /// Output a separator, may be a line break or a space.
    PrettyStream& operator<<(const separator& sep);

    /// Close a bracketed expression
    PrettyStream& operator<<(const endtoken& end);

    /// Output a string view
    PrettyStream& operator<<(const std::string_view& st);

    /// Output a string
    PrettyStream& operator<<(const std::string& st);

    /// Output a c-string
    PrettyStream& operator<<(const char* st);

    /// Flush the output of this stream
    void flush();
  };
}
