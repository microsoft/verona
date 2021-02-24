// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "pretty.h"

#include <cassert>
#include <deque>
#include <variant>

namespace verona::parser
{
  /// Indent by this number of characters.
  constexpr size_t indent_count = 2;

  struct indent_token
  {};
  /// Print this to the `IndentStream` to increase the indent level.
  constexpr indent_token indent;

  struct undent_token
  {};
  /// Print this to the `IndentStream` to decrease the indent level.
  constexpr undent_token undent;

  /**
   * This class encapsulates a few simple features for pretty printing
   *  - Indentation
   *  - Consolidating new lines
   *  - Swollowing newlines
   */
  class IndentStream
  {
    /// Stream for outputting to
    std::ostream& underlying_output;

    /// Current indent level
    uint16_t indent = 0;

    /// Width of lines for output
    uint16_t width;

    /// Set if we are on a new line
    bool empty_line = true;

    /// If false, then new lines will be ignored.
    bool ignore_new_lines = false;

  public:
    IndentStream(std::ostream& underlying_output, uint16_t width)
    : underlying_output(underlying_output), width(width)
    {}

    /// How long is a line given current indent levels.
    int line_width()
    {
      return width - indent;
    }

    /// Set if new lines should be ignored.
    void set_ignore_new_lines(bool b)
    {
      ignore_new_lines = b;
    }

    IndentStream& operator<<(const indent_token& _)
    {
      indent += indent_count;
      return *this;
    }

    IndentStream& operator<<(const undent_token& _)
    {
      indent -= indent_count;
      return *this;
    }

    // For new line.
    IndentStream& operator<<(std::ostream& (*f)(std::ostream&))
    {
      if (!ignore_new_lines)
      {
        if (!empty_line) // Remove repeated newlines.
        {
          empty_line = true;
          underlying_output << std::endl;
        }
      }
      return *this;
    }

    template<typename T>
    IndentStream& operator<<(T t)
    {
      if (empty_line)
      {
        if (indent != 0)
          underlying_output << std::string(indent, ' ');
        empty_line = false;
      }
      underlying_output << t;
      return *this;
    }
  };

  /// Boiler plate for using variants with lambdas.
  template<class... Ts>
  struct overload : Ts...
  {
    using Ts::operator()...;
  };
  template<class... Ts>
  overload(Ts...)->overload<Ts...>;

  /**
   * Implementation of pretty printing.
   *
   * This uses a queue of tokens that are to be printed with special
   *  - start
   *  - separator
   *  - end
   * to bracket a term.
   */
  class PrettyStreamImpl
  {
    using Tokens = std::
      variant<start, endtoken, separator, std::string_view, std::string, char>;

    /// output stream to print to
    IndentStream underlying_output;

    /// Currently unprinted tokens. We need a buffer of these to decide how
    /// to pring the brackets, i.e. whether it can fit on a single line or not.
    std::deque<Tokens> tokens;

    /// Is the last token a separator, as we ignore repeated separators.
    bool last_separator = false;

    /// Current length of underlying_output in characters when printed on a
    /// single line.
    size_t length = 0;

    /// Returns the length of a particular token in characters.
    size_t token_length(Tokens t)
    {
      return std::visit(
        overload{[](start& s) { return s.label.length() + 1; },
                 [](endtoken& _) { return (size_t)1; },
                 [](separator& _) { return (size_t)1; },
                 [](std::string_view& s) { return s.length(); },
                 [](std::string& s) { return s.size(); },
                 [](char& _) { return (size_t)1; }},
        t);
    }

    /**
     * Looks at the queue and decides how many tokens should be
     * printing in a single go.
     */
    uint16_t tokens_to_print()
    {
      size_t line_budget = underlying_output.line_width();
      int nesting = 0;
      uint16_t count = 0;

      if (tokens.empty())
        return 0;

      for (auto token : tokens)
      {
        count++;
        if (line_budget < token_length(token))
          // We cannot print this as a whole term, so print just it.
          return 1;
        line_budget -= token_length(token);

        if (std::holds_alternative<endtoken>(token))
          nesting--;

        if (std::holds_alternative<start>(token))
          nesting++;

        if (nesting <= 0)
          return count;
      }
      // Don't know what to do, need more input
      return 0;
    }

    /// Prints the front character in the queue and removes it.
    void print_front(bool break_lines)
    {
      assert(!tokens.empty());
      // Pop a token
      Tokens token = tokens.front();
      tokens.pop_front();
      length -= token_length(token);

      // Disable or enable new lines in output stream
      // This means we can have less branching in the
      // printing code below.
      underlying_output.set_ignore_new_lines(!break_lines);

      std::visit(
        [this, break_lines](auto&& token) {
          using T = std::decay_t<decltype(token)>;
          if constexpr (std::is_same_v<T, separator>)
          {
            underlying_output << std::endl;
            if (!break_lines)
              underlying_output << ' ';
          }
          else if constexpr (std::is_same_v<T, endtoken>)
            underlying_output << undent << token.bracket;
          else if constexpr (std::is_same_v<T, start>)
            underlying_output << std::endl
                              << token.bracket << token.label << std::endl
                              << indent;
          else
            underlying_output << token;
        },
        token);
    }

  public:
    PrettyStreamImpl(std::ostream& output, uint16_t width = 80)
    : underlying_output(output, width)
    {}

    /// Append a character into the stream.
    void append(Tokens t)
    {
      // Skip repeated separators.
      if (std::exchange(last_separator, std::holds_alternative<separator>(t)))
      {
        if (last_separator)
          return;
      }

      length += token_length(t);
      tokens.push_back(t);

      if (length >= underlying_output.line_width())
      {
        flush();
      }
    }

    /// Flush as much of the stream of tokens as we currently can.
    void flush()
    {
      uint16_t to_print;
      while ((to_print = tokens_to_print()) != 0)
      {
        bool break_lines = to_print == 1;
        for (int i = 0; i < to_print; i++)
        {
          print_front(break_lines);
        }
      }
    }
  };

  PrettyStream& PrettyStream::operator<<(const quote& q)
  {
    impl->append('"');
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const separator& sep)
  {
    impl->append(sep);
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const endtoken& end)
  {
    impl->append(end);
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const start& st)
  {
    impl->append(sep);
    impl->append(st);
    impl->append(sep);
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const std::string_view& st)
  {
    impl->append(st);
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const std::string& st)
  {
    impl->append(st);
    return *this;
  }

  PrettyStream& PrettyStream::operator<<(const char* st)
  {
    impl->append((std::string_view)st);
    return *this;
  }

  void PrettyStream::flush()
  {
    impl->flush();
  }

  PrettyStream::~PrettyStream() {}

  PrettyStream::PrettyStream(std::ostream& out, uint16_t width)
  {
    /// Allocation for abstraction, thanks C++
    impl = std::make_unique<PrettyStreamImpl>(out, width);
  }
}
