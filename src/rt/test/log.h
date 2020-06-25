// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <iostream>
#include <sstream>

namespace logger
{
  class Log
  {
  private:
    // The only purpose of this is to accumulate a write to cout until the
    // whole thing can be written at once, to prevent multithreaded output
    // from being interleaved.
    std::stringstream ss;
    std::ostream& o;

  public:
    Log(std::ostream& o) : o(o) {}

    template<typename T>
    inline Log& operator<<(T const& value)
    {
      ss << value;
      return *this;
    }

    inline Log& operator<<(std::ostream& (*f)(std::ostream&))
    {
      ss << f;
      o << ss.str();
      std::stringstream().swap(ss);
      return *this;
    }
  };

  inline Log& cout()
  {
    static thread_local Log cout_log(std::cout);
    return cout_log;
  }
}
