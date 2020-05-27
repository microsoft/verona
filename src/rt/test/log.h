// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "aal/aal.h"

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

    template<typename T>
    inline Log& trace(
      const char* prefix,
      const T* label,
      size_t value,
      uint64_t timestamp = snmalloc::Aal::tick())
    {
      ss << prefix << "," << label << "," << timestamp << "," << value << "\n";
      return *this;
    }
  };

  inline Log& cout()
  {
    static thread_local Log cout_log(std::cout);
    return cout_log;
  }
}
