// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/ostream.h>
#include <iostream>
#include <type_traits>

class InternalError
{
public:
  template<typename S, typename... Args>
  [[noreturn]] static void print(const S& format_str, Args&&... args)
  {
    fmt::print(std::cerr, format_str, std::forward<Args>(args)...);
    abort();
  }
};