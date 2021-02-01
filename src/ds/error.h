#pragma once

#include <iostream>
#include <type_traits>
#include <fmt/ostream.h>

class InternalError
{
  public:
    template <typename S, typename... Args>
    [[ noreturn ]] static void print(const S& format_str, Args&&... args) {
        fmt::print(std::cerr, format_str, std::forward<Args>(args)...);
        abort();
    }
};