#pragma once

#include <iostream>
#include <type_traits>
#include <fmt/ostream.h>


class InternalError
{
  public:
    InternalError() {}

    [[ noreturn ]] ~InternalError()
    {
        abort();
    }

    template <typename S, typename... Args>
    [[ noreturn ]] void print(const S& format_str, Args&&... args) {
        fmt::print(std::cerr, format_str, std::forward<Args>(args)...);
        abort();
    }
};

inline InternalError operator<<(InternalError out, 
    std::ostream& (*f)(std::ostream&)
)
{
    std::cerr << f;
    return out;
}

template <typename T>
inline InternalError operator<<(InternalError out, T& p)
{
    std::cerr << p;
    return out;
}

template <typename T>
inline InternalError operator<<(InternalError out, const T& p)
{
    std::cerr << p;
    return out;
}
