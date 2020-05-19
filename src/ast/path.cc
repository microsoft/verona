#include "path.h"

#if defined(__linux__)
#  include <limits.h>
#  include <stdlib.h>
#endif

namespace path
{
  std::string executable()
  {
#if defined(__linux__)
    constexpr auto link = "/proc/self/exe";
    char buffer[PATH_MAX];
    return realpath(link, buffer);
#elif defined(_WIN32)
    char p[MAX_PATH];
    GetModuleFileName(NULL, p, MAX_PATH);
    return std::string(p);
#else
#  error "path::executable not supported on this target."
#endif
  }

  std::string directory(const std::string& path)
  {
    constexpr auto delim =
#if defined(__linux__)
      "/"
#elif defined(_WIN32)
      "\\"
#else
#  error "path::directory not supported on this target."
#endif
      ;

    size_t pos = path.rfind(delim);

    if ((pos == std::string::npos) || (pos >= (path.size() - 1)))
      return path;

    return path.substr(0, pos);
  }
}
