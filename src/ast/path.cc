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
#else
#  error "path::executable not supported on this target."
#endif
  }

  std::string directory(const std::string& path)
  {
#if defined(__linux__)
    size_t pos = path.rfind("/");

    if ((pos == std::string::npos) || (pos >= (path.size() - 1)))
      return path;

    return path.substr(0, pos);
#else
#  error "path::directory not supported on this target."
#endif
  }
}
