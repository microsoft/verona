// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "path.h"

#if defined(__linux__)
#  include <limits.h>
#  include <stdlib.h>
#elif defined(__APPLE__)
#  include <sys/syslimits.h>
#  include <mach-o/dyld.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace path
{
  std::string executable()
  {
#ifdef WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(NULL, buf, MAX_PATH);
    return std::string(buf);
#elif defined(__linux__) || defined(__FreeBSD__)
#  ifdef __linux__
    constexpr auto link = "/proc/self/exe";
#  elif defined(__FreeBSD__)
    constexpr auto link = "/proc/curproc/file";
#  endif
    char buf[PATH_MAX];
    return std::string(realpath(link, buf));
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    uint32_t size = PATH_MAX;
    auto result = _NSGetExecutablePath(buf, &size);
    if (result == -1)
      buf[0] = '\0';

    return std::string(buf);
#else
#  error "Unsupported platform"
#endif
  }

  std::string directory(const std::string& path)
  {
    constexpr auto delim =
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
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
