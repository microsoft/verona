// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "path.h"

#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__linux__)
#  include <dirent.h>
#  include <unistd.h>
#elif defined(__FreeBSD__)
#  include <dirent.h>
#  include <sys/sysctl.h>
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <dirent.h>
#  include <mach-o/dyld.h>
#  include <sys/syslimits.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace path
{
  constexpr size_t length(const char* str)
  {
    return *str ? 1 + length(str + 1) : 0;
  }

  constexpr auto delim =
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    "/"
#elif defined(_WIN32)
    "\\"
#else
#  error "path::delim not supported on this target."
#endif
    ;
  constexpr auto delim_len = length(delim);

  constexpr auto src_delim =
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    delim
#else
    "/"
#endif
    ;
  constexpr auto src_delim_len = length(src_delim);

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
#  error "path::executable not supported on this target."
#endif
  }

  std::string directory(const std::string& path)
  {
    if (is_directory(path))
      return path;

    size_t pos = path.rfind(delim);

    if (pos == std::string::npos)
      return {};

    return path.substr(0, pos + delim_len);
  }

  std::string filename(const std::string& path)
  {
    if (is_directory(path))
      return {};

    size_t pos = path.rfind(delim);

    if (pos == std::string::npos)
      return path;

    return path.substr(pos + delim_len);
  }

  std::string join(const std::string& path1, const std::string& path2)
  {
    if (!is_relative(path2))
      return path2;

    return directory(path1).append(path2);
  }

  std::string to_directory(const std::string& path)
  {
    if (is_directory(path))
      return path;

    return path + delim;
  }

  std::string extension(const std::string& path)
  {
    auto pos = path.rfind('.');

    if (pos == std::string::npos)
      return {};

    auto d = path.rfind(delim);

    if ((d != std::string::npos) && (pos <= d))
      return {};

    return path.substr(pos + 1);
  }

  std::string to_platform(const std::string& path)
  {
    auto result = path;

    if constexpr (delim != src_delim)
    {
      auto pos = result.find(src_delim, src_delim_len);

      while (pos != std::string::npos)
      {
        result.replace(pos, src_delim_len, delim, delim_len);
        pos = result.find(delim, pos);
      }
    }

    return result;
  }

  std::string canonical(const std::string& path)
  {
#ifdef _WIN32
    char resolved[FILENAME_MAX];

    if (
      (GetFullPathName(path.c_str(), FILENAME_MAX, resolved, NULL) == 0) ||
      (GetFileAttributes(resolved) == INVALID_FILE_ATTRIBUTES))
    {
      return {};
    }
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    char resolved[PATH_MAX];

    if (realpath(path.c_str(), resolved) == NULL)
      return {};

    // Win32 includes a trailing delimiter but POSIX does not.
    if (is_directory(path))
      ::strcat(resolved, delim);
#endif

    return std::string(resolved);
  }

  std::vector<std::string> files(const std::string& path)
  {
    if (!is_directory(path))
      return {};

    std::vector<std::string> r;

#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    DIR* dir = opendir(path.c_str());

    if (dir == nullptr)
      return {};

    dirent* e;

    while ((e = readdir(dir)) != nullptr)
    {
      switch (e->d_type)
      {
        case DT_REG:
        {
          r.push_back(e->d_name);
          break;
        }

        case DT_LNK:
        case DT_UNKNOWN:
        {
          auto name = join(path, e->d_name);

          if (is_file(name))
            r.push_back(name);
          break;
        }
      }
    }

    closedir(dir);
#elif defined(_WIN32)
    constexpr auto mask = FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_DIRECTORY |
      FILE_ATTRIBUTE_REPARSE_POINT;
    WIN32_FIND_DATAA ffd;
    auto search = path + "*";
    auto handle = FindFirstFileA(search.c_str(), &ffd);

    while (handle != INVALID_HANDLE_VALUE)
    {
      if ((ffd.dwFileAttributes & mask) == 0)
        r.push_back(ffd.cFileName);

      if (!FindNextFile(handle, &ffd))
      {
        FindClose(handle);
        handle = INVALID_HANDLE_VALUE;
      }
    }
#else
#  error "path::files not supported on this target."
#endif

    return r;
  }

  bool is_relative(const std::string& path)
  {
    if (path.empty())
      return true;

    if (path.compare(0, delim_len, delim) == 0)
      return false;

#if defined(_WIN32)
    if ((path.size() >= (delim_len + 2)) && (path[1] == ':'))
    {
      auto c = path[0];

      if (((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z')))
        return false;
    }
#endif

    return true;
  }

  bool is_directory(const std::string& path)
  {
    // A directory name always has a trailing delimiter.
    if (path.size() < delim_len)
      return false;

    auto pos = path.size() - delim_len;

    if (path.compare(pos, delim_len, delim) == 0)
      return true;

    return false;
  }

  bool is_file(const std::string& path)
  {
    if (is_directory(path))
      return false;

#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    struct stat sb;
    return (stat(path.c_str(), &sb) == 0) && S_ISREG(sb.st_mode);
#elif defined(_WIN32)
    struct stat sb;
    return (_stat(path.c_str(), &sb) == 0) && S_ISREG(sb.st_mode);
#else
#  error "path::files not supported on this target."
#endif
  }

  bool is_hidden(const std::string& path)
  {
    return path.empty() || (path.front() == '.');
  }
}
