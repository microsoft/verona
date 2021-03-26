// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "path.h"

#include <algorithm>
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

namespace verona::parser::path
{
  constexpr size_t length(const char* str)
  {
    return *str ? 1 + length(str + 1) : 0;
  }

  constexpr auto delim = "/";
  constexpr auto delim_len = length(delim);

  constexpr auto platform_delim =
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    delim
#elif defined(_WIN32)
    "\\"
#endif
    ;
  constexpr auto platform_delim_len = length(platform_delim);

  std::string executable()
  {
#ifdef WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(NULL, buf, MAX_PATH);
    return from_platform(std::string(buf));
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

  std::string from_platform(const std::string& path)
  {
    if constexpr (delim != platform_delim)
    {
      auto result = path;
      auto pos = result.find(platform_delim, platform_delim_len);

      while (pos != std::string::npos)
      {
        result.replace(pos, platform_delim_len, delim, delim_len);
        pos = result.find(platform_delim, pos);
      }

      return result;
    }

    return path;
  }

  std::string canonical(const std::string& path)
  {
#ifdef _WIN32
    char resolved[FILENAME_MAX];

    if (GetFullPathNameA(path.c_str(), FILENAME_MAX, resolved, NULL) == 0)
      return {};

    // Check that this file or directory exists.
    DWORD attrib = GetFileAttributes(resolved);

    if (attrib == INVALID_FILE_ATTRIBUTES)
      return {};

    // Check that the trailing delimiter for `path` matches whether or not the
    // actual filesystem item is a directory or not.
    auto is_dir = (attrib & FILE_ATTRIBUTE_DIRECTORY) != 0;

    if (is_dir != is_directory(path))
      return {};
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    char resolved[PATH_MAX];

    if (realpath(path.c_str(), resolved) == NULL)
      return {};

    // Win32 includes a trailing delimiter but POSIX does not.
    if (is_directory(path))
      ::strcat(resolved, delim);
#endif

    return from_platform(std::string(resolved));
  }

  std::vector<std::string> contents(const std::string& path, bool files)
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
          if (files)
            r.push_back(e->d_name);
          break;
        }

        case DT_DIR:
        {
          if (!files)
            r.push_back(e->d_name);
          break;
        }

        case DT_LNK:
        case DT_UNKNOWN:
        {
          auto name = join(path, e->d_name);

          if (files && (type(name) == Type::File))
            r.push_back(name);
          else if (!files && (type(name) == Type::Directory))
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
    auto att = files ? 0 : FILE_ATTRIBUTE_DIRECTORY;

    while (handle != INVALID_HANDLE_VALUE)
    {
      if ((ffd.dwFileAttributes & mask) == att)
        r.push_back(ffd.cFileName);

      if (!FindNextFileA(handle, &ffd))
      {
        FindClose(handle);
        handle = INVALID_HANDLE_VALUE;
      }
    }
#else
#  error "path::files not supported on this target."
#endif

    // Sort files to avoid a non-deterministic build.
    std::sort(r.begin(), r.end());
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

  std::vector<std::string> files(const std::string& path)
  {
    return contents(path, true);
  }

  std::vector<std::string> directories(const std::string& path)
  {
    return contents(path, false);
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

  bool is_hidden(const std::string& path)
  {
    return path.empty() || (path.front() == '.');
  }

  Type type(const std::string& path)
  {
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    struct stat sb;

    if (stat(path.c_str(), &sb) != 0)
      return Type::NotFound;

    if (S_ISREG(sb.st_mode))
      return Type::File;

    if (S_ISDIR(sb.st_mode))
      return Type::Directory;
#elif defined(_WIN32)
    struct _stat sb;

    if (_stat(path.c_str(), &sb) != 0)
      return Type::NotFound;

    if (((sb.st_mode) & _S_IFMT) == _S_IFREG)
      return Type::File;

    if (((sb.st_mode) & _S_IFMT) == _S_IFDIR)
      return Type::Directory;
#else
#  error "path::type not supported on this target."
#endif
    return Type::Other;
  }
}
