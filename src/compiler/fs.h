// Copyright (c) Contributers to Project Verona. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef CXX_FILESYSTEM
#  include <filesystem>
namespace fs = std::filesystem;
#elif defined(CXX_FILESYSTEM_EXPERIMENTAL)
#  include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#  include <string>
#  ifdef _WIN32
#    error("Supported windows should provide filesystem library.")
#  else
#    include <libgen.h>
// This is the only feature from filesystem we are currently using.
// As support spreads, we can remove this.
namespace fs
{
  struct path : std::string
  {
    path(const std::string& p) : std::string(p) {}
    path(const char* p) : std::string(p) {}

    std::string string()
    {
      return *this;
    }

    path remove_filename()
    {
      std::unique_ptr<char, decltype(free)*> str = {strdup(c_str()), free};
      return path(dirname(str.get()));
    }
  };
}
#  endif
#endif
