// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

namespace verona::parser::path
{
  // All path names use / as the delimiter, regardless of platform. Windows
  // absolute paths (returned by path::executable() and path::canonical() have a
  // drive letter prefix, but still use / as the delimiter. Directories always
  // have a trailing delimiter - any path with no trailing delimiter is always
  // interpreted as referencing a file.

  // Returns the absolute path to the currently executing program.
  std::string executable();

  // Returns the directory portion of the path, if any.
  std::string directory(const std::string& path);

  // Returns the filename portion of the path, if any.
  std::string filename(const std::string& path);

  // Joins the two paths. If path2 is not relative, this returns path2.
  std::string join(const std::string& path1, const std::string& path2);

  // Appends a delimiter if path is not a directory.
  std::string to_directory(const std::string& path);

  // Returns the extension (the part after the dot) of the filename. If there is
  // no filename, no extension is returned.
  std::string extension(const std::string& path);

  // Turns platform-specific delimiters into /.
  std::string from_platform(const std::string& path);

  // Returns a canonical absolute path.
  std::string canonical(const std::string& path);

  // If path is a directory, returns the files (not subdirectories), if any.
  std::vector<std::string> files(const std::string& path);

  // If path is a directory, returns the subdirectories (not files), if any.
  std::vector<std::string> directories(const std::string& path);

  bool is_relative(const std::string& path);
  bool is_directory(const std::string& path);
  bool is_hidden(const std::string& path);

  enum class Type
  {
    NotFound,
    File,
    Directory,
    Other,
  };

  Type type(const std::string& path);
}
