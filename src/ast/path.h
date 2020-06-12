// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

namespace path
{
  std::string executable();
  std::string directory(const std::string& path);
  std::string filename(const std::string& path);
  std::string join(const std::string& path1, const std::string& path2);
  std::string to_directory(const std::string& path);
  std::string extension(const std::string& path);
  std::string to_platform(const std::string& path);
  std::string canonical(const std::string& path);

  std::vector<std::string> files(const std::string& path);

  bool is_relative(const std::string& path);
  bool is_directory(const std::string& path);
  bool is_file(const std::string& path);
  bool is_hidden(const std::string& path);
}
