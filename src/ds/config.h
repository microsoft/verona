// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * Appends command line options from a file.
 *
 * Syntax: -config <config-file>
 *
 * File format: cmdline options verbatim
 *
 * Example:
 *   Cmdline: -foo -config config.txt -baz
 *   config.txt: -bar
 *   Result: -foo -bar -baz
 */
class CmdLineAppend
{
  /// Cmdline from config file
  std::vector<char*> args;
  /// Paths to the config files, if any
  std::vector<std::string> paths;

  /// Add new option to arguments array, allocating enough space on each vector
  /// element
  void addArgOption(const char* arg, size_t len)
  {
    args.push_back(new char[len + 1]);
    auto& inplace = args[args.size() - 1];
    std::copy(arg, arg + len, inplace);
    inplace[len] = 0;
  }

public:
  CmdLineAppend() = default;

  ~CmdLineAppend()
  {
    // Manual cleanup because char* doesn't have a destructor
    for (auto arg : args)
      delete arg;
  }

  /// Parse config files and append internal argument to the main list
  bool parse(int argc, char** argv)
  {
    std::string pathName;
    std::string_view flag("-config");
    for (int i = 0; i < argc; i++)
    {
      std::string_view arg(argv[i]);
      // If not config, just copy the argv as is
      if (flag.compare(arg))
      {
        addArgOption(arg.data(), arg.length());
        continue;
      }

      // Else, append all arguments from the file to args
      auto path = argv[++i];
      paths.push_back(path);
      std::ifstream file(path);
      if (!file.good())
      {
        return false;
      }

      // For each arg, append the options to the command line
      std::string buffer;
      while (file >> quoted(buffer))
      {
        addArgOption(buffer.data(), buffer.size());
      }
      file.close();
    }

    return true;
  }

  /// Return the number of arguments as an int
  int argc() const
  {
    return args.size();
  }

  /// Return the list of arguments as char**
  char** argv()
  {
    return args.data();
  }

  /// Return the list of config files, if any, that were parsed
  const std::vector<std::string>& configPaths() const
  {
    return paths;
  }
};
