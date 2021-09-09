// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <llvm/Support/VirtualFileSystem.h>
#include <string>

/**
 * Helper for LLVM FS functionality
 */
struct FSHelper
{
  /**
   * Resolves a path and returns its absolute canonical path
   */
  static std::string getRealPath(const std::string_view path)
  {
    llvm::SmallVector<char, 16> out;
    llvm::sys::fs::real_path(path, out, /*expand_tilde*/ true);
    std::string res(out.data(), out.size());
    return res;
  }

  /**
   * Returns the directory name that contains this path
   */
  static std::string getDirName(const llvm::StringRef path)
  {
    return llvm::sys::path::parent_path(path).str();
  }
};
