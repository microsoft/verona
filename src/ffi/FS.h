// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/VirtualFileSystem.h>

namespace verona::ffi
{
  /**
   * Creates an in-memory overlay file-system, so we can create the interim
   * compile unit (that includes the user file) alongside the necessary
   * headers to include (built-in, etc).
   *
   * This will expand to lookup for built-in headers, potentially using
   * the driver's logic to find them.
   */
  struct FileSystem
  {
    /// Main overlay FS initialised with the OS FS
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay;
    /// InMemory FS to store the compiled sources and built-in headers
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inMemory;

  public:
    FileSystem()
    : inMemory(new llvm::vfs::InMemoryFileSystem()),
      overlay(new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()))
    {
      overlay->pushOverlay(inMemory);
    }

    /// Get the overlay file system
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> get()
    {
      return overlay;
    }

    /// Add the unit to the file system
    void
    addFile(const llvm::Twine& name, std::unique_ptr<llvm::MemoryBuffer> Buf)
    {
      inMemory->addFile(name, time(nullptr), std::move(Buf));
    }
  };
} // namespace verona::ffi
