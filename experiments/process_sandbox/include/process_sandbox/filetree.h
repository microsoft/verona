// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This file contains the abstractions required for a lightweight virtual
 * filesystem exported to a sandbox.  The code outside the sandbox is
 * responsible for defining the structure of a file tree and providing either
 * handles to directories in a real filesystem or to individual file objects.
 *
 * In a Capsicum world, this can mostly live entirely inside the sandbox
 * because the `*at()` calls are safe.  If sandboxes are implemented with
 * system-call filtering, this must live outside the sandbox and provide
 * handles inside.  In theory, sandboxes could be implemented using a jail-like
 * mechanism and in the VFS would be replaced with a real directory tree
 * containing nullfs mounts and hard links.
 *
 * TODO: Eventually, we should add permissions masks on entities in this
 * filesystem.
 */

#pragma once
#include "path.h"
#include "platform/platform.h"

#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace sandbox
{
  /**
   * Class encapsulating a filesystem exported to a sandbox.  This is a very
   * lightweight simulation of a real filesystem.  This provides a directory
   * structure where each leaf node is either a directory handle to a real
   * directory or a file handle to a file (or something equivalent, such as
   * an anonymous shared memory object).
   *
   * Note: The directories in this structure that are not represented by
   * handles cannot be opened or otherwise inspected in the sandbox.  If the
   * tree stores `/tmp/example/somefile`, where only `somefile` is
   * represented by a real file descriptor, the sandbox will not see `/tmp`
   * or `/tmp/example`.  This restriction may be lifted in the future if it
   * causes significant problems.
   *
   * TODO: The current implementation provides all-or-nothing permissions.
   * Files or directories are delegated with ambient authority.  This is fine
   * for simple cases but eventually we should provide finer-grained
   * controls.  For example, when running a Verona program as `root`,
   * sandboxed code should not have read-write access to `/lib`, even if the
   * `root` user does.
   */
  class ExportedFileTree
  {
    /**
     * A single directory within the exported tree.  This class exists mostly
     * because C++ does not allow recursive types without an explicitly named
     * class somewere in the cycle, so we can't define a map whose value type
     * is the map type itself.
     */
    class ExportedDirectory
    {
      /**
       * This class is an implementation detail of `ExportedFileTree`, which
       * is the only thing allowed to interact with it directly.
       */
      friend class ExportedFileTree;

      /**
       * Handle representing a file.  This is used so that we can statically
       * differentiate between handles to files and handles to directories in
       * the type system but is in no other way different from a generic handle.
       */
      struct FileHandle : public platform::Handle
      {
        using platform::Handle::Handle;
        FileHandle(Handle&& h) : Handle(std::move(h)) {}
      };

      /**
       * Handle representing a directory.  This is used so that we can
       * statically differentiate between handles to files and handles to
       * directories in the type system but is in no other way different from a
       * generic handle.
       */
      struct DirHandle : public platform::Handle
      {
        using platform::Handle::Handle;
        DirHandle(Handle&& h) : Handle(std::move(h)) {}
      };

      /**
       * A pointer to the child directory.  This could probably be a unique
       * pointer as it's currently used, but `std::variant` can't contain
       * references and so we'd have to use bare pointers during traversal.
       */
      using DirPtr = std::shared_ptr<ExportedDirectory>;

      /**
       * The contents of this virtual directory.  This maps from file names to
       * another instance of this class or a file / directory handle.
       */
      std::
        unordered_map<std::string, std::variant<DirPtr, FileHandle, DirHandle>>
          directory;

      /**
       * The result of a directory lookup.  This is either a handle to a real
       * directory, or a pointer to another instance of this class.
       */
      using DirLookupResult =
        std::variant<std::monostate, platform::handle_t, DirPtr>;

      /**
       * Look up a directory (either a real directory handle or a pointer to an
       * instance of this class).  The variant contains `std::monostate` if
       * the path is not a directory.
       */
      DirLookupResult get_dir(const std::string& dir)
      {
        auto it = directory.find(dir);
        if (it == directory.end())
        {
          return {};
        }
        return std::visit(
          [](auto&& arg) -> DirLookupResult {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, FileHandle>)
            {
              return {};
            }
            else if constexpr (std::is_same_v<T, DirPtr>)
            {
              return arg;
            }
            else if constexpr (std::is_same_v<T, DirHandle>)
            {
              return arg.fd;
            }
          },
          it->second);
      }

      /**
       * Look up a file in this directory.  Returns the handle to the file if
       * it exists.
       */
      std::optional<platform::handle_t> get_file(const std::string& file)
      {
        auto it = directory.find(file);
        if (it == directory.end())
        {
          return {};
        }
        if (auto* h = std::get_if<FileHandle>(&it->second))
        {
          return h->fd;
        }
        return {};
      }

      /**
       * Add a child directory tree to this directory.
       */
      void add_tree(const std::string& name, DirPtr& tree)
      {
        directory[name] = tree;
      }

      /**
       * Add a directory tree represented by a directory handle.  Takes
       * ownership of the handle.
       */
      void add_directory(const std::string& name, platform::Handle&& h)
      {
        directory[name] = DirHandle(std::move(h));
      }

      /**
       * Add a file, represented by a file handle. Takes ownership of the
       * handle.
       */
      void add_file(const std::string& name, platform::Handle&& h)
      {
        directory[name] = FileHandle(std::move(h));
      }
    };

    /**
     * Pointer to a single directory in the tree.
     */
    using DirPtr = ExportedDirectory::DirPtr;

    /**
     * The root of this exported file tree.
     */
    DirPtr root = std::make_shared<ExportedDirectory>();

    /**
     * Add a handle at a specific page.  This is a helper method for the public
     * interfaces that add a file or directory handle.  The `add` parameter
     * contains a callable object that takes the base file name and the
     * directory pointer as arguments and adds it.
     *
     * Returns true on success, false on failure.  This can fail if one of the
     * path components is a directory handle.
     */
    template<typename T>
    bool add_handle(const std::string& path, T&& add)
    {
      DirPtr dir = root;
      size_t it = 0;
      size_t end = path.find_last_of('/');
      while (it < end)
      {
        size_t next = path.find_first_of('/', it + 1);
        auto component = path.substr(it + 1, next - it - 1);
        it = next;
        auto result = dir->get_dir(component);
        if (std::holds_alternative<DirPtr>(result))
        {
          dir = std::get<DirPtr>(result);
        }
        else if (std::holds_alternative<std::monostate>(result))
        {
          auto newdir = std::make_shared<ExportedDirectory>();
          dir->add_tree(component, newdir);
          dir = newdir;
        }
        else
        {
          return false;
        }
      }
      auto component = path.substr(it + 1);
      add(component, dir);
      return true;
    }

  public:
    /**
     * Look up the file at a path.  There are three possible outcomes:
     *
     *  - This exported tree does not contain the file, in which case the
     *    lookup returns nothing.
     *  - This exported tree contains a handle to a directory somewhere along
     *    the path. In this case, the return value is the directory handle
     *    and the remainder of the path.
     *  - This exported tree contains the file handle, in which case the
     *    return value is just that handle.
     *
     * The argument is a canonicalised path.
     */
    std::optional<std::pair<platform::handle_t, Path>>
    lookup_file(const Path& path)
    {
      DirPtr dir = root;
      auto i = path.begin(), e = path.end();
      if (std::distance(i, e) > 1)
      {
        auto last_dir = e;
        --last_dir;
        for (; i != last_dir; ++i)
        {
          auto result = dir->get_dir(*i);
          if (std::holds_alternative<DirPtr>(result))
          {
            dir = std::get<DirPtr>(result);
          }
          else if (std::holds_alternative<platform::handle_t>(result))
          {
            return std::make_pair(
              std::get<platform::handle_t>(result), Path(++i, e));
          }
          else
          {
            return {};
          }
        }
      }
      if (i != e)
      {
        auto file = dir->get_file(*i);
        if (file)
        {
          return std::make_pair(file.value(), Path());
        }
      }
      return {};
    }

    /**
     * Add a directory, specified by a directory handle, returning true on
     * success or false on failure.  Once a part of an exported tree is
     * represented by a directory handle, it cannot be replaced with a virtual
     * tree.  The directory descriptor may already be exposed in the sandbox
     * and cannot be revoked.
     */
    bool add_directory(const std::string& path, platform::Handle&& file)
    {
      return add_handle(path, [&](const std::string& filename, DirPtr& dir) {
        dir->add_directory(filename, std::move(file));
      });
    }

    /**
     * Add a file, represented by a handle, to the exported tree, at the
     * specified path.  Returns true on success, false on failure.
     */
    bool add_file(const std::string& path, platform::Handle&& file)
    {
      return add_handle(path, [&](const std::string& filename, DirPtr& dir) {
        dir->add_file(filename, std::move(file));
      });
    }
  };
}
