// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "helpers.h"

#include <list>
#include <sstream>
#include <string>
#include <string_view>

namespace sandbox
{
  /**
   * Class encapsulating a path.  A path is a route through a tree.  This class
   * works on an abstract filesystem namespace, it never accesses the
   * filesystem.
   */
  class Path
  {
    /**
     * Is this an absolute path.
     */
    bool isAbsolute = false;

    /**
     * List of path components.
     */
    std::list<std::string> components;

    /**
     * Helper function to find the next path separator.
     */
    template<typename It, typename It2>
    bool next_separator(It&& cur, const It2& end)
    {
      while (cur != end)
      {
        // Note: This does not handle UTF-8 strings correctly.
        if (*cur == '/')
        {
          return true;
        }
        ++cur;
        if (cur == end)
        {
          return false;
        }
        if (*cur == '\\')
        {
          ++cur;
        }
      }
      return false;
    }

  public:
    /**
     * The iterator type that is used to iterate over path components.
     */
    using component_iterator = std::list<std::string>::const_iterator;

    /**
     * Construct a new path object from a string view containing the text.
     */
    Path(std::string_view raw_path)
    {
      auto begin = raw_path.begin();
      if (*begin == '/')
      {
        isAbsolute = true;
        ++begin;
      }
      auto next = begin;
      auto end = raw_path.end();
      while (next_separator(next, end))
      {
        components.emplace_back(begin, next);
        ++next;
        begin = next;
      }
      if (begin != end)
      {
        components.emplace_back(begin, end);
      }
    }

    /**
     * Default constructor, constructs an empty path.
     */
    Path() = default;

    /**
     * Construct a path from a range of path components.
     */
    Path(component_iterator b, component_iterator e)
    {
      components.insert(components.begin(), b, e);
    }

    /**
     * Returns true if this is an empty path (no elements).
     */
    bool is_empty()
    {
      return components.empty();
    }

    /**
     * Return a string representation of this path.
     */
    std::string str() const
    {
      std::stringstream s;
      bool insertSlash = isAbsolute;
      for (auto component : components)
      {
        if (insertSlash)
        {
          s << '/';
        }
        insertSlash = true;
        s << component;
      }
      return s.str();
    }

    /**
     * Construct a path from the current working directory.
     */
    static Path getcwd()
    {
      unique_c_ptr<char> rawcwd{::getcwd(nullptr, 0)};
      return Path(rawcwd.get());
    }

    /**
     * Remove all empty path components.  In a POSIX filesystem, `a///b` is
     * equivalent to `a/b`.
     */
    void remove_empty()
    {
      components.remove_if(
        [](auto& component) { return component.size() == 0; });
    }

    /**
     * Eliminate all path components that are `..` by removing the `..` element
     * and the
     */
    bool remove_dotdot()
    {
      auto current = components.begin();
      auto end = components.end();
      bool success = true;
      while (current != end)
      {
        if (*current != "..")
        {
          current++;
          continue;
        }
        // We can't remove a leading '..'.
        if (current == components.begin())
        {
          return false;
        }

        auto parent = current;
        --parent;
        ++current;
        components.erase(parent, current);
      }
      return success;
    }

    /**
     * Canonicalise the path.  This removes empty elements and eliminates `..`
     * elements but does not inspect the filesystem and so the path may still
     * contain elements that resolve to symbolic links.
     *
     * Returns true on success, false otherwise.  This function can fail if
     * there are too many `..` elements in the path.
     */
    bool canonicalise()
    {
      remove_empty();
      return remove_dotdot();
    }

    /**
     * Make a relative path absolute by combining it with a path representing
     * the location that this path is relative to.
     */
    void make_absolute(const Path& base = getcwd())
    {
      SANDBOX_DEBUG_INVARIANT(
        !isAbsolute, "Trying to make an absolute path absolute");
      components.insert(
        components.begin(), base.components.begin(), base.components.end());
      isAbsolute = true;
    }

    /**
     * Returns true if this is an absolute path, false otherwise.
     */
    bool is_absolute() const
    {
      return isAbsolute;
    }

    /**
     * Returns a path-component iterator for the first path component in this
     * path.
     */
    component_iterator begin() const
    {
      return components.begin();
    }

    /**
     * Returns a path-component iterator for the end of the path.
     */
    component_iterator end() const
    {
      return components.end();
    }
  };
}
