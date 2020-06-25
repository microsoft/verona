// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <clang-c/Index.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

using std::cerr;
using std::cout;

namespace
{
  /**
   * RAIICXString wraps a CXString and handles automatic deallocation.
   */
  class RAIICXString
  {
    /**
     * The string that this wraps.
     */
    CXString cxstr;

  public:
    /**
     * Construct from a libclang string.
     */
    RAIICXString(CXString string) : cxstr(string) {}
    /**
     * Extract the C string from this string when cast to a C string.
     */
    operator const char*()
    {
      return clang_getCString(cxstr);
    }
    /**
     * Extract the C string from this string and convert it to a `std::string`.
     */
    std::string str()
    {
      return std::string(clang_getCString(cxstr));
    }
    /**
     * Allow casts to a `std::string`.
     */
    operator std::string()
    {
      return str();
    }
    /**
     * Allow comparisons to strings.
     */
    bool operator==(const std::string& s)
    {
      return str() == s;
    }
    /**
     * Allow comparisons to C strings.
     */
    bool operator==(const char* s)
    {
      return str() == s;
    }
    /**
     * Destroy the underlying string.
     */
    ~RAIICXString()
    {
      clang_disposeString(cxstr);
    }
  };

  /**
   * Type for visitors passed to `visitChildren`.
   */
  typedef std::function<CXChildVisitResult(CXCursor, CXCursor)> Visitor;

  /**
   * Trampoline used by visitChildren to call a `std::function` instead of a C
   * function.
   */
  CXChildVisitResult visitChildrenTrampoline(
    CXCursor cursor, CXCursor parent, CXClientData client_data)
  {
    return (*reinterpret_cast<Visitor*>(client_data))(cursor, parent);
  }

  /**
   * `clang_visitChildren` wrapper that takes a `std::function`.
   */
  unsigned visitChildren(CXCursor cursor, Visitor v)
  {
    return clang_visitChildren(
      cursor, visitChildrenTrampoline, (CXClientData*)&v);
  }

  /**
   * std::hash specialisation for libclang cursors.  Forwards to the
   * corresponding libclang function.
   */
  struct hash_cursor
  {
    typedef CXCursor argument_type;
    typedef unsigned result_type;
    result_type operator()(argument_type const& c) const noexcept
    {
      return clang_hashCursor(c);
    }
  };

  /**
   * std::equal_to specialisation for libclang cursors.  Forwards to the
   * corresponding libclang function.
   */
  struct equal_to_cursor
  {
    typedef CXCursor first_argument;
    typedef CXCursor second_argument;
    typedef bool result_type;
    result_type
    operator()(first_argument const& c, second_argument const& c1) const
      noexcept
    {
      return clang_equalCursors(c, c1);
    }
  };
}

int main(int, char**)
{
  // Command line arguments for our compile.  Includes paths to the include
  // files.
  std::vector<const char*> args = {
    "-std=c++17", "-mcx16", "-I", "../src", "-I", "../external/snmalloc/src"};
  // Define a simple .cc file as the root of our compilation unit.  This just
  // includes the master header.
  const char* fileName = "verona_doc.cc";
  const char* fileContents = "#include \"verona.h\"";
  CXUnsavedFile stub = {fileName, fileContents, strlen(fileContents)};
  // Create the index and parse the file
  CXIndex idx = clang_createIndex(1, 1);
  CXTranslationUnit translationUnit = clang_createTranslationUnitFromSourceFile(
    idx, fileName, args.size(), args.data(), 1, &stub);

  if (!translationUnit)
  {
    cerr << "Unable to parse file\n";
    return EXIT_FAILURE;
  }

  // Visit all AST nodes (recursively) looking for comments that start /**!
  visitChildren(
    clang_getTranslationUnitCursor(translationUnit), [](CXCursor c, CXCursor) {
      // Helper to trim leading whitespace from a string.
      auto trimLeadingWhitespace = [](std::string& line) {
        line.erase(
          line.begin(),
          std::find_if(
            line.begin(),
            line.end(),
            std::not1(std::function<int(int)>(::isspace))));
      };
      // Helper to check if a string starts with another string.  C++20 has a
      // method to do this, but C++17 does not.
      auto startsWith = [](const std::string& str, const std::string& pattern) {
        return (
          (str.size() >= pattern.size()) &&
          (str.substr(0, pattern.size()) == pattern));
      };
      // Skip anything that's not a declaration.
      if (clang_isDeclaration(clang_getCursorKind(c)))
      {
        // If it is a declaration, find out if there's an associated comment.
        CXSourceRange r = clang_Cursor_getCommentRange(c);
        if (!clang_Range_isNull(r))
        {
          RAIICXString comment = clang_Cursor_getRawCommentText(c);
          std::stringstream commentString(comment);
          // Get the first line of the comment
          std::string firstLine;
          std::getline(commentString, firstLine);
          trimLeadingWhitespace(firstLine);
          // We're expecting to find comments of the form /**!{filename}
          if (startsWith(firstLine, "/**!"))
          {
            auto outFileName = firstLine.substr(4);
            std::ofstream s(outFileName);
            cerr << "Generating " << outFileName << std::endl;
            // Trim leading stars (and whitespace before stars) from each line
            // in the comment
            for (std::string line; std::getline(commentString, line);)
            {
              trimLeadingWhitespace(line);
              size_t star = line.find('*');
              if (star < line.size())
              {
                bool allSpace = true;
                for (size_t i = 0; i < star; i++)
                {
                  allSpace &= isspace(line.at(i));
                }
                if (allSpace)
                {
                  bool skip = false;
                  for (auto i = line.begin() + star, e = line.end(); i != e;
                       ++i)
                  {
                    if (*i == '/')
                    {
                      skip = true;
                      break;
                    }
                    if (*i != '*')
                    {
                      break;
                    }
                  }
                  if (skip)
                  {
                    continue;
                  }
                  line = line.substr(star + 1);
                }
              }
              s << line << '\n';
            }
          }
        }
      }
      return CXChildVisit_Recurse;
    });
  return EXIT_SUCCESS;
}
