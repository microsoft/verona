// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <process_sandbox/helpers.h>
#include <process_sandbox/path.h>

using sandbox::Path;

namespace
{
  Path get(std::string_view str)
  {
    Path p{str};
    SANDBOX_INVARIANT(p.str() == str, "{} as a path became {}.", str, p.str());
    return p;
  }

  void check_canonicalisation(std::string_view path, std::string_view expected)
  {
    Path p = get(path);
    snmalloc::UNUSED(p.canonicalise());
    SANDBOX_INVARIANT(
      p.str() == expected,
      "{} canonicalised as {}, expected {}",
      path,
      p.str(),
      expected);
  }

  void expect(const Path& p, std::string_view expected)
  {
    SANDBOX_INVARIANT(
      p.str() == expected, "Expected {}, got {}", expected, p.str());
  }
}

int main()
{
  const char* cwd = getcwd(nullptr, 0);

  {
    Path p = get("/foo/../bar");
    SANDBOX_INVARIANT(p.is_absolute(), "{} is an absolute path.", p.str());
    snmalloc::UNUSED(p.canonicalise());
    expect(p, "/bar");
  }

  expect(Path::getcwd(), cwd);

  {
    Path p = get("foo/../bar");
    SANDBOX_INVARIANT(!p.is_absolute(), "{} is not an absolute path.", p.str());
    snmalloc::UNUSED(p.canonicalise());
    SANDBOX_INVARIANT(
      p.str() == "bar", "Canonicalised incorrectly as {}.", p.str());

    std::string expected{cwd};
    expected += "/bar";
    p.make_absolute();
    expect(p, expected);
  }

  {
    const char* leading_dotdot = "../../foo";
    Path p = get(leading_dotdot);
    SANDBOX_INVARIANT(
      !p.canonicalise(), "Path with leading .. should be canonicalisable");
    expect(p, leading_dotdot);

    Path base = get("/1/2/3/4");
    p.make_absolute(base);
    SANDBOX_INVARIANT(p.is_absolute(), "{} is not an absolute path.", p.str());
    snmalloc::UNUSED(p.canonicalise());
    expect(p, "/1/2/foo");
  }

  {
    auto p = get("/1/2/../../../4");
    SANDBOX_INVARIANT(
      !p.canonicalise(),
      "Unexpected success canonicalising path with too many ..s");
    expect(p, "/../4");
  }

  {
    auto p = get("1/2/../../../4");
    SANDBOX_INVARIANT(
      !p.canonicalise(),
      "Unexpected success canonicalising path with too many ..s");
    expect(p, "../4");
  }

  check_canonicalisation("//foo//bar", "/foo/bar");
  check_canonicalisation(
    "//directory with spaces//bar", "/directory with spaces/bar");
  check_canonicalisation("//fo\\/o//bar", "/fo\\/o/bar");
}
