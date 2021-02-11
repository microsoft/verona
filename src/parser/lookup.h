// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // Find a definition of `name` in `ast` only.
  // This is safe to use during parsing if `ast` has been fully parsed.
  Ast look_in(Ast& ast, const Location& name);

  // Find parameter or local variable called `name`.
  // This is safe to use during parsing.
  Ast look_up_local(AstPath& path, const Location& name);

  // Find all visible definitions of `name`, sorted from closest to furthest.
  // This shouldn't be used until after parsing is complete.
  AstPaths look_up(AstPath& path, const Location& name);

  // Find all visible definitions of `names`, sorted from closest to furthest.
  // Set the `from_using` flag if you are looking up the TypeRef in a `using`
  // node. This shouldn't be used until after parsing is complete.
  AstPaths
  look_up(AstPath& path, List<TypeName>& names, bool from_using = false);
}
