// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

// This is a bag of utility functions to handle AST lookups and fail-safe
// operation. While the AST design is still in flux, we can keep this around,
// but once we're set on its structure, this should move to src/ast instead.
//
// Current Memory Model
//
// The memory of all AST nodes is owned but the AST. Here, we only query
// temporary values for reading purposes only. The AST uses weak_ptr for
// temporary variables, so some of our methods use it too to pass areguments.
//
// Once we need to actually read the value, we use weak_ptr.lock() as usual.
//
// Values returned are either new strings, weak_ptr or a new vector of weak
// pointers. The idea is to detach the structures of the AST and have a flat
// representation for the specific types of nodes we need in each call.

namespace mlir::verona::ASTInterface
{
  // We need this because the ""_ operator doens't stack well outside of the
  // peg namespace, so we need to call str2tag directly. Easier to do so in a
  // constexpr enum type creation and let the rest be unsigned comparisons.
  // The AST code needs to be flexible, so using the operator directly is more
  // convenient. But we need to be very strict (with MLIR generation), so this
  // also creates an additional layer of safety.
  using NodeType = unsigned int;
  enum NodeKind : NodeType
  {
    None = 0,
    Module = peg::str2tag("module"),
    ClassDef = peg::str2tag("classdef"),
    Function = peg::str2tag("function"),
    FuncName = peg::str2tag("funcname"),
    Sig = peg::str2tag("sig"),
    Block = peg::str2tag("block"),
    OfType = peg::str2tag("oftype"),
    Constraints = peg::str2tag("constraints"),
    Constraint = peg::str2tag("constraint"),
    Type = peg::str2tag("type"),
    TypeRef = peg::str2tag("type_ref"),
    TypeBody = peg::str2tag("typebody"),
    Params = peg::str2tag("params"),
    NamedParam = peg::str2tag("namedparam"),
    ID = peg::str2tag("id"),
    Seq = peg::str2tag("seq"),
    Assign = peg::str2tag("assign"),
    Let = peg::str2tag("let"),
    Call = peg::str2tag("call"),
    Args = peg::str2tag("args"),
    Integer = peg::str2tag("int"),
    Local = peg::str2tag("local"),
    Localref = peg::str2tag("localref"),
    // TODO: Add all
  };

  // Find a sub-node of tag 'type'
  ::ast::WeakAst findNode(::ast::WeakAst ast, NodeType type);

  // Get token value
  llvm::StringRef getTokenValue(::ast::WeakAst ast);

  // Type helpers
  ::ast::WeakAst getType(::ast::WeakAst ast);
  const std::string getTypeDesc(::ast::WeakAst ast);

  // Function helpers
  llvm::StringRef getFunctionName(::ast::WeakAst ast);
  ::ast::WeakAst getFunctionType(::ast::WeakAst ast);
  std::vector<::ast::WeakAst> getFunctionArgs(::ast::WeakAst ast);
  std::vector<::ast::WeakAst> getFunctionConstraints(::ast::WeakAst ast);
  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast);
}
