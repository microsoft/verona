// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

/**
 * This is a bag of utility functions to handle AST lookups and fail-safe
 * operations. While the AST design is still in flux, we can keep this around,
 * but once we're set on its structure, this should move to src/ast instead.
 *
 * Current Memory Model
 *
 * The memory of all AST nodes is owned by the AST. Here, we only query
 * temporary values for reading purposes only. The AST uses weak_ptr for
 * temporary variables, so some of our methods use it too to pass areguments.
 *
 * Once we need to actually read the value, we use weak_ptr.lock() as usual.
 *
 * Values returned are either new strings, weak_ptr or a new vector of weak
 * pointers. The idea is to detach the structures of the AST and have a flat
 * representation for the specific types of nodes we need in each call.
 */

namespace mlir::verona::ASTInterface
{
  /**
   * We need this because the ""_ operator doens't stack well outside of the
   * peg namespace, so we need to call str2tag directly. Easier to do so in a
   * constexpr enum type creation and let the rest be unsigned comparisons.
   * The AST code needs to be flexible, so using the operator directly is more
   * convenient. But we need to be very strict (with MLIR generation), so this
   * also creates an additional layer of safety.
   */
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
    Float = peg::str2tag("float"),
    Condition = peg::str2tag("cond"),
    If = peg::str2tag("if"),
    Else = peg::str2tag("else"),
  };

  // ================================================= Generic Helpers
  /// Ast&MLIR independent path component
  struct NodePath
  {
    const std::string& file;
    const size_t line;
    const size_t column;
  };
  /// Return the path of the ast node
  NodePath getPath(::ast::WeakAst ast);
  /// Return the 'name' of a node, for error reporting
  const std::string& getName(::ast::WeakAst ast);
  /// Return the 'kind' of a node, for comparison
  unsigned int getKind(::ast::WeakAst ast);
  /// Return true if node is of a certain kind
  bool isA(::ast::WeakAst ast, NodeKind kind);
  /// Return true if node is of any kind in a list
  bool isAny(::ast::WeakAst ast, llvm::ArrayRef<NodeKind> kind);
  /// Find a sub-node of tag 'type'
  ::ast::WeakAst findNode(::ast::WeakAst ast, NodeType type);
  /// Return a list of sub-nodes
  std::vector<::ast::WeakAst> getSubNodes(::ast::WeakAst ast);

  // ================================================= Value Helpers
  /// Return true if node is a value
  bool isValue(::ast::WeakAst ast);
  /// Return true if node is a value
  bool isConstant(::ast::WeakAst ast);
  /// Return true if node is a local variable reference
  bool isLocalRef(::ast::WeakAst ast);
  /// Return true if node is a new variable definition
  bool isLet(::ast::WeakAst ast);
  /// Get the string value of a token
  const std::string& getTokenValue(::ast::WeakAst ast);
  /// Return true if node is a variable definition
  const std::string& getLocalName(::ast::WeakAst ast);
  /// Return true if node is an ID (func, var, type names)
  const std::string& getID(::ast::WeakAst ast);

  // ================================================= Type Helpers
  /// Return true if node is a type
  bool isType(::ast::WeakAst ast);
  /// Get the ast type representaiton of an ast node
  ::ast::WeakAst getType(::ast::WeakAst ast);
  /// Get the string description of a type node
  const std::string getTypeDesc(::ast::WeakAst ast);

  // ================================================= Function Helpers
  /// Return true if node is a function
  bool isFunction(::ast::WeakAst ast);
  /// Get the string name of a function node
  llvm::StringRef getFunctionName(::ast::WeakAst ast);
  /// Get the return type of a function node
  ::ast::WeakAst getFunctionType(::ast::WeakAst ast);
  /// Get the ast nodes for the function arguments
  std::vector<::ast::WeakAst> getFunctionArgs(::ast::WeakAst ast);
  /// Get the ast nodes for the function constraints
  std::vector<::ast::WeakAst> getFunctionConstraints(::ast::WeakAst ast);
  /// Get the ast node for the function body
  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast);

  // ================================================= Class Helpers
  /// Return true if node is a class/module
  bool isClass(::ast::WeakAst ast);
  /// Get the body of a class/module declaration
  ::ast::WeakAst getClassBody(::ast::WeakAst ast);

  // ================================================= Operation Helpers
  /// Return true if node is an operation/call
  bool isCall(::ast::WeakAst ast);
  /// Return true if node is an assignment
  bool isAssign(::ast::WeakAst ast);
  /// Return the left-hand side of an assignment
  ::ast::WeakAst getLHS(::ast::WeakAst ast);
  /// Return the right-hand side of an assignment
  ::ast::WeakAst getRHS(::ast::WeakAst ast);
  /// Return the number of operands in an operation
  size_t numOperands(::ast::WeakAst ast);
  /// Return true if node an unary operation
  bool isUnary(::ast::WeakAst ast);
  /// Return true if node an binary operation
  bool isBinary(::ast::WeakAst ast);
  /// Return the n-th operand of the operation
  ::ast::WeakAst getOperand(::ast::WeakAst ast, size_t n);

  // ================================================= Condition Helpers
  /// Return true if node is an if statement
  bool isIf(::ast::WeakAst ast);
  /// Return true if node is a condition
  bool isCondition(::ast::WeakAst ast);
  /// Return true if node is an else block
  bool isBlock(::ast::WeakAst ast);
  /// Return true if node is an else block
  bool isElse(::ast::WeakAst ast);
  /// Return the condition form an if statement
  ::ast::WeakAst getCond(::ast::WeakAst ast);
  /// Return the block form an if statement
  ::ast::WeakAst getIfBlock(::ast::WeakAst ast);
  /// Return the else block form an if statement
  ::ast::WeakAst getElseBlock(::ast::WeakAst ast);
}
