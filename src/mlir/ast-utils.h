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
    Module = peg::str2tag("module"), // = 73005690
    ClassDef = peg::str2tag("classdef"), // = 983016457
    Function = peg::str2tag("function"), // = 89836898
    FuncName = peg::str2tag("funcname"), // = 90200697
    Sig = peg::str2tag("sig"), // = 124317
    Block = peg::str2tag("block"), // = 117895113
    OfType = peg::str2tag("oftype"), // = 3504561
    Constraints = peg::str2tag("constraints"), // = 4070926742
    Constraint = peg::str2tag("constraint"), // = 2466070853
    Type = peg::str2tag("type"), // = 4058008
    TypeRef = peg::str2tag("type_ref"), // = 2115269750
    TypeBody = peg::str2tag("typebody"), // = 2117081800
    Params = peg::str2tag("params"), // = 4271185724
    NamedParam = peg::str2tag("namedparam"), // = 1544471404
    ID = peg::str2tag("id"), // = 3565
    Seq = peg::str2tag("seq"), // = 124167
    Assign = peg::str2tag("assign"), // = 3930587681
    Let = peg::str2tag("let"), // = 114397
    Call = peg::str2tag("call"), // = 3519010
    Return = peg::str2tag("return"), // = 210835850
    Args = peg::str2tag("args"), // = 3609031
    Integer = peg::str2tag("int"), // = 117427
    Local = peg::str2tag("local"), // = 124201581
    Localref = peg::str2tag("localref"), // = 961318684
    Float = peg::str2tag("float"), // = 122359824
    Condition = peg::str2tag("cond"), // = 3533542
    If = peg::str2tag("if"), // = 3567
    Else = peg::str2tag("else"), // = 3742303
    While = peg::str2tag("while"), // = 140358143
    Continue = peg::str2tag("continue"), // = 2929012833
    Break = peg::str2tag("break"), // = 117842911
  };

  // ================================================= Path Helpers
  /// Ast&MLIR independent path component
  struct NodePath
  {
    const std::string& file;
    const size_t line;
    const size_t column;
  };
  /// Return the path of the ast node
  NodePath getPath(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    return {ptr->path, ptr->line, ptr->column};
  }

  // ================================================= Generic Helpers
  /// Return the 'name' of a node, for error reporting
  const std::string& getName(::ast::WeakAst ast)
  {
    return ast.lock()->name;
  }

  /// Return the 'kind' of a node, for comparison
  unsigned int getKind(::ast::WeakAst ast)
  {
    return ast.lock()->tag;
  }

  /// Return true if node is of a certain kind
  bool isA(::ast::WeakAst ast, NodeKind kind)
  {
    return getKind(ast) == kind;
  }

  /// Return true if node is of any kind in a list
  bool isAny(::ast::WeakAst ast, llvm::ArrayRef<NodeKind> kinds)
  {
    return std::find(kinds.begin(), kinds.end(), getKind(ast)) != kinds.end();
  }

  /// Return true if node has a certain kind sub-node
  bool hasA(::ast::WeakAst ast, NodeKind kind)
  {
    auto nodes = ast.lock()->nodes;
    return std::find_if(nodes.begin(), nodes.end(), [&](::ast::Ast& node) {
             return isA(node, kind);
           }) != nodes.end();
  }

  /// Return true if node is a value
  bool isValue(::ast::WeakAst ast)
  {
    return ast.lock()->is_token;
  }

  /// Return true if node is a value
  bool isConstant(::ast::WeakAst ast)
  {
    return isValue(ast) && isAny(ast, {NodeKind::Integer, NodeKind::Float});
  }

  /// Return true if node is a local variable reference
  bool isLocalRef(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Localref);
  }

  /// Return true if node is a new variable definition
  bool isLet(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Let);
  }

  /// Return true if node is a type
  bool isType(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::OfType);
  }

  /// Return true if node is a function
  bool isFunction(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Function);
  }

  /// Return true if node is a class/module
  bool isClass(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::ClassDef);
  }

  /// Return true if node is an operation/call
  bool isCall(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Call);
  }

  /// Return true if node is a return
  bool isReturn(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Return);
  }

  /// Return true if node is an assignment
  bool isAssign(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Assign);
  }

  /// Return true if node is an if statement
  bool isIf(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::If);
  }

  /// Return true if node is a condition
  bool isCondition(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Condition);
  }

  /// Return true if node is a block
  bool isBlock(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Block);
  }

  /// Return true if node is an else block
  bool isElse(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Else);
  }

  /// Return true if node is a while loop
  bool isWhile(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::While);
  }

  /// Return true if node is any kind of loop
  bool isLoop(::ast::WeakAst ast)
  {
    // Add isFor when we support it
    return isWhile(ast);
  }

  /// Return true if node is a loop continue
  bool isContinue(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Continue);
  }

  /// Return true if node is a loop break
  bool isBreak(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Break);
  }

  /// Find a sub-node of tag 'type'
  ::ast::WeakAst findNode(::ast::WeakAst ast, NodeType type)
  {
    assert(!isValue(ast) && "Bad node");
    // Match tag with NodeKind's enum value
    auto ptr = ast.lock();
    auto sub = std::find_if(
      ptr->nodes.begin(), ptr->nodes.end(), [type](::ast::Ast& sub) {
        return (sub->tag == type);
      });
    // TODO: Make this into a soft error
    assert(sub != ptr->nodes.end());
    return *sub;
  }

  /// Return the single sub-node, error out if more than one
  ::ast::WeakAst getSingleSubNode(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->nodes.size() == 1 && "Wrong number of nodes");
    return ptr->nodes[0];
  }

  /// Return a list of sub-nodes
  template<class T>
  void getSubNodes(T& nodes, ::ast::WeakAst ast)
  {
    // If single node, but of type 'seq', descend
    auto ptr = ast.lock();
    if (ptr->nodes.size() == 1 && isA(ptr->nodes[0], NodeKind::Seq))
      ptr = ptr->nodes[0];

    // Return the nodes
    nodes.insert(nodes.end(), ptr->nodes.begin(), ptr->nodes.end());
  }

  // ================================================= Value Helpers
  /// Get the string value of a token
  const std::string& getTokenValue(::ast::WeakAst ast)
  {
    assert(isValue(ast) && "Bad node");
    auto ptr = ast.lock();
    assert(!ptr->token.empty());
    return ptr->token;
  }

  /// Return true if node is a variable definition
  const std::string& getLocalName(::ast::WeakAst ast)
  {
    // Local variables can be new 'local' or existing 'localref'
    if (isLocalRef(ast))
      return getTokenValue(ast);
    assert(isLet(ast) && "Bad node");
    return getTokenValue(findNode(ast, NodeKind::Local));
  }

  /// Return true if node is an ID (func, var, type names)
  const std::string& getID(::ast::WeakAst ast)
  {
    // FIXME: Why is the call ID 'function' while all others 'id'?
    if (isA(ast, NodeKind::Call))
      return getTokenValue(findNode(ast, NodeKind::Function));
    return getTokenValue(findNode(ast, NodeKind::ID));
  }

  // ================================================= Type Helpers
  /// Get the ast type representaiton of an ast node
  ::ast::WeakAst getType(::ast::WeakAst ast)
  {
    return findNode(ast, NodeKind::OfType);
  }

  /// Get the string description of a type node
  const std::string getTypeDesc(::ast::WeakAst ast)
  {
    assert(isType(ast) && "Bad node");
    auto ptr = ast.lock();
    if (ptr->nodes.empty())
      return "";

    std::string desc;
    // This undoes the work that the ast did to split the types
    // and it should be rethought, but MLIR doens't allow multiple
    // types on a single node. Perhaps attributes?
    auto type = findNode(ptr, NodeKind::Type).lock();
    for (auto sub : type->nodes)
    {
      if (sub->is_token)
      {
        desc += sub->token;
      }
      else
      {
        auto ref = findNode(sub, NodeKind::TypeRef).lock();
        desc += ref->nodes[0]->token;
      }
    }
    assert(!desc.empty());

    return desc;
  }

  // ================================================= Function Helpers
  /// Get the string name of a function node
  llvm::StringRef getFunctionName(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Empty function name is "apply"
    auto funcname = findNode(ast, NodeKind::FuncName).lock();
    assert(!funcname->nodes.empty() && "Bad function");

    // Else, get function name
    assert(funcname->nodes.size() == 1);
    return getTokenValue(funcname->nodes[0]);
  }

  /// Get the return type of a function node
  ::ast::WeakAst getFunctionType(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Return type is in the sig / oftype / type
    auto sig = findNode(ast, NodeKind::Sig);
    return getType(sig);
  }

  /// Get the ast nodes for the function arguments
  template<class T>
  void getFunctionArgs(T& args, ::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Arguments are in sig / params
    auto sig = findNode(ast, NodeKind::Sig).lock();
    auto params = findNode(sig, NodeKind::Params).lock();
    for (auto param : params->nodes)
      args.push_back(findNode(param, NodeKind::NamedParam));
  }

  /// Get the ast nodes for the function constraints
  template<class T>
  void getFunctionConstraints(T& constraints, ::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Constraints are in sig / params
    auto sig = findNode(ast, NodeKind::Sig).lock();
    auto consts = findNode(sig, NodeKind::Constraints).lock();
    for (auto c : consts->nodes)
      constraints.push_back(c);
  }

  /// Get the ast node for the function body
  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Body is just a block
    return findNode(ast, NodeKind::Block);
  }

  // ================================================= Class Helpers
  /// Get the body of a class/module declaration
  ::ast::WeakAst getClassBody(::ast::WeakAst ast)
  {
    assert(isClass(ast) && "Bad node");

    // TypeBody is just a block node in the class
    return findNode(ast, NodeKind::TypeBody);
  }

  // ================================================= Operation Helpers
  /// Return the number of operands in an operation
  size_t numOperands(::ast::WeakAst ast)
  {
    assert(isCall(ast) && "Bad node");
    if (!isValue(ast.lock()->nodes[2]))
      return 0;
    auto args = findNode(ast, NodeKind::Args);
    return args.lock()->nodes.size() + 1;
  }

  /// Return true if node an unary operation
  bool isUnary(::ast::WeakAst ast)
  {
    return numOperands(ast) == 1;
  }

  /// Return true if node an binary operation
  bool isBinary(::ast::WeakAst ast)
  {
    return numOperands(ast) == 2;
  }

  /// Return the left-hand side of an assignment
  ::ast::WeakAst getLHS(::ast::WeakAst ast)
  {
    // LHS is the assignable 'let' or 'localref'
    assert(isAssign(ast) && "Bad node");
    auto lhs = ast.lock()->nodes[0];
    assert((isLocalRef(lhs) || isLet(lhs)) && "Bad node");
    return lhs;
  }

  /// Return the right-hand side of an assignment
  ::ast::WeakAst getRHS(::ast::WeakAst ast)
  {
    // RHS is the expression on the second node
    assert(isAssign(ast) && "Bad node");
    return ast.lock()->nodes[1];
  }

  /// Return the n-th operand of the operation
  ::ast::WeakAst getOperand(::ast::WeakAst ast, size_t n)
  {
    assert(n < numOperands(ast) && "Bad offset");
    auto ptr = ast.lock();
    // Calls have the first operand after 'typeargs' (3rd place)
    if (n == 0)
      return ptr->nodes[2];
    // All others in 'args'
    auto args = findNode(ast, NodeKind::Args);
    return args.lock()->nodes[n - 1];
  }

  /// Return all operands of the operation
  template<class T>
  void getAllOperands(T& ops, ::ast::WeakAst ast)
  {
    auto numOps = numOperands(ast);
    for (size_t i = 0; i < numOps; i++)
      ops.push_back(getOperand(ast, i));
  }

  // ================================================= Condition Helpers
  /// Return the condition form an if statement
  bool hasElse(::ast::WeakAst ast)
  {
    // Else nodes always exist inside `if` nodes, but if there was no `else`
    // block, they're empty. We should only return true if they're not.
    return isA(ast, NodeKind::If) && hasA(ast, NodeKind::Else) &&
      !findNode(ast, NodeKind::Else).lock()->nodes.empty();
  }

  /// Return the block form an if statement
  ::ast::WeakAst getCond(::ast::WeakAst ast)
  {
    // These are the nodes that could have conditions as subnodes
    assert((isIf(ast) || isLoop(ast)) && "Bad node");
    // Cond nodes are always seq, even with just one cond
    auto cond = findNode(ast, NodeKind::Condition);
    return findNode(cond, NodeKind::Seq);
  }

  /// Return true if the 'if' node has an else block
  ::ast::WeakAst getIfBlock(::ast::WeakAst ast)
  {
    assert(isIf(ast) && "Bad node");
    return findNode(ast, NodeKind::Block);
  }

  /// Return the else block form an if statement
  ::ast::WeakAst getElseBlock(::ast::WeakAst ast)
  {
    assert(isIf(ast) && "Bad node");
    // Else has either a single node or a seq
    auto node = findNode(ast, NodeKind::Else);
    return node.lock()->nodes[0];
  }

  // ================================================= Loop Helpers
  /// Return the block form a loop
  ::ast::WeakAst getLoopBlock(::ast::WeakAst ast)
  {
    assert(isLoop(ast) && "Bad node");
    return findNode(ast, NodeKind::Block);
  }
}
