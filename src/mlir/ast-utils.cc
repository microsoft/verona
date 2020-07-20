// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "ast-utils.h"

#include "symbol.h"

namespace mlir::verona::ASTInterface
{
  // ================================================= Generic Helpers
  NodePath getPath(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    return {ptr->path, ptr->line, ptr->column};
  }

  const std::string& getName(::ast::WeakAst ast)
  {
    return ast.lock()->name;
  }

  unsigned int getKind(::ast::WeakAst ast)
  {
    return ast.lock()->tag;
  }

  bool isA(::ast::WeakAst ast, NodeKind kind)
  {
    return getKind(ast) == kind;
  }

  bool isAny(::ast::WeakAst ast, llvm::ArrayRef<NodeKind> kinds)
  {
    return std::find(kinds.begin(), kinds.end(), getKind(ast)) != kinds.end();
  }

  bool hasA(::ast::WeakAst ast, NodeKind kind)
  {
    auto nodes = ast.lock()->nodes;
    return std::find_if(nodes.begin(), nodes.end(), [&](::ast::Ast& node) {
             return isA(node, kind);
           }) != nodes.end();
  }

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

  ::ast::WeakAst getSingleSubNode(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->nodes.size() == 1 && "Wrong number of nodes");
    return ptr->nodes[0];
  }

  std::vector<::ast::WeakAst> getSubNodes(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    std::vector<::ast::WeakAst> nodes;

    // Single node, but of type 'seq', descend
    if (ptr->nodes.size() == 1 && isA(ptr->nodes[0], NodeKind::Seq))
      ptr = ptr->nodes[0];

    // Return the nodes
    nodes.insert(nodes.end(), ptr->nodes.begin(), ptr->nodes.end());
    return nodes;
  }

  // ================================================= Value Helpers
  bool isValue(::ast::WeakAst ast)
  {
    return ast.lock()->is_token;
  }

  bool isConstant(::ast::WeakAst ast)
  {
    return isValue(ast) && isAny(ast, {NodeKind::Integer, NodeKind::Float});
  }

  bool isLocalRef(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Localref);
  }

  bool isLet(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Let);
  }

  const std::string& getTokenValue(::ast::WeakAst ast)
  {
    assert(isValue(ast) && "Bad node");
    auto ptr = ast.lock();
    assert(!ptr->token.empty());
    return ptr->token;
  }

  const std::string& getLocalName(::ast::WeakAst ast)
  {
    // Local variables can be new 'local' or existing 'localref'
    if (isLocalRef(ast))
      return getTokenValue(ast);
    assert(isLet(ast) && "Bad node");
    return getTokenValue(findNode(ast, NodeKind::Local));
  }

  const std::string& getID(::ast::WeakAst ast)
  {
    // FIXME: Why is the call ID 'function' while all others 'id'?
    if (isA(ast, NodeKind::Call))
      return getTokenValue(findNode(ast, NodeKind::Function));
    return getTokenValue(findNode(ast, NodeKind::ID));
  }

  // ================================================= Type Helpers
  bool isType(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::OfType);
  }

  ::ast::WeakAst getType(::ast::WeakAst ast)
  {
    return findNode(ast, NodeKind::OfType);
  }

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
  bool isFunction(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Function);
  }

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

  ::ast::WeakAst getFunctionType(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Return type is in the sig / oftype / type
    auto sig = findNode(ast, NodeKind::Sig);
    return getType(sig);
  }

  std::vector<::ast::WeakAst> getFunctionArgs(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Arguments is in sig / params
    std::vector<::ast::WeakAst> args;
    auto sig = findNode(ast, NodeKind::Sig).lock();
    auto params = findNode(sig, NodeKind::Params).lock();
    for (auto param : params->nodes)
      args.push_back(findNode(param, NodeKind::NamedParam));
    return args;
  }

  std::vector<::ast::WeakAst> getFunctionConstraints(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    std::vector<::ast::WeakAst> constraints;
    auto sig = findNode(ast, NodeKind::Sig).lock();
    auto consts = findNode(sig, NodeKind::Constraints).lock();
    for (auto c : consts->nodes)
      constraints.push_back(c);
    return constraints;
  }

  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Body is just a block
    return findNode(ast, NodeKind::Block);
  }

  // ================================================= Class Helpers
  bool isClass(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::ClassDef);
  }

  ::ast::WeakAst getClassBody(::ast::WeakAst ast)
  {
    assert(isClass(ast) && "Bad node");

    // TypeBody is just a block node in the class
    return findNode(ast, NodeKind::TypeBody);
  }

  // ================================================= Operation Helpers
  bool isCall(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Call);
  }

  bool isReturn(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Return);
  }

  bool isAssign(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Assign);
  }

  ::ast::WeakAst getLHS(::ast::WeakAst ast)
  {
    // LHS is the assignable 'let' or 'localref'
    assert(isAssign(ast) && "Bad node");
    auto lhs = ast.lock()->nodes[0];
    assert((isLocalRef(lhs) || isLet(lhs)) && "Bad node");
    return lhs;
  }

  ::ast::WeakAst getRHS(::ast::WeakAst ast)
  {
    // RHS is the expression on the second node
    assert(isAssign(ast) && "Bad node");
    return ast.lock()->nodes[1];
  }

  size_t numOperands(::ast::WeakAst ast)
  {
    assert(isCall(ast) && "Bad node");
    if (!isValue(ast.lock()->nodes[2]))
      return 0;
    auto args = findNode(ast, NodeKind::Args);
    return args.lock()->nodes.size() + 1;
  }

  bool isUnary(::ast::WeakAst ast)
  {
    return numOperands(ast) == 1;
  }
  bool isBinary(::ast::WeakAst ast)
  {
    return numOperands(ast) == 2;
  }

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

  std::vector<::ast::WeakAst> getAllOperands(::ast::WeakAst ast)
  {
    auto numOps = numOperands(ast);
    std::vector<::ast::WeakAst> ops;
    for (size_t i = 0; i < numOps; i++)
      ops.push_back(getOperand(ast, i));
    return ops;
  }

  // ================================================= Condition Helpers
  bool isIf(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::If);
  }

  bool isCondition(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Condition);
  }

  bool isBlock(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Block);
  }

  bool isElse(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Else);
  }

  bool hasElse(::ast::WeakAst ast)
  {
    // Else nodes always exist inside `if` nodes, but if there was no `else`
    // block, they're empty. We should only return true if they're not.
    return isA(ast, NodeKind::If) && hasA(ast, NodeKind::Else) &&
      !findNode(ast, NodeKind::Else).lock()->nodes.empty();
  }

  ::ast::WeakAst getCond(::ast::WeakAst ast)
  {
    // These are the nodes that could have conditions as subnodes
    assert((isIf(ast) || isLoop(ast)) && "Bad node");
    // Cond nodes are always seq, even with just one cond
    auto cond = findNode(ast, NodeKind::Condition);
    return findNode(cond, NodeKind::Seq);
  }

  ::ast::WeakAst getIfBlock(::ast::WeakAst ast)
  {
    assert(isIf(ast) && "Bad node");
    return findNode(ast, NodeKind::Block);
  }

  ::ast::WeakAst getElseBlock(::ast::WeakAst ast)
  {
    assert(isIf(ast) && "Bad node");
    // Else has either a single node or a seq
    auto node = findNode(ast, NodeKind::Else);
    return node.lock()->nodes[0];
  }

  // ================================================= Condition Helpers
  bool isWhile(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::While);
  }

  bool isLoop(::ast::WeakAst ast)
  {
    // Add isFor when we support it
    return isWhile(ast);
  }

  bool isContinue(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Continue);
  }

  bool isBreak(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Break);
  }

  ::ast::WeakAst getLoopBlock(::ast::WeakAst ast)
  {
    assert(isLoop(ast) && "Bad node");
    return findNode(ast, NodeKind::Block);
  }
}
