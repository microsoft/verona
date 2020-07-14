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

  bool isLocalRef(::ast::WeakAst ast)
  {
    return isValue(ast) && isA(ast, NodeKind::Localref);
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

  bool isAssign(::ast::WeakAst ast)
  {
    return isA(ast, NodeKind::Assign);
  }

  ::ast::WeakAst getLHS(::ast::WeakAst ast)
  {
    // LHS is the assignable 'let'
    assert(isAssign(ast) && "Bad node");
    return findNode(ast, NodeKind::Let);
  }

  ::ast::WeakAst getRHS(::ast::WeakAst ast)
  {
    // RHS is the expression on the second node
    assert(isAssign(ast) && "Bad node");
    return ast.lock()->nodes[1];
  }

  ::ast::WeakAst getOperand(::ast::WeakAst ast, size_t n)
  {
    assert(isCall(ast) && "Bad node");
    auto ptr = ast.lock();
    // Calls have the first operand after 'typeargs' (3rd place)
    if (n == 0)
      return ptr->nodes[2];
    // All others in 'args'
    auto args = ptr->nodes[3];
    assert(n <= args->nodes.size() && "Bad offset");
    return args->nodes[n - 1];
  }
}
