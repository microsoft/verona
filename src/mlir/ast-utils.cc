// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "ast-utils.h"

#include "symbol.h"

namespace mlir::verona::ASTInterface
{
  // ================================================= Generic Helpers
  ::ast::WeakAst findNode(::ast::WeakAst ast, NodeType type)
  {
    auto ptr = ast.lock();
    assert(!ptr->is_token && "Bad node");
    // Match tag with NodeKind's enum value
    auto sub = std::find_if(
      ptr->nodes.begin(), ptr->nodes.end(), [type](::ast::Ast& sub) {
        return (sub->tag == type);
      });
    assert(sub != ptr->nodes.end());
    return *sub;
  }

  llvm::StringRef getTokenValue(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->is_token && "Bad node");
    assert(!ptr->token.empty());
    return ptr->token;
  }

  ::ast::WeakAst getType(::ast::WeakAst ast)
  {
    return findNode(ast, NodeKind::OfType);
  }

  // ================================================= Type Helpers
  const std::string getTypeDesc(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::OfType && "Bad node");
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
  llvm::StringRef getFunctionName(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::Function && "Bad node");

    // Empty function name is "apply"
    auto funcname = findNode(ptr, NodeKind::FuncName).lock();
    assert(!funcname->nodes.empty() && "Bad function");

    // Else, get function name
    assert(funcname->nodes.size() == 1);
    return getTokenValue(funcname->nodes[0]);
  }

  ::ast::WeakAst getFunctionType(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::Function && "Bad node");

    // Return type is in the sig / oftype / type
    auto sig = findNode(ast, NodeKind::Sig);
    return getType(sig);
  }

  std::vector<::ast::WeakAst> getFunctionArgs(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::Function && "Bad node");

    // Arguments is in sig / params
    std::vector<::ast::WeakAst> args;
    auto sig = findNode(ptr, NodeKind::Sig).lock();
    auto params = findNode(sig, NodeKind::Params).lock();
    for (auto param : params->nodes)
      args.push_back(findNode(param, NodeKind::NamedParam));
    return args;
  }

  std::vector<::ast::WeakAst> getFunctionConstraints(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::Function && "Bad node");

    std::vector<::ast::WeakAst> constraints;
    auto sig = findNode(ptr, NodeKind::Sig).lock();
    auto consts = findNode(sig, NodeKind::Constraints).lock();
    for (auto c : consts->nodes)
      constraints.push_back(c);
    return constraints;
  }

  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast)
  {
    auto ptr = ast.lock();
    assert(ptr->tag == NodeKind::Function && "Bad node");

    // Body is just a block node in the function
    return findNode(ast, NodeKind::Block);
  }
}
