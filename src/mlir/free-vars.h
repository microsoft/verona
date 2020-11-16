// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast-utils.h"
#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"
#include "error.h"

#include <map>
#include <set>
#include <string>

// Only for debug, remove later
#include <iostream>

namespace mlir::verona
{
  /**
   * Free Variable Analysis: scans nodes for new definitions and writes
   * to variables to identify which blocks or regions will need for return
   * values and arguments.
   *
   * The algorithm is recursive, using an intermediate block information to
   * allow collection of write/def variables (by name) and annotate the
   * nodes that define new basic blocks or regions to help the generator
   * define arguments and return values without having to back-track.
   *
   * Those names will only be associated with SSA values once lowering start,
   * and can be used directly to associate the arguments in each block.
   *
   * We don't consider reading of variables directly because in Verona every
   * write is also a read, and read-only operations use the variable scope
   * that will be updated by the basic block arguments, so keeping that
   * information up-to-date is irrelevant in the AST.
   */

  class FreeVariableAnalysis
  {
    /// Simple struct with writes and defs.
    using Vars = std::set<llvm::StringRef>;
    struct BlockInfo
    {
      Vars write;
      Vars def;
    };

    /// The map that will contain all arguments and returns, by name.
    /// Verona forbids shadowing, and the context of this analysis is within
    /// a function, so there is no context to worry about and no risk of
    /// having two different variables with the same name.
    std::map<::ast::Ast, Vars> freeVars;

    /// Merge variables from one block onto another.
    void mergeInfo(BlockInfo& base, BlockInfo&& sub)
    {
      base.write.insert(sub.write.begin(), sub.write.end());
      base.def.insert(sub.def.begin(), sub.def.end());
    }

    /// Get arguments from BlockInfo.
    /// Subtracting defs from writes gives us the block arguments.
    Vars getArgs(BlockInfo& block)
    {
      Vars args = block.write;
      for (auto def : block.def)
        args.erase(def);
      return args;
    }

    /// Return true if the node is a block.
    /// These are the merge points that will be cached in the freeVars map
    /// for the generator to query the list of arguments.
    bool isBlockNode(::ast::Ast node)
    {
      switch (node->tag)
      {
        case AST::NodeKind::If:
        case AST::NodeKind::While:
          return true;
      }
      return false;
    }

    /// Recursively run on a node (and its sub-nodes).
    BlockInfo runOnNode(::ast::Ast node)
    {
      BlockInfo info;

      // First, descend into all sub-nodes and take their info
      std::vector<::ast::Ast> nodes;
      AST::getSubNodes(nodes, node);
      for (auto sub : nodes)
      {
        mergeInfo(info, runOnNode(sub));
      }

      // Then see if there's any info this node can give
      switch (node->tag)
      {
        // Nodes that define new variables
        case AST::NodeKind::NamedParam:
        case AST::NodeKind::Let:
        {
          // Defines a variable with the name in the ID node.
          // Reads from sub-expressions have already been added to the
          // list by recursion and merge above.
          info.def.insert(AST::getID(node));
          break;
        }
        // Nodes that assign to variables
        case AST::NodeKind::Assign:
        {
          // Writes to the variable in the first node.
          // Reads from sub-expressions have already been added to the
          // list by recursion and merge above.
          info.write.insert(AST::getLocalName(node->nodes[0]));
          break;
        }
      }

      // Finally, update the map if need args/rets.
      if (isBlockNode(node))
        freeVars.insert({node, getArgs(info)});

      return info;
    }

  public:
    /// Run the analysis on a function, returns a map with all the nodes that
    /// create, use or consume values.
    void runOnFunction(::ast::Ast func)
    {
      assert(AST::isFunction(func) && "Bad node");
      // Clear any previous data and scan the whole function.
      freeVars.clear();
      runOnNode(func);
    }

    /// Append returns to a list, for building basic-block arguments.
    /// T must be a list<StringRef> type with push_back.
    template <class T>
    void appendArguments(::ast::Ast node, T& list)
    {
      auto args = freeVars.find(node);
      assert(args != freeVars.end() && "Node doesn't have associated Vars");
      for (auto ret : args->second)
        list.push_back(ret);
    }
  };
}
