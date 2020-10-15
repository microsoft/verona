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
   * Free Variable Analysis: scans nodes for new definitions, reads and writes
   * to variables to identify which blocks or regions will need for return
   * values and arguments.
   *
   * The algorithm is recursive, using an intermediate block information to
   * allow collection of read/write/def variables (by name) and annotate the
   * nodes that define new basic blocks or regions to help the generator
   * define arguments and return values without having to back-track.
   *
   * Those names will only be associated with SSA values once lowering start,
   * and can be used directly to associate the arguments in each block.
   */

  class FreeVariableAnalysis
  {
    using Vars = std::set<llvm::StringRef>;

    /// Simple struct with reads, writes and defs.
    struct BlockInfo
    {
      Vars read;
      Vars write;
      Vars def;
    };

    /// Simple struct to hold args and rets for each region.
    struct ArgRet
    {
      Vars args; // read minus def
      Vars rets; // write minus def
    };

  private:
    /// The map that will contain all arguments and returns, by name.
    /// Verona forbids shadowing, and the context of this analysis is within
    /// a function, so there is no context to worry about and no risk of
    /// having two different variables with the same name.
    std::map<::ast::Ast, ArgRet> freeVars;

    /// Cache of the function being evaluated (for dumping purposes)
    ::ast::Ast function;

    /// Merge variables from one block onto another.
    void mergeInfo(BlockInfo& base, BlockInfo&& sub)
    {
      base.read.insert(sub.read.begin(), sub.read.end());
      base.write.insert(sub.write.begin(), sub.write.end());
      base.def.insert(sub.def.begin(), sub.def.end());
    }

    /// Get arguments and return values from read/write/def.
    ArgRet getArgsRets(BlockInfo& block)
    {
      ArgRet ar;
      ar.args = block.read;
      ar.rets = block.write;
      // TODO: this is inefficient but will do for now
      for (auto def : block.def)
      {
        ar.args.erase(def);
        ar.rets.erase(def);
      }
      return ar;
    }

    /// Return true if the node is a block (and needs arguments, rets).
    bool isBlockNode(::ast::Ast node)
    {
      switch (node->tag)
      {
        case AST::NodeKind::If:
        case AST::NodeKind::While:
        case AST::NodeKind::For:
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
      // TODO: This may be incomplete.
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
        // Nodes that access existing variables.
        case AST::NodeKind::Localref:
        {
          // All reads are localref at some (nested) level, and since we're
          // recursing earlier (and merging the data) we don't need to search
          // for them here.
          info.read.insert(AST::getLocalName(node));
          break;
        }
      }

      // Finally, update the map if need args/rets.
      if (isBlockNode(node))
        freeVars.insert({node, getArgsRets(info)});

      return info;
    }

    /// Dump an ArgRet structure.
    void dump(ArgRet& ar)
    {
      std::cerr << "    Args:";
      for (auto var : ar.args)
        std::cerr << "  " << var.str();
      std::cerr << std::endl;
      std::cerr << "    Rets:";
      for (auto var : ar.rets)
        std::cerr << "  " << var.str();
      std::cerr << std::endl;
    }

  public:
    /// Run the analysis on a function, returns a map with all the nodes that
    /// create, use or consume values.
    void runOnFunction(::ast::Ast func)
    {
      assert(AST::isFunction(func) && "Bad node");
      // Cache for debug purposes, get rid once we're happy with the impl.
      function = func;
      // Clear any previous data and scan the whole function.
      freeVars.clear();
      runOnNode(func);
    }

    /// Return the ArgsRet information from an Ast node
    const ArgRet& getArgRet(::ast::Ast node)
    {
      assert(!freeVars.empty() && "Not in a function");
      auto argRet = freeVars.find(node);
      assert(argRet != freeVars.end() && "Node doesn't have associated ArgRet");
      return argRet->second;
    }

    /// Dumps the whole map, per function.
    void dump()
    {
      auto funcName = AST::getFunctionName(function);
      std::cerr << "Free Var Map: " << funcName.str() << std::endl;
      for (auto pair : freeVars)
      {
        auto node = pair.first;
        std::cerr << "Node: " << node->name << std::endl;
        auto ar = pair.second;
        dump(ar);
      }
      std::cerr << std::endl;
    }
  };
}
