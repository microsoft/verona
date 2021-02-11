// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::verona
{
  /**
   * This is a bag of utility functions to handle AST lookups and fail-safe
   * operations. While the AST design is still in flux, we can keep this around,
   * but once we're set on its structure, this should move to src/ast instead.
   *
   * We hold no values here, so we don't need to worry about ownership. All
   * methods are static and work on Ast nodes directly. The ownership of those
   * nodes is up to the caller.
   *
   * Values returned are either new strings, Ast or a new vector of Ast nodes.
   * The idea is to detach the structures of the AST and have a flat
   * representation for the specific types of nodes we need in each call.
   *
   * This is a static struct because we're declaring the functions directly on
   * the header (so we can use templates) and we don't want to end up with
   * duplicated definitions when included more than once. We also don't want
   * to make it super complex as it will go away some day in favour of a more
   * structured interface.
   */
  struct AST
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
      Field = peg::str2tag("field"), // = 122469826
      Function = peg::str2tag("function"), // = 89836898
      FuncName = peg::str2tag("funcname"), // = 90200697
      Sig = peg::str2tag("sig"), // = 124317
      Qualifier = peg::str2tag("qualifier"), // = 3224348024
      Static = peg::str2tag("static"), // = 156887736
      Block = peg::str2tag("block"), // = 117895113
      OfType = peg::str2tag("oftype"), // = 3504561
      Constraints = peg::str2tag("constraints"), // = 4070926742
      Constraint = peg::str2tag("constraint"), // = 2466070853
      Type = peg::str2tag("type"), // = 4058008
      TypeTuple = peg::str2tag("type_tuple"), // = 1428239871
      TypeRef = peg::str2tag("type_ref"), // = 2115269750
      Typeref = peg::str2tag("typeref"), // = 4098803273
      TypeOp = peg::str2tag("type_op"), // = 4098764984
      TypeOne = peg::str2tag("type_one"), // = 2115257603
      TypeBody = peg::str2tag("typebody"), // = 2117081800
      QualType = peg::str2tag("qualtype"), // = 1660450641
      Bool = peg::str2tag("bool"), // = 3565102
      Hex = peg::str2tag("hex"), // = 110293
      Binary = peg::str2tag("binary"), // = 3884723791
      Params = peg::str2tag("params"), // = 4271185724
      NamedParam = peg::str2tag("namedparam"), // = 1544471404
      ID = peg::str2tag("id"), // = 3565
      Seq = peg::str2tag("seq"), // = 124167
      Assign = peg::str2tag("assign"), // = 3930587681
      Let = peg::str2tag("let"), // = 114397
      Call = peg::str2tag("call"), // = 3519010
      Invoke = peg::str2tag("invoke"), // = 4219878096
      StaticCall = peg::str2tag("static-call"), // = 1256961527
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
      For = peg::str2tag("for"), // = 112155
      Continue = peg::str2tag("continue"), // = 2929012833
      Break = peg::str2tag("break"), // = 117842911
      New = peg::str2tag("new"), // = 120796
      InRegion = peg::str2tag("inregion"), // = 4267592959
      InitExpr = peg::str2tag("initexpr"), // = 3935567717
      Member = peg::str2tag("member"), // = 75189616
      Lookup = peg::str2tag("lookup"), // = 4099072514
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
    static NodePath getPath(::ast::Ast ast)
    {
      return {ast->path, ast->line, ast->column};
    }

    // ================================================= Generic Helpers
    /// Return the 'name' of a node, for error reporting
    static const std::string& getName(::ast::Ast ast)
    {
      return ast->name;
    }

    /// Return the 'kind' of a node, for comparison
    static unsigned int getKind(::ast::Ast ast)
    {
      return ast->tag;
    }

    /// Return true if node is of a certain kind
    static bool isA(::ast::Ast ast, NodeKind kind)
    {
      return getKind(ast) == kind;
    }

    /// Return true if node is of any kind in a list
    static bool isAny(::ast::Ast ast, llvm::ArrayRef<NodeKind> kinds)
    {
      return std::find(kinds.begin(), kinds.end(), getKind(ast)) != kinds.end();
    }

    /// Return true if node has a certain kind sub-node
    static bool hasA(::ast::Ast ast, NodeKind kind)
    {
      auto nodes = ast->nodes;
      return std::find_if(nodes.begin(), nodes.end(), [&](::ast::Ast& node) {
               return isA(node, kind);
             }) != nodes.end();
    }

    /// Return true if node has subnodes
    static bool hasSubs(::ast::Ast ast)
    {
      return !ast->nodes.empty();
    }

    /// Return true if node is a value
    static bool isValue(::ast::Ast ast)
    {
      return ast->is_token;
    }

    /// Return true if node is a value
    static bool isConstant(::ast::Ast ast)
    {
      return isValue(ast) &&
        isAny(ast,
              {NodeKind::Integer,
               NodeKind::Float,
               NodeKind::Bool,
               NodeKind::Hex,
               NodeKind::Binary});
    }

    /// Return true if node is a local variable reference
    static bool isLocalRef(::ast::Ast ast)
    {
      return isValue(ast) && isA(ast, NodeKind::Localref);
    }

    /// Return true if node is a new variable definition
    static bool isLet(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Let);
    }

    /// Return true if node is a type
    static bool isType(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Type);
    }

    /// Return true if node is a type holder
    static bool isTypeHolder(::ast::Ast ast)
    {
      return isAny(ast, {NodeKind::OfType, NodeKind::TypeTuple, NodeKind::New});
    }

    /// Return true if node is a type holder
    static bool isQualType(::ast::Ast ast)
    {
      return isA(ast, NodeKind::QualType);
    }

    /// Return true if node is final type (tuple element)
    static bool IsTypeTupleElement(::ast::Ast ast)
    {
      return isA(ast, NodeKind::TypeOne);
    }

    /// Return true if node is a field
    static bool isField(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Field);
    }

    /// Return true if node is a function
    static bool isFunction(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Function);
    }

    /// Return true if node is a static qualifier
    static bool isStatic(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Static);
    }

    /// Return true if node is a class/module
    static bool isClass(::ast::Ast ast)
    {
      return isA(ast, NodeKind::ClassDef);
    }

    /// Return true if node is a member access
    static bool isMember(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Member);
    }

    /// Return true if node is an operation/call
    static bool isCall(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Call);
    }

    /// Return true if node is a direct invoke (obj.method)
    static bool isInvoke(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Invoke);
    }

    /// Return true if node is a static call
    static bool isStaticCall(::ast::Ast ast)
    {
      return isA(ast, NodeKind::StaticCall);
    }

    /// Return true if node is a return
    static bool isReturn(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Return);
    }

    /// Return true if node is an assignment
    static bool isAssign(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Assign);
    }

    /// Return true if node is an if statement
    static bool isIf(::ast::Ast ast)
    {
      return isA(ast, NodeKind::If);
    }

    /// Return true if node is a condition
    static bool isCondition(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Condition);
    }

    /// Return true if node is a block
    static bool isBlock(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Block);
    }

    /// Return true if node is an else block
    static bool isElse(::ast::Ast ast)
    {
      return isA(ast, NodeKind::Else);
    }

    /// Return true if node is a while loop
    static bool isWhile(::ast::Ast ast)
    {
      return isA(ast, NodeKind::While);
    }

    /// Return true if node is a for loop
    static bool isFor(::ast::Ast ast)
    {
      return isA(ast, NodeKind::For);
    }

    /// Return true if node is any kind of loop
    static bool isLoop(::ast::Ast ast)
    {
      return isWhile(ast) || isFor(ast);
    }

    /// Return true if node is a loop continue
    static bool isContinue(::ast::Ast ast)
    {
      return isValue(ast) && isA(ast, NodeKind::Continue);
    }

    /// Return true if node is a loop break
    static bool isBreak(::ast::Ast ast)
    {
      return isValue(ast) && isA(ast, NodeKind::Break);
    }

    /// Return true if node is a new allocator
    static bool isNew(::ast::Ast ast)
    {
      return isA(ast, NodeKind::New);
    }

    /// Return true if node defined a region to allocate
    static bool hasInRegion(::ast::Ast ast)
    {
      return hasA(ast, NodeKind::InRegion);
    }

    /// Find a sub-node of tag 'type'
    static ::ast::Ast findNode(::ast::Ast ast, NodeType type)
    {
      assert(!isValue(ast) && "Bad node");
      // Match tag with NodeKind's enum value
      auto ptr = ast;
      auto sub = std::find_if(
        ptr->nodes.begin(), ptr->nodes.end(), [type](::ast::Ast& sub) {
          return (sub->tag == type);
        });
      // TODO: Make this into a soft error
      assert(sub != ptr->nodes.end());
      return *sub;
    }

    /// Return the single sub-node, error out if more than one
    static ::ast::Ast getSingleSubNode(::ast::Ast ast)
    {
      assert(ast->nodes.size() == 1 && "Wrong number of nodes");
      return ast->nodes[0];
    }

    /// Get a list of sub-nodes.
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getSubNodes(T& nodes, ::ast::Ast ast)
    {
      // If single node, but of type 'seq', descend
      if (ast->nodes.size() == 1 && isA(ast->nodes[0], NodeKind::Seq))
        ast = ast->nodes[0];

      // Return the nodes
      nodes.insert(nodes.end(), ast->nodes.begin(), ast->nodes.end());
    }

    // ================================================= Value Helpers
    /// Get the string value of a token
    static llvm::StringRef getTokenValue(::ast::Ast ast)
    {
      assert(isValue(ast) && "Bad node");
      assert(!ast->token.empty());
      return ast->token;
    }

    /// Get the local reference (local variable) from an expression
    static llvm::StringRef getLocalRef(::ast::Ast ast)
    {
      assert(!isValue(ast) && "Bad node");
      return getTokenValue(findNode(ast, NodeKind::Localref));
    }

    /// Return true if node is a variable definition
    static llvm::StringRef getLocalName(::ast::Ast ast)
    {
      // Local variables can be new 'local' or existing 'localref'
      if (isLocalRef(ast))
        return getTokenValue(ast);
      if (isMember(ast))
        return getTokenValue(findNode(ast, NodeKind::Localref));
      assert(isLet(ast) && "Bad node");
      return getTokenValue(findNode(ast, NodeKind::Local));
    }

    /// Return true if node is an ID (func, var, type names)
    static llvm::StringRef getID(::ast::Ast ast)
    {
      if (isAny(ast, {NodeKind::Call, NodeKind::StaticCall}))
        return getTokenValue(findNode(ast, NodeKind::Function));
      if (isMember(ast))
        return getTokenValue(findNode(ast, NodeKind::Lookup));
      if (isQualType(ast))
        return getTokenValue(findNode(ast, NodeKind::Typeref));
      if (isInvoke(ast))
        return getID(findNode(ast, NodeKind::Member));
      return getTokenValue(findNode(ast, NodeKind::ID));
    }

    // ================================================= Type Helpers
    /// Get the ast type representaiton of an ast node
    static ::ast::Ast getType(::ast::Ast ast)
    {
      // This can be any of the type holders
      for (auto node : ast->nodes)
      {
        if (isTypeHolder(node))
          return node;
      }
      // TODO: Make this into a soft error
      assert(false && "AST node doesn't have a type holder");
      // Avoid no-return warnings, until we fix the TODO above
      return ::ast::Ast();
    }

    /// Return true if node has non-empty (oftype / type)
    static bool hasType(::ast::Ast ast)
    {
      if (!isTypeHolder(ast))
        return false;
      return hasA(ast, NodeKind::Type);
    }

    /// Get the list of types grouped by the same operator (| or &) or
    /// a single type, if no grouping.
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getTypeElements(::ast::Ast ast, char& sep, T& nodes)
    {
      assert(isTypeHolder(ast) && "Bad node");

      // (type) is a tuple of (type_one)s and/or (type_op)s. The type ops must
      // be all the same (& or |), return a list of (type_one)s and the group
      // operator (in sep).
      sep = 0;
      auto type = findNode(ast, NodeKind::Type);
      for (auto node : type->nodes)
      {
        switch (node->tag)
        {
          case NodeKind::TypeOp:
            // Check if type operator is the same all over
            if (!sep)
              sep = node->token[0];
            else
              assert(sep == node->token[0] && "Invalid type grouping");
            break;
          case NodeKind::TypeOne:
            // If this node is a tuple, it will have a type holder in it.
            // otherwise, it will be a (type_ref)
            nodes.push_back(node->nodes[0]);
            break;
        }
      }
    }

    // ================================================= Function Helpers
    /// Get the string name of a function node
    static llvm::StringRef getFunctionName(::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      auto funcname = findNode(ast, NodeKind::FuncName);
      assert(hasSubs(funcname) && "Bad function");
      return getTokenValue(funcname->nodes[0]);
    }

    /// Get the return type of a function node
    static ::ast::Ast getFunctionType(::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      // Return type is in the sig / oftype / type
      auto sig = findNode(ast, NodeKind::Sig);
      return getType(sig);
    }

    /// Get the ast nodes for the function arguments.
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getFunctionArgs(T& args, ::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      // Arguments are in sig / params
      auto sig = findNode(ast, NodeKind::Sig);
      auto params = findNode(sig, NodeKind::Params);
      for (auto param : params->nodes)
        args.push_back(findNode(param, NodeKind::NamedParam));
    }

    /// Get the ast nodes for the function constraints
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getFunctionConstraints(T& constraints, ::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      // Constraints are in sig / params
      auto sig = findNode(ast, NodeKind::Sig);
      auto consts = findNode(sig, NodeKind::Constraints);
      for (auto c : consts->nodes)
        constraints.push_back(c);
    }

    /// Get the ast node for the function body
    static ::ast::Ast getFunctionBody(::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      // Body is just a block
      return findNode(ast, NodeKind::Block);
    }

    /// Returns true if the function has a body (definition)
    static bool hasFunctionBody(::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      // Body is just a block
      return hasA(ast, NodeKind::Block);
    }

    /// Get the function qualifiers
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getFunctionQualifiers(T& qual, ::ast::Ast ast)
    {
      assert(isFunction(ast) && "Bad node");

      auto quals = findNode(ast, NodeKind::Qualifier);
      for (auto c : quals->nodes)
        qual.push_back(c);
    }

    // ================================================= Class Helpers
    /// Get the body of a class/module declaration
    static ::ast::Ast getClassBody(::ast::Ast ast)
    {
      assert((isClass(ast) || isNew(ast)) && "Bad node");

      // TypeBody is just a block node in the class
      return findNode(ast, NodeKind::TypeBody);
    }

    /// Get the class allocation region if any
    static ::ast::Ast getInRegion(::ast::Ast ast)
    {
      assert(isNew(ast) && "Bad node");

      return findNode(ast, NodeKind::InRegion);
    }

    /// Get the field init expression
    static ::ast::Ast getInitExpr(::ast::Ast ast)
    {
      assert(isField(ast) && "Bad node");

      // First node of initexpr is the expression
      return findNode(ast, NodeKind::InitExpr)->nodes[0];
    }

    /// Get the class ref node from a type
    static ::ast::Ast getClassTypeRef(::ast::Ast ast)
    {
      return findNode(
        findNode(findNode(ast, NodeKind::Type), NodeKind::TypeOne),
        NodeKind::TypeRef);
    }

    /// Get the body of a class/module declaration
    static bool isClassType(::ast::Ast ast)
    {
      if (!isTypeHolder(ast))
        return false;

      auto typeID = findNode(
        findNode(
          findNode(findNode(ast, NodeKind::Type), NodeKind::TypeOne),
          NodeKind::TypeRef),
        NodeKind::ID);
      auto def = ast::get_def(typeID, typeID->token);
      if (!def)
        return false;
      return isClass(def);
    }

    // ================================================= Operation Helpers
    /// Return the number of operands in an operation
    static size_t numOperands(::ast::Ast ast)
    {
      assert((isCall(ast) || isInvoke(ast) || isStaticCall(ast)) && "Bad node");
      // Dynamic calls must have descriptor
      if (isCall(ast))
        assert(isValue(ast->nodes[2]) && "No descriptor for dynamic call");
      // Invokes must have a member
      if (isInvoke(ast))
        findNode(ast, NodeKind::Member);
      // Dynamic call's first argument is separate (descriptor), bool to int
      size_t firstArg = isCall(ast) || isInvoke(ast);
      auto args = findNode(ast, NodeKind::Args);
      return args->nodes.size() + firstArg;
    }

    /// Return the left-hand side of an assignment
    static ::ast::Ast getLHS(::ast::Ast ast)
    {
      // LHS is the assignable 'let' or 'localref'
      assert(isAssign(ast) && "Bad node");
      auto lhs = ast->nodes[0];
      assert((isLocalRef(lhs) || isLet(lhs) || isMember(lhs)) && "Bad node");
      return lhs;
    }

    /// Return the right-hand side of an assignment
    static ::ast::Ast getRHS(::ast::Ast ast)
    {
      // RHS is the expression on the second node
      assert(isAssign(ast) && "Bad node");
      return ast->nodes[1];
    }

    /// Return the n-th operand of the operation
    static ::ast::Ast getOperand(::ast::Ast ast, size_t n)
    {
      assert(n < numOperands(ast) && "Bad offset");
      auto args = findNode(ast, NodeKind::Args);
      // Static calls don't have special first argument
      if (isStaticCall(ast))
        return args->nodes[n];
      // Invoke/Calls have the first operand as 'localref'
      if (n == 0)
      {
        auto node = ast;
        if (isInvoke(ast))
          node = findNode(ast, NodeKind::Member);
        return findNode(node, NodeKind::Localref);
      }
      // All others in 'args'
      return args->nodes[n - 1];
    }

    /// Get all operands of the operation.
    /// T must be a list<ast> type with push_back.
    template<class T>
    static void getAllOperands(T& ops, ::ast::Ast ast)
    {
      auto numOps = numOperands(ast);
      for (size_t i = 0; i < numOps; i++)
        ops.push_back(getOperand(ast, i));
    }

    /// Get static call's qualified type
    static ::ast::Ast getStaticQualType(::ast::Ast ast)
    {
      assert(isStaticCall(ast) && "Bad node");
      return findNode(ast, NodeKind::QualType);
    }

    // ================================================= Condition Helpers
    /// Return the condition from an if statement
    static bool hasElse(::ast::Ast ast)
    {
      // Else nodes always exist inside `if` nodes, but if there was no `else`
      // block, they're empty. We should only return true if they're not.
      return isA(ast, NodeKind::If) && hasA(ast, NodeKind::Else) &&
        hasSubs(findNode(ast, NodeKind::Else));
    }

    /// Return the block from an if statement
    static ::ast::Ast getCond(::ast::Ast ast)
    {
      // These are the nodes that could have conditions as subnodes
      assert((isIf(ast) || isLoop(ast)) && "Bad node");
      // Cond nodes are always seq, even with just one cond
      auto cond = findNode(ast, NodeKind::Condition);
      return findNode(cond, NodeKind::Seq);
    }

    /// Return true if the 'if' node has an else block
    static ::ast::Ast getIfBlock(::ast::Ast ast)
    {
      assert(isIf(ast) && "Bad node");
      return findNode(ast, NodeKind::Block);
    }

    /// Return the else block from an if statement
    static ::ast::Ast getElseBlock(::ast::Ast ast)
    {
      assert(hasElse(ast) && "Bad node");
      // Else has either a single node or a seq
      auto node = findNode(ast, NodeKind::Else);
      return node->nodes[0];
    }

    // ================================================= Loop Helpers
    /// Return the block from a loop
    static ::ast::Ast getLoopBlock(::ast::Ast ast)
    {
      assert(isLoop(ast) && "Bad node");
      return findNode(ast, NodeKind::Block);
    }

    /// Return the sequence generator from a `for` loop
    static ::ast::Ast getLoopSeq(::ast::Ast ast)
    {
      assert(isFor(ast) && "Bad node");
      return findNode(findNode(ast, NodeKind::Seq), NodeKind::Localref);
    }

    /// Return the induction variable from a `for` loop
    static ::ast::Ast getLoopInd(::ast::Ast ast)
    {
      assert(isFor(ast) && "Bad node");
      return findNode(ast, NodeKind::Localref);
    }
  };
}
