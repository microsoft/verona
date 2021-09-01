// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/point.h"
#include "compiler/ir/type_arguments.h"
#include "compiler/ir/variable.h"
#include "compiler/source_manager.h"
#include "compiler/type.h"

#include <deque>
#include <queue>
#include <unordered_set>

/*
 * SSA-based Intermediate Representation of verona programs.
 *
 * The program AST is converted to IR form early on in the process. Most of the
 * compiler passes, such as type inference, region checking and code
 * generation, operate directly on the IR rather than the AST.
 *
 * With flow-typing, an AST variable could have multiple types, for each
 * assignement. Moreover, the type of the variable could also change on
 * control-flow joins. The SSA form makes this very natural, as each variable
 * has a single definition and a single type. Control-flow joins create phi
 * nodes to merge the definitions, which creates a new variable and new type.
 *
 * The IR and SSA themselves are pretty classic. The IR is made of basic blocks
 * which have a single entry-point and a single terminator. The terminator can
 * jump to zero (return terminator), one (branch terminator), or many (if/match
 * terminators) other basic blocks.
 *
 * Phi nodes can only be used if all the predecessors have a unique successor.
 * This simplifies various analyses, as well as code-generation.
 *
 * Because Phi nodes might need to be inserted at the top of any block with
 * multiple predecessors, we require that all such block have predecessors with
 * a unique successor.
 *
 * Essentially, the following two graphs are allowed. Because BBs 1 and 2 have
 * a single predecessor, BB 0, they will never need phi nodes to be inserted.
 * It's therefore acceptable for BB 0 to have multiple successors. Conversly,
 * since BB 5 has multiple predecessors, it may require phi nodes to be
 * inserted, thus BBs 3 and 4 must have a single successor.
 *
 *       +-----+        +-----+   +-----+
 *       |  0  |        |  3  |   |  4  |
 *       +-+-+-+        +---+-+   +-+---+
 *         | |              |       |
 *      +--+ +--+           +--+ +--+
 *      ↓       ↓              ↓ ↓
 *  +-----+   +-----+        +-----+
 *  |  1  |   |  2  |        |  5  |
 *  +-----+   +-----+        +-----+
 *
 * In contrast, the following graph is not allowed, because BB 9 has multiple
 * predecessors, but one of them, BB 6, has multiple successors.
 *
 *            +-----+   +-----+
 *            |  6  |   |  7  |
 *            +-+-+-+   +---+-+
 *              | |         |
 *              | +-------+ |
 *              ↓         ↓ ↓
 *            +-----+   +-----+
 *            |  8  |   |  9  |
 *            +-----+   +-----+
 *
 */
namespace verona::compiler
{
  struct TypecheckResults;
  struct BasicBlock;
  struct FunctionIR;

  struct IRInput
  {
    Variable variable;
    SourceManager::SourceRange source_range;

    IRInput() = default;
    IRInput(Variable variable, SourceManager::SourceRange source_range)
    : variable(variable), source_range(source_range)
    {}

    operator Variable() const
    {
      return variable;
    }
  };

  struct BranchTerminator
  {
    BasicBlock* target;
    std::vector<Variable> phi_arguments;
  };
  struct IfTerminator
  {
    IRInput input;
    BasicBlock* true_target;
    BasicBlock* false_target;
  };

  /**
   * Match expressions are translated in two different structures.
   * The match terminator decides which blocks to jump to, based on the
   * descriptor of the input. Each target block begins with a "match bind"
   * statement, which is essentially an unchecked downcast of the input to the
   * right type
   */
  struct MatchTerminator
  {
    IRInput input;
    struct Arm
    {
      TypePtr type;
      BasicBlock* target;
      Variable binding;
    };
    std::vector<Arm> arms;
  };
  struct ReturnTerminator
  {
    IRInput input;
  };
  typedef std::
    variant<BranchTerminator, IfTerminator, MatchTerminator, ReturnTerminator>
      Terminator;

  struct BaseStatement
  {
    BaseStatement(SourceManager::SourceRange source_range)
    : source_range(source_range)
    {}

    // Source range of the expression from which this statement was generated.
    SourceManager::SourceRange source_range;

    BaseStatement(const BaseStatement&) = delete;
    BaseStatement(BaseStatement&&) = default;
    BaseStatement& operator=(const BaseStatement&) = delete;
    BaseStatement& operator=(BaseStatement&&) = default;
  };

  struct StaticTypeStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    const Entity* definition;
    TypeArgumentsId type_arguments;
  };

  struct NewStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    std::optional<IRInput> parent;
    const Entity* definition;
    TypeArgumentsId type_arguments;
  };

  struct CallStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput receiver;
    std::string method;
    std::vector<IRInput> arguments;
    TypeArgumentsId type_arguments;
  };

  struct WhenStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output; // Promise goes here, will be UNIT to start with
    std::vector<IRInput> cowns;
    std::vector<IRInput> captures;
    size_t closure_index; // Body of the function to call.
  };

  struct ReadFieldStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput base;
    std::string name;
  };

  struct WriteFieldStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput base;
    IRInput right;
    std::string name;
  };

  struct IntegerLiteralStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    uint64_t value;
  };

  struct StringLiteralStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    std::string value;
  };

  struct MatchBindStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput input;
    TypePtr type;
  };

  struct CopyStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput input;
  };

  struct UnitStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
  };

  struct ViewStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable output;
    IRInput input;
  };

  /**
   * Generated at the end of a source-language scope.
   * Contains the list of local variables that go out of scope here. The
   * EndScopeStmt may be omitted if the list is empty.
   *
   * This might not be the end of a BB, for example when a BlockExpr is used.
   */
  struct EndScopeStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    // List of variables that go out of scope here.
    std::vector<Variable> dead_variables;
  };

  /**
   * Generated when a local is overwritten. This indicates the old SSA variable
   * that was bound to that local.
   *
   * TODO: Should this be part of BaseStatement? Or part of every statement that
   * may overwrite a local?
   */
  struct OverwriteStmt : public BaseStatement
  {
    using BaseStatement::BaseStatement;

    Variable dead_variable;
  };

  typedef std::variant<
    NewStmt,
    StaticTypeStmt,
    CallStmt,
    WhenStmt,
    ReadFieldStmt,
    WriteFieldStmt,
    IntegerLiteralStmt,
    StringLiteralStmt,
    CopyStmt,
    MatchBindStmt,
    ViewStmt,
    EndScopeStmt,
    OverwriteStmt,
    UnitStmt>
    Statement;

  struct BasicBlock
  {
    size_t index;

    BasicBlock* immediate_dominator;
    std::vector<BasicBlock*> predecessors;
    std::vector<Variable> phi_nodes;

    std::vector<Statement> statements;
    std::optional<Terminator> terminator;

    // This will fail if not a branch terminator
    BranchTerminator& branch_terminator()
    {
      return std::get<BranchTerminator>(terminator.value());
    }
    const BranchTerminator& branch_terminator() const
    {
      return std::get<BranchTerminator>(terminator.value());
    }

    /**
     * Invokes the given function on each successor of the basic block.
     */
    void visit_successors(const std::function<void(BasicBlock*)> visitor) const;

    IRPoint entry_point() const
    {
      return IRPoint::entry(this);
    }

    IRPoint terminator_point() const
    {
      return IRPoint::terminator(this);
    }

    /**
     * BasicBlock can be iterated over, returning a pair of IRPoint and
     * Statement.
     */
    struct iterator
    {
      typedef std::pair<IRPoint, const Statement&> value_type;

      // prefix increment
      iterator& operator++()
      {
        it_++;
        return *this;
      }

      value_type operator*() const
      {
        size_t index = it_ - basic_block_->statements.begin();
        return {IRPoint::statement(basic_block_, index), *it_};
      }

      bool operator==(const iterator& other) const
      {
        // We shouldn't be comparing iterators that come from different
        // BasicBlocks
        assert(basic_block_ == other.basic_block_);
        return it_ == other.it_;
      }

      bool operator!=(const iterator& other) const
      {
        // We shouldn't be comparing iterators that come from different
        // BasicBlocks
        assert(basic_block_ == other.basic_block_);
        return it_ != other.it_;
      }

    private:
      iterator(const BasicBlock* bb, std::vector<Statement>::const_iterator it)
      : basic_block_(bb), it_(it)
      {}
      friend BasicBlock;

      const BasicBlock* basic_block_;
      std::vector<Statement>::const_iterator it_;
    };

    iterator begin() const
    {
      return iterator(this, statements.begin());
    }
    iterator end() const
    {
      return iterator(this, statements.end());
    }
  };

  struct FunctionIR
  {
    std::optional<Variable> receiver;
    std::vector<Variable> parameters;

    // Use deque in order to have stable addresses.
    std::deque<BasicBlock> basic_blocks;

    BasicBlock* entry;

    // Set of basic blocks which return from the function
    std::unordered_set<BasicBlock*> exits;

    // Point in the IR at which each SSA Variable is defined.
    std::unordered_map<Variable, IRPoint> variable_definitions;

    /**
     * Add a BasicBlock to the IR, with the given immediate dominator
     */
    BasicBlock* add_block(BasicBlock* dominator)
    {
      basic_blocks.push_back(BasicBlock{});

      BasicBlock* bb = &basic_blocks.back();
      bb->index = basic_blocks.size() - 1;
      bb->immediate_dominator = dominator;

      return bb;
    }
  };

  struct MethodIR
  {
    std::vector<std::unique_ptr<FunctionIR>> function_irs;

    /**
     * Add a FunctionIR to this method
     */
    FunctionIR* create_function_ir()
    {
      auto ir = std::make_unique<FunctionIR>();
      auto irp = ir.get();
      function_irs.push_back(std::move(ir));
      return irp;
    }
  };

  /**
   * Currently a breath-first traversal of the CFG.
   *
   * A reverse-postorder would be nicer, but there doesn't seem to be any
   * iterative algorithm for it.
   */
  class IRTraversal
  {
  public:
    IRTraversal(const FunctionIR& ir);

    BasicBlock* next();

  private:
    void enqueue_block(BasicBlock* bb);

    std::unordered_set<BasicBlock*> visited_;
    std::queue<BasicBlock*> queue_;
  };

  std::ostream& operator<<(std::ostream& s, const BasicBlock& bb);
  std::ostream& operator<<(std::ostream& s, const Variable& v);
}
