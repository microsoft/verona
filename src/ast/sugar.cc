// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "sugar.h"

// Include PEG's ""_ operator for string-switches
using namespace peg::udl;

namespace
{
  /// Create an element, append to ast and return it.
  /// Used to simplify building nested structures.
  ::ast::Ast append_descend(::ast::Ast& ast, const std::string& name)
  {
    auto sub = node(ast, name.c_str());
    push_back(ast, sub);
    return sub;
  }

  /// Append an atom to `ast`, appending `arg` to the atom.
  void atom(::ast::Ast& ast, ::ast::Ast& arg)
  {
    auto atom = node(ast, "atom");
    move_back(atom, arg);
    push_back(ast, atom);
  }

  /// Append an atom to `ast`, appending a token to the atom.
  void
  token_atom(::ast::Ast& ast, const std::string& name, const std::string& value)
  {
    auto tok = token(ast, name.c_str(), value.c_str());
    atom(ast, tok);
  }

  /// Append an atom to `ast`, appending an empty node to the atom.
  void node_atom(::ast::Ast& ast, const std::string& name)
  {
    auto sub = node(ast, name.c_str());
    atom(ast, sub);
  }

  /// Add dynamic call with one argument (self) to the node `ast`
  void call_1(::ast::Ast& ast, const std::string& obj, const std::string& func)
  {
    // object
    token_atom(ast, "id", obj);
    // dot
    token_atom(ast, "sym", ".");
    // function name
    token_atom(ast, "id", func);
    // tuple
    node_atom(ast, "tuple");
  }

  /// Add assign pattern (var =) to the node `ast`
  void let_assign(::ast::Ast& ast, ::ast::Ast& var)
  {
    // let
    if (var->tag == "atom"_)
    {
      push_back(ast, var);
    }
    else
    {
      atom(ast, var);
    }
    // sym (=)
    token_atom(ast, "sym", "=");
  }

  /// Add assign to a variable pattern to the `ast` node, creating a new `let`
  void assign(::ast::Ast& ast, const std::string& name)
  {
    // create a new let
    auto let = node(ast, "let");
    auto id = token(let, "id", name.c_str());
    push_back(let, id);
    auto type = node(let, "oftype");
    push_back(let, type);
    // assign
    let_assign(ast, let);
  }

  /// Replace 'for' loop with 'while' loop.
  void sugar_for(::ast::Ast ast)
  {
    // block > seq > term > blockexpr > for (insert iterator in seq before loop)
    auto seq = get_closest(ast, "seq"_);
    auto term = get_closest(ast, "term"_);
    // for > expr > atom (expression that declares the iteration value)
    auto id = ast->nodes[0]->nodes[0];
    assert(id->tag == "atom"_);
    // for > seq > expr (expression that gives an iterator)
    auto iterExpr = ast->nodes[1]->nodes[0];
    assert(iterExpr->tag == "expr"_);
    // for > block > seq (sequence of expressions in the body)
    auto body = ast->nodes[2]->nodes[0];
    assert(body->tag == "seq"_);

    // Iterator term ($iter = iterator()), insert before loop's term
    auto iterTerm = node(ast, "term");
    std::string iter = hygienic_id(ast, "iter");
    assign(iterTerm, iter);
    move_children(iterExpr, iterTerm);
    insert_before(iterTerm, term);
    remove(iterExpr);

    // While loop (new, to replace the current ast)
    auto loop = node(ast, "while");

    // Condition (entirely new iter.has_value call)
    auto cond = append_descend(loop, "cond");
    auto seqCond = append_descend(cond, "seq");
    auto exprCond = append_descend(seqCond, "expr");
    call_1(exprCond, iter, "has_value");

    // Block (new, with apply, next and remainder of for block)
    auto block = append_descend(loop, "block");
    auto seqBlock = append_descend(block, "seq");

    // Apply (new, take the value of the iterator)
    auto termApply = append_descend(seqBlock, "term");
    let_assign(termApply, id);
    call_1(termApply, iter, "apply");

    // Next (new, increment the iterator)
    auto termNext = append_descend(seqBlock, "term");
    call_1(termNext, iter, "next");

    // Body (moving, add all terms from old body)
    move_children(body, seqBlock);
    remove(body);

    // Replacing old 'for' term 'with' while term
    replace(ast, loop);
  }
}

namespace sugar
{
  void build(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "for"_:
        // Replaces the for loop with a while loop.
        //
        // Change:
        // for (tuple in expr) {
        //   body(tuple);
        // }
        //
        // To:
        // $iter = expr;
        // while ($iter.has_value()) {
        //   tuple = $iter.apply();
        //   $iter.next();
        //   body(tuple);
        // }
        sugar_for(ast);
        break;
    }

    ast::for_each(ast, build, err);
  }
}
