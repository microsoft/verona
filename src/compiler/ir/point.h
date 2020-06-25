// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <iosfwd>
#include <variant>

namespace verona::compiler
{
  struct BasicBlock;

  /**
   * Represents a position in a function's IR.
   *
   * It pairs up a BasicBlock and an offset within that block.
   *
   * The offset is one of:
   * - Entry: the entrypoint of the BasicBlock, used for Phi nodes and function
   *   argument definitions.
   * - Statement(index): `index` is the position of the statement within the
   *   BasicBlock's `statements` field.
   * - Terminator
   *
   * Offsets within the same basic block can be compared using the usual
   * operators.
   */
  struct IRPoint
  {
  private:
    struct Entry
    {
      bool operator<(const Entry& other) const
      {
        return false;
      }
      bool operator<=(const Entry& other) const
      {
        return true;
      }
      bool operator==(const Entry& other) const
      {
        return true;
      }
      bool operator!=(const Entry& other) const
      {
        return false;
      }
    };
    struct Statement
    {
      size_t index;
      bool operator<(const Statement& other) const
      {
        return index < other.index;
      }
      bool operator<=(const Statement& other) const
      {
        return index <= other.index;
      }
      bool operator==(const Statement& other) const
      {
        return index == other.index;
      }
      bool operator!=(const Statement& other) const
      {
        return index != other.index;
      }
    };
    struct Terminator
    {
      bool operator<(const Terminator& other) const
      {
        return false;
      }
      bool operator<=(const Terminator& other) const
      {
        return true;
      }
      bool operator==(const Terminator& other) const
      {
        return true;
      }
      bool operator!=(const Terminator& other) const
      {
        return false;
      }
    };

  public:
    /*
     * The order of the variants is picked in way that gets a sane comparison
     * operators, that is:
     *
     *   Entry < Statement(i) < Statement(j) < Terminator
     *
     * for i < j
     */
    typedef std::variant<Entry, Statement, Terminator> Offset;

    const BasicBlock* basic_block;
    Offset offset;

    static IRPoint entry(const BasicBlock* bb)
    {
      return IRPoint(bb, Entry());
    }

    static IRPoint statement(const BasicBlock* bb, size_t index)
    {
      return IRPoint(bb, Statement{index});
    }

    static IRPoint terminator(const BasicBlock* bb)
    {
      return IRPoint(bb, Terminator());
    }

    friend std::ostream& operator<<(std::ostream& out, const IRPoint& point);

  private:
    explicit IRPoint(const BasicBlock* basic_block, Offset offset)
    : basic_block(basic_block), offset(offset)
    {}
  };
}
