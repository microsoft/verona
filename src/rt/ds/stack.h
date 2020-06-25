// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <snmalloc.h>

namespace verona::rt
{
  template<class T, class Alloc>
  class Stack
  {
  private:
    static constexpr size_t STACK_COUNT = 63;

    struct Block
    {
      Block* prev;
      T data[STACK_COUNT];
    };

    Alloc* alloc;
    Block* block;
    Block* backup;
    size_t index;

  public:
    Stack(Alloc* alloc)
    : alloc(alloc), block(nullptr), backup(nullptr), index(STACK_COUNT)
    {}

    ~Stack()
    {
      auto local_block = block;
      while (local_block)
      {
        auto prev = local_block->prev;
        alloc->template dealloc<sizeof(Block)>(local_block);
        local_block = prev;
      }
      if (backup != nullptr)
        alloc->template dealloc<sizeof(Block)>(backup);
    }

    ALWAYSINLINE bool empty()
    {
      return block == nullptr;
    }

    ALWAYSINLINE T peek()
    {
      assert(!empty());
      return block->data[index - 1];
    }

    ALWAYSINLINE T pop()
    {
      assert(!empty());

      index--;
      T item = block->data[index];

      if (index == 0)
        pop_slow_path();

      return item;
    }

    ALWAYSINLINE void push(T item)
    {
      if (index < STACK_COUNT)
      {
        block->data[index] = item;
        index++;
      }
      else
      {
        push_slow_path(item);
      }
    }

    template<void apply(T t)>
    void forall()
    {
      Block* curr = block;
      size_t i = index;

      while (curr != nullptr)
      {
        do
        {
          i--;
          apply(curr->data[i]);
        } while (i > 0);

        curr = curr->prev;
        i = STACK_COUNT;
      }
    }

  private:
    void pop_slow_path()
    {
      Block* prev = block->prev;

      if (backup != nullptr)
        alloc->template dealloc<sizeof(Block)>(backup);

      backup = block;
      block = prev;
      index = STACK_COUNT;
    }

    void push_slow_path(T item)
    {
      Block* next;

      if (backup != nullptr)
      {
        next = backup;
        backup = nullptr;
      }
      else
      {
        next = (Block*)alloc->template alloc<sizeof(Block)>();
      }

      index = 1;
      next->data[0] = item;
      next->prev = block;
      block = next;
    }
  };
} // namespace verona::rt
