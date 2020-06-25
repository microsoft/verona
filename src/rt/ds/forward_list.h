// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <utility>

namespace verona::rt
{
  /**
   * Simple implementation of a subset of `std::forward_list`.
   * The standard Windows version of `std::forward_list` uses more than one
   * pointer's worth of state in the class and so is not useable in places
   * where you care about space but want to be able to chain a linked list.
   */
  template<typename T, int Size = 1>
  class forward_list
  {
    /**
     * One node in the linked list.
     */
    struct Node
    {
      /**
       * Pointer to the next element.
       */
      Node* next;
      /**
       * The value in this list.
       */
      T value;
      /**
       * Construct a node by copying the value.
       */
      Node(Node* n, const T& v) : next(n), value(v) {}
      /**
       * Construct a node by moving the value.
       */
      Node(Node* n, T&& v) : next(n), value(std::move(v)) {}
    };
    /**
     * List iterator.  Wrapper around a pointer to a `Node`.
     */
    class iterator
    {
      /**
       * Only the enclosing class is allowed to create iterators.
       */
      friend class forward_list;
      /**
       * The current node.  This will be `nullptr` for the end iterator.
       */
      Node* val;
      /**
       * Constructor.  Private to guarantee that only the `forward_list` class
       * can create iterators.
       */
      iterator(Node* v) : val(v) {}

    public:
      /**
       * Equality comparison, compares nodes by address.
       */
      bool operator==(const iterator& other)
      {
        return val == other.val;
      }
      /**
       * Inequality comparison, compares nodes by address.
       */
      bool operator!=(const iterator& other)
      {
        return val != other.val;
      }
      /**
       * Preincrement operator.  Moves to the next entry in the list.
       */
      iterator& operator++()
      {
        val = val->next;
        return *this;
      }
      /**
       * Dereference operator.  Returns the current entry.  Undefined if this
       * is not a valid node pointer.
       */
      T& operator*()
      {
        return val->value;
      }
    };
    /**
     * Pointer to the head of the list.  This is the only state stored inside
     * of an instance of this class.
     */
    Node* head = nullptr;

  public:
    /**
     * Returns true if this list is empty, false otherwise.
     */
    bool empty()
    {
      return head == nullptr;
    }
    /**
     * Returns an iterator to the start of the list.
     */
    iterator begin()
    {
      return iterator(head);
    }
    /**
     * Returns a forward iterator to the end of the list.  This can be used for
     * comparison, but does not allow reaching the end of the list.
     */
    iterator end()
    {
      return iterator(nullptr);
    }
    /**
     * Insert a new element at the start of the list, by copy.
     */
    void push_front(const T& val)
    {
      Node* n = new Node(head, val);
      head = n;
    }
    /**
     * Insert a new element the start of the list, by move.
     */
    void push_front(T&& val)
    {
      Node* n = new Node(head, std::move(val));
      head = n;
    }
    /**
     * Destructor.  Deletes the list.
     */
    ~forward_list()
    {
      while (head)
      {
        Node* tmp = head;
        head = head->next;
        delete tmp;
      }
    }
  };
} // namespace verona::rt
