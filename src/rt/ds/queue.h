// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
namespace verona
{
  template<class T>
  class Queue
  {
  private:
    T* first = nullptr;
    T* last = nullptr;

  public:
    bool is_empty()
    {
      return first == nullptr;
    }

    void enqueue(T* next)
    {
      assert(next != nullptr);
      assert((first != nullptr && last != nullptr) || last == first);

      if (is_empty())
      {
        last = next;
        first = next;
      }
      else
      {
        last->next = next;
        last = next;
      }

      assert((first != nullptr && last != nullptr) || last == first);
    }

    T* dequeue()
    {
      assert(first != nullptr);
      assert((first != nullptr && last != nullptr) || last == first);

      auto result = first;
      if (first == last)
      {
        first = nullptr;
        last = nullptr;
      }
      else
      {
        first = first->next;
      }

      assert((first != nullptr && last != nullptr) || last == first);
      return result;
    }

    size_t length()
    {
      T* p = first;

      if (p == nullptr)
        return 0;

      size_t len = 1;

      while (p != last)
      {
        len++;
        p = p->next;
      }

      return len;
    }
  };
} // namespace verona::rt
