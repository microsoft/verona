// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/mpscq.h"
#include "../object/object.h"
#include "behaviour.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class MultiMessage
  {
  public:
    /**
     * This represents a message that is sent to a behaviour.
     *
     * The layout requires that after the allocation there is space for
     * `count` cown pointers, and then the behaviour's body.
     *
     * This layout allows the message body to be single allocation even though
     * there are multiple different sized pieces.
     */
    struct Body
    {
      std::atomic<size_t> exec_count_down;
      size_t count;

    private:
      Body(size_t count) : exec_count_down(count), count(count) {}

    public:
      /**
       * TODO When we move to C++20, convert to returning a span.
       */
      Request* get_requests_array()
      {
        return snmalloc::pointer_offset<Request>(this, sizeof(Body));
      }

      Behaviour& get_behaviour()
      {
        return *snmalloc::pointer_offset<Behaviour>(
          this, sizeof(Body) + sizeof(Request) * count);
      }

      /**
       * Remove one from the exec_count_down.
       *
       * Returns true if this call makes the count_down_zero
       */
      bool count_down()
      {
        // Note that we don't actually perform the last decrement as it is not
        // required.
        return (exec_count_down.load(std::memory_order_acquire) == 1) ||
          (exec_count_down.fetch_sub(1) == 1);
      }

      /**
       * Allocates a message body with sufficient space for the
       * cowns_array and the behaviour.  This does not initialise the cowns
       * array.
       */
      template<typename Be, typename... Args>
      static Body* make(Alloc& alloc, size_t count, Args... args)
      {
        size_t size = sizeof(Body) + (sizeof(Request) * count) + sizeof(Be);

        // Create behaviour
        auto body = new (alloc.alloc(size)) Body(count);
        new ((Be*)&(body->get_behaviour())) Be(std::forward<Args>(args)...);

        static_assert(
          alignof(Be) <= sizeof(void*), "Alignment not supported, yet!");
        static_assert(
          sizeof(Body) % alignof(Be) == 0, "Alignment not supported, yet!");

        return body;
      }
    };

    class Delivered
    {
      // If is_last is true, then this is an owning reference.
      Body& body;
      bool is_read;
      bool is_last;
      Alloc& alloc;

    public:
      Body& get_body()
      {
        return body;
      }

      bool is_last_reference()
      {
        return is_last;
      }

      bool is_read_request()
      {
        return is_read;
      }

      ~Delivered()
      {
        if (is_last)
        {
          alloc.dealloc(&body);
        }
      }

      Delivered(const Delivered& other) = delete;
      Delivered(Delivered&& other) = delete;
    };

  private:
    friend verona::rt::MPSCQ<MultiMessage>;

    // The body of the actual message.
    // uses the bottom bit to determine if the request is a read.
    uintptr_t body_and_mode;

    std::atomic<MultiMessage*> next{nullptr};

  public:
    static MultiMessage* make(Alloc& alloc, Body* body, bool is_read)
    {
      auto msg = (MultiMessage*)alloc.alloc<sizeof(MultiMessage)>();
      msg->body_and_mode = ((uintptr_t)body) | (is_read ? 1 : 0);
      Logging::cout() << "MultiMessage " << msg << " payload " << body
                      << Logging::endl;
      return msg;
    }

    inline size_t size()
    {
      return sizeof(MultiMessage);
    }

    inline bool is_read()
    {
      return (body_and_mode & 1) == 1;
    }

    /**
     * Processes the message, if it is the last message then
     * it takes ownership of the body.
     */
    Delivered deliver(Alloc& alloc)
    {
      auto body = (Body*)(body_and_mode & ~Object::MARK_MASK);
      auto is_read_ = is_read();
      body_and_mode = 0;
      auto is_last = body->count_down();
      return {*(body), is_read_, is_last, alloc};
    }

    bool next_is_null()
    {
      return next.load(std::memory_order_acquire) == nullptr;
    }
  };
} // namespace verona::rt
