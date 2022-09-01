// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "cown.h"
#include "promise.h"
#include "when.h"

#include <atomic>
#include <cstdint>

namespace verona::cpp
{
  /**
   * An approach to limiting the work in the system.
   *
   * It provides a bounded set of Tokens, that can be requested
   * from a Token::Source.  If the there are no tokens available,
   * then the lambda is delayed, until a Token has been destroyed.
   *
   * Tokens must be treated linearly, once a token is no longer required
   * its destructor runs, and that enables the source to provide another token.
   */
  class Token
  {
    /**
     * Internal function that provides a cown that can be waited on.
     * This is like a create promise, just the minimal thing required for this
     * example.
     */
    static cown_ptr<size_t> make_wait()
    {
      auto a = verona::cpp::make_cown<size_t>((size_t)0);
      a.underlying_cown()->wake();
      return a;
    }

    /**
     * Used to signal that any messages waiting on the cown can be executed.
     *
     * This is like fulfill on a promise.
     */
    static void signal_wait(cown_ptr<size_t> w)
    {
      Cown::acquire(w.underlying_cown());
      w.underlying_cown()->schedule(false);
    }

    /**
     * Represents the persistent state of a Token source. This is the long lived
     * internal state, for which both Token and Token::Source are smart pointers
     * for accessing it.
     */
    struct State
    {
      /**
       * The combined count of Tokens and Sources.
       * There may be at most one Source, and there may be many tokens.
       */
      std::atomic<size_t> inflight{1};

      /**
       * The maximum value we allow inflight to reach.
       */
      size_t max_inflight;

      /**
       * A cown that is used to wait for Tokens to returned.
       */
      cown_ptr<size_t> wait{};

      /**
       * Set the up wait objects
       * Used at construction, and once a wait has occured to create a new one.
       */
      void setup_wait()
      {
        // Smart pointer assignment means the original wait is deallocated if
        // required.
        wait = make_wait();
      }

      State(size_t max_inflight) : max_inflight(max_inflight + 1)
      {
        setup_wait();
      }
    };

  public:
    /**
     * This represents a bounded Source of tokens.
     */
    class Source
    {
      // Reference to the underlying state.
      State* state;

      // Local cache of tokens.  These tokens can be used without
      // synchronising with the shared state.
      size_t local_tokens = 0;

      Source(State* state) : state(state) {}

      /**
       * Clears the reference to the state.
       */
      void clear()
      {
        if (state)
        {
          // Return any locally held tokens.
          state->inflight -= local_tokens;
          local_tokens = 0;

          // Return the sources count.
          auto old = state->inflight--;

          // If this is the last reference to the state,
          // then it is responsible for deallocating it.
          if (old == 1)
          {
            delete (state);
          }
          state = nullptr;
        }
      }

    public:
      /**
       * Returns the number of tokens that be be successfully acquired
       * by calling `get_token`.
       *
       * This will synchronise with the global state, and locally caches
       * the tokens to reduce the number of synchronising operations.
       */
      size_t available_tokens()
      {
        if (state->inflight.load() == state->max_inflight)
        {
          return local_tokens;
        }
        auto available = state->max_inflight - state->inflight.load() - 1;

        state->inflight += available;

        local_tokens += available;

        return local_tokens;
      }

      /**
       * Only call previous call to available_tokens() has returned 0;
       *
       * This will call f once there is at least one available token.
       * f may be called synchronously or asynchronously.
       */
      template<typename F>
      void wait_for_token(F f) &&
      {
        assert(local_tokens == 0);

        auto old = state->inflight++;

        // We have already performed an increment, so the local_tokens
        // needs increasing to reflect this.  The token may or mat not be
        // available yet, but will definitely be by the time we call f.
        local_tokens++;

        if (old + 1 == state->max_inflight)
        {
          // There are no more Tokens available, so when on wait
          // to be notified when more are available.
          when(this->state->wait)
            << [that = std::move(*this), f = std::move(f)](auto) mutable {
                 // Create a new thing to wait on.
                 that.state->setup_wait();
                 // We are guaranteed that a token is available, so proceed.
                 f(std::move(that));
               };
        }
        else
        {
          // Token was already available so proceed.
          f(std::move(*this));
        }
      }

      /**
       * Gets a token, requires that there is at least one available token.
       * This should be ensured by calling `available_tokens` first.
       */
      Token get_token()
      {
        assert(local_tokens > 0);
        local_tokens--;
        return {state};
      }

      /**
       * Remove copy operators as we need to treat the source linearly.
       * @{
       */
      Source(const Source&) = delete;
      const Source& operator=(const Source&) = delete;
      /**
       * @}
       */

      /**
       * Move constructor
       */
      Source(Source&& b)
      {
        state = b.state;
        local_tokens = b.local_tokens;
        b.state = nullptr;
        b.local_tokens = 0;
      }

      /**
       * Move assignment operator
       */
      Source& operator=(Source&& b)
      {
        clear();
        state = b.state;
        local_tokens = b.local_tokens;
        b.state = nullptr;
        b.local_tokens = 0;
        return *this;
      }

      /**
       * @brief Creates a new Token::Source
       *
       * @param max_inflight the maximum number of tokens that can be provided.
       * @return Source the new Source of Tokens that must be handled linearly.
       */
      static Source create(size_t max_inflight)
      {
        auto state = new State{max_inflight};
        return Source(state);
      }

      ~Source()
      {
        clear();
      }
    };

  private:
    State* src;

    Token(State* src) : src(src) {}

    void clear()
    {
      if (src)
      {
        auto old = src->inflight--;
        if (old == src->max_inflight)
        {
          signal_wait(src->wait);
        }
        else if (old == 1)
        {
          delete src;
        }
        src = nullptr;
      }
    }

  public:
    Token() : src(nullptr) {}

    ~Token()
    {
      clear();
    }

    /**
     * Remove copy operators as we need to treat the Tokens linearly.
     * @{
     */
    Token(const Token&) = delete;
    const Token& operator=(const Token&) = delete;
    /**
     * @}
     */

    Token(Token&& b)
    {
      src = b.src;
      b.src = nullptr;
    }

    Token& operator=(Token&& b)
    {
      clear();
      src = b.src;
      b.src = nullptr;
      return *this;
    }
  };
} // namespace verona::rt