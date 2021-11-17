// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <variant>

namespace verona::rt
{
  /*
   * This class defines a Promise object on top of the verona runtime.
   * A promise is a cown whose lifetime is controlled by the read and write
   * end point (PromiseR and PromiseW). There can be a single writer and
   * multiple readers.
   *
   * The cown has a local value (val) which is the value returned by the
   * promise. To read this value, one should call then() on the read end-point
   * and pass a lambda that takes as an argument the promise value.
   *
   * From a Verona runtime point of view, creating a promise corresponds to
   * creating a cown with an empty queue, whose state is not SLEEPING, though,
   * but it is AWAKE instead. This prevents putting the cown on a scheduler
   * thread when sending messages to it. Instead, the cown is only put on a
   * scheduler thread once the cown is fulfilled.
   */
  template<typename T>
  class Promise : public VCown<Promise<T>>
  {
  public:
    class PromiseErr
    {
      friend class Promise;

      int err_code;
      PromiseErr(int code) : err_code(code) {}
    };

    /**
     * The promise read end-point
     * This is a smart pointer that points to the promise cown
     */
    class PromiseR
    {
    private:
      friend class Promise;

      Promise* promise;

      PromiseR& operator=(const PromiseR&) = delete;

    public:
      template<
        typename F,
        typename =
          std::enable_if_t<std::is_invocable_v<F, std::variant<T, PromiseErr>>>>
      void then(F&& fn)
      {
        promise->then(std::forward<F>(fn));
      }

      PromiseR() : promise(nullptr) {}

      PromiseR(Promise* p, TransferOwnership transfer = NoTransfer) : promise(p)
      {
        if (transfer == NoTransfer)
          Cown::acquire(p);
      }

      /**
       * A promise can have multiple readers
       */
      PromiseR(const PromiseR& other)
      {
        promise = other.promise;
        Cown::acquire(promise);
      }

      PromiseR(PromiseR&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
      }

      ~PromiseR()
      {
        if (promise)
          Cown::release(ThreadAlloc::get(), promise);
      }

      PromiseR& operator=(PromiseR&& old)
      {
        if (promise)
          Cown::release(ThreadAlloc::get(), promise);
        promise = old.promise;
        old.promise = nullptr;
        return *this;
      }

      template<TransferOwnership transfer = NoTransfer>
      Promise* get_promise()
      {
        Promise* ret = promise;

        if constexpr (transfer == NoTransfer)
        {
          if (promise)
            Cown::acquire(promise);
        }
        else
        {
          promise = nullptr;
        }
        return ret;
      }
    };

    /**
     * The write end-point of the promise
     * This is a smart pointer that points to the promise cown
     */
    class PromiseW
    {
    private:
      friend class Promise;

      Promise* promise;

      PromiseW(Promise* p) : promise(p)
      {
        Cown::acquire(p);
      }

      /*
       * A promise can have a single writer that can fulfill the promise
       * only once.
       */
      PromiseW(const PromiseW&) = delete;

      PromiseW& operator=(const PromiseW&) = delete;

    public:
      PromiseW() : promise(nullptr) {}

      PromiseW(PromiseW&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
      }

      PromiseW& operator=(PromiseW&& old)
      {
        if (promise)
          Cown::release(ThreadAlloc::get(), promise);
        promise = old.promise;
        old.promise = nullptr;
        return *this;
      }

      ~PromiseW()
      {
        if (promise)
        {
          if (!promise->fulfilled)
            promise->schedule();
          else
            Cown::release(ThreadAlloc::get(), promise);
        }
      }
    };

  private:
    T val;
    bool fulfilled;

    template<
      typename F,
      typename =
        std::enable_if_t<std::is_invocable_v<F, std::variant<T, PromiseErr>>>>
    void then(F&& fn)
    {
      schedule_lambda(this, [fn = std::move(fn), this] {
        if (fulfilled)
        {
          if constexpr (std::is_trivially_copy_constructible<T>::value)
          {
            fn(val);
          }
          else
          {
            fn(std::move(val));
          }
        }
        else
          fn(PromiseErr(-1));
      });
    }

    /**
     * Create an empty cown and call wake() on it. This will move the cown's
     * queue from the SLEEPING state to WAKE and prevent subsequent messages
     * from putting the cown on a scheduler thread queue. This cown can only
     * be scheduled through an explicit call to schedule(). schedule() is
     * called when the promise is fulfilled.
     */
    Promise()
    {
      VCown<Promise<T>>::wake();
      fulfilled = false;
    }

  public:
    /**
     * Create a promise and get its read and write end-points
     */
    static std::pair<PromiseR, PromiseW> create_promise()
    {
      Promise* p = new Promise<T>;
      PromiseR r(p);
      PromiseW w(p);
      Cown::release(ThreadAlloc::get(), p);

      return std::make_pair(std::move(r), std::move(w));
    }

    /**
     * Fulfill the promise with a value and put the promise cown in a
     * scheduler thread queue. A PromiseW can be fulfilled only once.
     */
    static void fulfill(PromiseW&& wp, T&& v)
    {
      PromiseW tmp = std::move(wp);
      tmp.promise->val = std::move(v);
      tmp.promise->fulfilled = true;
      Cown::acquire(tmp.promise);
      tmp.promise->schedule();
    }
  };
}
