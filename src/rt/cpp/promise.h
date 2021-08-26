// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::rt
{
  template<typename T>
  class Promise : public VCown<Promise<T>>
  {
  public:
    class PromiseR
    {
    private:
      friend class Promise;

      Promise* promise;

      PromiseR(Promise* p) : promise(p)
      {
        Cown::acquire(p);
      }

      PromiseR& operator=(PromiseR&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
        return *this;
      }
      PromiseR& operator=(const PromiseR&) = delete;

    public:
      template<
        typename F,
        typename = std::enable_if_t<std::is_invocable_v<F, T>>>
      void then(F&& fn)
      {
        promise->then(std::forward<F>(fn));
      }

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
    };

    class PromiseW
    {
    private:
      friend class Promise;

      Promise* promise;

      PromiseW(Promise* p) : promise(p)
      {
        Cown::acquire(p);
      }

      PromiseW(const PromiseW&) = delete;

      PromiseW& operator=(PromiseW&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
        return *this;
      }
      PromiseW& operator=(const PromiseW&) = delete;

    public:
      PromiseW(PromiseW&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
      }

      ~PromiseW()
      {
        if (promise)
          Cown::release(ThreadAlloc::get(), promise);
      }
    };

  private:
    T val;

    template<typename F, typename = std::enable_if_t<std::is_invocable_v<F, T>>>
    void then(F&& fn)
    {
      schedule_lambda(this, [fn = std::move(fn), this] { fn(val); });
    }

    /**
     * Create an empty cown and call wake() on it. This will move the cown's
     * queue from the SLEEPING state to WAKE and prevent subsequent messages
     * from putting the cown on a scheduler thread queue. This cown can only
     * be scheduled through an explicit call to schedule(). schedule() is
     * called when the promise is fulfilled
     */
    Promise()
    {
      VCown<Promise<T>>::wake();
    }

  public:
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
    static void fulfill(PromiseW&& wp, T v)
    {
      PromiseW tmp = std::move(wp);
      tmp.promise->val = v;
      Cown::acquire(tmp.promise);
      tmp.promise->schedule();
    }
  };
}
