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

      PromiseR(const PromiseR&) = delete;

      PromiseR& operator=(PromiseR&& old)
      {
        promise = old.promise;
        old.promise = nullptr;
        return *this;
      }
      PromiseR& operator=(const PromiseR&) = delete;

    public:
      /** This function consumes the read endpoint so that only one entity can
       * wait on it
       */
      template<
        typename F,
        typename = std::enable_if_t<std::is_invocable_v<F, T>>>
      static void then(PromiseR&& rp, F&& fn)
      {
        PromiseR tmp = std::move(rp);
        tmp.promise->then(std::forward<F>(fn));
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
      /* fulfil consumes the write end point so that the promise
       * so that the promise will be fulfilled only once
       */
      static void fulfill(PromiseW&& wp, T v)
      {
        PromiseW tmp = std::move(wp);
        tmp.promise->fulfill(v);
      }

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

    void fulfill(T v)
    {
      val = v;
      Cown::acquire(this);
      VCown<Promise<T>>::schedule();
    }

    template<typename F, typename = std::enable_if_t<std::is_invocable_v<F, T>>>
    void then(F&& fn)
    {
      schedule_lambda(this, [fn = std::move(fn), this] { fn(val); });
    }

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
  };
}
