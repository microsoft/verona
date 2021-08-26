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

      PromiseR(Promise* p) : promise(p) {}

    public:
      /** This function consumes the read endpoint so that only one entity can
       * wait on it
       */
      template<
        typename F,
        typename = std::enable_if_t<std::is_invocable_v<F, T>>>
      static void then(std::unique_ptr<PromiseR> rp, F fn)
      {
        rp->promise->then(std::forward<F>(fn));
      }
    };

    class PromiseW
    {
    private:
      friend class Promise;

      Promise* promise;

      PromiseW(Promise* p) : promise(p) {}

    public:
      /* fulfil consumes the write end point so that the promise
       * so that the promise will be fulfilled only once
       */
      static void fulfill(std::unique_ptr<PromiseW> w, T v)
      {
        w->promise->fulfill(v);
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
      schedule_lambda<YesTransfer>(
        this, [fn = std::move(fn), this] { fn(val); });
    }

    Promise()
    {
      VCown<Promise<T>>::wake();
    }

  public:
    static std::pair<std::unique_ptr<PromiseR>, std::unique_ptr<PromiseW>>
    create_promise()
    {
      // FIXME: Should this be 3 allocations?
      Promise* p = new Promise<T>;
      std::unique_ptr<PromiseR> r = std::unique_ptr<PromiseR>(new PromiseR(p));
      std::unique_ptr<PromiseW> w = std::unique_ptr<PromiseW>(new PromiseW(p));
      return std::make_pair(std::move(r), std::move(w));
    }
  };
}
