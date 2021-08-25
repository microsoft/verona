// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::rt
{
  template<typename T>
  class PromiseR;

  template<typename T>
  class PromiseW;

  template<typename T>
  class Promise : public VCown<Promise<T>>
  {
  private:
    friend class PromiseR<T>;
    friend class PromiseW<T>;

    T val;

    void fulfill(T v)
    {
      val = v;
      Cown::acquire(this);
      VCown<Promise<T>>::schedule();
    }

    template<typename F, typename = std::enable_if_t<std::is_invocable_v<F, T>>>
    void then(F fn)
    {
      // FIXME: Am I using the right capture for fn?
      schedule_lambda<YesTransfer>(this, [=]() { fn(val); });
    }

    Promise()
    {
      VCown<Promise<T>>::wake();
    }

  public:
    static std::pair<std::unique_ptr<PromiseR<T>>, std::unique_ptr<PromiseW<T>>>
    create_promise()
    {
      // FIXME: Should this be 3 allocations?
      Promise<T>* p = new Promise<T>;
      std::unique_ptr<PromiseR<T>> r =
        std::unique_ptr<PromiseR<T>>(new PromiseR<T>(p));
      std::unique_ptr<PromiseW<T>> w =
        std::unique_ptr<PromiseW<T>>(new PromiseW<T>(p));
      return std::make_pair(std::move(r), std::move(w));
    }
  };

  template<typename T>
  class PromiseR
  {
  private:
    Promise<T>* promise;

  public:
    PromiseR(Promise<T>* p) : promise(p) {}

    /** This function consumes the read endpoint so that only one entity can
     * wait on it
     */
    template<typename F, typename = std::enable_if_t<std::is_invocable_v<F, T>>>
    static void then(std::unique_ptr<PromiseR<T>> rp, F fn)
    {
      rp->promise->then(std::forward<F>(fn));
    }
  };

  template<typename T>
  class PromiseW
  {
  private:
    Promise<T>* promise;

  public:
    PromiseW(Promise<T>* p) : promise(p) {}

    /* fulfil consumes the write end point so that the promise
     * so that the promise will be fulfilled only once
     */
    static void fulfill(std::unique_ptr<PromiseW<T>> w, T v)
    {
      w->promise->fulfill(v);
    }
  };
}
