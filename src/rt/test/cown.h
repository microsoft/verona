// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <functional>
#include <tuple>
#include <utility>
#include <verona.h>

class cown_ptr_trait
{
  private:
    cown_ptr_trait() {}

  template<typename T>
  friend class cown_ptr;
};

/**
 * Smart pointer to represent shared access to a cown.
 * Can only be used asychronously with `when` to get
 * underlying access.
 */
template<typename T>
class cown_ptr : cown_ptr_trait
{
private:
  /**
   * Internal Verona runtime cown for this type.
   */
  class ActualCown : public VCown<ActualCown>
  {
  private:
    T value;

  public:
    ActualCown(T&& t) : value(std::forward<T>(t))
    {}

    template<typename TT>
    friend class acquired_cown;
  };

  /**
   * Internal Verona runtime cown for this type.
   */
  ActualCown* allocated_cown = nullptr;

  /**
   * Accesses the internal Verona runtime cown for this handle.
   */
  Cown* underlying_cown()
  {
    return allocated_cown;
  }

  /**
   * Construct a new cown ptr object, actually allocates a runtime cown.
   *
   * This is internal, and the `make_cown` below is the public interface,
   * which has better behaviour for implicit template arguments.
   */
  cown_ptr(T&& t) : allocated_cown(new ActualCown(std::forward<T>(t))) {}

public:
  /**
   * Copy an existing cown ptr.  Shares the underlying cown.
   */
  cown_ptr(const cown_ptr& other)
  {
    allocated_cown = other.allocated_cown;
    verona::rt::Cown::acquire(allocated_cown);
  }

  /**
   * Move an existing cown ptr.  Does not create a new cown,
   * and is more efficient than copying, as it does not need
   * to perform reference count operations.
   */
  cown_ptr(cown_ptr&& other)
  {
    allocated_cown = other.allocated_cown;
    other.allocated_cown = nullptr;
  }

  ~cown_ptr()
  {
    // Condition to handle moved cown ptrs.
    // And a defence against the destructor being called twice.
    if (allocated_cown != nullptr)
    {
      auto& alloc = verona::rt::ThreadAlloc::get();
      verona::rt::Cown::release(alloc, allocated_cown);
      allocated_cown = nullptr;
    }
  }

  // Required as acquired_cown has to reach inside.
  template<typename TT>
  friend class acquired_cown;

  template<typename TT>
  friend cown_ptr<TT> make_cown(TT&& t);

  template<typename...>
  friend class When;
};

/**
 * Used to construct a new cown_ptr to `t`.
 *
 * TODO: Need to improve the forwarding versus copy behaviour here.
 */
template<typename T>
cown_ptr<T> make_cown(T&& t)
{
  return cown_ptr<T>(std::forward<T>(t));
}

/**
 * Represents a cown that has been acquired in a `when` clause.
 *
 * Can only be constructed by a `when`.
 *
 * The acquired_cown should not be persisted beyond the lifetime of the `when`
 */
template<typename T>
class acquired_cown
{
  // Needed to build one from inside a `when`
  template<typename...>
  friend class When;

private:
  // Underlying cown that has been acquired.
  // TODO: Look to optimise away the reference count here, as the
  // runtime already holds one for the duration of the `when`.
  cown_ptr<T> origin_cown;

  acquired_cown(const cown_ptr<T>& origin) : origin_cown(origin) {}

public:
  cown_ptr<T> cown() const
  {
    return origin_cown;
  }

  T* operator->()
  {
    return &(origin_cown.allocated_cown->value);
  }

  T& operator*()
  {
    return origin_cown.allocated_cown->value;
  }

  operator T&()
  {
    return origin_cown.allocated_cown->value;
  }

  acquired_cown(acquired_cown&&) = delete;
  acquired_cown& operator=(acquired_cown&&) = delete;
  acquired_cown(const acquired_cown&) = delete;
  acquired_cown& operator=(const acquired_cown&) = delete;
};