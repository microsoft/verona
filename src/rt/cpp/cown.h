// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <functional>
#include <tuple>
#include <utility>
#include <verona.h>

/**
 * Used in static asserts to check passed values are cown_ptr types.
 *
 * This class is used so a single simple check can be used
 *   std::is_base_of<cown_ptr_base, T>
 * To check T is a cown_ptr.
 */
class cown_ptr_base
{
private:
  cown_ptr_base() {}

  /**
   * Only cown_ptr can construct one of these,
   * so anything that has this base class will be a cown_ptr.
   */
  template<typename T>
  friend class cown_ptr;
};

/**
 * Smart pointer to represent shared access to a cown.
 * Can only be used asychronously with `when` to get
 * underlying access.
 *
 * Note using lower case name to match C++ std library
 * as this is one of the exposed types.
 */
template<typename T>
class cown_ptr : cown_ptr_base
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
    template<typename... Args>
    ActualCown(Args&&... ts) : value(std::forward<Args>(ts)...)
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
  cown_ptr(ActualCown* cown) : allocated_cown(cown) {}

public:
  /**
   * Copy an existing cown ptr.  Shares the underlying cown.
   */
  cown_ptr(const cown_ptr& other)
  {
    allocated_cown = other.allocated_cown;
    verona::rt::Cown::acquire(allocated_cown);
  }

  cown_ptr& operator=(cown_ptr&& other)
  {
    allocated_cown = other.allocated_cown;
    verona::rt::Cown::acquire(allocated_cown);
    return *this;
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
    if (allocated_cown != nullptr)
    {
      auto& alloc = verona::rt::ThreadAlloc::get();
      verona::rt::Cown::release(alloc, allocated_cown);
    }
  }

  // Required as acquired_cown has to reach inside.
  // Note only requires friend when implicit typename is T
  // but C++ doesn't like this.
  template<typename>
  friend class acquired_cown;

  // Note only requires friend when TT is T
  // but C++ doesn't like this.
  template<typename TT, typename... Args>
  friend cown_ptr<TT> make_cown(Args&&...);

  template<typename...>
  friend class When;
};

/**
 * Used to construct a new cown_ptr.
 *
 * Forwards arguments to construct the underlying data contained in the cown.
 */
template<typename T, typename... Args>
cown_ptr<T> make_cown(Args&&... ts)
{
  return cown_ptr<T>(
    new typename cown_ptr<T>::ActualCown(std::forward<Args>(ts)...));
}

/**
 * Represents a cown that has been acquired in a `when` clause.
 *
 * Can only be constructed by a `when`.
 *
 * The acquired_cown should not be persisted beyond the lifetime of the `when`
 *
 * Note using lower case name to match C++ std library
 * as this is one of the exposed types.
 */
template<typename T>
class acquired_cown
{
  /// Needed to build one from inside a `when`
  template<typename...>
  friend class When;

private:
  /// Underlying cown that has been acquired.
  /// Runtime is actually holding this reference count.
  cown_ptr<T> origin_cown;

  /// Constructor is private, as only `When` can construct one.
  /// TODO: Consider if we can reduce the reference count here, as the
  /// runtime is holding references too.
  acquired_cown(const cown_ptr<T>& origin) : origin_cown(origin) {}

public:
  /// Get a handle on the underlying cown.
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

  /**
   * Deleted to prevent accidental copying or
   * moving.  The lifetime is tied to the `when`,
   * so the cown should not be put somewhere else.
   * @{
   */
  acquired_cown(acquired_cown&&) = delete;
  acquired_cown& operator=(acquired_cown&&) = delete;
  acquired_cown(const acquired_cown&) = delete;
  acquired_cown& operator=(const acquired_cown&) = delete;
  /// @}
};