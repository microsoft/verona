// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <tuple>
#include <utility>
#include <verona.h>

namespace verona::cpp
{
  using namespace verona::rt;

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

  template<typename T>
  class cown_ptr;

  /**
   * Internal Verona runtime cown for the type T.
   *
   * This class is used to prevent access to the representation except
   * through the correct usage of cown_ptr and when.
   */
  template<typename T>
  class ActualCown : public VCown<ActualCown<T>>
  {
  private:
    T value;

    template<typename... Args>
    ActualCown(Args&&... ts) : value(std::forward<Args>(ts)...)
    {}

    template<typename TT>
    friend class acquired_cown;

    template<typename TT>
    friend class cown_ptr;

    template<typename TT, typename... Args>
    friend cown_ptr<TT> make_cown(Args&&... ts);
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
  public:
    class weak
    {
      friend cown_ptr;

      /**
       * Internal Verona runtime cown for this type.
       */
      ActualCown<T>* allocated_cown{nullptr};

      weak(ActualCown<T>* c) : allocated_cown(c) {}

    public:
      /**
       * Sets the cown_ptr::weak to nullptr, and decrements the reference count
       * if it was not already nullptr.
       */
      void clear()
      {
        // Condition to handle moved weak cown ptrs.
        if (allocated_cown != nullptr)
        {
          auto& alloc = verona::rt::ThreadAlloc::get();
          allocated_cown->weak_release(alloc);
          allocated_cown = nullptr;
        }
      }

      constexpr weak() = default;

      /**
       * Copy an existing weak cown ptr.  Shares the underlying cown.
       */
      weak(const weak& other)
      {
        allocated_cown = other.allocated_cown;
        if (allocated_cown != nullptr)
          allocated_cown->weak_acquire();
      }

      /**
       * Copy an existing cown ptr to a weak ptr.  Shares the underlying cown.
       */
      weak(const cown_ptr& other)
      {
        allocated_cown = other.allocated_cown;
        if (allocated_cown != nullptr)
          allocated_cown->weak_acquire();
      }

      /**
       * Copy an existing weak cown ptr.  Shares the underlying cown.
       */
      weak& operator=(const weak& other)
      {
        clear();
        allocated_cown = other.allocated_cown;
        if (allocated_cown != nullptr)
          allocated_cown->weak_acquire();
        return *this;
      }

      /**
       * Nullptr assignment for a weak cown.
       */
      weak& operator=(std::nullptr_t)
      {
        clear();
        return *this;
      }

      /**
       * Move an existing weak cown ptr.  Does not create a new cown,
       * and is more efficient than copying, as it does not need
       * to perform reference count operations.
       */
      weak(weak&& other)
      {
        allocated_cown = other.allocated_cown;
        other.allocated_cown = nullptr;
      }

      /**
       * Move an existing weak cown ptr.  Does not create a new cown,
       * and is more efficient than copying, as it does not need
       * to perform reference count operations.
       */
      weak& operator=(weak&& other)
      {
        clear();
        allocated_cown = other.allocated_cown;
        other.allocated_cown = nullptr;
        return *this;
      }

      operator bool() const
      {
        return allocated_cown != nullptr;
      }

      cown_ptr promote()
      {
        if (
          (allocated_cown != nullptr) &&
          allocated_cown->acquire_strong_from_weak())
        {
          return {allocated_cown};
        }

        return nullptr;
      }

      ~weak()
      {
        clear();
      }
    };

  private:
    template<typename TT>
    friend class Access;

    /**
     * Internal Verona runtime cown for this type.
     */
    ActualCown<T>* allocated_cown{nullptr};

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
    cown_ptr(ActualCown<T>* cown) : allocated_cown(cown) {}

  public:
    constexpr cown_ptr() = default;

    /**
     * Copy an existing cown ptr.  Shares the underlying cown.
     */
    cown_ptr(const cown_ptr& other)
    {
      allocated_cown = other.allocated_cown;
      if (allocated_cown != nullptr)
        verona::rt::Cown::acquire(allocated_cown);
    }

    /**
     * Copy an existing cown ptr.  Shares the underlying cown.
     */
    cown_ptr& operator=(const cown_ptr& other)
    {
      clear();
      allocated_cown = other.allocated_cown;
      if (allocated_cown != nullptr)
        verona::rt::Cown::acquire(allocated_cown);
      return *this;
    }

    /**
     * Nullptr assignment for a cown.
     */
    cown_ptr& operator=(std::nullptr_t)
    {
      clear();
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

    /**
     * Move an existing cown ptr.  Does not create a new cown,
     * and is more efficient than copying, as it does not need
     * to perform reference count operations.
     */
    cown_ptr& operator=(cown_ptr&& other)
    {
      clear();
      allocated_cown = other.allocated_cown;
      other.allocated_cown = nullptr;
      return *this;
    }

    operator bool() const
    {
      return allocated_cown != nullptr;
    }

    bool operator==(const cown_ptr& other)
    {
      return allocated_cown == other.allocated_cown;
    }

    bool operator!=(const cown_ptr& other)
    {
      return !((*this) == other);
    }

    bool operator==(std::nullptr_t)
    {
      return allocated_cown == nullptr;
    }

    /**
     * Sets the cown_ptr to nullptr, and decrements the reference count
     * if it was not already nullptr.
     */
    void clear()
    {
      // Condition to handle moved cown ptrs.
      if (allocated_cown != nullptr)
      {
        auto& alloc = verona::rt::ThreadAlloc::get();
        verona::rt::Cown::release(alloc, allocated_cown);
        allocated_cown = nullptr;
      }
    }

    weak get_weak()
    {
      if (allocated_cown != nullptr)
        allocated_cown->weak_acquire();
      return {allocated_cown};
    }

    ~cown_ptr()
    {
      clear();
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

  /* A cown_ptr<const T> is used to mark that the cown is being accessed as
   * read-only. (This combines the type as the capability. We do not have deep
   * immutability in C++, so acquired_cown<const T> is an approximation.)
   *
   * We use inheritance to allow us to construct a cown_ptr<const T> from a
   * cown_ptr<T>.
   */
  template<typename T>
  class cown_ptr<const T> : public cown_ptr<T>
  {
  public:
    cown_ptr(const cown_ptr<T>& other) : cown_ptr<T>(other){};
  };

  template<typename T>
  cown_ptr<const T> read(cown_ptr<T> cown)
  {
    return cown;
  }

  /**
   * Used to construct a new cown_ptr.
   *
   * Forwards arguments to construct the underlying data contained in the cown.
   */
  template<typename T, typename... Args>
  cown_ptr<T> make_cown(Args&&... ts)
  {
    static_assert(
      !std::is_const_v<T>,
      "Cannot make a cown of const type as this conflicts with read acquire "
      "encoding trick. If we hit this assertion, raise an issue explaining the "
      "use case.");
    return cown_ptr<T>(new ActualCown<T>(std::forward<Args>(ts)...));
  }

  template<typename T>
  bool operator==(std::nullptr_t, const cown_ptr<T>& rhs)
  {
    return rhs == nullptr;
  }

  template<typename T>
  bool operator!=(const cown_ptr<T>& lhs, std::nullptr_t)
  {
    return !(lhs == nullptr);
  }

  template<typename T>
  bool operator!=(std::nullptr_t, const cown_ptr<T>& rhs)
  {
    return !(rhs == nullptr);
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
    ActualCown<std::remove_const_t<T>>& origin_cown;

    /// Constructor is private, as only `When` can construct one.
    acquired_cown(ActualCown<std::remove_const_t<T>>& origin)
    : origin_cown(origin)
    {}

  public:
    /// Get a handle on the underlying cown.
    cown_ptr<std::remove_const_t<T>> cown() const
    {
      verona::rt::Cown::acquire(&origin_cown);
      return cown_ptr<T>(&origin_cown);
    }

    T& get_ref() const
    {
      if constexpr (std::is_const<T>())
        return const_cast<T&>(origin_cown.value);
      else
        return origin_cown.value;
    }

    T& operator*()
    {
      return get_ref();
    }

    T* operator->()
    {
      return &get_ref();
    }

    operator T&()
    {
      return get_ref();
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
} // namespace verona::rt