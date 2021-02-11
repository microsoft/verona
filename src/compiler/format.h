// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

namespace verona::compiler
{
  namespace format
  {
    namespace internal
    {
      template<typename T, typename = void>
      struct can_dereference : std::false_type
      {};

      template<typename T>
      struct can_dereference<
        T,
        std::void_t<decltype(*std::declval<const T&>())>> : std::true_type
      {};

      static_assert(can_dereference<const int*>::value);
      static_assert(can_dereference<const std::unique_ptr<int>&>::value);
      static_assert(can_dereference<const std::shared_ptr<int>&>::value);
      static_assert(can_dereference<const std::optional<int>&>::value);
      static_assert(!can_dereference<int>::value);
      static_assert(!can_dereference<const int&>::value);

      /**
       * Helper method which dereferences its argument if possible, or returns
       * it unchanged otherwise.
       */
      template<typename T>
      const auto& deref(const T& value)
      {
        if constexpr (can_dereference<T>::value)
          return *value;
        else
          return value;
      }

      template<typename T>
      using element_type = typename std::remove_reference_t<T>::value_type;

      template<typename T, typename = void>
      struct has_value_method : std::false_type
      {};

      template<typename T>
      struct has_value_method<
        T,
        std::void_t<decltype(std::declval<const T&>().has_value())>>
      : std::true_type
      {};

      /**
       * Check whether something has a "useful" value, using either a has_value
       * method or by converting to a boolean.
       */
      template<typename T>
      bool has_value(const T& p)
      {
        if constexpr (has_value_method<T>::value)
          return p.has_value();
        else
          return bool(p);
      }
    }

    /**
     * Printable object which displays a string prefix followed by another
     * object.
     */
    template<typename T>
    struct prefixed
    {
      explicit prefixed(std::string prefix, T&& value)
      : prefix_(prefix), value_(std::forward<T>(value))
      {}

      friend std::ostream& operator<<(std::ostream& out, const prefixed& self)
      {
        return out << self.prefix_ << internal::deref(self.value_);
      }

      bool has_value() const
      {
        return internal::has_value(value_);
      }

    private:
      std::string prefix_;
      T value_;
    };

    /**
     * Printable object which displays a value surrounded by a prefix and a
     * suffix.
     */
    template<typename T>
    struct surrounded
    {
      explicit surrounded(std::string prefix, std::string suffix, T&& value)
      : prefix_(prefix), suffix_(suffix), value_(std::forward<T>(value))
      {}

      friend std::ostream& operator<<(std::ostream& out, const surrounded& self)
      {
        return out << self.prefix_ << internal::deref(self.value_)
                   << self.suffix_;
      }

      bool has_value() const
      {
        return internal::has_value(value_);
      }

    private:
      std::string prefix_;
      std::string suffix_;
      T value_;
    };

    /**
     * Printable object which only displays its contents if the contents have
     * a value, as defined by their `has_value` method or by converting to a
     * boolean.
     *
     * The contents would usually be an std::optional, a pointer or a vector
     * inside of a separated_by printer. Combinators such a prefixed and
     * surrounded evaluate to whatever their contents evaluate, allowing
     * expressions such as `optional(prefixed(prefix, value))`. This either
     * prints nothing, or prefix followed by value.
     *
     *
     */
    template<typename T>
    struct optional
    {
      explicit optional(T&& value) : value_(std::forward<T>(value)) {}

      bool has_value() const
      {
        return internal::has_value(value_);
      }

      friend std::ostream& operator<<(std::ostream& out, const optional& self)
      {
        if (self.has_value())
          return out << internal::deref(self.value_);
        else
          return out;
      }

    private:
      T value_;
    };

    /**
     * Printable object which displays `value` if it is true when converted to
     * bool, or `fallback` otherwise.
     */
    template<typename T>
    struct defaulted
    {
      explicit defaulted(T&& value, std::string fallback)
      : value_(std::forward<T>(value)), fallback_(fallback)
      {}

      friend std::ostream& operator<<(std::ostream& out, const defaulted& self)
      {
        if (self.value_)
          return out << internal::deref(self.value_);
        else
          return out << self.fallback_;
      }

      bool has_value() const
      {
        return true;
      }

    private:
      T value_;
      std::string fallback_;
    };

    /**
     * Printable object which displays each element of a list, witht he given
     * separator.
     *
     * An optional function can be applied to each element before it gets
     * printed.
     */
    template<typename T, typename U = void>
    struct separated_by
    {
      using element_type = internal::element_type<T>;
      using getter_fn = std::function<U(const element_type&)>;

      explicit separated_by(T&& value, std::string separator)
      : value_(std::forward<T>(value)), separator_(separator), getter_(nullptr)
      {}

      explicit separated_by(T&& value, std::string separator, getter_fn getter)
      : value_(std::forward<T>(value)), separator_(separator), getter_(getter)
      {}

      bool has_value() const
      {
        return !value_.empty();
      }

      friend std::ostream&
      operator<<(std::ostream& out, const separated_by& self)
      {
        bool first = true;
        for (const auto& elem : self.value_)
        {
          if (first)
            first = false;
          else
            out << self.separator_;

          if constexpr (std::is_same_v<U, void>)
          {
            out << internal::deref(elem);
          }
          else
          {
            out << internal::deref((self.getter_)(elem));
          }
        }
        return out;
      }

    private:
      T value_;
      std::string separator_;
      getter_fn getter_;
    };

    /**
     * Displays each element in the list on a separate line.
     */
    template<typename T>
    struct lines
    {
      explicit lines(T&& value) : value_(std::forward<T>(value)) {}

      bool has_value() const
      {
        return !value_.empty();
      }

      friend std::ostream& operator<<(std::ostream& out, const lines& self)
      {
        bool first = true;
        for (const auto& elem : self.value_)
        {
          out << internal::deref(elem) << "\n";
        }
        return out;
      }

    private:
      T value_;
    };

    /**
     * Displays a value surrounded with parenthesis.
     */
    template<typename T>
    auto parens(T&& value)
    {
      return surrounded("(", ")", std::forward<T>(value));
    }

    /**
     * Displays a value surrounded with square brackets.
     */
    template<typename T>
    auto brackets(T&& value)
    {
      return surrounded("[", "]", std::forward<T>(value));
    }

    /**
     * Displays each element of a list, separated by commas.
     */
    template<typename T>
    auto comma_sep(T&& value)
    {
      return separated_by(std::forward<T>(value), ", ");
    }

    /**
     * Displays the result of applying `fn` to each element of a list, separated
     * by commas.
     */
    template<typename T, typename Fn>
    auto comma_sep(T&& value, Fn&& fn)
    {
      return separated_by(std::forward<T>(value), ", ", std::forward<Fn>(fn));
    }

    /**
     * Displays each element of a list, separated by commas, and surrounded with
     * square brackets.
     */
    template<typename T>
    auto list(T&& value)
    {
      return brackets(comma_sep(std::forward<T>(value)));
    }

    /**
     * Displays the result of applying `fn` to each element of a list, separated
     * by commas, and surrounded with square brackets.
     */
    template<typename T, typename Fn>
    auto list(T&& value, Fn&& fn)
    {
      return brackets(comma_sep(std::forward<T>(value), std::forward<Fn>(fn)));
    }

    /**
     * Displays each element of a list, separated by commas, and surrounded with
     * square brackets.
     *
     * Does nothing if the list is empty.
     */
    template<typename T>
    auto optional_list(T&& value)
    {
      return optional(list(std::forward<T>(value)));
    }

    /**
     * Displays the result of applying `fn` to each element of a list, separated
     * by commas, and surrounded with square brackets.
     *
     * Does nothing if the list is empty.
     */
    template<typename T, typename Fn>
    auto optional_list(T&& value, Fn&& fn)
    {
      return optional(list(std::forward<T>(value), std::forward<Fn>(fn)));
    }

    /**
     * Some deduction guides to help the compiler figure out the right type
     * arguments.
     */
    template<typename T>
    explicit prefixed(std::string, T &&)->prefixed<T>;

    template<typename T>
    explicit surrounded(std::string, std::string, T &&)->surrounded<T>;

    template<typename T>
    explicit optional(T &&)->optional<T>;

    template<typename T>
    explicit defaulted(T&&, std::string)->defaulted<T>;

    template<typename T>
    explicit separated_by(T&&, std::string)->separated_by<T>;

    template<typename T, typename Fn>
    explicit separated_by(T&&, std::string, Fn &&)
      ->separated_by<
        T,
        std::invoke_result_t<Fn, const internal::element_type<T>&>>;

    template<typename T>
    explicit lines(T &&)->lines<T>;

    /**
     * Print an object to a string.
     */
    template<typename T>
    std::string to_string(const T& value)
    {
      std::ostringstream buf;
      buf << internal::deref(value);
      return buf.str();
    }

    template<typename T, typename Fn>
    std::string to_string(const T& value, const Fn& fn)
    {
      return to_string(fn(value));
    }

    /**
     * Format each element in a list as a string, and sort the resulting vector
     * based on the string representation.
     *
     * This is useful to have a stable ordering when printing sets.
     */
    template<typename T, typename... Args>
    std::vector<std::string> sorted(const T& collection, Args... args)
    {
      std::vector<std::string> result;
      for (const auto& elem : collection)
      {
        result.push_back(to_string(elem, args...));
      }
      std::sort(result.begin(), result.end());
      return result;
    }
  }
}
