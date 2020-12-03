// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>
#include <utility>

namespace verona::compiler
{
  namespace traits
  {
    /*
     * Set-like types.
     * Has identical key and value types.
     */
    template<class T, class = void>
    struct is_set_like : std::false_type
    {};
    template<class T>
    struct is_set_like<
      T,
      std::enable_if_t<
        std::is_same_v<typename T::key_type, typename T::value_type>>>
    : std::true_type
    {};
    template<class T>
    constexpr bool is_set_like_v = is_set_like<T>::value;

    /*
     * Map-like types.
     * Has value = pair<key, mapped>
     */
    template<class T, class = void>
    struct is_map_like : std::false_type
    {};
    template<class T>
    struct is_map_like<
      T,
      std::enable_if_t<std::is_same_v<
        std::pair<const typename T::key_type, typename T::mapped_type>,
        typename T::value_type>>> : std::true_type
    {};
    template<class T>
    constexpr bool is_map_like_v = is_map_like<T>::value;

    /*
     * Vector-like types.
     * Has a value type, but no key.
     */
    template<class T, class = void, class = void>
    struct is_vector_like : std::false_type
    {};
    template<class T, class V>
    struct is_vector_like<T, std::void_t<typename T::value_type>, V>
    : std::true_type
    {};
    template<class T>
    struct is_vector_like<
      T,
      std::void_t<typename T::key_type>,
      std::void_t<typename T::value_type>> : std::false_type
    {};
    template<class T>
    constexpr bool is_vector_like_v = is_vector_like<T>::value;
  }
}
