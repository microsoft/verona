// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::compiler
{
  template<typename Left, typename Right>
  struct zip
  {
    using left_iterator = decltype(std::begin(std::declval<Left&>()));
    using right_iterator = decltype(std::begin(std::declval<Right&>()));

    struct iterator
    {
      left_iterator left;
      right_iterator right;

      bool operator!=(const iterator& other) const
      {
        return left != other.left && right != other.right;
      }

      void operator++()
      {
        left++;
        right++;
      }

      std::pair<
        typename left_iterator::reference,
        typename right_iterator::reference>
      operator*()
      {
        return {*left, *right};
      }
    };

    zip(Left&& left, Right&& right)
    : left(std::forward<Left>(left)), right(std::forward<Right>(right))
    {}

    iterator begin()
    {
      return iterator{left.begin(), right.begin()};
    }

    iterator end()
    {
      return iterator{left.end(), right.end()};
    }

    Left&& left;
    Right&& right;
  };

  template<typename Left, typename Right>
  zip<Left, Right> safe_zip(Left&& left, Right&& right)
  {
    assert(left.size() == right.size());
    return zip<Left, Right>(
      std::forward<Left>(left), std::forward<Right>(right));
  }
}
