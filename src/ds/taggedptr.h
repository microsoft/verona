// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace rt
{
  template<class T>
  struct TaggedPtr
  {
    T* ptr;

    TaggedPtr(T* p) : ptr(p)
    {
      assert(bits::address(p) & 7 == 0);
    }

    T* get_ptr()
    {
      return (T*)(ptr ^ (ptr & 7));
    }
  }
}