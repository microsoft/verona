// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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