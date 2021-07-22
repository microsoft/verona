// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"

namespace verona::rt
{
  class Cown;

  template<TransferOwnership transfer = NoTransfer, typename T>
  static void schedule_lambda(Cown* c, T f);
}
