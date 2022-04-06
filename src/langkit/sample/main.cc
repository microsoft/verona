// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

int main(int argc, char** argv)
{
  return sample::driver().run(argc, argv) ? 0 : -1;
}
