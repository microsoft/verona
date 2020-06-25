// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ext_ref_basic.h"
#include "ext_ref_merge.h"

int main(int argc, char** argv)
{
  (void)argc;
  (void)argv;

  ext_ref_basic::run_test();
  ext_ref_merge::run_test();

  return 0;
}
