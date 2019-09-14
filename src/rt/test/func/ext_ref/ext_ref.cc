// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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
