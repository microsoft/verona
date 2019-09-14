// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "./noticeboard_basic.h"
#include "./noticeboard_primitive_weak.h"
#include "./noticeboard_weak.h"

#include <test/harness.h>

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(noticeboard_basic::run_test);
  harness.run(noticeboard_weak::run_test);
  harness.run(noticeboard_primitive_weak::run_test);
  return 0;
}
