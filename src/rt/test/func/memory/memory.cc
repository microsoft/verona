// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "memory.h"

#include "memory_alloc.h"
#include "memory_gc.h"
#include "memory_iterator.h"
#include "memory_merge.h"
#include "memory_subregion.h"
#include "memory_swap_root.h"

#include <test/harness.h>
#include <test/opt.h>

/**
 * Tests memory management, including the region functionality.
 *
 * Other tests to look at include finalisers and the various cowngc tests.
 **/

void test_alloc_pool()
{
#ifndef SNMALLOC_PASS_THROUGH
  auto* a1 = current_alloc_pool()->acquire();
  auto* a2 = current_alloc_pool()->acquire();
  check(a1 != a2);

  current_alloc_pool()->release(a1);
  auto* a3 = current_alloc_pool()->acquire();
  check(a3 == a1);

  current_alloc_pool()->release(a2);
  current_alloc_pool()->release(a3);
  snmalloc::current_alloc_pool()->debug_check_empty();
#endif
}

void test_dealloc()
{
  auto* alloc = ThreadAlloc::get();

  size_t size = 1 << 25;
  void* p = alloc->alloc(size);
  alloc->dealloc(p, size);

  snmalloc::current_alloc_pool()->debug_check_empty();
}

size_t do_nothing(size_t x);

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  size_t seed = opt.is<size_t>("--seed", 0);
  seed = do_nothing(seed);
#ifdef USE_SYSTEMATIC_TESTING
#  ifdef WIN32
//  TODO: Not currently supported by snmalloc
//  default_memory_provider().systematic_bump_ptr() += seed << 17;
#  endif
#endif

#ifdef CI_BUILD
  auto log = true;
#else
  auto log = opt.has("--log-all");
#endif

  if (log)
    Systematic::enable_logging();

  memory_alloc::run_test();
  memory_iterator::run_test();
  memory_swap_root::run_test();
  memory_merge::run_test();
  memory_gc::run_test();
  memory_subregion::run_test();

  test_alloc_pool();
  test_dealloc();

  return 0;
}
