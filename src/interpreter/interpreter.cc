// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "interpreter/code.h"
#include "interpreter/vm.h"
#include "options.h"

#include <iterator>
#include <verona.h>

namespace verona::interpreter
{
  Code load_file(std::istream& input)
  {
    input.unsetf(std::ios_base::skipws);

    std::vector<uint8_t> data;
    std::copy(
      std::istream_iterator<uint8_t>(input),
      std::istream_iterator<uint8_t>(),
      std::back_inserter(data));

    return Code(data);
  }

  class EmptyCown : public rt::VCown<EmptyCown>
  {
  public:
    EmptyCown() {}
  };

  void
  instantiate(size_t cores, const Code& code, bool verbose, size_t seed = 1234)
  {
#ifdef USE_SYSTEMATIC_TESTING
    Systematic::set_seed(seed);
#endif
    rt::Scheduler& sched = rt::Scheduler::get();
    sched.init(cores);

    size_t ip = code.entrypoint();

    rt::Cown* cown = new EmptyCown();

    // The entrypoint is a static function, so we pass the Main descriptor as
    // the receiver. This matches the usual calling convention for static
    // methods.
    // TODO: Should this contain command line arguments in the future?
    std::vector<Value> args;
    args.push_back(Value::descriptor(code.special_descriptors().main));

    rt::Cown::schedule<ExecuteMessage>(cown, ip, std::move(args), 0);

    rt::Alloc* alloc = rt::ThreadAlloc::get();
    rt::Cown::release(alloc, cown);

    sched.run_with_startup<const Code*, bool>(VM::init_vm, &code, verbose);

    snmalloc::current_alloc_pool()->debug_check_empty();
  }

  void instantiate(InterpreterOptions& options, const Code& code)
  {
#ifdef USE_SYSTEMATIC_TESTING
    if (options.run_seed.has_value())
    {
      if (options.debug_runtime)
        Systematic::enable_logging();
      if (options.run_seed_upper.has_value())
      {
        for (size_t i = options.run_seed.value();
             i < options.run_seed_upper.value();
             i++)
        {
          std::cout << "Seed: " << i << std::endl;
          interpreter::instantiate(options.cores, code, options.verbose, i);
        }
      }
      else
      {
        interpreter::instantiate(
          options.cores, code, options.verbose, options.run_seed.value());
      }
    }
    else
    {
      interpreter::instantiate(options.cores, code, options.verbose);
    }
#else
    interpreter::instantiate(options.cores, code, options.verbose);
#endif
  }
}
