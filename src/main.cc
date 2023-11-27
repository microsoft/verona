// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "trieste/driver.h"

namespace verona
{
  void Options::configure(CLI::App& cli)
  {
    cli.add_flag("--no-std", no_std, "Don't import the standard library.");
  }
}

int main(int argc, char** argv)
{
  trieste::Driver d(
    "Verona", &verona::options(), verona::parser(), verona::passes());

  return d.run(argc, argv);
}
