// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ds/hashmap.h"

#include "test/harness.h"
#include "test/opt.h"
#include "test/xoroshiro.h"
#include "verona.h"

#include <unordered_map>

using namespace snmalloc;
using namespace verona::rt;

template<typename Entry, typename Model>
bool model_check(
  const ObjectMap<Entry>& map, const Model& model, std::stringstream& err)
{
  map.debug_layout(err) << "\n";

  if (map.size() != model.size())
  {
    err << "map size (" << map.size() << ") is not expected (" << model.size()
        << ")"
        << "\n";
    return false;
  }

  for (const auto& e : model)
  {
    auto it = map.find(e.first);
    if (it == map.end())
    {
      err << "not found: " << e.first << "\n";
      return false;
    }
    else if (it.is_marked())
    {
      err << "marked: " << e.first << "\n";
      return false;
    }
  }

  auto iter_model = model;
  for (auto it = map.begin(); it != map.end(); ++it)
    iter_model.erase(it.key());

  if (!iter_model.empty())
  {
    for (const auto& e : iter_model)
      err << "not found: " << e.first << "\n";

    return false;
  }

  return true;
}

struct Key : public VCown<Key>
{};

bool test(size_t seed)
{
  auto& alloc = ThreadAlloc::get();
  ObjectMap<std::pair<Key*, int32_t>> map(alloc);
  std::unordered_map<Key*, int32_t> model;

  xoroshiro::p128r64 rng{seed};
  std::stringstream err;

  map.debug_layout(err) << "\n";

  static constexpr size_t entries = 100;
  for (size_t i = 0; i < entries; i++)
  {
    auto* key = new (alloc) Key();
    auto entry = std::make_pair(key, (int32_t)i);
    err << "insert " << key
#ifdef USE_SYSTEMATIC_TESTING
        << " (" << key->id() << ")"
#endif
        << "\n";
    model.insert(entry);
    auto insertion = map.insert(alloc, entry);
    if ((insertion.first != true) || (insertion.second.key() != key))
    {
      map.debug_layout(err)
        << "\n"
        << "incorrect return from insert: " << insertion.first << ", "
        << insertion.second.key() << "\n";
      std::cout << err.str() << std::flush;
      return false;
    }
    if (!model_check(map, model, err))
    {
      std::cout << err.str() << std::flush;
      return false;
    }

    if (!insertion.first)
    {
      std::cout << err.str() << "not inserted: " << key << std::endl;
      return false;
    }

    if ((rng.next() % 10) == 0)
    {
      err << "update " << key << "\n";
      entry.second = -entry.second;
      model.insert(entry);
      insertion = map.insert(alloc, entry);
      if (!model_check(map, model, err))
      {
        std::cout << err.str() << std::flush;
        return false;
      }

      if (insertion.first)
      {
        std::cout << err.str() << "not updated: " << key << std::endl;
        return false;
      }
    }

    if ((rng.next() % 10) == 0)
    {
      err << "erase " << key << "\n";
      model.erase(key);
      auto erased = map.erase(key);
      if (!model_check(map, model, err))
      {
        std::cout << err.str() << std::flush;
        return false;
      }

      if (!erased)
      {
        std::cout << err.str() << "not erased: " << key << std::endl;
        return false;
      }
      Cown::release(alloc, key);
    }
  }

  map.clear(alloc);
  if (map.size() != 0)
  {
    map.debug_layout(std::cout) << "not empty" << std::endl;
    return false;
  }

  for (auto e : model)
    Cown::release(alloc, e.first);

  return true;
}

int main(int argc, char** argv)
{
  // Use harness for consistent API to seeds for randomness.
  SystematicTestHarness harness(argc, argv);

  for (size_t seed = harness.seed_lower; seed <= harness.seed_upper; seed++)
  {
    std::cout << "seed: " << seed << std::endl;
    if (!test(seed))
      return 1;

    debug_check_empty<snmalloc::Alloc::Config>();
  }

  return 0;
}
