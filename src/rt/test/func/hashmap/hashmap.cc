// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ds/hashmap.h"

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

bool test(size_t seed)
{
  auto* alloc = ThreadAlloc::get();
  ObjectMap<std::pair<Object*, int32_t>> map(alloc);
  std::unordered_map<Object*, int32_t> model;

  xoroshiro::p128r64 rng{seed};
  std::stringstream err;

  map.debug_layout(err) << "\n";

  static constexpr size_t entries = 100;
  for (size_t i = 0; i < entries; i++)
  {
    auto* key = (Object*)(rng.next() * Object::ALIGNMENT);
    auto entry = std::make_pair(key, (int32_t)i);
    err << "insert " << key << "\n";
    model.insert(entry);
    auto inserted = map.insert(alloc, entry).first;
    if (!model_check(map, model, err))
    {
      std::cout << err.str() << std::flush;
      return false;
    }

    if (!inserted)
    {
      std::cout << err.str() << "not inserted: " << entry.first << std::endl;
      return false;
    }

    if ((rng.next() % 10) == 0)
    {
      err << "update " << key << "\n";
      entry.second = -entry.second;
      model.insert(entry);
      inserted = map.insert(alloc, entry).first;
      if (!model_check(map, model, err))
      {
        std::cout << err.str() << std::flush;
        return false;
      }

      if (inserted)
      {
        std::cout << err.str() << "not updated: " << entry.first << std::endl;
        return false;
      }
    }

    if ((rng.next() % 10) == 0)
    {
      err << "erase " << key << "\n";
      model.erase(entry.first);
      auto erased = map.erase(entry.first);
      if (!model_check(map, model, err))
      {
        std::cout << err.str() << std::flush;
        return false;
      }

      if (!erased)
      {
        std::cout << err.str() << "not erased: " << entry.first << std::endl;
        return false;
      }
    }
  }

  map.clear(alloc);
  if (map.size() != 0)
  {
    map.debug_layout(std::cout) << "not empty" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  auto seed = opt.is<size_t>("--seed", 5489);
  const auto seed_upper = opt.is<size_t>("--seed_upper", seed);

  for (; seed <= seed_upper; seed++)
  {
    std::cout << "seed: " << seed << std::endl;
    if (!test(seed))
      return 1;

    current_alloc_pool()->debug_check_empty();
  }

  return 0;
}
