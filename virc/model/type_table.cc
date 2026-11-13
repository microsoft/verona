#include "type_table.h"

namespace virc
{
  size_t TypeInfoHash::operator()(const TypeInfo& type) const noexcept
  {
    auto hash = size_t(14695981039346656037ull);
    hash = (hash ^ static_cast<size_t>(type.kind)) * size_t(1099511628211ull);

    for (auto element : type.elements)
      hash = (hash ^ element) * size_t(1099511628211ull);

    return hash;
  }
}