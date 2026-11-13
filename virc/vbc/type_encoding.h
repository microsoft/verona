#pragma once

#include "../model/type_table.h"
#include "encoder.h"

#include <vir.h>

namespace virc::vbc_backend
{
  uleb<size_t> encode_region(trieste::Node node);
  void
  encode_type_table(ByteBuffer& output, const std::vector<TypeInfo>& types);
}