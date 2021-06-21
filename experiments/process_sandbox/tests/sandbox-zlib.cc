// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>
#include <zlib.h>

using namespace sandbox;

/**
 * The structure that represents an instance of the sandbox.
 */
struct SandboxZlib
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(public_name, private_name) \
  decltype(make_sandboxed_function<decltype(private_name)>(lib)) public_name = \
    make_sandboxed_function<decltype(private_name)>(lib);
#include "zlib.inc"
};

struct UnsandboxedZlib
{
  struct
  {
    template<typename T>
    T* alloc(size_t count = 1)
    {
      T* array = static_cast<T*>(calloc(sizeof(T), count));
      for (size_t i = 0; i < count; i++)
      {
        new (&array[i]) T();
      }
      return array;
    }
    void free(void* ptr)
    {
      ::free(ptr);
    }
    char* strdup(const char* ptr)
    {
      return ::strdup(ptr);
    }
  } lib;
  int deflateInit_(
    z_streamp strm, int level, const char* version, int stream_size)
  {
    return ::deflateInit_(strm, level, version, stream_size);
  }
  int deflate(z_streamp strm, int flush)
  {
    return ::deflate(strm, flush);
  }
  int deflateEnd(z_streamp strm)
  {
    return ::deflateEnd(strm);
  }
};

template<typename ZLib>
void test(ZLib& sandbox, const char* file, std::vector<char>& result)
{
  int fd = open(file, O_RDONLY);
  SANDBOX_INVARIANT(fd >= 0, "Failed to open {}", file);
  static const size_t out_buffer_size = 1024;
  static const size_t in_buffer_size = 1024;
  char* in = sandbox.lib.template alloc<char>(in_buffer_size);
  char* out = sandbox.lib.template alloc<char>(out_buffer_size);
  // This is needed because deflateInit is a macro that implicitly passes
  // a string literal.
  char* version = sandbox.lib.strdup(ZLIB_VERSION);
#undef ZLIB_VERSION
#define ZLIB_VERSION version
  z_stream* zs = sandbox.lib.template alloc<z_stream>();
  memset(zs, 0, sizeof(*zs));
  zs->zalloc = Z_NULL;
  zs->zfree = Z_NULL;
  int ret = sandbox.deflateInit(zs, Z_DEFAULT_COMPRESSION);
  SANDBOX_INVARIANT(
    ret == Z_OK, "deflateInit returned {}, expected {}", ret, Z_OK);

  zs->next_out = reinterpret_cast<Bytef*>(out);
  zs->avail_out = out_buffer_size;

  while ((zs->avail_in = read(fd, in, in_buffer_size)))
  {
    zs->next_in = reinterpret_cast<Bytef*>(in);
    SANDBOX_INVARIANT(
      sandbox.deflate(zs, Z_PARTIAL_FLUSH) != Z_STREAM_ERROR,
      "deflate returned Z_STREAM_ERROR");
    size_t avail_out = std::min<size_t>(zs->avail_out, out_buffer_size);
    if (avail_out < out_buffer_size)
    {
      result.insert(result.end(), out, out + (out_buffer_size - avail_out));
      zs->next_out = reinterpret_cast<Bytef*>(out);
      zs->avail_out = out_buffer_size;
    }
  }
  sandbox.deflate(zs, Z_FINISH);
  sandbox.deflateEnd(zs);
  size_t avail_out = std::min<size_t>(zs->avail_out, out_buffer_size);
  if (avail_out < out_buffer_size)
  {
    result.insert(
      result.end(), zs->next_out, zs->next_out + (out_buffer_size - avail_out));
  }
  sandbox.lib.free(zs);
  sandbox.lib.free(in);
  sandbox.lib.free(out);
  close(fd);
}

int main(int, char** argv)
{
  SandboxZlib sandbox;
  UnsandboxedZlib nosb;
  std::vector<char> sb_compressed;
  std::vector<char> compressed;
  test(nosb, argv[0], compressed);
  try
  {
    test(sandbox, argv[0], sb_compressed);
  }
  catch (std::runtime_error& e)
  {
    printf("Sandbox exception: %s while running zlib compress\n", e.what());
    return -1;
  }
  SANDBOX_INVARIANT(
    sb_compressed.size() == compressed.size(),
    "Compression in the sandbox gave {} bytes, outside gave {} bytes",
    sb_compressed.size(),
    compressed.size());
  SANDBOX_INVARIANT(
    sb_compressed == compressed,
    "Compressing inside and outside of the sandbox gave different results");

  return 0;
}
