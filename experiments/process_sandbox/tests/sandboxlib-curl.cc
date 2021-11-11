// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <curl/curl.h>
#include <stdio.h>

namespace
{
  /**
   * Buffer containing the fetched URL.
   */
  std::vector<char> buffer;

  /**
   * Callback used by curl when data are received.
   */
  size_t write_callback(char* ptr, size_t size, size_t nmemb, void*)
  {
    size_t bytes = size * nmemb;
    fprintf(stderr, "Curl received %zd bytes\n", bytes);
    buffer.insert(buffer.end(), ptr, ptr + bytes);
    return bytes;
  }

  /**
   * Fetch a URL in a sandbox, returning the result.
   */
  std::pair<char*, size_t> fetch(char* url)
  {
    buffer.clear();
    CURL* easy_handle = curl_easy_init();
    fprintf(stderr, "Sandbox fetching %s\n", url);
    auto err = [](CURLcode err, const char* msg) {
      if (err == CURLE_OK)
      {
        fprintf(stderr, "%s succeeded\n", msg);
      }
      else
      {
        fprintf(stderr, "%s failed: %s\n", msg, curl_easy_strerror(err));
      }
    };

    auto res = curl_easy_setopt(easy_handle, CURLOPT_URL, url);
    err(res, "curl_easy_setopt(CURLOPT_URL)");
    res = curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, write_callback);
    err(res, "curl_easy_setopt(CURLOPT_WRITEFUNCTION)");
    res = curl_easy_perform(easy_handle);
    err(res, "curl_easy_perform");
    return {buffer.data(), buffer.size()};
  }

}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::fetch);
}
