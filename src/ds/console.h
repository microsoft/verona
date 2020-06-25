// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#ifdef _MSC_VER
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#ifdef _MSC_VER
UINT original_code_page = 0;
DWORD original_console_mode = 0;

void restore_console()
{
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  SetConsoleOutputCP(original_code_page);
  SetConsoleMode(hOut, original_console_mode);
}
#endif

void enable_colour_console()
{
#ifdef _MSC_VER
  // Enable console colours.
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD dwMode = 0;
  original_console_mode = GetConsoleMode(hOut, &dwMode);
  original_console_mode = dwMode;
  original_code_page = GetConsoleOutputCP();
  SetConsoleOutputCP(65001);
  dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
  SetConsoleMode(hOut, dwMode);
  atexit(restore_console);
  // TODO something for Ctrl-C and abort like exits.
#endif
}