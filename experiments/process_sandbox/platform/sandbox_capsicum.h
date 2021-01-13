// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#if __has_include(<sys/capsicum.h>)
#  include "../sandbox_fd_numbers.h"

#  include <sys/capsicum.h>
#  include <sys/procctl.h>
namespace sandbox
{
  namespace platform
  {
    struct SandboxCapsicum
    {
      template<typename T, typename U>
      void restrict_file_descriptors(const T&, const U& libdirfds)
      {
        // If we're compiling with Capsicum support, then restrict the
        // permissions on all of the file descriptors that are available to
        // untrusted code.
        auto limit_fd = [&](int fd, auto... permissions) {
          cap_rights_t rights;
          if (
            cap_rights_limit(fd, cap_rights_init(&rights, permissions...)) != 0)
          {
            err(1, "Failed to limit rights on file descriptor %d", fd);
          }
        };
        // Standard in is read only
        limit_fd(STDIN_FILENO, CAP_READ);
        // Standard out and error are write only
        limit_fd(STDOUT_FILENO, CAP_WRITE);
        limit_fd(STDERR_FILENO, CAP_WRITE);
        // The socket is used with a call-return protocol for requesting
        // services for malloc.
        limit_fd(PageMapUpdates, CAP_WRITE, CAP_READ);
        // The shared heap can be mapped read-write, but can't be truncated.
        limit_fd(SharedMemRegion, CAP_MMAP_RW);
        limit_fd(PageMapPage, CAP_MMAP_R);
        // The library must be parseable and mappable by rtld
        limit_fd(MainLibrary, CAP_READ, CAP_FSTAT, CAP_SEEK, CAP_MMAP_RX);
        // The libraries implicitly opened from the library directories inherit
        // the permissions from the parent directory descriptors.  These need
        // the permissions required to map a library and also the permissions
        // required to search the directory to find the relevant libraries.
        for (auto libfd : libdirfds)
        {
          limit_fd(libfd, CAP_READ, CAP_FSTAT, CAP_LOOKUP, CAP_MMAP_RX);
        }
      }

      /**
       * Apply the sandboxing policy.  This enters capability mode at which
       * point the process loses access to the global namespace and is unable
       * to open new sockets or files, except by being passed them from the
       * parent or via calls such as `openat` that take a sufficiently
       * authorized file descriptor.
       */
      static void apply_sandboxing_policy_postexec()
      {
        fprintf(stderr, "Entering capability mode!\n");
        int arg = PROC_TRAPCAP_CTL_ENABLE;
        int ret = procctl(P_PID, getpid(), PROC_TRAPCAP_CTL, &arg);
        assert(ret == 0);
        cap_enter();
      }
      void apply_sandboxing_policy_preexec() {}
    };
  }
}
#endif
