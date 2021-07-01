// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#ifdef __unix__
#  include <fcntl.h>
#  include <memory>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/types.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * Detail namespace, contains classes that are not part of the public
     * interface.  This provides implementations of a `SharedMemoryObject`
     * interface, which defines a single field, `Handle fd` containing a shared
     * memory object.
     */
    namespace detail
    {
      /**
       * Anonymous shared memory objects used for sandboxes can be very large,
       * if we dump them in core files the cores can be many gigabytes even if
       * the sandbox only touches a few megabytes.  To avoid this, we exclude
       * the shared regions from core dumps on mapping.  We avoid `#ifdef`s
       * around `mmap` calls by providing a no-op version of this flag for
       * platforms where it isn't supported.
       */
      constexpr int map_nocore =
#  ifdef MAP_NOCORE
        MAP_NOCORE
#  else
        0
#  endif
        ;

      /**
       * Generic POSIX shared memory object.  This uses the POSIX `shm_open`
       * interface to create a new shared memory object and then makes it
       * anonymous by unlinking it once it has been opened.  POSIX specifies
       * that the shared memory object will remain in existence until after it
       * has been unlinked and all file descriptors to it have been closed.
       *
       * There is a small window in between creating and unlinking the file
       * where it can leak.  This is unavoidable with the POSIX APIs: there is
       * no way of atomically creating and unlinking any kind of file.
       * Non-standard APIs for doing this should be used in preference.
       */
      struct SharedMemoryObjectPOSIX
      {
        /**
         * Helper that constructs a deleter from a C function, so that it can
         * be used with `std::unique_ptr`.
         */
        template<auto fn>
        using deleter_from_fn = std::integral_constant<decltype(fn), fn>;

        /**
         * The file descriptor for this object.
         */
        Handle fd;

        /**
         * Constructor, tries to construct an anonymous shared memory object by
         * trying to pick a name that doesn't collide and retrying until it
         * finds one.  In theory, this might not terminate if another process
         * keeps creating files with the same sequence of random numbers or if
         * all 2^64 names exist in the shared memory namespace.
         */
        SharedMemoryObjectPOSIX()
        {
          // RIAA wrapper around some C malloc'd memory for use with aspintf.
          std::unique_ptr<char, deleter_from_fn<::free>> name;
          do
          {
            // Create a new random name.  Note that this *looks* like a path,
            // but POSIX is ambiguous about whether this is or is not a path
            // and in most systems is a separate namespace from the
            // filesystem namespace.
            {
              char* name_raw = nullptr;
              int ret =
                asprintf(&name_raw, "/verona_sandbox_alloc_%lx", random());
              SANDBOX_INVARIANT(
                ret > 0,
                "aspintf failed trying to create unique name for POSIX shared "
                "memory object");
              name.reset(name_raw);
            }
            // Try to atomically create-and-open the shared memory object with
            // this name.  If it fails with an error indicating that the name
            // is in use, retry.
            fd = shm_open(
              name.get(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
          } while ((!fd.is_valid() && (errno == EEXIST)));
          // If we *did* create the object, unlink it so that we end up with an
          // anonymous shared memory object.
          if (fd.is_valid())
          {
            shm_unlink(name.get());
          }
        }
      };

#  ifdef USE_MEMFD
      /**
       * A shared memory object created with the MemFD interface.  This was
       * added for Linux to allow anonymous shared memory objects and, on Linux,
       * exposes some extra functionality for dynamic immutability that we don't
       * use.
       */
      struct SharedMemoryObjectMemFD
      {
        /**
         * The file descriptor for the memory object.
         */
        Handle fd;
        /**
         * Construct a memory file descriptor.  If this fails, try creating a
         * generic POSIX shared object instead.
         */
        SharedMemoryObjectMemFD() : fd(memfd_create("Verona Sandbox", 0))
        {
          // If memfd_create fails (e.g. on WSL1, where it isn't
          // implemented), fall back to the POSIX code path.
          if (!fd.is_valid())
          {
            SharedMemoryObjectPOSIX p;
            fd = std::move(p.fd);
          }
        }
      };
#  endif

#  ifdef SHM_ANON
      /**
       * A shared memory object using anonymous POSIX shared memory.  This is a
       * tiny extension to POSIX added to FreeBSD for Capsicum sandboxed
       * processes (which are not allowed to access global namespaces) and is
       * functionally equivalent to and atomic `shm_open` and `shm_unlink`: It
       * creates a POSIX shared memory object and never exposes it into the
       * global namespace for POSIX shared memory objects.
       */
      struct SharedMemoryObjectShmAnon
      {
        /**
         * The shared memory object.
         */
        Handle fd;
        /**
         * Create an anonymous POSIX shared memory object.
         */
        SharedMemoryObjectShmAnon()
        : fd(shm_open(SHM_ANON, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR))
        {}
      };
#  endif
    }

    /**
     * Base class for POSIX shared memory mappings.  Handles resizing a POSIX
     * file descriptor to the correct size, but not mapping the object.  This
     * is instantiated with a concrete class that constructs and owns the
     * shared memory object.
     *
     * Subclasses are expected to handle mapping the shared memory object.
     */
    template<typename SharedMemoryObject>
    class SharedMemoryMapPOSIXBase
    {
    protected:
      /**
       * The shared memory object.  Owns the file descriptor and is
       * responsible for destroying it.
       */
      SharedMemoryObject mem_object;

      /**
       * The location at which the memory is mapped.  Set to the `mmap` return
       * value in error conditions.
       */
      void* map_address = MAP_FAILED;

      /**
       * The size (in bytes) of the mapping.
       */
      size_t size;

      /**
       * Helper to get the file descriptor for the memory object.
       */
      handle_t get_fd()
      {
        return mem_object.fd.fd;
      }

      /**
       * Constructor.  Resizes the underlying file descriptor but does *not*
       * map the memory.  Protected so that it can be called only from subclass
       * constructors, which are expected to map the memory.
       */
      SharedMemoryMapPOSIXBase(uint8_t log2_size)
      : size(snmalloc::bits::one_at_bit(log2_size))
      {
        assert(mem_object.fd.is_valid());
        assert(size >= snmalloc::OS_PAGE_SIZE);
        int ret = ftruncate(mem_object.fd.fd, size);
        SANDBOX_INVARIANT(ret == 0, "ftruncate failed {}", strerror(errno));
      }

      /**
       * Destructor, unmaps the memory.
       */
      ~SharedMemoryMapPOSIXBase()
      {
        if (map_address != MAP_FAILED)
        {
          munmap(map_address, size);
        }
      }

    public:
      /**
       * Helper method to access the handle to the underlying memory.  The
       * handle is not used after the mapping is created and so this can be
       * `std::move`d out if required.  It is therefore possible that this
       * returns an invalid handle.
       */
      Handle& get_handle()
      {
        return mem_object.fd;
      }

      /**
       * The size, in bytes, of the mapped region.
       */
      size_t get_size()
      {
        return size;
      }

      /**
       * The base address of the mapping.  This is guaranteed to be naturally
       * aligned.
       */
      void* get_base()
      {
        assert(map_address != MAP_FAILED);
        return map_address;
      }
    };

    /**
     * Generic mapping object for POSIX shared memory.  Provides large
     * naturally aligned mappings without using any non-standard extensions.
     *
     * `SharedMemoryObject` is the class that provides the memory object (file
     * descriptor).
     */
    template<typename SharedMemoryObject>
    class SharedMemoryMapPOSIX
    : public SharedMemoryMapPOSIXBase<SharedMemoryObject>
    {
      /**
       * The base class type.  C++ does not allow us to refer to any members in
       * a template base class without explicit qualification, this declaration
       * reduces boilerplate.
       */
      using Base = SharedMemoryMapPOSIXBase<SharedMemoryObject>;

    public:
      /**
       * Constructor.  Takes the base-2 logarithm of the size of the mapping.
       */
      SharedMemoryMapPOSIX(uint8_t log2_size) : Base(log2_size)
      {
        // Set the object size to the desired size.
        int ret = ftruncate(Base::get_fd(), Base::size);
        SANDBOX_INVARIANT(ret == 0, "ftruncate failed {}", strerror(errno));

        if (Base::size > snmalloc::OS_PAGE_SIZE)
        {
          // Allocate a region twice as big as we need so that there is a
          // naturally aligned region of the correct space inside it.
          void* region = mmap(
            0,
            Base::size * 2,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANON | detail::map_nocore,
            -1,
            0);
          void* region_end = snmalloc::pointer_offset(region, Base::size * 2);
          // Find the first naturally aligned block  in this space
          void* aligned_start = snmalloc::pointer_align_up(region, Base::size);
          // Map the shared memory object at that block
          Base::map_address = mmap(
            aligned_start,
            Base::size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_FIXED | detail::map_nocore,
            Base::get_fd(),
            0);
          void* map_end =
            snmalloc::pointer_offset(Base::map_address, Base::size);
          // Trim the unused space
          if (aligned_start > region)
          {
            munmap(region, snmalloc::pointer_diff(region, aligned_start));
          }
          if (map_end < region_end)
          {
            munmap(map_end, snmalloc::pointer_diff(map_end, region_end));
          }
        }
        else
        {
          Base::map_address = mmap(
            0,
            Base::size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | detail::map_nocore,
            Base::get_fd(),
            0);
        }
      }
    };

#  ifdef MAP_ALIGNED
    template<typename SharedMemoryObject>
    struct SharedMemoryMapMMapAligned
    : SharedMemoryMapPOSIXBase<SharedMemoryObject>
    {
      /**
       * The base class type.  C++ does not allow us to refer to any members in
       * a template base class without explicit qualification, this declaration
       * reduces boilerplate.
       */
      using Base = SharedMemoryMapPOSIXBase<SharedMemoryObject>;
      /**
       * Constructor.  Takes the base-2 logarithm of the size of the mapping.
       */
      SharedMemoryMapMMapAligned(uint8_t log2_size) : Base(log2_size)
      {
        Base::map_address = mmap(
          0,
          Base::size,
          PROT_READ | PROT_WRITE,
          MAP_ALIGNED(log2_size) | MAP_SHARED | detail::map_nocore,
          Base::get_fd(),
          0);
      }
    };
#  endif

  }
}
#endif
