// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef __unix__
#  include <functional>
#  include <optional>
#  include <sys/socket.h>
#  include <sys/types.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * Type for creating pairs of connected sockets.
     */
    class SocketPairPosix
    {
    public:
      /**
       * The wrapper around a single socket.  This inherits from `Handle` and so
       * all lifetime management works automatically.
       */
      class Socket : public Handle
      {
        /**
         * Send a message containing `data_len` bytes of data starting at
         * `data`.  Optionally also send a file descriptor.
         */
        bool
        send_msg(const void* data, size_t data_len, int flags, int send_fd = -1)
        {
          struct msghdr header;
          struct iovec iov
          {
            const_cast<void*>(data), data_len
          };
          char ctrl_buf[CMSG_SPACE(sizeof(int))] = {0};
          // Point-to-point socket, no destination address
          header.msg_name = nullptr;
          header.msg_namelen = 0;
          header.msg_flags = 0;
          header.msg_iov = &iov;
          header.msg_iovlen = 1;
          if (send_fd >= 0)
          {
            header.msg_controllen = sizeof(ctrl_buf);
            header.msg_control = ctrl_buf;

            struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
            cmsg->cmsg_level = SOL_SOCKET;
            cmsg->cmsg_type = SCM_RIGHTS;
            cmsg->cmsg_len = CMSG_LEN(sizeof(int));
            int* fd_ptr = reinterpret_cast<int*>(CMSG_DATA(cmsg));
            *fd_ptr = send_fd;
          }
          else
          {
            header.msg_controllen = 0;
            header.msg_control = nullptr;
          }
          int ret;
          do
          {
            ret = sendmsg(fd, &header, MSG_NOSIGNAL | flags);
          } while ((ret == -1) && (errno == EINTR));
          return static_cast<size_t>(ret) == data_len;
        }

        /**
         * Receive a message, `data_len` bytes long into `data`.  If `recv_fd`
         * is not a null pointer, try to receive a file descriptor there.  The
         * `int` pointed to by this parameter is set to -1 if no file
         * descriptor is received.
         */
        bool receive_msg(
          void* data, size_t data_len, int flags, int* recv_fd = nullptr)
        {
          struct msghdr header;
          struct iovec iov
          {
            const_cast<void*>(data), data_len
          };
          char ctrl_buf[CMSG_SPACE(sizeof(int))] = {0};
          // Point-to-point socket, no source address
          header.msg_name = nullptr;
          header.msg_namelen = 0;
          header.msg_flags = 0;
          header.msg_iov = &iov;
          header.msg_iovlen = 1;
          if (recv_fd != nullptr)
          {
            header.msg_controllen = sizeof(ctrl_buf);
            header.msg_control = ctrl_buf;
            *recv_fd = -1;
          }
          else
          {
            header.msg_controllen = 0;
            header.msg_control = nullptr;
          }
          int ret;
          do
          {
            ret = recvmsg(fd, &header, flags | MSG_NOSIGNAL);
          } while ((ret == -1) && (errno == EINTR));
          if (
            (ret != -1) && (recv_fd != nullptr) && (header.msg_controllen > 0))
          {
            struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
            if (
              (cmsg->cmsg_level == SOL_SOCKET) &&
              (cmsg->cmsg_type == SCM_RIGHTS) &&
              (cmsg->cmsg_len == CMSG_LEN(sizeof(int))))
            {
              int* fd_ptr = reinterpret_cast<int*>(CMSG_DATA(cmsg));
              *recv_fd = *fd_ptr;
            }
          }
          return static_cast<size_t>(ret) == data_len;
        }

      public:
        /**
         * Export all constructors from `Handle`.
         */
        using Handle::Handle;

        enum ShouldBlock
        {
          Block,
          DoNotBlock
        };

        /**
         * Send a message containing `data`.  Accompanied by a copy of the file
         * descriptor owned by `h` (if one is provided).  If `B` is `Block` then
         * this will block until completion.  If `B` is `DoNotBlock` then it
         * will return `false` if the operation cannot complete immediately.
         */
        template<ShouldBlock B, typename T>
        [[nodiscard]] bool send(
          const T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          int fd = h.has_value() ? h->get().fd : -1;
          return send_msg(
            &data, sizeof(T), (B == Block) ? MSG_WAITALL : MSG_DONTWAIT, fd);
        }

        /**
         * Send a message containing `data`.  Accompanied by a copy of the file
         * descriptor owned by `h` (if one is provided).  Returns `false` if
         * the operation cannot complete immediately.
         */
        template<typename T>
        [[nodiscard]] bool nonblocking_send(
          const T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          return send<Socket::DoNotBlock>(data, h);
        }

        /**
         * Send a message containing `data`.  Accompanied by a copy of the file
         * descriptor owned by `h` (if one is provided).  This will block until
         * completion.
         * */
        template<typename T>
        [[nodiscard]] bool blocking_send(
          const T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          return send<Socket::Block>(data, h);
        }

        /**
         * Receive a message into `data`.  If `B` is set to `Block` then this
         * will block until completion, otherwise it will return failure if the
         * operation cannot complete.  If `h` is provided then it will be set
         * to a received file descriptor.
         */
        template<ShouldBlock B, typename T>
        [[nodiscard]] bool receive(
          T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          int flags = (B == Block) ? MSG_WAITALL : MSG_DONTWAIT;
          if (h.has_value())
          {
            int recv_fd;
            bool ret = receive_msg(&data, sizeof(T), flags, &recv_fd);
            if (ret)
            {
              h->get().reset(recv_fd);
            }
            return ret;
          }
          else
          {
            return receive_msg(&data, sizeof(T), flags);
          }
        }

        /**
         * Receive a message into `data`.  This will block until completion.  If
         * `h` is provided then it will be set to a received file descriptor.
         */
        template<typename T>
        [[nodiscard]] bool blocking_receive(
          T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          return receive<Socket::Block>(data, h);
        }

        /**
         * Receive a message into `data`.  Returns failure if the operation
         * cannot complete.  If `h` is provided then it will be set to a
         * received file descriptor.
         */
        template<typename T>
        [[nodiscard]] bool nonblocking_receive(
          T& data,
          const std::optional<std::reference_wrapper<Handle>> h = std::nullopt)
        {
          return receive<Socket::DoNotBlock>(data, h);
        }
      };

      /**
       * Create a pair of connected sockets.
       */
      static std::pair<Socket, Socket> create()
      {
        int socks[2];
        if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, socks))
        {
          err(1, "Failed to create socket pair");
        }
        return {Socket(socks[0]), Socket(socks[1])};
      }
    };
  }
}
#endif
