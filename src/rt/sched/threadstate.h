// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  class ThreadState
  {
  public:
    // The global state proceeds:
    // NotInLD -> PreScan -> Scan -> BelieveDone -> ReallyDone -> Sweep
    // -> Finished
    // In the ReallyDone state, the retracted flag may be true or false. If
    // it is true, the global state proceeds to Scan. If it is false, the
    // global state proceeds to Sweep, and then Finished.

    // Scheduler thread state proceeds:
    // NotInLD -> WantLD (may be skipped) -> PreScan (may be skipped) ->
    // Scan

    // The scheduler thread is responsible for moving from NotInLD to WantLD.

    // WantLD triggers moving into PreScan. Once all threads have observed
    // PreScan, then the global state moves to Scan, and thus the last thread
    // skips PreScan and immediately enters Scan.

    // At this point, the scheduler thread is responsible for changing its own
    // state to BelieveDone_Vote when it believes it has finished scanning.
    // BelieveDone_Vote -> BelieveDone_Voted (may be skipped) -> BelieveDone

    // At this point, the scheduler thread is responsible for changing its own
    // state to either BelieveDone_Confirm or BelieveDone_Retract.
    // BelieveDone_Confirm -> BelieveDone_Ack (may be skipped) -> ReallyDone
    // BelieveDone_Retract -> BelieveDone_Ack (may be skipped) -> ReallyDone

    // If all scheduler threads issued BelieveDone_Confirm, then the retracted
    // flag will be false.
    // ReallyDone -> ReallyDone_Confirm -> Sweep -> Finished (may be skipped) ->
    // -> NotInLD (may be skipped)
    // It's possible to go from Finished directly to PreScan if
    // another LD has been started and the thread is the last to find out.

    // If any scheduler thread issued BelieveDone_Retract, then the retracted
    // flag will be true.
    // ReallyDone -> ReallyDone_Retract -> Scan

    enum State
    {
      NotInLD,

      // A scheduler thread can change its own state from NotInLD to WantLD if
      // it would like to begin a Leak Detector Cycle.
      WantLD,

      // At this point, we are about to start scanning cowns to determine
      // reachability.  As the system does not hit a hard barrier, we begin
      // counting work that will need to be addressed later.  In particular, any
      // messages that are sent when in PreScan, may land on a cown that is in
      // the scan state.  We count these messages, so we know once, we have
      // completed them all.
      PreScan,

      // All threads are at least in PreScan, we can proceed with scanning
      // cowns and messages they are processing. Any messages sent from PreScan
      // or earlier must be scanned.
      Scan,

      // All threads are at least in Scan, we can consider terminating the
      // protocol as everyone is now performing this protocol.
      AllInScan,

      // When a scheduler thread has both reached its marker and has not
      // scheduled an unmarked cown, it reports that it believes it is done.
      BelieveDone_Vote,

      // A scheduler thread may be in the BelieveDone_Voted state on the way to
      // being in the BelieveDone state.
      BelieveDone_Voted,

      // Once a scheduler thread is in BelieveDone, it must issue exactly one
      // response of either BelieveDone_Confirm or BelieveDone_Retract.
      // If the global message count is non-zero, then the last thread to enter
      // BelieveDone must retract, so the process can continue.
      BelieveDone,
      BelieveDone_Confirm,
      BelieveDone_Retract,

      // A scheduler thread may be in the BelieveDone_Ack state on the way to
      // being in the ReallyDone state.
      BelieveDone_Ack,

      // Once in the ReallyDone state, a scheduler thread will transition
      // either to ReallyDone_Confirm, in which case it should collect any
      // unmarked cowns and will next transition into NotInLD, or
      // ReallyDone_Retract, in which case it will next transition to Scan,
      // where it should continue the protocol.
      ReallyDone,
      ReallyDone_Confirm,
      ReallyDone_Retract,

      // This LD epoch is now in the sweep phase
      Sweep,

      // This LD epoch has completed.
      Finished,
    };

  private:
#define IDX_BARRIER 0ULL
#define IDX_VOTE 29ULL
#define IDX_STATE 58ULL
#define IDX_RETRACTED 63ULL

#define U_BARRIER (1ULL << IDX_BARRIER)
#define U_VOTE (1ULL << IDX_VOTE)
#define U_RETRACTED (1ULL << IDX_RETRACTED)
#define ADD_THREAD (U_VOTE + U_BARRIER)

#define MASK_BARRIER ((1ULL << (IDX_VOTE)) -1ULL)
#define MASK_VOTE ((((1ULL << IDX_STATE) - 1ULL) >> IDX_VOTE) << IDX_VOTE)
#define MASK_STATE ((((1ULL << (IDX_RETRACTED)) -1ULL) >> IDX_STATE) << IDX_STATE)
#define MASK_RETRACTED (1ULL << IDX_RETRACTED)


#define GET_BARRIER(x) uint64_t(((x) & MASK_BARRIER) >> IDX_BARRIER)
#define GET_VOTE(x) uint64_t(((x) & MASK_VOTE) >> IDX_VOTE)
#define GET_STATE(x) uint64_t(((x) & MASK_STATE) >> IDX_STATE)
#define GET_RETRACTED(x) uint64_t(((x) & MASK_RETRACTED) >> IDX_RETRACTED)

#define SET_BARRIER(x, y) (CLEAR_BARRIER(x) | ((y) << IDX_BARRIER)) 
#define SET_VOTE(x, y) (CLEAR_VOTE(x) | ((y) << IDX_VOTE))
#define SET_STATE(x, y) (CLEAR_STATE(x) | ((y) << IDX_STATE))

#define CLEAR_BARRIER(x) ((x) & ~MASK_BARRIER)
#define CLEAR_VOTE(x) ((x) & ~MASK_VOTE)
#define CLEAR_STATE(x) ((x) & ~MASK_STATE)
#define CLEAR_RETRACTED(x) ((x) & ~MASK_RETRACTED)

    // Barrier_count should not be decremented until the end when threads
    // do not participate any longer in the ld protocol. Is that correct?
    // Here is the layout
    //    63    | 62 ... 58 | 57 ... 29 | 28 ... 0
    // retracted   state      vote        barrier_count 
    std::atomic_uint64_t atomic_state = (uint64_t(NotInLD) << IDX_STATE);

  public:
    constexpr ThreadState() = default;

    State get_state()
    {
      State state = static_cast<State>(GET_STATE(atomic_state));
      return state;
    }

    State next(State s, size_t total_votes)
    {
      switch (GET_STATE(atomic_state))
      {
        case NotInLD:
        {
          switch (s)
          {
            case NotInLD:
            case Finished:
              return NotInLD;

            case WantLD:
            {
              atomic_state = SET_STATE(atomic_state, uint64_t(PreScan));
              return vote_one<PreScan, Scan>(total_votes);
            }

            default:
              abort();
          }
        }

        case PreScan:
        {
          switch (s)
          {
            case NotInLD:
            case WantLD:
            case Finished:
              return vote_one<PreScan, Scan>(total_votes);

            case PreScan:
              return PreScan;

            default:
              abort();
          }
        }

        case Scan:
        {
          switch (s)
          {
            case PreScan:
              return vote<Scan, AllInScan>(total_votes);

            case Scan:
              return Scan;

            default:
              abort();
          }
        }

        case AllInScan:
        {
          switch (s)
          {
            case Scan:
            case AllInScan:
            case ReallyDone_Retract:
              return AllInScan;

            case BelieveDone_Vote:
              return vote<BelieveDone_Voted, BelieveDone>(total_votes);

            case BelieveDone_Voted:
              return BelieveDone_Voted;

            default:
              abort();
          }
        }

        case BelieveDone:
        {
          switch (s)
          {
            case BelieveDone_Voted:
              return BelieveDone;

            case BelieveDone_Confirm:
              return vote<BelieveDone_Ack, ReallyDone>(total_votes);

            case BelieveDone_Retract:
            {
              atomic_state.fetch_or(U_RETRACTED);
              return vote<BelieveDone_Ack, ReallyDone>(total_votes);
            }

            case BelieveDone_Ack:
              return BelieveDone_Ack;

            default:
              abort();
          }
        }

        case ReallyDone:
        {
          switch (s)
          {
            case BelieveDone_Ack:
            case ReallyDone:
            {
              if (atomic_state & U_RETRACTED)
              {
                vote<ReallyDone_Retract, AllInScan>(total_votes);
                return ReallyDone_Retract;
              }

              vote<ReallyDone_Confirm, Sweep>(total_votes);
              return ReallyDone_Confirm;
            }

            case ReallyDone_Confirm:
            case ReallyDone_Retract:
              return s;

            default:
              abort();
          }
        }

        case Sweep:
        {
          switch (s)
          {
            case ReallyDone_Confirm:
              return Sweep;

            case Sweep:
              return vote<Finished, NotInLD>(total_votes);

            case Finished:
              return Finished;

            default:
              abort();
          }
        }

        default:
          abort();
      }
    }

    template<State next>
    void reset()
    {
      if (next == AllInScan)
        atomic_state = CLEAR_RETRACTED(atomic_state);

      bool res = false;
      do
      {
        auto read = atomic_state.load();
        auto val = SET_STATE(read | (GET_BARRIER(read) << IDX_VOTE), uint64_t(next));
        res = atomic_state.compare_exchange_strong(read, val);
      }
      while(!res);
    }

    template<State next>
    void reset_one()
    {
      if (next == AllInScan)
        atomic_state = CLEAR_RETRACTED(atomic_state); 

      bool res = false;
      do
      {
        auto read = atomic_state.load(); 
        assert(GET_VOTE(read) == 0);
        auto val = SET_STATE(read | ((GET_BARRIER(read)-1) << IDX_VOTE), uint64_t(next));
        res = atomic_state.compare_exchange_strong(read, val);
      }
      while(!res);
    }

  private:
    template<State intermediate, State next>
    State vote(size_t total_votes)
    {
      UNUSED(total_votes);
      auto val = atomic_state.fetch_sub(U_VOTE); 
      if (GET_VOTE(val) == 1)
      {
        reset<next>();
        return next;
      }
      return intermediate;
    }

    template<State intermediate, State next>
    State vote_one(size_t total_votes)
    {
      UNUSED(total_votes);
      auto val = atomic_state.fetch_sub(U_VOTE);
      if (GET_VOTE(val) == 1)
      {
        reset_one<next>();
        return next;
      }
      return intermediate;
    }
  public:
    void set_barrier(size_t thread_count)
    {
      atomic_state = SET_BARRIER(atomic_state, thread_count);
      atomic_state = SET_VOTE(atomic_state, thread_count);
    }

    bool add_thread()
    {
      auto value = atomic_state.load();
      if (GET_STATE(value) >= State::Scan)
        return false;
      auto update = value + ADD_THREAD;
      return atomic_state.compare_exchange_strong(value, update);
    }

    uint64_t exit_thread()
    {
      return GET_BARRIER(atomic_state.fetch_sub(U_BARRIER));
    }

    uint64_t barrier_count()
    {
      return GET_BARRIER(atomic_state);
    }
  };



  inline std::ostream&
  operator<<(std::ostream& os, const ThreadState::State& obj)
  {
    os.width(20);
    os << std::left;
    switch (obj)
    {
      case ThreadState::NotInLD:
        os << "NotInLD";
        break;
      case ThreadState::WantLD:
        os << "WantLD";
        break;
      case ThreadState::PreScan:
        os << "PreScan";
        break;
      case ThreadState::Scan:
        os << "Scan";
        break;
      case ThreadState::AllInScan:
        os << "AllInScan";
        break;
      case ThreadState::BelieveDone_Vote:
        os << "BelieveDone_Vote";
        break;
      case ThreadState::BelieveDone_Voted:
        os << "BelieveDone_Voted";
        break;
      case ThreadState::BelieveDone:
        os << "BelieveDone";
        break;
      case ThreadState::BelieveDone_Confirm:
        os << "BelieveDone_Confirm";
        break;
      case ThreadState::BelieveDone_Retract:
        os << "BelieveDone_Retract";
        break;
      case ThreadState::BelieveDone_Ack:
        os << "BelieveDone_Ack";
        break;
      case ThreadState::Finished:
        os << "Finished";
        break;
      case ThreadState::ReallyDone:
        os << "ReallyDone";
        break;
      case ThreadState::ReallyDone_Confirm:
        os << "ReallyDone_Confirm";
        break;
      case ThreadState::ReallyDone_Retract:
        os << "ReallyDone_Retract";
        break;
      case ThreadState::Sweep:
        os << "Sweep";
        break;
    }
    return os;
  }
} // namespace verona::rt
