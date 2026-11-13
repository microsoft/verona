#pragma once

#include "drag.h"
#include "location.h"
#include "region.h"
#include "register.h"
#include "thread.h"

namespace vbci::writebarrier
{
  template<bool is_move>
  struct write_ops
  {
    Location store_loc = Location::immortal();
    Location out_loc = Location::immortal();
    Location in_loc = Location::immortal();

    bool ok = true;

    // Operations on the incoming value.
    bool need_in_drag = false;
    Region* in_drag_value = nullptr;
    bool need_in_inc = false;
    bool need_in_clear = false;
    int8_t need_in_stack_inc = 0;
    bool need_in_parent = false;
    Region* in_parent_value = nullptr;

    // Operations on the outgoing value.
    bool need_out_dec = false;
    int8_t need_out_stack_inc = 0;
    bool need_out_clear_parent = false;
    Region* out_clear_parent_region = nullptr;

    write_ops& prepare_store(Location loc)
    {
      store_loc = loc;

      // Can't store anything to an immutable or immortal location.
      if (store_loc.is_immutable() || store_loc.is_immortal())
        return fail();

      return *this;
    }

    write_ops& prepare_out(Location loc)
    {
      out_loc = loc;

      if (!ok)
        return *this;

      if (out_loc.is_region())
      {
        // If storage wasn't in the stack or a frame-local region, add a stack
        // RC.
        if (
          !store_loc.is_stack() &&
          !(store_loc.is_region() && store_loc.to_region()->is_frame_local()))
          out_stack_inc();

        // If out was in a different region, unparent it.
        if (store_loc != out_loc)
          out_clear_parent();
      }

      return *this;
    }

    write_ops& prepare_in(Location loc)
    {
      in_loc = loc;

      if (!ok)
        return *this;

      // Can store an immortal value anywhere.
      if (in_loc.is_immortal())
        return *this;

      if (in_loc.is_stack())
      {
        // Not even a frame-local region can reference the stack.
        if (!store_loc.is_stack())
          return fail();

        // Incoming value must outlive the storage location.
        if (store_loc.stack_index() < in_loc.stack_index())
          return fail();

        // We don't care if this is a move or not.
        return *this;
      }

      // Either clear or RC inc the incoming value.
      if constexpr (is_move)
        in_clear();
      else
        in_inc();

      // Can store an immutable value anywhere.
      if (in_loc.is_immutable())
        return *this;

      assert(in_loc.is_region());
      auto in_r = in_loc.to_region();

      if (in_r->is_frame_local())
      {
        if (store_loc.is_stack())
        {
          // Frame-local stored to stack: drag if the frame is younger
          // than the stack location.
          if (store_loc.stack_index() < in_r->get_frame_depth())
            in_drag(Thread::frame_region_for_stack(store_loc));

          return *this;
        }

        assert(store_loc.is_region());
        auto store_r = store_loc.to_region();

        if (store_r->is_frame_local())
        {
          // Frame-local to frame-local: drag if incoming is younger.
          if (store_r->get_frame_depth() < in_r->get_frame_depth())
            in_drag(store_r);
        }
        else
        {
          // Frame-local to heap region: drag to the storage region.
          in_drag(store_r);
        }

        return *this;
      }

      // Non-frame-local region.

      // If we're moving into a region, remove a stack RC.
      if constexpr (is_move)
        in_stack_dec();

      // If we're putting this into a stack or frame-local location,
      // add a stack RC.
      if (store_loc.is_stack())
        return in_stack_inc();

      if (store_loc.is_region() && store_loc.to_region()->is_frame_local())
        return in_stack_inc();

      assert(store_loc.is_region());

      // If we're putting this into the same region, we're done.
      if (store_loc == in_loc)
        return *this;

      // Regions can only have a single entry point.
      if (in_r->has_owner())
      {
        // If in and out are in the same sub-region, it's fine.
        if (out_loc == in_loc)
          return *this;

        return fail();
      }

      // Regions can't form cycles.
      auto storage_r = store_loc.to_region();

      if (in_r->is_ancestor_of(storage_r))
        return fail();

      // Set the parent.
      return in_parent(storage_r);
    }

    bool apply_in(void* addr, ValueType t, Reg<is_move> in)
    {
      assert(ok);

      // Save the header for potential reparenting (before the register is
      // cleared during a move).
      Header* in_header = need_in_parent ? in->get_base_header() : nullptr;

      // Drag the incoming value.
      if (need_in_drag)
      {
        assert(!need_in_parent);

        if (!drag_allocation<is_move>(
              in_drag_value, in->get_base_header(), &out_clear_parent_region))
          return false;

        // If we drag a sub-region reference to the same region as the outgoing
        // value, we won't need to clear the out parent.
        if (!out_clear_parent_region)
          need_out_clear_parent = false;
      }

      // Write to the target address.
      in->to_addr(t, addr);

      if (out_loc == in_loc)
      {
        // Combine stack inc/dec.
        need_in_stack_inc += need_out_stack_inc;
        need_out_stack_inc = 0;
      }

      if constexpr (is_move)
      {
        assert(!need_in_inc);

        // Clear the incoming register.
        if (need_in_clear)
          in.clear_unsafe();
      }
      else
      {
        assert(!need_in_clear);

        // Increment the incoming value RC.
        if (need_in_inc)
          in->field_inc();
      }

      // Reparent the region before stack_dec. If both are needed, parenting
      // must happen first — stack_dec to 0 with no owner would free the
      // region prematurely.
      if (need_in_parent)
      {
        assert(!need_in_drag);
        assert(in_header);
        in_loc.to_region()->set_parent(in_parent_value, in_header);
      }

      // Change the region stack RC.
      assert(need_in_stack_inc >= -1);
      assert(need_in_stack_inc <= 1);

      if (need_in_stack_inc == 1)
        in_loc.to_region()->stack_inc();
      else if (need_in_stack_inc == -1)
        in_loc.to_region()->stack_dec();

      return true;
    }

    void apply_out(Value& out)
    {
      assert(ok);

      if (need_out_dec)
        out.field_dec();

      assert(need_out_stack_inc >= -1);
      assert(need_out_stack_inc <= 1);

      if (need_out_stack_inc == 1)
        out_loc.to_region()->stack_inc();
      else if (need_out_stack_inc == -1)
        out_loc.to_region()->stack_dec();

      if (need_out_clear_parent)
        out.location().to_region()->clear_parent();
    }

    write_ops& fail()
    {
      ok = false;
      return *this;
    }

    write_ops& in_inc()
    {
      need_in_inc = true;
      return *this;
    }

    write_ops& in_clear()
    {
      need_in_clear = true;
      return *this;
    }

    write_ops& in_stack_inc()
    {
      need_in_stack_inc++;
      return *this;
    }

    write_ops& in_stack_dec()
    {
      need_in_stack_inc--;
      return *this;
    }

    write_ops& in_drag(Region* r)
    {
      need_in_drag = true;
      in_drag_value = r;
      return *this;
    }

    write_ops& in_parent(Region* parent)
    {
      need_in_parent = true;
      in_parent_value = parent;
      return *this;
    }

    write_ops& out_dec()
    {
      need_out_dec = true;
      return *this;
    }

    write_ops& out_stack_inc()
    {
      need_out_stack_inc++;
      return *this;
    }

    write_ops& out_stack_dec()
    {
      need_out_stack_inc--;
      return *this;
    }

    write_ops& out_clear_parent()
    {
      need_out_clear_parent = true;
      out_clear_parent_region = out_loc.to_region();
      return *this;
    }
  };

  inline void init(Location store_loc, void* addr, ValueType t, Register next)
  {
    if (next->is_readonly())
      Value::error(Error::BadStore);

    auto ops =
      write_ops<true>().prepare_store(store_loc).prepare_in(next->location());

    if (!ops.ok)
      Value::error(Error::BadAllocTarget);

    if (!ops.apply_in(addr, t, std::forward<Register>(next)))
      Value::error(Error::BadStore);
  }

  template<bool is_move>
  ValueTransfer
  exchange(Location store_loc, void* addr, ValueType t, Reg<is_move> next)
  {
    auto prev = Value::from_addr(t, addr);
    auto ops = write_ops<is_move>()
                 .prepare_store(store_loc)
                 .prepare_out(prev.location())
                 .prepare_in(next->location());

    if (!ops.ok)
      Value::error(Error::BadStore);

    if (!ops.apply_in(addr, t, (std::forward<Reg<is_move>>(next))))
      Value::error(Error::BadStore);

    ops.apply_out(prev);
    return ValueTransfer(prev);
  }

  inline void drop(Location store_loc, Value& v)
  {
    auto loc = v.location();

    // Do nothing.
    if (loc.is_immortal() || loc.is_stack())
      return;

    // RC dec, no region operation.
    if (loc.is_immutable())
    {
      v.field_dec();
      return;
    }

    // Edges within a frozen SCC have already been removed from its ARC.
    // collect_scc marks all members pending before finalizing their fields.
    if (loc.is_pending())
      return;

    assert(loc.is_region());
    auto r = loc.to_region();

    // Frame-local regions skip stack RC accounting.
    if (r->is_frame_local())
    {
      v.field_dec();
      return;
    }

    if (store_loc.is_stack())
    {
      r->stack_dec();
      return;
    }

    if (store_loc.is_region() && store_loc.to_region()->is_frame_local())
    {
      r->stack_dec();
      return;
    }

    // Internal edge within the same region — no parent to clear.
    // Still decrement field RC so individual object collection cascades.
    if (store_loc == loc)
    {
      v.field_dec();
      return;
    }

    // Clear parent if the region has one (sub-region relationship).
    if (r->has_parent())
    {
      r->clear_parent();
      return;
    }

    // Different region, no parent — just decrement the value's RC.
    v.field_dec();
  }
}
