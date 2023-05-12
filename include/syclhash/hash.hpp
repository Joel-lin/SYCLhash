#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>
//#include <syclhash/alloc.hpp>

namespace syclhash {

template <typename, int, sycl::access::mode, sycl::access::target>
class DeviceHash;

/** Hash table. Store key:value pairs, where key is Ptr type.
 *
 * @tparam T type of values held in the table.
 * @tparam search_width 2**width = number of consecutive keys to scan linearly
 */
template <typename T, int search_width=4>
class Hash {
    template <typename, int, sycl::access::mode, sycl::access::target>
    friend class DeviceHash;

    static_assert(search_width < 6, "Width too large!");

    //Alloc                     alloc;
    sycl::buffer<T,1>         cell;
    sycl::buffer<Ptr,1>       keys; // key for each cell
    int size_expt;            ///< Base-2 log of the max hash-table size.

    /// Reset just keys and next pointers (sufficient for new allocator).
    void reset_k(sycl::queue &queue) {
        queue.submit([&](sycl::handler &cgh){
            sycl::accessor K{keys, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(K, null_ptr);
        });
    }

  public:

    static const int width = search_width;

    /** Allocate and initialize space for max 2**size_expt items.
     *
     * @arg size_expt exponent of the allocator's size
     * @arg queue SYCL queue to use when initializing free_list
     */
    Hash(int size_expt, sycl::queue &queue)
        : //alloc(size_expt, queue),
          cell(1 << size_expt)
        , keys(1 << size_expt)
        , size_expt(size_expt) {

        if(size_expt >= 32) {
            throw std::invalid_argument("2**size_expt is too large.");
        }

        reset_k(queue);
    }

    /** Reset this structure to empty.
     */
    void reset(sycl::queue &queue) {
        //alloc.reset(queue);
        reset_k(queue);
    }
};

enum class Step {
    Stop = 7,
    Continue,
    Complete,
};

struct SingletGroup;

template<typename Group, typename DeviceHashT>
class PreBucket;

/** A Bucket points to the set of values in the hash
 * table with the given key.
 *
 * The real work is done with iterators, created
 * through the bucket's begin() and end() calls.
 *
 * Example:
 *
 *     auto bucket = dh[g][key];
 *     for(const auto val : bucket) {
 *         printf("%lu %lu\n", key, val);
 *     } printf("\n");
 *
 * @tparam T type of values held in the Bucket
 * @tparam search_width number of bins to search linearly between random jumps
 * @tparam Mode accessor mode for interacting with the bucket
 * @tparam accessTarget where memory accessors will live
 *
 */
template<typename Group_, typename DeviceHashT_>
class Bucket {
    const DeviceHashT_ &dh;
    using T = typename DeviceHashT_::value_type;
    friend class PreBucket<Group_,DeviceHashT_>;

  public:
    using Group = Group_;
    using DeviceHashT = DeviceHashT_;
    using value_type = typename DeviceHashT_::value_type;
    const Group group;
    const Ptr key;
    //static constexpr sycl::access::target Target = accessTarget;

    /** Construct the Bucket for `dh`
     * that points at `key`.
     */
    Bucket(Group grp, const DeviceHashT &dh, Ptr key)
        : dh(dh), group(grp), key(key) {}

    /** A cursor pointing at a specific cell in the Bucket.
     */
    class iterator {
        friend class Bucket<Group,DeviceHashT>;

        const DeviceHashT *dh;
        const Group grp;
        Ptr key;
        Ptr index;
    
        /** Only friends can construct iterators.
         */
        iterator(Group g, const DeviceHashT *dh, Ptr key, Ptr index)
                : dh(dh), grp(g), key(key), index(index) {
            seek(false);
        }

        /** Seek forward to the next valid index --
         * where dh->keys[index] == key
         * or index == null_ptr
         *
         * if fwd == true, then seek will start by advancing
         * to next(index)
         */
        void seek(bool fwd) {
            if(index == null_ptr) return;
            const Ptr i0 = index;
            if( ! dh->run_op(grp, key, index, true, [=](Ptr i1, Ptr k1){
#               ifdef DEBUG_SYCLHASH
                printf("seeking for %u from %u (found %u at %u)\n", key, i0, k1, i1);
#               endif
                if(k1 == null_ptr)
                    return Step::Stop;
                if(fwd && i1 == i0) return Step::Continue;
                if(key == k1)
                    return Step::Complete;
                return Step::Continue;
            }) ) {
                index = null_ptr;
            }
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = size_t;

        using value_type = std::conditional_t<
                                DeviceHashT::Mode == sycl::access::mode::read,
                       const T, T >;
        using reference = std::conditional_t<
                                DeviceHashT::Mode == sycl::access::mode::read,
                       const T&, T& >;
        using pointer = std::conditional_t<
                                DeviceHashT::Mode == sycl::access::mode::read,
                       const T*, T* >;

        iterator(const iterator &x)
            : dh(x.dh), grp(x.grp), key(x.key), index(x.index) {}
        iterator &operator=(const iterator &x) {
            dh = x.dh;
            key = x.key;
            //grp = x.grp;
            index = x.index;
            return *this;
        }

        // Is this cursor over an empty cell?
        // (due to potential parallel acccess,
        //  this function would not be stable)
        //bool is_empty() {
        //    return dh->keys[index] == null_ptr;
        //}

        /// Is this cursor a null-pointer?
        bool is_null() const {
            return index == null_ptr;
        }

        /// Access the value this iterator refers to.
        reference operator *() const {
            return dh->get_cell(index);
        }

        /// Erase the key:value pair this iterator refers to.
        bool erase() {
            bool ret;
            //return apply_leader<bool>(grp, dh->erase_key, index, key);
            apply_leader(ret, grp, dh->erase_key(index, key));
            return ret;
        }

        /// pre-increment
        iterator &operator++() {
            if(index != null_ptr) {
                seek(true);
            }
            return *this;
        }
        /// post-increment
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator== (const iterator& b) const {
            return index == b.index;
        };
        bool operator!= (const iterator& b) const {
            return index != b.index;
        };
    };

    /** Create an iterator pointing to the first cell
     *  contained in this Bucket.
     *
     * @arg grp collective group that will access the iterator
     */
    iterator begin() const {
        return iterator(group, &dh, key, dh.mod(key));
    }

    /** Create an iterator pointing to the end
     *  of this Bucket.
     *
     * @arg grp collective group that will access the iterator
     */
    iterator end() const {
        return iterator(group, &dh, key, null_ptr);
    }

    /** Put a new value into this bucket.
     *
     *  This is implemented via a call to DeviceHash::set_key,
     *  which acquires the key (replacing it with `reserved`),
     *  sets the value, then releases the key (replacing it with `key`).
     *
     *  @return iterator where value was successfully placed,
     *          or end(grp) on failure.
     *
     *  .. note::
     *
     *      The bucket is unordered, so there are no guarantees
     *      as to what relative location the value will be inserted.
     *
     *  .. note::
     *
     *      If value read/write and key deletion are happening concurrently
     *      by different threads, there is a chance that previous
     *      threads accessing the cell may still be using it
     *      (if they are unaware that someone had deleted it).
     */
    template <bool use=true,
			    std::enable_if_t< use &&
                    DeviceHashT::Mode != sycl::access::mode::read, bool> = true
			  , typename ...Args>
    iterator insert(Args ... args) const {
        Ptr index = dh.insert_cell(group, key, false, args...);
        return iterator(group, &dh, key, index);
    }

    template <bool use=true,
			    std::enable_if_t< use &&
                    DeviceHashT::Mode != sycl::access::mode::read, bool> = true
			 , typename ...Args>
    iterator insert_unique(Args ... args) const {
        Ptr index = dh.insert_cell(group, key, true, args...);
        return iterator(group, &dh, key, index);
    }

    /** Return true if index was deleted, false if it was not present.
     *
     *  Internally, we set the `key` to `null_ptr`
     *  for deleted keys. These are skipped over
     *  during bucket iteration.
     *
     *  This happens via a call to DeviceHash::erase_key, which
     *  has release memory-ordering semantics.
     *
     *  Regardless of the output of this call,
     *  your cursor is now invalid.
     *
     * .. warning::
     * 
     *     If value accesses, inserting and erasing are all happening
     *     concurrently by different threads, it is up to you to stop other
     *     readers & writers of the cell's value from accessing it.
     *     You must ensure this before you delete it!
     *     Those other potential accessors include everyone with
     *     an iterator pointing at this same position (since
     *     the iterator can be dereferenced).
     *
     *     Otherwise, you run the risk that the cell may be allocated
     *     again (potentially with a different key), and written
     *     into.
     *
     *     Technically, erasing without any concurrent insertions
     *     would leave the value dedicated, but no longer referenced.
     */
    template <bool use=true,
			    std::enable_if_t< use &&
              DeviceHashT::Mode != sycl::access::mode::read, bool> = true>
    bool erase(iterator position) const {
        return position.erase();
    }
};

/** A Pre-Bucket just stores the group which will be used
 * to access a particular bucket.
 *
 * @tparam Group type of the group (SingletGroup,
 *                                  sycl::group, or sycl::sub_group)
 *
 */
template<typename Group_, typename DeviceHashT_>
class PreBucket {
    using T = typename DeviceHashT_::value_type;
  public:
    using Group = Group_;
    using DeviceHashT = DeviceHashT_;
    using BucketT = Bucket<Group,DeviceHashT>;
    // TODO ensure only allowed group/device combinations:
    //   static_assert(std::is_same_t<Group,SingletGroup> ^ (DeviceHashT::accessTarget==HostDevice));

    const Group group;
    const DeviceHashT &dh;
    PreBucket(const DeviceHashT &dhash, const Group grp)
        : dh(dhash), group(grp) {}

    BucketT operator[](Ptr key) const {
        return BucketT(group, dh, key);
    }

    /** Convenience function to insert `value`
     * to the :class:`Bucket` at the given `key`
     *
     * @arg uniq true if keys are uniq (ignores value
     *                if key is found)
     */
    typename BucketT::iterator
    insert(Ptr key, const T&value, bool uniq) const {
        BucketT bucket(group, dh, key);
        return bucket.insert(value, uniq);
    }
};


/** Device side data structure for hash table.
 *
 * Max capacity = 1<<size_expt key/value pairs.
 *
 * Requires size_expt < 32
 *
 * @tparam T type of values held in the table.
 * @tparam Mode access mode for DeviceHash
 */
template <typename T,
          int search_width,
          sycl::access::mode accessMode,
          sycl::access::target accessTarget
              = sycl::access::target::global_buffer>
class DeviceHash {
    template <typename,int, sycl::access::mode, sycl::access::target>
    friend class DeviceHash;

    sycl::accessor<Ptr, 1, accessMode, accessTarget> keys;  // key for each cell
    sycl::accessor<T, 1, accessMode, accessTarget>   cell;

    //const DeviceAlloc<accessMode,accessTarget> alloc;

    /*  Attempt to reserve the key at index (by overwriting `was`
     *  with reserved using relaxed semantics)
     *
     *  Sets `was` to the old value of the key on return
     *  @return true if successful, false if no change
     */
    bool reserve_key(Ptr index, Ptr &was, Ptr key) const {
        ADDRESS_CHECK(index, size_expt);
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        if(!v.compare_exchange_weak(was, reserved)) {
                            //sycl::memory_order::acquire))
            return false;
        }
        return true;
    }

  public:
    using DeviceHashT = DeviceHash<T,search_width,accessMode,accessTarget>;
    using value_type = T;

    static constexpr sycl::access::mode Mode = accessMode;
    static constexpr sycl::access::target Target = accessTarget;
    static constexpr int width = search_width;
    const int size_expt;

    /** Construct from the host Hash class.
     *
     * @arg h host hash class
     * @arg cgh SYCL handler
     */
    DeviceHash(Hash<T,search_width> &h, sycl::handler &cgh)
        : keys(h.keys, cgh)
        , cell(h.cell, cgh)
        //, alloc(h.alloc, cgh)
        , size_expt(h.size_expt)
        //, count(h.cell.get_count()) // max capacity
        { }

    template <typename Device>
    DeviceHash(Hash<T,search_width> &h, sycl::handler &cgh, const Device &)
        : DeviceHash(h, cgh) { }

    template <bool use=true, std::enable_if_t< use &&
              accessTarget == sycl::access::target::host_buffer,bool> = true>
    DeviceHash(Hash<T,search_width> &h)
        : keys(h.keys)
        , cell(h.cell)
        , size_expt(h.size_expt)
        { }

    /** Increment index in a pseudo-random way
     *
     * Implementation note: Consumers of this function
     * require it to make progress even for accum == 0.
     *
     * This is a simple congruential generator with
     * the property that it is full-period for any
     * power of 2 modulus.
     */
    uint32_t next_hash(uint32_t seed) const {
        const uint32_t a = 1664525;
        const uint32_t c = 1; //1013904223;
        seed = (uint32_t) ( (uint64_t)(a)*(uint64_t)mod_w(seed>>search_width) + c );
        return mod_w(seed) << search_width;
    }

    /** Apply the function to every key,value pair.
     *  Each key is processed by a whole group.
     *
     *  The function should have type:
     *
     *  void fn(sycl::nd_item<Dim> it, Ptr key, T &value, Args ... args);
     *
     *  A generic nd_range<1>(1024, 32) is recommended for looping
     *  over its key-space.
     */
    template <int Dim, typename Fn, typename ...Args>
    void parallel_for(sycl::handler &cgh,
                      sycl::nd_range<Dim> rng,
                      Fn fn, Args ... args) const {
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng, [=, keys=this->keys, cell=this->cell]
                (sycl::nd_item<Dim> it) {
            sycl::group<Dim> g = it.get_group();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += g.get_group_linear_range()) {
                //Ptr key = sycl::group_broadcast(g, keys[i], 0);
                Ptr key = keys[i];
                if((key>>31) & 1) continue;
                fn(it, key, cell[i], args ...);
            }
        });
    }

    /** Apply the function to every key,value pair.
     *  Each key is processed by a whole group.
     *
     *  The function should have type:
     *
     *  R fn(sycl::nd_item<Dim> it, Ptr key, T &value);
     *
     *  A generic nd_range<1>(1024, 32) is recommended for looping
     *  over its key-space.
     */
    template <typename R, int Dim, typename Fn, typename ...Args>
    void parallel_for(sycl::handler &cgh,
                      sycl::nd_range<Dim> rng,
                      sycl::buffer<R,1> &ret,
                      Fn fn, Args ... args) const {
        //sycl::accessor<R, 1>  d_ret(ret, cgh, sycl::read_write);
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng,
                sycl::reduction(ret, cgh, sycl::plus<R>()),
                [=, keys=this->keys, cell=this->cell]
                (sycl::nd_item<Dim> it, auto &ans) {
            sycl::group<Dim> g = it.get_group();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += g.get_group_linear_range()) {
                //Ptr key = sycl::group_broadcast(g, keys[i], 0);
                Ptr key = keys[i];
                if((key>>31) & 1) continue;
                R tmp = fn(it, key, cell[i]); //, args ...);
                ans += tmp;
            }
        });
    }

    template <typename U, int Dim, sycl::access::mode Mode2, typename Fn,
              typename ...Args>
    void map(sycl::handler &cgh,
             sycl::nd_range<Dim> rng,
             const DeviceHash<U, search_width, Mode2, accessTarget> &out,
             Fn fn, Args ... args) const {
        sycl::accessor<T, 1, Mode>    cell(this->cell);
        sycl::accessor<U, 1, Mode2>   cell2(out.cell);
        sycl::accessor<Ptr, 1, Mode>  keys(this->keys);
        sycl::accessor<Ptr, 1, Mode2> keys2(out.keys);
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng,
                [=](sycl::nd_item<Dim> it) {
            sycl::group<Dim> g = it.get_group();
            const int ngrp = g.get_group_linear_range();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += ngrp) {
                Ptr key = keys[i];
                keys2[i] = key;
                if((key>>31) & 1) continue;
                fn(it, key, cell[i], cell2[i], args ...);
            }
        });
    }

    /** Wrap the index into the valid range, [0, 2**size_expt).
     */
    Ptr mod(Ptr index) const {
        return index & ((1<<size_expt) - 1);
    }

    /** Wrap the super-index into the valid range, [0, 2**(size_expt-width)).
     */
    Ptr mod_w(Ptr si) const {
        return si & ((1<<(size_expt-search_width)) - 1);
    }

    /// bucket = (key % N) is the first index.
    template <typename Group>
    PreBucket<Group,DeviceHashT> operator[](Group g) const {
        return PreBucket<Group,DeviceHashT>(*this, g);
    }

    /** Low-level function inserting a key,value pair
     *  and returning the Ptr index.
     *
     *  @arg args value to set on insert -- either value or empty
     *  @returns Ptr index where insertion took place / key was found
     *           or null_ptr if table is full
     */

    //template <typename Group, typename ...Args>
    template <typename Group, bool use=true,
			    std::enable_if_t< use &&
                    Mode != sycl::access::mode::read, bool> = true
			   , typename ...Args>
    Ptr insert_cell(Group g, Ptr key, bool uniq, Args ... args) const {
        Ptr index = mod(key);
        if(! run_op(g, key, index, uniq, [=](Ptr i1, Ptr k1) {
#           ifdef DEBUG_SYCLHASH
            printf("inserting %u at %u (found %u)\n", key, i1, k1);
#           endif
            // Note: we can't over-write erased keys
            // in uniq mode (since insert_uniq k2, delete k2, insert_uniq k1)
            // when k2 collides with k1 would leave an erased slot in front.
            while(k1 == null_ptr || (!uniq && k1 == erased)) {
                if(set_key(i1, k1, key, args...)) {
                    return Step::Complete;
                }
#               ifdef DEBUG_SYCLHASH
                printf("failed set_key (found %u)\n", k1);
#               endif
            }
            if(uniq && k1 == key) {
                return Step::Complete;
            }
            return Step::Continue;
        })) {
            return null_ptr;
        }
        return index;
    }

    /** Run through the `canonical` sequence of keys,
     * searching for null_ptr, deleted, or
     * (if uniq == true) key.
     *
     * If found, runs `Step op(Ptr index, Ptr key)`
     * on the index,keys[index] pair.
     *
     * That returns false of `Stop`, true on `Complete`, or,
     * on `Continue`, continues searching until all keys are exhausted,
     * then return false.
     */
    template <typename Group, typename Fn>
    bool run_op(Group g,
                Ptr key,
                Ptr &index,
                bool uniq,
                Fn op) const {
        Ptr i0 = (index >> search_width) << search_width; // align reads

        uint32_t cap = (1 << (size_expt-search_width)) + (i0 != index);

        // It would be better if we could fix the group size
        // to (1<<width).  However, this setup mimicks
        // that by disabling threads past (1<<width).
        const int tid = g.get_local_linear_id();
        const int ntid = g.get_local_linear_range();
        // Max number of usable threads in this group.
        const int ngrp = ntid < (1<<search_width)
                       ? ntid : (1<<search_width);
		// This must be a multiple of ntid so that all threads
		// will call ballot (preventing deadlock).
        const int max_sz = //ntid * ceil( (1<<width)/ntid)
                           ntid*( ((1<<search_width)+ntid-1)/ntid );
        for(int trials = 0
           ; trials < cap
           ; ++trials, i0 = next_hash(i0)) {
            // read next `2**width` keys
            for(int i = tid; i < max_sz; i += ngrp) {
                bool check = false;
                Ptr ahead;
                //if(i0+i < (1<<size_expt))
                if(i < (1<<search_width)) {
                    ahead = keys[i0+i];
                    check = ahead == null_ptr
                          || ahead == erased
                          || (uniq && ahead == key);
                    // ensure we move forward of index on trial 0
                    if(trials == 0) check = check && i0+i >= index;
                }
                warpMaskT mask = ballot(g, check);

                for(int winner=0; (mask>>winner) > 0; winner++) {
                    if(((mask>>winner)&1) == 0) continue;
                    Ptr found = sycl::group_broadcast(g, ahead, winner);

                    int idx = i0+i-tid+winner;
                    Step step;
                    apply_leader(step, g, op(idx, found));
                    switch(step) {
                    case Step::Stop:
                        index = idx;
                        return false;
                    case Step::Complete:
                        index = idx;
                        return true;
                    case Step::Continue:
                        break;
                    }
                }
            }
        }
        return false;
    }

    /// Low-level function used to read a cell value using a Ptr index.
    std::conditional_t<Mode == sycl::access::mode::read,
                       const T&, T& >
    get_cell(Ptr loc) const {
        ADDRESS_CHECK(loc, size_expt);
        return cell[loc];
    }

    /** Where key was null_ptr, set to key
     *  and fill its value atomically.
     *
     * @arg index hash-table index where key is set
     * @return true if set is successful, false otherwise
     */
    template <bool use=true, std::enable_if_t< use &&
			  Mode != sycl::access::mode::read, bool> = true>
    bool set_key(Ptr index, Ptr &was, Ptr key, const T &value) const {
        ADDRESS_CHECK(index, size_expt);
        if(!reserve_key(index, was, key)) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = reserved;
        cell[index] = value;
        while(!v.compare_exchange_strong(val, key,
                            sycl::memory_order::release)) {
            // this is an error
        };
        return true;
    }

    template <bool use=true, std::enable_if_t< use &&
			  Mode != sycl::access::mode::read, bool> = true>
    bool set_key(Ptr index, Ptr &was, Ptr key) const {
        ADDRESS_CHECK(index, size_expt);
        if(!reserve_key(index, was, key)) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = reserved;
        while(!v.compare_exchange_strong(val, key)) {
            // this is an error
        };
        return true;
    }

    /*  Erase the key at index (by overwriting key with `erased`)
     *
     *  @return true if successful, false if no change
     */
    bool erase_key(Ptr index, Ptr key) const {
        if(index == null_ptr) return false;
        ADDRESS_CHECK(index, size_expt);
#       ifdef DEBUG_SYCLHASH
        printf("Erasing %u at %u\n", key, index);
#       endif
        auto v = sycl::atomic_ref<
                            Ptr,
							//sycl::memory_order::release,
							sycl::memory_order::acq_rel,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = key;
        return v.compare_exchange_strong(val, erased);
    }
};

template <typename T, int search_width, sycl::access::mode Mode>
class HostHash : public DeviceHash<T, search_width, Mode,
                                   sycl::access::target::host_buffer>
{
  public:
    template <typename ...Args>
    HostHash(Hash<T,search_width> &hash, Args... deduction_helpers)
        : DeviceHash<T,search_width,Mode,sycl::access::target::host_buffer>(hash)
        {}
};

/* SYCL doesn't re-export the template params (access or mode) from an accessor.
template <typename Buffer, typename Descriptor...>
using AccessMode = typename std::invoke_result_t<sycl::accessor, Buffer, sycl::handler&, Descriptor...>::AccessMode; // C++17
*/

/* SYCL doesn't guarantee the descriptor type can be used either
template<typename T, int width, class Descriptor>
HostHash(Hash<T,width>&, Descriptor)
    -> HostHash<T, width, ::mode>;
*/

/* So, we have to re-implement SYCL's logic to determine
 * mode tags.  However, these don't have unique types,
 * so the mode tag can't be determined at compile time.
 * Nevertheless, the SYCL implementation does determine it
 * at compile time.  Hence, they must have unique types, despite
 * the incomplete specification:
inline constexpr __unspecified__ read_only;
inline constexpr __unspecified__ read_write;
inline constexpr __unspecified__ write_only;
inline constexpr __unspecified__ read_only_host_task;
inline constexpr __unspecified__ read_write_host_task;
inline constexpr __unspecified__ write_only_host_task;


|  Tag value  | Access mode        |   Accessor target  |
| ------      | ----------------   | ------------------ |
| read_write  | access_mode::read_write | target::device |
| read_only   | access_mode::read       | target::device |
| write_only  | access_mode::write      | target::device |
| read_write_host_task | access_mode::read_write | target::host_task |
| read_only_host_task  | access_mode::read | target::host_task |
| write_only_host_task | access_mode::write | target::host_task |
*/
/*
constexpr sycl::access_mode AccessMode(auto x) {
	return x == sycl::read_only ?
		sycl::access_mode::read
		: (sycl::access_mode::read_write);
};*/

/*
template <typename Descriptor>
using AccessMode_t = typename
    std::conditional_t<            std::is_same_v<Descriptor, decltype(sycl::read_only)>,
		decltype(sycl::access_mode::read),
		std::conditional_t<        std::is_same_v<Descriptor, decltype(sycl::read_write)>,
		    decltype(sycl::access_mode::read_write),
			std::conditional_t<    std::is_same_v<Descriptor, decltype(sycl::write_only)>,
			    decltype(sycl::access_mode::write),
				void
			>
		>
	>;
*/

/*
template <typename Descriptor>
constexpr sycl::access_mode AccessMode_t() {
	return sycl::access_mode::read;
}
template<>
constexpr sycl::access_mode AccessMode_t<decltype(sycl::read_only)>() {
	return sycl::access_mode::read;
}
template<>
constexpr sycl::access_mode AccessMode_t<decltype(sycl::read_write)>() {
	return sycl::access_mode::read_write;
}
template<>
constexpr sycl::access_mode AccessMode_t<decltype(sycl::write_only)>() {
	return sycl::access_mode::write;
}*/

// Cheat, and retrieve the argument from the underlying Descriptor.
// "When you can snatch the enum argument from my templated type,
//  it will be time for you to leave."
template <sycl::access_mode mode, template<sycl::access_mode> class ctorType>
constexpr sycl::access_mode AccessMode_t(ctorType<mode>) {
	return mode;
}

template<typename T, int width, class Descriptor>
DeviceHash(Hash<T,width>&,
           sycl::handler&,
           Descriptor d)
    -> DeviceHash<T, width, AccessMode_t(d), sycl::target::device >;
    //-> DeviceHash<T, width, Descriptor::mode, Descriptor::target>;

template<typename T, int width, class Descriptor>
HostHash(Hash<T,width>&, Descriptor d)
    -> HostHash<T, width, AccessMode_t(d) >;
}
