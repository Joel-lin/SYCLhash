
SYCL hash is a lock-free concurrent hash
table written in the SYCL programming model.

Buckets are implemented using a linked-cell
list data structure. Cells are allocated
using an allocator that maintains a free-list
as a bit vector (1 = occupied).

The following example shows how to initialize
and use a hash table.

.. code-block:: c++

    sycl::queue Q;
    using T = double;
    syclhash::Hash<T> hash(6, Q); // size 2^6 = 64 cells

    Q.submit([&](sycl::handler &cgh) {
        DeviceHash dh(hash, cgh, sycl::read_write);

        cgh.parallel_for(sycl::nd_range<1>(20,4), [=](sycl::nd_item<1> it) {
            int gid = it.get_group(0); // 0, 1, 2, 3, or 4
            sycl::group<1> g = it.get_group();

            syclhash::Ptr my_key = gid+2;
            T my_value = 1.0 + 10.0*gid;

            auto bucket = dh[g][my_key];
            bucket.insert(my_value);
            auto bucket2 = dh[g][my_key+2];
            bucket2.insert_unique(my_value);

            for(const T val : dh[g][my_key+1]) {
                dh[g][my_key-1].insert(val-1.0);
            }
        });
    });

This example performs group operations, where all members
of the group make the same calls, but only one member succeeds
in performing the write operation.  For read operations,
one members does the reading, and broadcasts the result
to the others.

Using groups allows the hash table to optimize its load/store
bandwidth to the hash table.

Each of the 5 groups inserts `1+10*gid` at the position
`gid+2`.  Note that positions have type `syclhash::Ptr`,
which is an alias to `uint32_t`.

Next, it attempts to insert the same value at the key
`gid+4`.  However, `insert_unique` will only insert if the
bucket is unoccupied.  Otherwise, it will fail and leave
the bucket unaltered.  Because `gid+4` may or may not have
already been filled by thread `gid+2` in the first step,
the result is dependent on execution order.

Finally, the for loop at the end
iterates through all values stored in the bucket `gid+3`
and copies them to `gid+1`.  Because this code is executed
in parallel, the contents of both buckets may be modified
by other threads during the loop.  SYCLhash guarantees
that no deadlock will occur and that all members of the group
`g` will see values consistent with some state of the hash
table.
However, reads and writes are not ordered between threads,
so the result of this operation depends on how many
writes to `gid+3` are seen by the reading threads.
