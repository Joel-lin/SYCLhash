# Out of source build.  To use, run:
#
#    mkdir build && cd build
#    mk -f ../mkfile
#
# Edit compile flags inside the mkhdr file.

DESTDIR = ../inst
PROJ = syclhash
SRC = ..
<$SRC/mkhdr

TESTS = tests/alloc.x tests/num.x tests/hash.x

HFILES = `{ls -1 $SRC/include/$PROJ/*.hpp}
GENHDR = include/$PROJ/config.hpp

<$SRC/mkmodern
