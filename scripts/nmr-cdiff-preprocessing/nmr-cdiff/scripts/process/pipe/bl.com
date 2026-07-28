#!/bin/csh
#
nmrPipe \
| nmrPipe -fn POLY -auto \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 -di -ov
sleep 1