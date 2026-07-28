#!/bin/csh

bruk2pipe -in $1/fid \
  -bad 0.0 -ext -aswap -AMX -decim 552 -dspfvs 20 -grpdly 68.0126800537109  \
  -xN             16384  \
  -xT              8192  \
  -xMODE            DQD  \
  -xSW        36231.884  \
  -xOBS         150.906  \
  -xCAR          99.410  \
  -xLAB             13C  \
  -ndim               1  \
| nmrPipe -fn MULT -c 3.05176e-02 \
| nmrPipe -fn EM -lb 1 -c 1.0 \
| nmrPipe -fn ZF -zf 3 \
| nmrPipe -fn FT \
#| nmrPipe -fn POLY -auto
sleep 1
