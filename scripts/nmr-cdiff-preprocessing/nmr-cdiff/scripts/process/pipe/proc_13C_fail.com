#!/bin/csh

bruk2pipe -in $1/fid \
  -bad 0.0 -ext -aswap -AMX -decim 1386.66666666667 -dspfvs 20 -grpdly 67.9878387451172  \
  -xN             16384  \
  -xT              8192  \
  -xMODE            DQD  \
  -xSW        14423.077  \
  -xOBS    600.08584217  \
  -xCAR           4.736  \
  -xLAB             13C  \
  -ndim               1  \
| nmrPipe -fn MULT -c 3.05176e-02 \
| nmrPipe -fn EM -lb 1 -c 1.0 \
# | nmrPipe -fn ZF -zf 3 \
| nmrPipe -fn FT \
#| nmrPipe -fn POLY -auto
sleep 1
