#!/bin/csh

bruk2pipe -in $1/ser \
  -bad 0.0 -ext -aswap -AMX -decim 1386.66666666667 -dspfvs 20 -grpdly 67.9878387451172  \
  -xN              8192  \
  -xT              4096  \
  -xMODE            DQD  \
  -xSW        14423.077  \
  -xOBS         600.086  \
  -xCAR           4.657  \
  -xLAB              1H  \
  -ndim               1  \
| nmrPipe -fn MULT -c 4.06901e-02 \
# | nmrPipe -fn EM -lb 5.0 -c 1.0 \
| nmrPipe -fn ZF -size 32768 \
 #-zf 3 \
| nmrPipe -fn FT \
#| nmrPipe -fn POLY -auto