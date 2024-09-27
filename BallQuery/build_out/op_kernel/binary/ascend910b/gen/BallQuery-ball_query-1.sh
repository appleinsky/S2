#!/bin/bash
echo "[Ascend910B1] Generating BallQuery_8ef4e9c4ead330762be4d0b61630f102 ..."
opc $1 --main_func=ball_query --input_param=/home/ma-user/work/suanzi/BallQuery/build_out/op_kernel/binary/ascend910b/gen/BallQuery_8ef4e9c4ead330762be4d0b61630f102_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/BallQuery_8ef4e9c4ead330762be4d0b61630f102.json ; then
  echo "$2/BallQuery_8ef4e9c4ead330762be4d0b61630f102.json not generated!"
  exit 1
fi

if ! test -f $2/BallQuery_8ef4e9c4ead330762be4d0b61630f102.o ; then
  echo "$2/BallQuery_8ef4e9c4ead330762be4d0b61630f102.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating BallQuery_8ef4e9c4ead330762be4d0b61630f102 Done"
