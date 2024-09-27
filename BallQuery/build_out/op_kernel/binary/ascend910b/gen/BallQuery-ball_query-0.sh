#!/bin/bash
echo "[Ascend910B1] Generating BallQuery_15860cad4f0b02b13abda0a22403ac0e ..."
opc $1 --main_func=ball_query --input_param=/home/ma-user/work/suanzi/BallQuery/build_out/op_kernel/binary/ascend910b/gen/BallQuery_15860cad4f0b02b13abda0a22403ac0e_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/BallQuery_15860cad4f0b02b13abda0a22403ac0e.json ; then
  echo "$2/BallQuery_15860cad4f0b02b13abda0a22403ac0e.json not generated!"
  exit 1
fi

if ! test -f $2/BallQuery_15860cad4f0b02b13abda0a22403ac0e.o ; then
  echo "$2/BallQuery_15860cad4f0b02b13abda0a22403ac0e.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating BallQuery_15860cad4f0b02b13abda0a22403ac0e Done"
