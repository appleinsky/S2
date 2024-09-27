#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(BallQuery)
    .INPUT(xyz, ge::TensorType::ALL())
    .INPUT(center_xyz, ge::TensorType::ALL())
    .OPTIONAL_INPUT(xyz_batch_cnt, ge::TensorType::ALL())
    .OPTIONAL_INPUT(center_xyz_batch_cnt, ge::TensorType::ALL())
    .OUTPUT(idx, ge::TensorType::ALL())
    .REQUIRED_ATTR(min_radius, Float)
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(BallQuery);

}

#endif
