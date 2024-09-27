
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BallQueryTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_xyz);
  TILING_DATA_FIELD_DEF(uint32_t, num_center_xyz);
  TILING_DATA_FIELD_DEF(uint32_t, num_xyz_batch_cnt);
  TILING_DATA_FIELD_DEF(uint32_t, num_center_xyz_batch_cnt);
  TILING_DATA_FIELD_DEF(uint32_t, shape_b);
  TILING_DATA_FIELD_DEF(uint32_t, shape_m);
  TILING_DATA_FIELD_DEF(uint32_t, shape_n);
  TILING_DATA_FIELD_DEF(float, min_radius);
  TILING_DATA_FIELD_DEF(float, max_radius);
  TILING_DATA_FIELD_DEF(int32_t, sample_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BallQuery, BallQueryTilingData)
}
