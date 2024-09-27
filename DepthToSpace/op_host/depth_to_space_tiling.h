
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
  // TILING_DATA_FIELD_DEF(int32_t, shape_b);
  // TILING_DATA_FIELD_DEF(int32_t, shape_c);
  // TILING_DATA_FIELD_DEF(int32_t, shape_h);
  // TILING_DATA_FIELD_DEF(int32_t, shape_w);
  TILING_DATA_FIELD_DEF(int32_t, run_mode);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 6, shape_input);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape_output);
  TILING_DATA_FIELD_DEF(int32_t, block_size);
  
  // TILING_DATA_FIELD_DEF(int32_t, data_format);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
