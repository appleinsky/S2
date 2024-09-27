
#include "ball_query_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  BallQueryTilingData tiling;
//   const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
//   context->SetBlockDim(8);
//   tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//   context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

//   return ge::GRAPH_SUCCESS;

  uint32_t num_xyz = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t num_center_xyz = context->GetInputShape(1)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t num_xyz_batch_cnt = 0;//context->GetInputShape(2)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t num_center_xyz_batch_cnt = 0;//context->GetInputShape(3)->GetStorageShape().GetShapeSize(); //输入数量

    tiling.set_num_xyz(num_xyz);
    tiling.set_num_center_xyz(num_center_xyz);
    tiling.set_num_xyz_batch_cnt(num_xyz_batch_cnt);
    tiling.set_num_center_xyz_batch_cnt(num_center_xyz_batch_cnt);


    auto shape_xyz = context->GetInputTensor(0)->GetOriginShape();
    auto shape_center_xyz = context->GetInputTensor(1)->GetOriginShape();

    uint32_t shape_b,shape_m,shape_n;

    shape_b = shape_xyz.GetDim(0);
    shape_n = shape_xyz.GetDim(1);
    shape_m = shape_center_xyz.GetDim(1);

    tiling.set_shape_b(shape_b);
    tiling.set_shape_m(shape_m);
    tiling.set_shape_n(shape_n);

    float min_radius = *context->GetAttrs()->GetFloat(0);
    float max_radius = *context->GetAttrs()->GetFloat(1);
    int32_t sample_num = *context->GetAttrs()->GetInt(2);

    tiling.set_min_radius(min_radius);
    tiling.set_max_radius(max_radius);
    tiling.set_sample_num(sample_num);

    
    // x_dimensional = shape_x.GetDimNum();

    // for(int i = 0; i < x_dimensional; i++)
    // {
    //     x_ndarray[i] = shape_x.GetDim(i);
    //     size *= x_ndarray[i];
    // }


  if((shape_n%16==0)&&(shape_m%16==0))
    {
        context->SetBlockDim(shape_b);
    }
    else
    {
        context->SetBlockDim(1);
    }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class BallQuery : public OpDef {
public:
    explicit BallQuery(const char* name) : OpDef(name)
    {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min_radius").Float();
        this->Attr("max_radius").Float();
        this->Attr("sample_num").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(BallQuery);
}
