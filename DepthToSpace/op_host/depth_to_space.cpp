#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  DepthToSpaceTilingData tiling;

    

    int32_t shape_b;
    int32_t shape_c;
    int32_t shape_h;
    int32_t shape_w;
    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    

    // tiling.set_shape_b(shape_b);
    // tiling.set_shape_b(shape_b);
    // tiling.set_shape_h(shape_h);
    // tiling.set_shape_w(shape_w);
    int32_t shape_input[6];
    int32_t shape_output[4];


    int32_t block_size = *context->GetAttrs()->GetInt(0);
    const char *mode_str = context->GetAttrs()->GetStr(1);
    const char *data_format_str = context->GetAttrs()->GetStr(2);

    if(strcmp(data_format_str, "NCHW") == 0)
    {
        shape_b = shape_x.GetDim(0);
        shape_c = shape_x.GetDim(1);
        shape_h = shape_x.GetDim(2);
        shape_w = shape_x.GetDim(3);
        if ((strcmp(mode_str, "DCR") == 0))
        {
            tiling.set_run_mode(0);
            shape_input[0] = shape_b;
            shape_input[1] = block_size;
            shape_input[2] = block_size;
            shape_input[3] = shape_c / (block_size * block_size);
            shape_input[4] = shape_h;
            shape_input[5] = shape_w;

            shape_output[0] = shape_b;
            shape_output[1] = shape_c / (block_size * block_size);
            shape_output[2] = shape_h * block_size;
            shape_output[3] = shape_w * block_size;
        }
        else
        {
            tiling.set_run_mode(1);
            shape_input[0] = shape_b;
            shape_input[1] = shape_c / (block_size * block_size);
            shape_input[2] = block_size;
            shape_input[3] = block_size;
            shape_input[4] = shape_h;
            shape_input[5] = shape_w;

            shape_output[0] = shape_b;
            shape_output[1] = shape_c / (block_size * block_size);
            shape_output[2] = shape_h * block_size;
            shape_output[3] = shape_w * block_size;
        }
    }
    else
    {
        shape_b = shape_x.GetDim(0);
        shape_h = shape_x.GetDim(1);
        shape_w = shape_x.GetDim(2);
        shape_c = shape_x.GetDim(3);
        if ((strcmp(mode_str, "DCR") == 0))
        {
            tiling.set_run_mode(2);
            shape_input[0] = shape_b;
            shape_input[1] = shape_h;
            shape_input[2] = shape_w;
            shape_input[3] = block_size;
            shape_input[4] = block_size;
            shape_input[5] = shape_c / (block_size * block_size);

            shape_output[0] = shape_b;
            shape_output[1] = shape_h * block_size;
            shape_output[2] = shape_w * block_size;
            shape_output[3] = shape_c / (block_size * block_size);
        }
        else
        {
            tiling.set_run_mode(1);
            shape_input[0] = shape_b;
            shape_input[1] = shape_h;
            shape_input[2] = shape_w;
            shape_input[3] = shape_c / (block_size * block_size);
            shape_input[4] = block_size;
            shape_input[5] = block_size;

            shape_output[0] = shape_b;
            shape_output[1] = shape_h * block_size;
            shape_output[2] = shape_w * block_size;
            shape_output[3] = shape_c / (block_size * block_size);
        }
    }

    
    tiling.set_shape_input(shape_input);
    tiling.set_shape_output(shape_output);
    tiling.set_block_size(block_size);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
//   const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
//   context->SetBlockDim(8);
//   tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//   context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

//   return ge::GRAPH_SUCCESS;
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
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").Int();
        this->Attr("mode").AttrType(OPTIONAL).String("DCR");
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DepthToSpace);
}
