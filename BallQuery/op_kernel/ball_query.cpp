#include "kernel_operator.h"
#include<cmath>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename TYPE_XYZ> class KernelBallQuery {
    using T = TYPE_XYZ;
public:
    __aicore__ inline KernelBallQuery() {}
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx,
                                uint32_t num_xyz, uint32_t num_center_xyz, uint32_t num_xyz_batch_cnt, uint32_t num_center_xyz_batch_cnt,
                                uint32_t shape_b, uint32_t shape_m, uint32_t shape_n,
                                float min_radius, float max_radius, int32_t sample_num) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");


        this->num_xyz = num_xyz;
        this->num_center_xyz = num_center_xyz;
        this->num_xyz_batch_cnt = num_xyz_batch_cnt;
        this->num_center_xyz_batch_cnt = num_center_xyz_batch_cnt;
        this->shape_b = shape_b;
        this->shape_m = shape_m;
        this->shape_n = shape_n;
        this->min_radius = min_radius * min_radius;
        this->max_radius = max_radius * max_radius;
        this->sample_num = sample_num;

        xyzGm.SetGlobalBuffer((__gm__ DTYPE_XYZ*)xyz, this->num_xyz);
        center_xyzGm.SetGlobalBuffer((__gm__ DTYPE_CENTER_XYZ*)center_xyz, this->num_center_xyz);
        idxGm.SetGlobalBuffer((__gm__ DTYPE_IDX*)idx, this->sample_num);
        // if(this->num_xyz_batch_cnt > 0)
        // {
        //     xyz_batch_cntGm.SetGlobalBuffer((__gm__ DTYPE_XYZ_BATCH_CNT*)xyz_batch_cnt, this->num_xyz_batch_cnt);
        //     center_xyz_batch_cntGm.SetGlobalBuffer((__gm__ DTYPE_CENTER_XYZ_BATCH_CNT*)center_xyz_batch_cnt, this->num_center_xyz_batch_cnt);
        // }

        // pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        // pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));

        // if constexpr (std::is_same_v<T, half>)
        // {
        pipe.InitBuffer(QueueTmp1, 32 * sizeof(float));
        // pipe.InitBuffer(QueueTmp2, this->tileDataNum * sizeof(DTYPE_X));
        // pipe.InitBuffer(QueueTmp3, this->tileDataNum * sizeof(float));
        // pipe.InitBuffer(QueueTmp4, this->tileDataNum * sizeof(float));
        // pipe.InitBuffer(QueueTmp5, this->tileDataNum * sizeof(float));
        // }
    }
    __aicore__ inline void Process() {
        // if(this->num_xyz_batch_cnt == 0)
        if(GetBlockNum() > 1)
        {
            uint32_t i = GetBlockIdx();
            // for(uint32_t i = 0; i < this->shape_b; i++)
            {
                for(uint32_t j = 0; j < this->shape_m; j++)
                {
                    float center_x = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 0);// center_xyz[i][j][0]
                    float center_y = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 1);// center_xyz[i][j][1]
                    float center_z = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 2);// center_xyz[i][j][2]
                    int32_t cnt = 0;
                    for(uint32_t k = 0; k < this->shape_n; k++)
                    {
                        float x = xyzGm.GetValue(i*shape_n*3 + k*3 + 0);//xyz[i][k][0]
                        float y = xyzGm.GetValue(i*shape_n*3 + k*3 + 1);//xyz[i][k][1]
                        float z = xyzGm.GetValue(i*shape_n*3 + k*3 + 2);//xyz[i][k][2]
                        float dis1 = (center_x - x) * (center_x - x) + (center_y - y) *(center_y - y) + (center_z - z) *(center_z - z);
                        float dis = dis1;//sqrt(dis1);
                        if(dis == 0 || (this->min_radius <= dis && dis < this->max_radius))
                        {
                            if(cnt == 0)
                            {
                                for(uint32_t t = 0; t < this->sample_num; t++)
                                {
                                    idxGm.SetValue(i*shape_m*this->sample_num + j*this->sample_num + t, k);
                                }
                            }
                            idxGm.SetValue(i*shape_m*this->sample_num + j*this->sample_num + cnt, k);
                            cnt += 1;
                            if(cnt >= this->sample_num)
                                break;
                        }
                    }
                }
            }

        }
        else
        {
            for(uint32_t i = 0; i < this->shape_b; i++)
            {
                for(uint32_t j = 0; j < this->shape_m; j++)
                {
                    float center_x = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 0);// center_xyz[i][j][0]
                    float center_y = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 1);// center_xyz[i][j][1]
                    float center_z = center_xyzGm.GetValue(i*shape_m*3 + j*3 + 2);// center_xyz[i][j][2]
                    int32_t cnt = 0;
                    for(uint32_t k = 0; k < this->shape_n; k++)
                    {
                        float x = xyzGm.GetValue(i*shape_n*3 + k*3 + 0);//xyz[i][k][0]
                        float y = xyzGm.GetValue(i*shape_n*3 + k*3 + 1);//xyz[i][k][1]
                        float z = xyzGm.GetValue(i*shape_n*3 + k*3 + 2);//xyz[i][k][2]
                        float dis1 = (center_x - x) * (center_x - x) + (center_y - y) *(center_y - y) + (center_z - z) *(center_z - z);
                        float dis = dis1;//sqrt(dis1);
                        if(dis == 0 || (this->min_radius <= dis && dis < this->max_radius))
                        {
                            if(cnt == 0)
                            {
                                for(uint32_t t = 0; t < this->sample_num; t++)
                                {
                                    idxGm.SetValue(i*shape_m*this->sample_num + j*this->sample_num + t, k);
                                }
                            }
                            idxGm.SetValue(i*shape_m*this->sample_num + j*this->sample_num + cnt, k);
                            cnt += 1;
                            if(cnt >= this->sample_num)
                                break;
                        }
                    }
                }
            }
        }
    // b, m, _ = center_xyz.shape
    // b, n, _ = xyz.shape
    // res = np.zeros([b, m, sample_num], dtype=np.int32)
    // for i in range(b):
    //     for j in range(m):
    //         center_x = center_xyz[i][j][0]
    //         center_y = center_xyz[i][j][1]
    //         center_z = center_xyz[i][j][2]
    //         cnt = 0

    //         for k in range(n):
    //             x = xyz[i][k][0]
    //             y = xyz[i][k][1]
    //             z = xyz[i][k][2]
    //             dis = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2 + (center_z - z) ** 2)
    //             if dis == 0 or min_radius <= dis < max_radius:
    //                 if cnt == 0:
    //                     for t in range(sample_num):
    //                         res[i][j][t] = k
    //                 res[i][j][cnt] = k
    //                 cnt += 1
    //                 if cnt >= sample_num:
    //                     break

    // return np.array(res)


    //     else
    //     {
    //         float radius2 = this->max_radius * this->max_radius;
    //         for(uint32_t i = 0; i < this->shape_b; i++)
    //         {
    //             int32_t current_b_idx = 0;
    //             int32_t tmp_b = 0;
    //             for(uint32_t _b = 0; _b < this->num_center_xyz_batch_cnt; _b++)
    //             {
    //                 tmp_b += center_xyz_batch_cntGm.GetValue(_b);
    //                 if (tmp_b > i)
    //                 {
    //                     current_b_idx = _b;
    //                     break;
    //                 }
    //             }
    //             float new_x = center_xyz[i][0]
    //             float new_y = center_xyz[i][1]
    //             float new_z = center_xyz[i][2]
    //             n = xyz_batch_cnt[current_b_idx]
    //         }
    //     }

    //     radius2 = max_radius * max_radius
    // center_xyz_length = center_xyz.shape[0]
    // idx = torch.zeros((center_xyz_length, sample_num), dtype=torch.int32)
    // for i in range(center_xyz_length):
    //     current_b_idx = 0
    //     tmp_b = 0
    //     for _b in range(len(center_xyz_batch_cnt)):
    //         tmp_b += center_xyz_batch_cnt[_b]
    //         if tmp_b > i:
    //             current_b_idx = _b
    //             break
    //     new_x = center_xyz[i][0]
    //     new_y = center_xyz[i][1]
    //     new_z = center_xyz[i][2]
    //     n = xyz_batch_cnt[current_b_idx]

    //     xyz_offset = 0

    //     for _t in range(current_b_idx):
    //         xyz_offset += xyz_batch_cnt[_t]

    //     cnt = 0
    //     for j in range(n):
    //         x = xyz[xyz_offset + j][0]
    //         y = xyz[xyz_offset + j][1]
    //         z = xyz[xyz_offset + j][2]
    //         dis = (new_x - x) ** 2 + (new_y - y) ** 2 + (new_z - z) ** 2
    //         if dis < radius2:
    //             if cnt == 0:
    //                 for f in range(sample_num):
    //                     idx[i][f] = j
    //             idx[i][cnt] = j
    //             cnt += 1
    //             if cnt >= sample_num:
    //                 break
    // return idx.numpy()
        // int32_t loopCount = this->tileNum;
        // this->processDataNum = this->tileDataNum;
        // for (int32_t i = 0; i < loopCount; i++) {
        //     if (i == this->tileNum - 1) {
        //       this->processDataNum = this->tailDataNum;
        //     }
        //     CopyIn(i);
        //     Compute(i);
        //     CopyOut(i);
        // }
    }

private:
    TPipe pipe;
    // TBuf<QuePosition::VECCALC> tmpBuffer, signbitBuffer;
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueTmp1;// QueueTmp2;// QueueTmp3;// QueueTmp4, QueueTmp5;

    GlobalTensor<DTYPE_XYZ> xyzGm;
    GlobalTensor<DTYPE_CENTER_XYZ> center_xyzGm;
    GlobalTensor<DTYPE_XYZ_BATCH_CNT> xyz_batch_cntGm;
    GlobalTensor<DTYPE_CENTER_XYZ_BATCH_CNT> center_xyz_batch_cntGm;
    GlobalTensor<DTYPE_IDX> idxGm;
    uint32_t num_xyz;
    uint32_t num_center_xyz;
    uint32_t num_xyz_batch_cnt;
    uint32_t num_center_xyz_batch_cnt;
    uint32_t shape_b;
    uint32_t shape_m;
    uint32_t shape_n;
    float min_radius;
    float max_radius;
    int32_t sample_num;
};
extern "C" __global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelBallQuery<DTYPE_XYZ> op;
    op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx,
            tiling_data.num_xyz, tiling_data.num_center_xyz, tiling_data.num_xyz_batch_cnt, tiling_data.num_center_xyz_batch_cnt,
            tiling_data.shape_b, tiling_data.shape_m, tiling_data.shape_n,
            tiling_data.min_radius, tiling_data.max_radius, tiling_data.sample_num
            );  
    op.Process();
}