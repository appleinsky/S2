#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename TYPE_X> class KernelDepthToSpace {
    using T = TYPE_X;
public:
    __aicore__ inline KernelDepthToSpace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int32_t run_mode, int32_t shape_input[], int32_t shape_output[], int32_t block_size
                                ){
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");


        this->run_mode = run_mode;
        this->shape_input[0] = shape_input[0];
        this->shape_input[1] = shape_input[1];
        this->shape_input[2] = shape_input[2];
        this->shape_input[3] = shape_input[3];
        this->shape_input[4] = shape_input[4];
        this->shape_input[5] = shape_input[5];
        this->shape_output[0] = shape_output[0];
        this->shape_output[1] = shape_output[1];
        this->shape_output[2] = shape_output[2];
        this->shape_output[3] = shape_output[3];
        this->block_size = block_size;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, 1000);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, 1000);

    }
    __aicore__ inline void Process() {
        int32_t _N = this->shape_input[0];
        int32_t _H = this->shape_input[1];
        int32_t _W = this->shape_input[2];
        int32_t _C0 = this->shape_input[3];
        int32_t _C1 = this->shape_input[4];
        int32_t _C2 = this->shape_input[5];
        if(this->run_mode == 2) //0, 1, 3, 2, 4, 5
        {
            for(int32_t n=0; n<_N; n++)
            {
                for(int32_t h=0; h<_H; h++)
                {
                    for(int32_t w=0; w<_W; w++)
                    {
                        for(int32_t c0=0; c0<_C0; c0++)
                        {
                            for(int32_t c1=0; c1<_C1; c1++)
                            {
                                for(int32_t c2=0; c2<_C2; c2++)
                                {
                                    int32_t m = n*(_H*_W*_C0*_C1*_C2) + h*(_W*_C0*_C1*_C2) + w*(_C0*_C1*_C2) + c0*(_C1*_C2) + c1*_C2 + c2;
                                    int32_t k = n*(_H*_C0*_W*_C1*_C2) + h*(_C0*_W*_C1*_C2) + c0*(_W*_C1*_C2) + w*(_C1*_C2) + c1*_C2 + c2;
                                    yGm.SetValue(k, (DTYPE_Y)xGm.GetValue(m));
                                }
                            }
                        }
                    }
                }
            }
        }
        else if(this->run_mode == 0) //0, 3, 4, 1, 5, 2
        {
            for(int32_t n=0; n<_N; n++)
            {
                for(int32_t h=0; h<_H; h++)
                {
                    for(int32_t w=0; w<_W; w++)
                    {
                        for(int32_t c0=0; c0<_C0; c0++)
                        {
                            for(int32_t c1=0; c1<_C1; c1++)
                            {
                                for(int32_t c2=0; c2<_C2; c2++)
                                {
                                    int32_t m = n*(_H*_W*_C0*_C1*_C2) + h*(_W*_C0*_C1*_C2) + w*(_C0*_C1*_C2) + c0*(_C1*_C2) + c1*_C2 + c2;
                                    int32_t k = n*(_C0*_C1*_H*_C2*_W) + c0*(_C1*_H*_C2*_W) + c1*(_H*_C2*_W) + h*(_C2*_W) + c2*_W + w;
                                    yGm.SetValue(k, (DTYPE_Y)xGm.GetValue(m));
                                }
                            }
                        }
                    }
                }
            }
        }
        else//0, 1, 4, 2, 5, 3
        {
            for(int32_t n=0; n<_N; n++)
            {
                for(int32_t h=0; h<_H; h++)
                {
                    for(int32_t w=0; w<_W; w++)
                    {
                        for(int32_t c0=0; c0<_C0; c0++)
                        {
                            for(int32_t c1=0; c1<_C1; c1++)
                            {
                                for(int32_t c2=0; c2<_C2; c2++)
                                {
                                    int32_t m = n*(_H*_W*_C0*_C1*_C2) + h*(_W*_C0*_C1*_C2) + w*(_C0*_C1*_C2) + c0*(_C1*_C2) + c1*_C2 + c2;
                                    int32_t k = n*(_H*_C1*_W*_C2*_C0) + h*(_C1*_W*_C2*_C0) + c1*(_W*_C2*_C0) + w*(_C2*_C0) + c2*_C0 + c0;
                                    yGm.SetValue(k, (DTYPE_Y)xGm.GetValue(m));
                                }
                            }
                        }
                    }
                }
            }
        }
        



        // yGm.SetValue(0, (DTYPE_Y)this->run_mode);
    }

private:
    TPipe pipe;
    // TBuf<QuePosition::VECCALC> tmpBuffer, signbitBuffer;
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    // TBuf<QuePosition::VECCALC> QueueTmp1;// QueueTmp2;// QueueTmp3;// QueueTmp4, QueueTmp5;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t run_mode;
    int32_t shape_input[6];
    int32_t shape_output[4];
    int32_t block_size;
};
extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelDepthToSpace<DTYPE_X> op;
    op.Init(x, y,
            tiling_data.run_mode, tiling_data.shape_input, tiling_data.shape_output, tiling_data.block_size
            );  
    op.Process();
}