
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BALL_QUERY_H_
#define ACLNN_BALL_QUERY_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBallQueryGetWorkspaceSize
 * parameters :
 * xyz : required
 * centerXyz : required
 * xyzBatchCntOptional : optional
 * centerXyzBatchCntOptional : optional
 * minRadius : required
 * maxRadius : required
 * sampleNum : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBallQueryGetWorkspaceSize(
    const aclTensor *xyz,
    const aclTensor *centerXyz,
    const aclTensor *xyzBatchCntOptional,
    const aclTensor *centerXyzBatchCntOptional,
    double minRadius,
    double maxRadius,
    int64_t sampleNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBallQuery
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBallQuery(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
