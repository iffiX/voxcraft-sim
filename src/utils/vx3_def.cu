//
// Created by iffi on 11/19/22.
//
#include "vx3_def.h"

//!< Returns the link axis of the specified link direction.
__host__ __device__ LinkAxis linkDirectionToAxis(LinkDirection direction) {
    return (LinkAxis)((int)direction / 2);
}

//!< Returns the link direction of the specified link axis and sign.
__host__ __device__ LinkDirection linkAxisToDirection(LinkAxis axis, bool positiveDirection) {
    return (LinkDirection)(2 * ((int)axis) + positiveDirection ? 0 : 1);
}

//!< Returns true if the specified link direction is negative.
__host__ __device__ bool isLinkDirectionNegative(LinkDirection direction) {
    return direction % 2 == 1;
}

//!< Returns true if the specified link direction is positive.
__host__ __device__ bool isLinkDirectionisPositive(LinkDirection direction) {
    return direction % 2 == 0;
}

//!< Returns the opposite (negated) link direction of the specified
//!< direction.
__host__ __device__ LinkDirection oppositeLinkDirection(LinkDirection direction) {
    return (LinkDirection)(direction - direction % 2 + (direction + 1) % 2);
}