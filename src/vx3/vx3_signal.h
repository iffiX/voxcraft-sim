#ifndef VX3_SIGNAL_H
#define VX3_SIGNAL_H

#include "utils/vx3_def.h"
#include "utils/vx3_soa.h"

struct VX3_Signal {
    Vfloat value = 0;
    Vfloat active_time = 0;
};

REFL_AUTO(type(VX3_Signal), field(value), field(active_time))

#endif // VX3_SIGNAL_H
