#ifndef VX3_DEF_H
#define VX3_DEF_H

// V is a short form for VX
#include <cstdint>
#include <limits>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CEIL(x, y) ((x) + (y) - 1) / (y)
#define FLOOR(x, y) ((x) / (y))
#define ALIGN(x, y) (CEIL(x, y) * y)
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define BOUND(x, min, max) (MAX(MIN(x, max), min))

#define VF(float_value) Vfloat(float_value)

using Vfloat = float;
using Vindex = unsigned int;
using Vsize = unsigned int;
constexpr Vindex NULL_INDEX = std::numeric_limits<Vindex>::max();

//! The collision detection scheme to use if enabled.
enum ColSystem {
    COL_BASIC,   //!< Basic n2 collision detection. Extremely slow.
    COL_SURFACE, //!< Basic n2 collision detection, but only considering voxels on the
                 //!< surface of the object. Very slow.
    COL_BASIC_HORIZON,  //!< Hierarchical collision detection between all voxels. (One
                        //!< level) Updates potential collision list only when aggregated
                        //!< motion requires it.
    COL_SURFACE_HORIZON //!< Hierarchical collision detection between surface voxels. (One
                        //!< level) This is the fastest collision scheme implemented, and
                        //!< suffice for most cases. Updates potential collision list only
                        //!< when aggregated motion requires it.
};

enum MatBlendModel {
    MB_LINEAR,      // blended materials combine stiffness linearly (x)
    MB_EXPONENTIAL, // blended materials combine stiffness exponentially (2^x-1) (-1 puts
                    // in Y range of 0 to 1)
    MB_POLYNOMIAL // blended materials combine stiffness polynomially (x^n) with n stored
                  // seperately
};

//! Determines optional condition for the simulation to stop.
enum StopCondition {
    SC_NONE,           //!< Runs indefinitely
    SC_MAX_TIME_STEPS, //!< Runs to a set maximum number of timesteps
    SC_MAX_SIM_TIME,   //!< Runs for a set number of simulation seconds
    SC_TEMP_CYCLES,    //!< IF temperature is varying, runs for a set number of cycles.
                    //!< Otherwise runs indefinitely.
    SC_CONST_MAXENERGY, //!< runs until kinetic+potential energy stabilizes to threshhold.
                        //!< Begins checking after 50 time steps, energy must stay within
                        //!< StopConditionValue threshold for 10 consecutive readings at 50
                        //!< simulation steps apart.
    SC_MIN_KE, //!< runs until kinetic energy is below a threshhold. Begins checking after
               //!< 10 time steps, energy must stay below StopConditionValue threshold for
               //!< 10 consecutive readings at 50 simulation steps apart.
    SC_MIN_MAXMOVE //!< runs until maximum voxel displacement/timestep (of any voxel in
                   //!< the simulation) is below a threshhold (in mm?)
};

enum Axis { // which axis do we refer to?
    AXIS_NONE,
    AXIS_X,
    AXIS_Y,
    AXIS_Z
};

// What statistics to calculate
enum SimStat : int {
    CALCSTAT_NONE = 0,
    CALCSTAT_ALL = 0xffff,
    CALCSTAT_COM = 1 << 0,
    CALCSTAT_DISP = 1 << 1,
    CALCSTAT_VEL = 1 << 2,
    CALCSTAT_KINE = 1 << 3,
    CALCSTAT_STRAINE = 1 << 4,
    CALCSTAT_ENGSTRAIN = 1 << 5,
    CALCSTAT_ENGSTRESS = 1 << 6,
    CALCSTAT_PRESSURE = 1 << 7

};

// Simulation features that can be turned on and off
enum SimFeature : int {
    VXSFEAT_NONE = 0,
    VXSFEAT_COLLISIONS = 1 << 0,
    VXSFEAT_GRAVITY = 1 << 1,
    VXSFEAT_FLOOR = 1 << 2,
    VXSFEAT_TEMPERATURE = 1 << 3,
    VXSFEAT_TEMPERATURE_VARY = 1 << 4,
    VXSFEAT_PLASTICITY = 1 << 5,
    VXSFEAT_FAILURE = 1 << 6,
    VXSFEAT_BLENDING = 1 << 7,
    VXSFEAT_VOLUME_EFFECTS = 1 << 8,
    VXSFEAT_MAX_VELOCITY = 1 << 9,
    VXSFEAT_EQUILIBRIUM_MODE = 1 << 10,
    VXSFEAT_IGNORE_ANG_DAMP = 1 << 11

};

// VOXELS

// old
enum BondDir { // Defines the direction of a link
    BD_PX = 0, // Positive X direction
    BD_NX = 1, // Negative X direction
    BD_PY = 2, // Positive Y direction
    BD_NY = 3, // Negative Y direction
    BD_PZ = 4, // Positive Z direction
    BD_NZ = 5  // Negative Z direction
};
#define NO_BOND -1 // if there is no bond present on a specified direction

enum LinkFlags { // default of each should be zero for easy clearing
    LOCAL_VELOCITY_VALID =
        1
        << 0 // has something changes to render local velocity calculations (for damping)
             // invalid? (this happens when small angle or global base size has changed)
};

typedef int LinkState;

//! Defines the direction of a link relative to a given voxel.
enum LinkDirection : int {
    X_POS = 0, //!< Positive X direction
    X_NEG = 1, //!< Negative X direction
    Y_POS = 2, //!< Positive Y direction
    Y_NEG = 3, //!< Negative Y direction
    Z_POS = 4, //!< Positive Z direction
    Z_NEG = 5  //!< Negative Z direction
};
//! Defines each of 8 corners of a voxel.
enum VoxelCorner : int {
    NNN = 0, // 0b000
    NNP = 1, // 0b001
    NPN = 2, // 0b010
    NPP = 3, // 0b011
    PNN = 4, // 0b100
    PNP = 5, // 0b101
    PPN = 6, // 0b110
    PPP = 7  // 0b111
};

using VoxState = int;

//!< FLOOR_STATIC_FRICTION:
//!< If set, then the voxel is in contact with the floor and
//!< stationary in the horizontal directions. This corresponds to that
//!< voxel being in the mode of static friction (as opposed to kinetic)
//!< with the floor.
enum VoxFlags : int {   // default of each should be zero for easy clearing
    SURFACE = 1 << 1,       // on the surface?
    FLOOR_ENABLED = 1 << 2, // interact with a floor at z=0?
    FLOOR_STATIC_FRICTION =
        1 << 3, // is the voxel in a state of static friction with the floor?
    COLLISIONS_ENABLED = 1 << 5
};

enum LinkAxis : int {
    X_AXIS = 0, //!< X Axis
    Y_AXIS = 1, //!< Y Axis
    Z_AXIS = 2  //!< Z Axis
};

constexpr float HYSTERESIS_FACTOR = 1.2f; // Amount for small angle bond calculations
constexpr const float SA_BOND_BEND_RAD = 0.05f; // Amount for small angle bond
                                                // calculations
constexpr const float SA_BOND_EXT_PERC = 0.50f; // Amount for small angle bond
                                                // calculations

//!< Returns the link axis of the specified link direction.
__host__ __device__ LinkAxis linkDirectionToAxis(LinkDirection direction);

//!< Returns the link direction of the specified link axis and sign.
__host__ __device__ LinkDirection linkAxisToDirection(LinkAxis axis, bool positiveDirection);

//!< Returns true if the specified link direction is negative.
__host__ __device__ bool isLinkDirectionNegative(LinkDirection direction);

//!< Returns true if the specified link direction is positive.
__host__ __device__ bool isLinkDirectionisPositive(LinkDirection direction);

//!< Returns the opposite (negated) link direction of the specified
//!< direction.
__host__ __device__ LinkDirection oppositeLinkDirection(LinkDirection direction);

#endif // VX3_DEF_H