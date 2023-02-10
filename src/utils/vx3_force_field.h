#ifndef VX3_FORCEFIELD_H
#define VX3_FORCEFIELD_H

#include "vx3_math_tree.h"

struct VX3_ForceField {
    VX3_MathTreeToken x_force_field[VX3_MATH_TREE_MAX_EXPRESSION_TOKENS];
    VX3_MathTreeToken y_force_field[VX3_MATH_TREE_MAX_EXPRESSION_TOKENS];
    VX3_MathTreeToken z_force_field[VX3_MATH_TREE_MAX_EXPRESSION_TOKENS];
    VX3_ForceField() {
        x_force_field[0].op = mtCONST;
        x_force_field[0].value = 0;
        x_force_field[1].op = mtEND;
        y_force_field[0].op = mtCONST;
        y_force_field[0].value = 0;
        y_force_field[1].op = mtEND;
        z_force_field[0].op = mtCONST;
        z_force_field[0].value = 0;
        z_force_field[1].op = mtEND;
    }
    bool validate() {
        return VX3_MathTree::validate(x_force_field) &&
               VX3_MathTree::validate(y_force_field) &&
               VX3_MathTree::validate(z_force_field);
    }
    __device__ __host__ Vfloat x_forcefield(Vfloat x, Vfloat y, Vfloat z, Vfloat hit,
                                            Vfloat t, Vfloat angle, Vfloat closeness,
                                            int numClosePairs, int num_voxel) {
        return VX3_MathTree::eval(x, y, z, hit, t, angle, closeness, numClosePairs,
                                  num_voxel, x_force_field);
    }
    __device__ __host__ Vfloat y_forcefield(Vfloat x, Vfloat y, Vfloat z, Vfloat hit,
                                            Vfloat t, Vfloat angle, Vfloat closeness,
                                            int numClosePairs, int num_voxel) {
        return VX3_MathTree::eval(x, y, z, hit, t, angle, closeness, numClosePairs,
                                  num_voxel, y_force_field);
    }
    __device__ __host__ Vfloat z_forcefield(Vfloat x, Vfloat y, Vfloat z, Vfloat hit,
                                            Vfloat t, Vfloat angle, Vfloat closeness,
                                            int numClosePairs, int num_voxel) {
        return VX3_MathTree::eval(x, y, z, hit, t, angle, closeness, numClosePairs,
                                  num_voxel, z_force_field);
    }
};

#endif // VX3_FORCEFIELD_H
