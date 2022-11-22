#if !defined(VX3_QUAT3D_H)
#define VX3_QUAT3D_H

/*******************************************************************************
Copyright (c) 2015, Jonathan Hiller
To cite academic use of Voxelyze: Jonathan Hiller and Hod Lipson "Dynamic
Simulation of Soft Multimaterial 3D-Printed Objects" Soft Robotics. March 2014,
1(1): 88-101. Available at
http://online.liebertpub.com/doi/pdfplus/10.1089/soro.2013.0010

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. Voxelyze is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details. See <http://www.opensource.org/licenses/lgpl-3.0.html> for license
details.
*******************************************************************************/
//
// Learning resource: refer to https://eater.net/quaternions/video/intro , The
// visualization and tutorial on Quanernions.
//

#include "vx3_vec3d.h"

#include <cfloat>
#include <cmath>

#define PI 3.14159265358979
#define DBL_EPSILONx24 5.328e-15 // DBL_EPSILON*24
//#define DBL_EPSILON_SQ_x6xROOT2 4.17e-31 //DBL_EPSILON*DBL_EPSILON*6*sqrt(2.0)

// The following values are calculated based on: MAX_ERROR_PERCENT 1e-4
#define DISCARD_ANGLE_RAD 1e-7 // Anything less than this angle can be considered 0
#define SMALL_ANGLE_RAD                                                                  \
    1.732e-2 // Angles less than this get small angle approximations. To get:
             // Root solve atan(t)/t-1+MAX_ERROR_PERCENT. From:
             // MAX_ERROR_PERCENT = (t-atan(t))/t
#define SMALL_ANGLE_W                                                                    \
    0.9999625 // quaternion W value corresponding to a SMALL_ANGLE_RAD. To
              // calculate, cos(SMALL_ANGLE_RAD*0.5).
#define W_THRESH_ACOS2SQRT                                                               \
    0.9988 // Threshhold of w above which we can approximate acos(w) with
           // sqrt(2-2w). To get: Root solve 1-sqrt(2-2wt)/acos(wt) -
           // MAX_ERROR_PERCENT. From MAX_ERROR_PERCENT =
           // (acos(wt)-sqrt(2-2wt))/acos(wt)
//#define SLTHRESH_DISCARD_ANGLE 1e-14; //SquareLength (x^2+y^2+z^2 comparison)
// threshhold for what constitutes a discard-small angle. From DISCARD_ANGLE_RAD
//~= acos(w) and SquareLength also = 1-w*w. Must upcast to double or else
// truncated floats.
#define SLTHRESH_ACOS2SQRT                                                               \
    2.4e-3 // SquareLength threshhold for when we can use square root
           // optimization for acos. From SquareLength = 1-w*w. (calculate
           // according to 1.0-W_THRESH_ACOS2SQRT*W_THRESH_ACOS2SQRT

// quaternion properties (for reference)
// 1) To rotate a vector V, form a quaternion with w = 0; To rotate by
// Quaternion Q, do Q*V*Q.Conjugate() and trim off the w component. 2) To do
// multiple rotations: To Rotate by Q1 THEN Q2,
// Q2*Q1*V*Q1.Conjugate*Q2.Conjugate(), or make a Qtot = Q2*Q1 and do
// Qtot*V*Qtot.Conjucate() 3) Q1*Q1.Conjugate - Identity 4) To do a reverse
// rotation Q1, just do Q1.conjugate*V*Q1
// http://www.cprogramming.com/tutorial/3d/quaternions.html

//! A generic 3D quaternion template
/*!
The template parameter is assumed to be either float or double depending on the
desired numerical resolution.
*/
template <typename T = Vfloat> class VX3_Quat3D {
  public:
    T w; //!< The current W value.
    T x; //!< The current X value.
    T y; //!< The current Y value.
    T z; //!< The current Z value.

    // constructors
    __host__ __device__ VX3_Quat3D()
        : w(1), x(0), y(0), z(0) {} //!< Constructor. Initialzes w, x, y, z to zero.
    __host__ __device__ VX3_Quat3D(const T dw, const T dx, const T dy, const T dz) {
        w = dw;
        x = dx;
        y = dy;
        z = dz;
    } //!< Constructor with specified individual values.
    __host__ __device__ VX3_Quat3D(const VX3_Quat3D &QuatIn) {
        w = QuatIn.w;
        x = QuatIn.x;
        y = QuatIn.y;
        z = QuatIn.z;
    } //!< Copy constructor

    //!< Constructs this quaternion from rotation vector VecIn. See
    //!< FromRotationVector(). @param[in] VecIn A rotation vector.
    __host__ __device__ explicit VX3_Quat3D(const VX3_Vec3D<T> &VecIn) {
        FromRotationVector(VecIn);
    }

    //!< Constructs this quaternion from an angle in radians and a unit axis.
    //!< @param[in] angle An angle in radians @param[in] axis A normalize
    //!< rotation axis.
    __host__ __device__ VX3_Quat3D(const T angle, const VX3_Vec3D<T> &axis) {
        const T a = angle * (T)0.5;
        const T s = sin(a);
        const T c = cos(a);
        w = c;
        x = axis.x * s;
        y = axis.y * s;
        z = axis.z * s;
    }

    //!< Constructs this quaternion to represent the rotation from two
    //!< vectors. The vectors need not be normalized and are not modified.
    //!< @param[in] RotateFrom A vector representing a pre-rotation
    //!< orientation. @param[in] RotateTo A vector representing a
    //!< post-rotation orientation.
    __host__ __device__ VX3_Quat3D(const VX3_Vec3D<T> &RotateFrom,
                                   const VX3_Vec3D<T> &RotateTo) {
        T theta = acos(
            RotateFrom.dot(RotateTo) /
            sqrt(RotateFrom.length2() * RotateTo.length2())); // angle between vectors.
                                                              // from A.B=|A||B|cos(theta)
        //		if (theta < DISCARD_ANGLE_RAD) {*this =
        // VX3_Quat3D(1,0,0,0); return;} //very small angle, return no rotation
        if (theta <= 0) {
            *this = VX3_Quat3D(1, 0, 0, 0);
            return;
        } // very small angle, return no rotation
        VX3_Vec3D<T> Axis = RotateFrom.cross(RotateTo); // Axis of rotation
        Axis.normalizeFast();
        if (theta > PI - DISCARD_ANGLE_RAD) {
            *this = VX3_Quat3D(Axis);
            return;
        } // 180 degree rotation (180 degree rot about axis ax, ay, az is
          // Quat(0,ax,ay,az))
        *this = VX3_Quat3D(theta, Axis); // otherwise for the quaternion from angle-axis.
    }

    __host__ __device__ inline void debug() {
        printf("w:%f, x:%f, y:%f, z:%f\t", w, x, y, z);
    }

    // functions to make code with mixed template parameters work...
    template <typename U>
    __host__ __device__ explicit VX3_Quat3D<T>(const VX3_Quat3D<U> &QuatIn) {
        w = QuatIn.w;
        x = QuatIn.x;
        y = QuatIn.y;
        z = QuatIn.z;
    } //!< Copy constructor from another template type
    template <typename U>
    __host__ __device__ explicit VX3_Quat3D<T>(const VX3_Vec3D<U> &VecIn) {
        w = 0;
        x = VecIn.x;
        y = VecIn.y;
        z = VecIn.z;
    } //!< Copies x, y, z from the specified vector and sets w to zero.
    template <typename U> __host__ __device__ explicit operator VX3_Quat3D<U>() const {
        return VX3_Quat3D<U>(w, x, y, z);
    } //!< overload conversion operator for different template types
    template <typename U>
    __host__ __device__ VX3_Quat3D<T> operator=(const VX3_Quat3D<U> &s) {
        w = s.w;
        x = s.x;
        y = s.y;
        z = s.z;
        return *this;
    } //!< Equals operator for different template types
    template <typename U>
    __host__ __device__ const VX3_Quat3D<T> operator+(const VX3_Quat3D<U> &s) {
        return VX3_Quat3D<T>(w + s.w, x + s.x, y + s.y, z + s.z);
    } //!< Addition operator for different template types
    template <typename U>
    __host__ __device__ const VX3_Quat3D<T> operator*(const U &f) const {
        return VX3_Quat3D<T>(f * w, f * x, f * y, f * z);
    } //!< Scalar multiplication operator for different template types
    template <typename U>
    __host__ __device__ const VX3_Quat3D<T> operator*(const VX3_Quat3D<U> &f) const {
        return VX3_Quat3D(
            w * f.w - x * f.x - y * f.y - z * f.z, w * f.x + x * f.w + y * f.z - z * f.y,
            w * f.y - x * f.z + y * f.w + z * f.x, w * f.z + x * f.y - y * f.x + z * f.w);
    } //!< Quaternion multplication operator for different template types

    // overload operators
    __host__ __device__ VX3_Quat3D &operator=(const VX3_Quat3D &s) {
        w = s.w;
        x = s.x;
        y = s.y;
        z = s.z;
        return *this;
    } //!< overload equals
    __host__ __device__ const VX3_Quat3D operator+(const VX3_Quat3D &s) const {
        return VX3_Quat3D(w + s.w, x + s.x, y + s.y, z + s.z);
    } //!< overload additoon
    __host__ __device__ const VX3_Quat3D operator-(const VX3_Quat3D &s) const {
        return VX3_Quat3D(w - s.w, x - s.x, y - s.y, z - s.z);
    } //!< overload subtraction
    __host__ __device__ const VX3_Quat3D operator*(const T f) const {
        return VX3_Quat3D(w * f, x * f, y * f, z * f);
    } //!< overload scalar multiplication
    __host__ __device__ const VX3_Quat3D friend operator*(const T f, const VX3_Quat3D v) {
        return VX3_Quat3D(v.w * f, v.x * f, v.y * f, v.z * f);
    } //!< overload scalar multiplication with number first.
    __host__ __device__ const VX3_Quat3D operator*(const VX3_Quat3D &f) const {
        return VX3_Quat3D(
            w * f.w - x * f.x - y * f.y - z * f.z, w * f.x + x * f.w + y * f.z - z * f.y,
            w * f.y - x * f.z + y * f.w + z * f.x, w * f.z + x * f.y - y * f.x + z * f.w);
    } //!< overload quaternion multiplication.
    __host__ __device__ bool operator==(const VX3_Quat3D &s) const {
        return (w == s.w && x == s.x && y == s.y && z == s.z);
    } //!< overload is equal.
    __host__ __device__ bool operator!=(const VX3_Quat3D &s) const {
        return (w != s.w || x != s.x || y != s.y || z != s.z);
    } //!< overload is not equal.
    __host__ __device__ const VX3_Quat3D &operator+=(const VX3_Quat3D &s) {
        w += s.w;
        x += s.x;
        y += s.y;
        z += s.z;
        return *this;
    } //!< overload add and set
    __host__ __device__ const VX3_Quat3D &operator-=(const VX3_Quat3D &s) {
        w -= s.w;
        x -= s.x;
        y -= s.y;
        z -= s.z;
        return *this;
    } //!< overload subtract and set

    //!< Explicit casting to a vector. Throws away w and copies x, y, z
    //!< directly.

    __host__ __device__ const VX3_Vec3D<T> ToVec() const { return VX3_Vec3D<T>(x, y, z); }
    // utilities

    //!< Returns the length (magnitude) of the quaternion.
    __host__ __device__ const T length() const { return sqrt(length2()); }

    //!< Returns the length (magnitude) squared of the quaternion.
    __host__ __device__ const T length2() const {
        return (w * w + x * x + y * y + z * z);
    }

    //!< Normalizes this quaternion. Returns the previous magnitude of this
    //!< quaternion before normalization. Note: function changes this
    //!< quaternion.
    __host__ __device__ const T normalize() {
        T l = length();
        if (l == 0) {
            w = 1;
            x = 0;
            y = 0;
            z = 0;
        } else if (l > 0) {
            T li = 1.0 / l;
            w *= li;
            x *= li;
            y *= li;
            z *= li;
        }
        return l;
    }

    //!< Normalizes this quaternion slightly faster than Normalize() by not
    //!< returning a value. Note: function changes this quaternion.
    __host__ __device__ void normalizeFast() {
        T l = sqrt(x * x + y * y + z * z + w * w);
        if (l != 0) {
            T li = 1.0 / l;
            w *= li;
            x *= li;
            y *= li;
            z *= li;
        }
        if (w >= 1.0) {
            w = 1.0;
            x = 0;
            y = 0;
            z = 0;
        }
    }
    //!< Returns a quaternion that is the inverse of this quaternion. This
    //!< quaternion is not modified.
    __host__ __device__ const VX3_Quat3D inverse() const {
        T n = w * w + x * x + y * y + z * z;
        return VX3_Quat3D(w / n, -x / n, -y / n, -z / n);
    }
    //!< Returns a quaternion that is the conjugate of this quaternion. This
    //!< quaternion is not modified.
    __host__ __device__ const VX3_Quat3D conjugate() const {
        return VX3_Quat3D(w, -x, -y, -z);
    }

    // angle and/or axis calculations
    //!< Returns the angular rotation of this quaternion in radians.
    __host__ __device__ const T angle() const { return 2.0 * acos(w > 1 ? 1 : w); }
    //!< Returns the angular rotation of this quaternion in degrees.
    __host__ __device__ const T angleDegrees() const {
        return angle() * 57.29577951308232;
    }
    //!< Returns true if the angular rotation of this quaternion is likely to
    //!< be considered negligible.
    __host__ __device__ bool isNegligibleAngle() const {
        return 2.0 * acos(w) < DISCARD_ANGLE_RAD;
    }

    //!< Returns true if the angular rotation of this quaternion is small
    //!< enough to be a good candidate for small angle approximations.
    __host__ __device__ bool isSmallAngle() const { return w > SMALL_ANGLE_W; }

    //!< Returns the normalized axis of rotation of this quaternion
    __host__ __device__ VX3_Vec3D<T> axis() const {
        T square_length = 1.0 - w * w; // because x*x + y*y + z*z + w*w = 1.0,
                                       // but more susceptible to w noise
        if (square_length <= 0) {
            return VX3_Vec3D<T>(1, 0, 0);
        } else {
            return VX3_Vec3D<T>(x, y, z) / sqrt(square_length);
        }
    }

    //!< Returns the un0normalized axis of rotation of this quaternion
    __host__ __device__ VX3_Vec3D<T> axisUnNormalized() const {
        return VX3_Vec3D<T>(x, y, z);
    }

    //!< Returns the angle and a normalize axis that represent this
    //!< quaternion's rotation. @param[out] angle The rotation angle in
    //!< radians. @param[out] axis The rotation axis in normalized vector
    //!< form.
    __host__ __device__ void angleAxis(T &angle, VX3_Vec3D<T> &axis) const {
        angleAxisUnNormalized(angle, axis);
        axis.normalizeFast();
    }

    //!< Returns the angle and an un-normalized axis that represent this
    //!< quaternion's rotation. @param[out] angle The rotation angle in
    //!< radians. @param[out] axis The rotation axis in un-normalized vector
    //!< form.
    __host__ __device__ void angleAxisUnNormalized(T &angle, VX3_Vec3D<T> &axis) const {
        if (w >= 1.0) {
            angle = 0;
            axis = VX3_Vec3D<T>(1, 0, 0);
            return;
        }
        angle = 2.0 * acos(w > 1 ? 1 : w);
        axis = VX3_Vec3D<T>(x, y, z);
    }

    //!< Returns a rotation vector representing this quaternion rotation.
    //!< Adapted from
    //!< http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
    __host__ __device__ const VX3_Vec3D<T> toRotationVector() const {
        if (w >= 1.0 || w <= -1.0)
            return VX3_Vec3D<T>(0, 0, 0);
        T square_length = 1.0 - w * w; // because x*x + y*y + z*z + w*w = 1.0,
                                       // but more susceptible to w noise (when
        if (square_length < SLTHRESH_ACOS2SQRT)
            return VX3_Vec3D<T>(x, y, z) * 2.0 *
                   sqrt((2 - 2 * w) /
                        square_length); // acos(w) = sqrt(2*(1-x)) for w close
                                        // to 1. for w=0.001, error is 1.317e-6
        else
            return VX3_Vec3D<T>(x, y, z) * 2.0 * acos(w) / sqrt(square_length);
    }

    //!< Overwrites this quaternion with values from the specified rotation
    //!< vector. Adapted from
    //!< http://physicsforgames.blogspot.com/2010/02/quaternions.html.  Note:
    //!< function changes this quaternion. @param[in] VecIn A rotation vector
    //!< to calculate this quaternion from.
    __host__ __device__ void FromRotationVector(const VX3_Vec3D<T> &vec_in) {
        VX3_Vec3D<T> theta = vec_in / 2;
        T s, theta_mag2 = theta.length2();
        if (theta_mag2 * theta_mag2 <
            DBL_EPSILONx24) { // if the 4th taylor expansion term is negligible
            w = 1.0 - 0.5 * theta_mag2;
            s = 1.0 - theta_mag2 / 6.0;
        } else {
            T theta_mag = sqrt(theta_mag2);
            w = cos(theta_mag);
            s = sin(theta_mag) / theta_mag;
        }
        x = theta.x * s;
        y = theta.y * s;
        z = theta.z * s;
    }

    //!< Overwrites this quaternion with the calculated rotation that would
    //!< transform the specified rotate_from vector to point in the positve X
    //!< direction. Note: function changes this quaternion.  @param[in]
    //!< rotate_from An arbitrary direction vector. Does not need to be
    //!< normalized.
    __host__ __device__ void
    fromAngleToPosX(const VX3_Vec3D<T>
                        &rotate_from) { // highly optimized at the expense of readability
        if (VX3_Vec3D<T>(0, 0, 0) == rotate_from)
            return; // leave off if it slows down too much!!

        // Catch and handle small angle:
        T y_over_x = rotate_from.y / rotate_from.x;
        T z_over_x = rotate_from.z / rotate_from.x;
        if (y_over_x < SMALL_ANGLE_RAD && y_over_x > -SMALL_ANGLE_RAD &&
            z_over_x < SMALL_ANGLE_RAD &&
            z_over_x > -SMALL_ANGLE_RAD) { // Intercept small angle and zero angle
            x = 0;
            y = 0.5 * z_over_x;
            z = -0.5 * y_over_x;
            w = 1 + 0.5 * (-y * y - z * z); // w=sqrt(1-x*x-y*y), small angle
                                            // sqrt(1+x) ~= 1+x/2 at x near zero.
            return;
        }

        // more accurate non-small angle:
        VX3_Vec3D<> rot_from_norm = rotate_from;
        rot_from_norm = rot_from_norm.normalized(); // Normalize the input...

        T theta =
            acos(rot_from_norm.x); // because rot_from_norm is normalized, 1,0,0 is
                                   // normalized, and A.B = |A||B|cos(theta) = cos(theta)
        if (theta > PI - DISCARD_ANGLE_RAD) {
            w = 0;
            x = 0;
            y = 1;
            z = 0;
            return;
        } // 180 degree rotation (about y axis, since the vector must be
          // pointing in -x direction

        const T axis_mag_inv = 1.0 / sqrt(rot_from_norm.z * rot_from_norm.z +
                                          rot_from_norm.y * rot_from_norm.y);
        // Here theta is the angle, axis is rot_from_norm.cross(VX3_Vec3D(1,0,0))
        // = VX3_Vec3D(0, rot_from_norm.z/axis_mag_inv, -rot_from_norm.y/axis_mag_inv),
        // which is still normalized. (super rolled together)
        const T a = 0.5 * theta;
        const T s = sin(a);
        w = cos(a);
        x = 0;
        y = rot_from_norm.z * axis_mag_inv * s;
        z = -rot_from_norm.y * axis_mag_inv * s; // angle axis function, reduced
    }

    //!< Returns a vector representing the specified vector "f" rotated by
    //!< this quaternion. @param[in] f The vector to transform.
    __host__ __device__ const VX3_Vec3D<T> rotateVec3D(const VX3_Vec3D<T> &f) const {
        T fx = f.x, fy = f.y, fz = f.z;
        T tw = fx * x + fy * y + fz * z;
        T tx = fx * w - fy * z + fz * y;
        T ty = fx * z + fy * w - fz * x;
        T tz = -fx * y + fy * x + fz * w;
        return VX3_Vec3D<T>(w * tx + x * tw + y * tz - z * ty,
                            w * ty - x * tz + y * tw + z * tx,
                            w * tz + x * ty - y * tx + z * tw);
    }

    //!< Returns a vector representing the specified vector "f" rotated by
    //!< this quaternion. Mixed template parameter version. @param[in] f The
    //!< vector to transform.
    template <typename U>
    __host__ __device__ const VX3_Vec3D<U> rotateVec3D(const VX3_Vec3D<U> &f) const {
        U fx = (U)(f.x), fy = (U)(f.y), fz = (U)(f.z);
        U tw = (U)(fx * x + fy * y + fz * z);
        U tx = (U)(fx * w - fy * z + fz * y);
        U ty = (U)(fx * z + fy * w - fz * x);
        U tz = (U)(-fx * y + fy * x + fz * w);
        return VX3_Vec3D<U>((U)(w * tx + x * tw + y * tz - z * ty),
                            (U)(w * ty - x * tz + y * tw + z * tx),
                            (U)(w * tz + x * ty - y * tx + z * tw));
    }

    //!< Returns a vector representing the specified vector "f" rotated by the
    //!< inverse of this quaternion. This is the opposite of RotateVec3D.
    //!< @param[in] f The vector to transform.
    __host__ __device__ const VX3_Vec3D<T> rotateVec3DInv(const VX3_Vec3D<T> &f) const {
        T fx = f.x, fy = f.y, fz = f.z;
        T tw = x * fx + y * fy + z * fz;
        T tx = w * fx - y * fz + z * fy;
        T ty = w * fy + x * fz - z * fx;
        T tz = w * fz - x * fy + y * fx;
        return VX3_Vec3D<T>(tw * x + tx * w + ty * z - tz * y,
                            tw * y - tx * z + ty * w + tz * x,
                            tw * z + tx * y - ty * x + tz * w);
    }

    //!< Returns a vector representing the specified vector "f" rotated by the
    //!< inverse of this quaternion. This is the opposite of RotateVec3D.
    //!< Mixed template parameter version. @param[in] f The vector to
    //!< transform.
    template <typename U>
    __host__ __device__ const VX3_Vec3D<U> rotateVec3DInv(const VX3_Vec3D<U> &f) const {
        T fx = f.x, fy = f.y, fz = f.z;
        T tw = x * fx + y * fy + z * fz;
        T tx = w * fx - y * fz + z * fy;
        T ty = w * fy + x * fz - z * fx;
        T tz = w * fz - x * fy + y * fx;
        return VX3_Vec3D<U>(tw * x + tx * w + ty * z - tz * y,
                            tw * y - tx * z + ty * w + tz * x,
                            tw * z + tx * y - ty * x + tz * w);
    }
};

using Quat3f = VX3_Quat3D<Vfloat>;
REFL_AUTO(type(Quat3f), field(x), field(y), field(z), field(w))
#endif // VX3_QUAT3D_H
