#ifndef VX3_MATH_TREE_H
#define VX3_MATH_TREE_H
#include "assert.h"
#include "utils/vx3_def.h"

#define MAX_STACK_SIZE              4
#define MAX_EXPRESSION_TOKENS       16

enum VX3_MathTreeOperator : unsigned int {
    mtEND,
    mtCONST,
    mtE,  // number e
    mtPI, // number pi
    mtVAR,
    mtADD,
    mtSUB,
    mtMUL,
    mtDIV,
    mtPOW,  // power
    mtSQRT, // sqrt
    mtSIN,
    mtCOS,
    mtTAN,
    mtATAN,
    mtLOG, // log_e
    mtINT, // round to nearest integer. e.g. 0.9 --> 1.0
    mtABS,
    mtNOT,
    mtGREATERTHAN,
    mtLESSTHAN,
    mtAND,
    mtOR,
    mtNORMALCDF, // normal CDF function
};
struct VX3_MathTreeToken {
    VX3_MathTreeOperator op = mtEND;
    Vfloat value = 0.0;

#ifndef __CUDACC__
    void set(VX3_MathTreeOperator inOp, Vfloat inValue = 0.0) {
        op = inOp;
        value = inValue;
    }
#else
    __host__ __device__ void set(VX3_MathTreeOperator inOp, Vfloat inValue = 0.0) {
        op = inOp;
        value = inValue;
    }
#endif
};

struct VX3_MathTree {

    static bool validate(const VX3_MathTreeToken *buff) {
        try {
            eval(1, 1, 1, 1, 1, 1, 1, 1, 1, buff);
        } catch (...) {
            return false;
        }
        return true;
    }

    static bool isExpressionValid(const VX3_MathTreeToken *buff) {
        int cursor = -1;
        for (int i = 0; i < MAX_EXPRESSION_TOKENS; i++) {
            switch (buff[i].op) {
                case mtEND:
                    return true;
                case mtCONST:
                case mtE:
                case mtPI:
                case mtVAR:
                    // Cursor writes value and move to next position (put 1)
                    if (cursor >= MAX_STACK_SIZE - 1)
                        return false;
                    cursor++;
                    break;
                case mtSIN:
                case mtCOS:
                case mtTAN:
                case mtATAN:
                case mtLOG:
                case mtINT:
                case mtNORMALCDF:
                case mtSQRT:
                case mtABS:
                case mtNOT:
                    // Cursor writes value to the input position (take 1 put 1)
                    break;
                case mtADD:
                case mtSUB:
                case mtMUL:
                case mtDIV:
                case mtPOW:
                case mtGREATERTHAN:
                case mtLESSTHAN:
                case mtAND:
                case mtOR:
                    // Cursor writes value to the first input position (take 2 put 1)
                    cursor--;
                    break;
                default:
                    return false;
            }
        }
        return false;
    }

    /* Standard implementation is CUDA Math API
    https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    */
    __host__ __device__ static Vfloat eval(Vfloat x, Vfloat y, Vfloat z,
                                           Vfloat hit, Vfloat t, Vfloat angle,
                                           Vfloat closeness, int num_close_pairs,
                                           int num_voxel,
                                           const VX3_MathTreeToken *buff) {
        Vfloat inputs[9] = {x,
                            y,
                            z,
                            hit,
                            t,
                            angle,
                            closeness,
                            (Vfloat)num_close_pairs,
                            (Vfloat)num_voxel};
        // Registers
        Vfloat values[MAX_STACK_SIZE] = {0};
        // Cursor points to the location of the last element
        int cursor = -1;
        for (int i = 0; i < MAX_EXPRESSION_TOKENS; i++) {
            switch (buff[i].op) {
            case mtEND:
                return cursor == -1 ? 0 : values[cursor];
            case mtCONST:
                assert(cursor < MAX_STACK_SIZE - 1);
                values[++cursor] = buff[i].value;
                break;
            case mtE:
                assert(cursor < MAX_STACK_SIZE - 1);
                values[++cursor] = 2.71828182845904523536;
                break;
            case mtPI:
                assert(cursor < MAX_STACK_SIZE - 1);
                values[++cursor] = 3.14159265358979323846;
                break;
            case mtVAR:
                // Checking of index is done by the parser
                assert(cursor < MAX_STACK_SIZE - 1);
                values[++cursor] = inputs[llrintf(buff[i].value)];
                break;
            case mtSIN:
                values[cursor] = sin(values[cursor]);
                break;
            case mtCOS:
                values[cursor] = cos(values[cursor]);
                break;
            case mtTAN:
                values[cursor] = tan(values[cursor]);
                break;
            case mtATAN:
                values[cursor] = atan(values[cursor]);
                break;
            case mtLOG:
                values[cursor] = log(values[cursor]);
                break;
            case mtINT:
                values[cursor] = rint(values[cursor]);
                break;
            case mtNORMALCDF:
                values[cursor] = normcdf(values[cursor]);
                break;
            case mtSQRT:
                values[cursor] = sqrt(values[cursor]);
                break;
            case mtABS:
                values[cursor] = abs(values[cursor]);
                break;
            case mtNOT:
                values[cursor] = values[cursor] < 0.5;
                break;
            case mtADD:
                values[cursor - 1] = values[cursor - 1] + values[cursor];
                cursor--;
                break;
            case mtSUB:
                values[cursor - 1] = values[cursor - 1] - values[cursor];
                cursor--;
                break;
            case mtMUL:
                values[cursor - 1] = values[cursor - 1] * values[cursor];
                cursor--;
                break;
            case mtDIV:
                values[cursor - 1] = values[cursor - 1] / values[cursor];
                cursor--;
                break;
            case mtPOW:
                values[cursor - 1] = pow(values[cursor - 1], values[cursor]);
                cursor--;
                break;
            case mtGREATERTHAN:
                values[cursor - 1] = values[cursor - 1] > values[cursor];
                cursor--;
                break;
            case mtLESSTHAN:
                values[cursor - 1] = values[cursor - 1] < values[cursor];
                cursor--;
                break;
            case mtAND:
                values[cursor - 1] = values[cursor - 1] > 0.5 && values[cursor] > 0.5;
                cursor--;
                break;
            case mtOR:
                values[cursor - 1] = values[cursor - 1] > 0.5 || values[cursor] > 0.5;
                cursor--;
                break;

            default:
#ifdef __CUDA_ARCH__
                printf("ERROR: not implemented.\n");
                return -1;
#else
                throw 0;
#endif
            }

        }
#ifdef __CUDA_ARCH__
        printf("ERROR: math tree overflow, expression not terminated.\n");
        return -1;
#else
        throw 2;
#endif
    }
};

#endif // VX3_MATH_TREE_H
