#include "vx3_link_material.h"
#include "vx3_voxelyze_kernel.cuh"

// Must include this at end because we need to define
// reflection macros in vx3_link, vx3_voxel, etc. first
#include "vx3/vx3_context.h"

void VX3_LinkMaterial::init(const VX3_InitContext &ictx) {
    auto &vox1_mat_ref = ictx.voxel_materials[vox1_mat];
    auto &vox2_mat_ref = ictx.voxel_materials[vox2_mat];

    nom_size = (Vfloat)0.5 * (vox2_mat_ref.nom_size + vox2_mat_ref.nom_size);
    r = (int)(0.5 * (vox1_mat_ref.r + vox2_mat_ref.r));
    g = (int)(0.5 * (vox1_mat_ref.g + vox2_mat_ref.g));
    b = (int)(0.5 * (vox1_mat_ref.b + vox2_mat_ref.b));
    a = (int)(0.5 * (vox1_mat_ref.a + vox2_mat_ref.a));

    // Since link material is not a real material
    material_id = -1;

    // TODO: are these relations correct?
    fixed = vox1_mat_ref.fixed and vox2_mat_ref.fixed;
    sticky = vox1_mat_ref.sticky and vox1_mat_ref.sticky;
    cilia = (Vfloat)0.5 * (vox1_mat_ref.cilia + vox2_mat_ref.cilia);

    // linear, E, nu, and fail_stress are computed later
    rho = (Vfloat)0.5 * (vox1_mat_ref.rho + vox2_mat_ref.rho);
    alpha_CTE = (Vfloat)0.5 * (vox1_mat_ref.alpha_CTE + vox2_mat_ref.alpha_CTE);
    u_static = (Vfloat)0.5 * (vox1_mat_ref.u_static + vox2_mat_ref.u_static);
    u_kinetic = (Vfloat)0.5 * (vox1_mat_ref.u_kinetic + vox2_mat_ref.u_kinetic);
    zeta_internal =
        (Vfloat)0.5 * (vox1_mat_ref.zeta_internal + vox2_mat_ref.zeta_internal);
    zeta_global = (Vfloat)0.5 * (vox1_mat_ref.zeta_global + vox2_mat_ref.zeta_global);
    zeta_collision =
        (Vfloat)0.5 * (vox1_mat_ref.zeta_collision + vox2_mat_ref.zeta_collision);

    // failure stress (f) is the minimum of the two failure stresses,
    // or if both are -1.0f
    // it should also be -1.0f to denote no failure specified
    Vfloat f1 = vox1_mat_ref.sigma_fail, f2 = vox2_mat_ref.sigma_fail;
    if (f1 == -1.0f)
        fail_stress = f2; //-1.0f or vox2Mat fail
    else if (f2 == -1.0f)
        fail_stress = f1; // vox1_mat fail
    else
        fail_stress = f1 < f2 ? f1 : f2; // the lesser stress denotes failure

    if (vox1_mat_ref.linear && vox2_mat_ref.linear)
        E = (Vfloat) 2.0 * vox1_mat_ref.E * vox2_mat_ref.E / (vox1_mat_ref.E + vox2_mat_ref.E);
    else {
        // Not implemented
        // at least 1 bilinear or data-based, so build up data points and apply it.
//        VX3_dVector<float> newStressValues, newStrainValues;
//        newStressValues.push_back(0.0f);
//        newStrainValues.push_back(0.0f);
//
//        // step up through ascending strains data points (could alternate randomly between
//        // vox1_mat and vox2Mat points
//        int dataIt1 = 1, dataIt2 = 1; // iterators through each data point of the model
//        while (dataIt1 < (int)vox1_mat.d_strainData.size() &&
//               dataIt2 < (int)vox2_mat.d_strainData.size()) {
//            float strain =
//                FLT_MAX; // strain for the next data point is the smaller of the two
//                         // possible next strain points (but we have to make sure we don't
//                         // access off the end of one of the arrays)
//            if (dataIt1 < (int)vox1_mat.d_strainData.size())
//                strain = vox1_mat.d_strainData[dataIt1];
//            if (dataIt2 < (int)vox2_mat.d_strainData.size() &&
//                vox2_mat.d_strainData[dataIt2] < strain)
//                strain = vox2_mat.d_strainData[dataIt2];
//            else
//                assert(strain != FLT_MAX); // this should never happen
//
//            if (strain == vox1_mat.d_strainData[dataIt1])
//                dataIt1++;
//            if (strain == vox2_mat.d_strainData[dataIt2])
//                dataIt2++;
//
//            float modulus1 = vox1_mat.modulus(strain - FLT_EPSILON);
//            float modulus2 = vox2_mat.modulus(strain - FLT_EPSILON);
//            float thisModulus = 2.0f * modulus1 * modulus2 / (modulus1 + modulus2);
//
//            // add to the new strain/stress values
//            int lastDataIndex = newStrainValues.size() - 1;
//
//            newStrainValues.push_back(strain);
//            newStressValues.push_back(
//                newStressValues[lastDataIndex] +
//                thisModulus *
//                    (strain -
//                     newStrainValues[lastDataIndex])); // springs in series equation
//        }
//
//        setModel(newStrainValues.size(), &newStrainValues[0], &newStressValues[0]);
//
//        // override failure points in case no failure was specified before (as possible in
//        // combos of linear and bilinear materials) yield point is handled correctly in
//        // setModel.
//        sigmaFail = stressFail;
//        epsilonFail = stressFail == -1.0f ? -1.0f : strain(stressFail);
    }

    // poissons ratio: choose such that Ehat ends up according to
    // spring in series of Ehat1 and EHat2
    if (vox1_mat_ref.nu == 0 && vox2_mat_ref.nu == 0)
        nu = 0;
    else {
        Vfloat tmp_E_hat =
            2 * vox1_mat_ref.E_hat * vox2_mat_ref.E_hat / (vox1_mat_ref.E_hat + vox2_mat_ref.E_hat);
        // completing the square algorithm to solve for nu.
        // eHat = E/((1-2nu)(1+nu)) -> E/EHat = -2nu^2-nu+1 -> nu^2+0.5nu =
        // (EHat+E)/(2EHat)

        // nu^2+0.5nu+0.0625 = c2 -> (nu+0.25)^2 = c2
        Vfloat c2 = (Vfloat)((tmp_E_hat - E) / (2 * tmp_E_hat) + 0.0625);
        nu = (Vfloat)(sqrt(c2) - 0.25); // from solving above
    }

    // Initialize base VX3_Material class derived variables
    VX3_VoxelMaterial::init(ictx);

    // Update derived attributes
    // stiffnesses terms for links
    Vfloat L = nom_size;
    a1 = E * L;                              // EA/L : Units of N/m
    a2 = E * L * L * L / (12.0f * (1 + nu)); // GJ/L : Units of N-m
    b1 = E * L;                              // 12EI/L^3 : Units of N/m
    // 6EI/L^2 : Units of N (or N-m/m: torque related to linear distance)
    b2 = E * L * L / 2.0f;
    b3 = E * L * L * L / 6.0f; // 2EI/L : Units of N-m

    // damping sqrt(mk) terms (with sqrt(m) factored out)
    sqA1 = sqrt(a1);
    sqA2xIp = sqrt(a2 * L * L / 6.0f);
    sqB1 = sqrt(b1);
    sqB2xFMp = sqrt(b2 * L / 2.0f);
    sqB3xIp = sqrt(b3 * L * L / 6.0f);
}
