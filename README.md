/* ******************************************************************
*
*    DESCRIPTION:Copyright(c) 2020-2030 Xiamen Yealink Network Technology Co,.Ltd
*
*    AUTHOR: QiuZhiHao. The venus project authors. All Rights Reserved.
*
*    HISTORY:
*
*    DATE:2024-10-11
*
*
****************************************************************** */
#include "SSL_GMM.h"

static void Complex_Distance (DSP_Float in1_Re, DSP_Float in1_Im, DSP_Float in2_Re, DSP_Float in2_Im, DSP_Float *out)
{
    /*
    Complex euclidean distance
    */
    in1_Re = in1_Re - in2_Re;
    in1_Im = in1_Im - in2_Im;
    *out   = in1_Re * in1_Re + in1_Im * in1_Im;
    
    return;
};

static void Probability_Gaussian(DSP_Float x_miu, DSP_Float sigma, DSP_Float *out)
{
    /*
    Probability density function of Gaussian distribution = 
    1/(sqrt(sigma*2*pi)) * exp(-(x-miu)^2/(2*sigma))
    
    Logarithmic operation log() = 
    -log(2*pi)*0.5-log(Sigma)*0.5 - (x-miu)^2/(2 *Sigma);
    
    x_miu = (x-miu)^2
    */
    DSP_Float Eps = 0.000001f;
    
    *out = -0.9189f - ((DSP_Float)(log(sigma)))*0.5f - x_miu / (2 * sigma+ Eps);
    *out = (DSP_Float)(exp(*out));
    
    return;
};

DSP_Word32 GMM_EM_Coarse(SSL_Instance *inst, DSP_Float(*x_re)[SSL_Fre_N], DSP_Float(*x_im)[SSL_Fre_N], DSP_Float(*miu_re)[SSL_MP_N][SSL_FreL_N], DSP_Float(*miu_im)[SSL_MP_N][SSL_FreL_N], DSP_Float *gmmweight)
{
    /*
    x            :Input,      N_Pair * N_Fre
    miu          :Mean Value, N_Cand_Coarse * N_Pair * N_Fre
    gmmweight    :Weight,     N_Cand_Coarse
    Psd          :Probability density function of Gaussian distribution
    Liklihood    :Liklihood
    Liklihood_Der:Derivative of Liklihood
    Regular      :Regularizer
    Gradient     :Gradient
    */
    DSP_Word16 i_pair  = 0;
    DSP_Word16 i_fre   = 0;
    DSP_Word16 i_fre_s = 0;
    DSP_Word16 i_cand  = 0;
    DSP_Word16 i_his   = 0;
    DSP_Float Eps      = 0.00000001f;

    DSP_Float X_Miu[SSL_Cand_Coarse_N][SSL_MP_N][SSL_FreL_N];
    DSP_Float X_Miu_Mean;
    DSP_Float Psd[SSL_Cand_Coarse_N][SSL_MP_N][SSL_FreL_N];
    DSP_Float Liklihood = 0;
    DSP_Float Liklihood_Sum[SSL_MP_N][SSL_FreL_N];
    DSP_Float PostLik_Sum;
    DSP_Float sigma_Sum;
    DSP_Float Liklihood_Der = 0;
    DSP_Float Regular = 0;
    DSP_Float Der = 0;
    DSP_Float Gradient = 0;
    DSP_Float GmmWeight_Sum = 0;
    
    DSP_Float Featere_N = (DSP_Float)(SSL_MP_N * inst->FreL_S_N);
    
    
    /*Expectation Step*/
    for (i_pair = 0; i_pair < SSL_MP_N; i_pair++)
    {
        for (i_fre = 0; i_fre < inst->FreL_S_N; i_fre++)
        {
            i_fre_s = inst->FreL_S_Index[i_fre];
            
            Liklihood_Sum[i_pair][i_fre] = 0;
            for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
            {
                Complex_Distance(x_re[i_pair][i_fre_s], x_im[i_pair][i_fre_s], miu_re[i_cand][i_pair][i_fre_s], miu_im[i_cand][i_pair][i_fre_s], &X_Miu[i_cand][i_pair][i_fre]);
                
                Probability_Gaussian(X_Miu[i_cand][i_pair][i_fre], inst->GMM_sigma_Coarse[i_cand], &Psd[i_cand][i_pair][i_fre]);
                Liklihood = Psd[i_cand][i_pair][i_fre] * gmmweight[i_cand];
                Liklihood_Sum[i_pair][i_fre] = Liklihood_Sum[i_pair][i_fre] + Liklihood;
            }
        }
    }
    
    /*Maximization Step*/
    GmmWeight_Sum = 0;
    for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
    {
        X_Miu_Mean = 0;
        Liklihood_Der = 0;
        for (i_pair = 0; i_pair < SSL_MP_N; i_pair++)
        {
            for (i_fre = 0; i_fre < inst->FreL_S_N; i_fre++)
            {
                X_Miu_Mean = X_Miu_Mean + X_Miu[i_cand][i_pair][i_fre];
                
                Liklihood_Der = Liklihood_Der +Psd[i_cand][i_pair][i_fre] / (Liklihood_Sum[i_pair][i_fre] + Eps);
            }
        }
        /*Euclidean History*/
        X_Miu_Mean = X_Miu_Mean / Featere_N;
        DSP_MEMCPY_F32(&inst->GMM_EM_Euclidean[i_cand][1], &inst->GMM_EM_Euclidean[i_cand][0], SSL_EM_N - 1);
        inst->GMM_EM_Euclidean[i_cand][0] = X_Miu_Mean;
        
        Liklihood_Der = -Liklihood_Der / Featere_N;
        
        /*PostLik History*/
        DSP_MEMCPY_F32(&inst->GMM_EM_PostLik[i_cand][1], &inst->GMM_EM_PostLik[i_cand][0], SSL_EM_N - 1);
        inst->GMM_EM_PostLik[i_cand][0] = -Liklihood_Der;
        
        Regular = -(1 + (DSP_Float)(log(gmmweight[i_cand])));
        Der = Liklihood_Der + inst->GMM_gamma_Coarse * Regular;
        Gradient = (DSP_Float)(exp(-inst->GMM_eta*Der));
        if (Gradient > 100)
        {
            Gradient = 100;
        }
        gmmweight[i_cand] = gmmweight[i_cand] * Gradient;
        GmmWeight_Sum = GmmWeight_Sum + gmmweight[i_cand];
    }
    
    for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
    {
        gmmweight[i_cand] = gmmweight[i_cand] / (GmmWeight_Sum + Eps);
        
        PostLik_Sum = 0;
        sigma_Sum   = 0;
        for (i_his = 0; i_his < SSL_EM_N; i_his++)
        {
            PostLik_Sum = PostLik_Sum + inst->GMM_EM_PostLik[i_cand][i_his];
            sigma_Sum = sigma_Sum + inst->GMM_EM_PostLik[i_cand][i_his] * inst->GMM_EM_Euclidean[i_cand][i_his];
        }
        inst->GMM_sigma_Coarse[i_cand] = sigma_Sum / (PostLik_Sum + Eps);
    }
    
    return 0;
};

DSP_Word32 GMM_EM_Zoom(SSL_Instance *inst, DSP_Float(*x_re)[SSL_Fre_N], DSP_Float(*x_im)[SSL_Fre_N], DSP_Float(*miu_re)[SSL_MP2_N][SSL_FreM_N], DSP_Float(*miu_im)[SSL_MP2_N][SSL_FreM_N], DSP_Float *gmmweight)
{
    /*
    x            :Input,      N_Pair * N_Fre
    miu          :Mean Value, N_Cand_Coarse * N_Pair * N_Fre
    gmmweight    :Weight,     N_Cand_Coarse
    Psd          :Probability density function of Gaussian distribution
    Liklihood    :Liklihood
    Liklihood_Der:Derivative of Liklihood
    Regular      :Regularizer
    Gradient     :Gradient
    */
    DSP_Word16 i_pair = 0;
    DSP_Word16 pair_index = 0;
    DSP_Word16 i_fre = 0;
    DSP_Word16 i_fre_x = 0;
    DSP_Word16 i_fre_miu = 0;
    DSP_Word16 i_cand = 0;
    DSP_Float Eps = 0.00000001f;

    DSP_Float X_Miu = 0.0f;
    DSP_Float Psd[SSL_Cand_Zoom_N][SSL_MP2_N][SSL_FreM_N];
    DSP_Float Liklihood = 0.0f;
    DSP_Float Liklihood_Sum[SSL_MP2_N][SSL_FreM_N];
    DSP_Float Liklihood_Der = 0.0f;
    DSP_Float Regular = 0.0f;
    DSP_Float Der = 0.0f;
    DSP_Float Gradient = 0.0f;
    DSP_Float GmmWeight_Sum = 0.0f;

    DSP_Float Featere_N = (DSP_Float)(SSL_MP2_N * inst->FreM_S_N);
    
    if (inst->GMM_Cand_Coarse_Index != inst->GMM_Cand_Coarse_Post)
    {
        for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
        {
            gmmweight[i_cand] = 1.0f / SSL_Cand_Zoom_N;
        }
    }
    
    /*Expectation Step*/
    for (i_pair = 0; i_pair < SSL_MP2_N; i_pair++)
    {
        pair_index = MicPair2_Index[i_pair]-1;
        for (i_fre = 0; i_fre < inst->FreM_S_N; i_fre++)
        {
            i_fre_x = inst->FreM_X_Index[i_fre];
            i_fre_miu = inst->FreM_Miu_Index[i_fre];
            Liklihood_Sum[i_pair][i_fre] = 0;
            for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
            {
                Complex_Distance(x_re[pair_index][i_fre_x], x_im[pair_index][i_fre_x], miu_re[i_cand][i_pair][i_fre_miu], miu_im[i_cand][i_pair][i_fre_miu], &X_Miu);
                Probability_Gaussian(X_Miu, inst->GMM_sigma_Zoom, &Psd[i_cand][i_pair][i_fre]);
                Liklihood = Psd[i_cand][i_pair][i_fre] * gmmweight[i_cand];
                Liklihood_Sum[i_pair][i_fre] = Liklihood_Sum[i_pair][i_fre] + Liklihood;
            }
        }
    }
    
    /*Maximization Step*/
    GmmWeight_Sum = 0.0f;
    for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
    {
        Liklihood_Der = 0.0f;
        for (i_pair = 0; i_pair < SSL_MP2_N; i_pair++)
        {
            for (i_fre = 0; i_fre < inst->FreM_S_N; i_fre++)
            {
                Liklihood_Der = Liklihood_Der + Psd[i_cand][i_pair][i_fre] / (Liklihood_Sum[i_pair][i_fre] + Eps);
            }
        }
        Liklihood_Der = -Liklihood_Der / Featere_N;
        Regular = -(1.0f + (DSP_Float)(log(gmmweight[i_cand])));
        Der = Liklihood_Der + inst->GMM_gamma_Zoom * Regular;
        Gradient = (DSP_Float)(exp(-inst->GMM_eta*Der));
        if (Gradient > 100)
        {
            Gradient = 100;
        }
        gmmweight[i_cand] = gmmweight[i_cand] * Gradient;
        GmmWeight_Sum = GmmWeight_Sum + gmmweight[i_cand];
    }
    for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
    {
        gmmweight[i_cand] = gmmweight[i_cand] / (GmmWeight_Sum + Eps);
    }
    
    return 0;
};

void GMM_Reset(SSL_Instance *inst)
{
    DSP_Word16 i_cand = 0;
    
    for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
    {
        inst->GMM_Weight_Coarse[i_cand] = 1.0f / SSL_Cand_Coarse_N;
        inst->GMM_sigma_Coarse[i_cand]  = 0.1f;
    }
    
    for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
    {
        inst->GMM_Weight_Zoom[i_cand] = 1.0f / SSL_Cand_Zoom_N;
    }
    
    inst->GMM_Cand_Coarse_Post = 0;
};

void GMM_Init(SSL_Instance *inst)
{
    DSP_Word16 i_cand = 0;
    DSP_Word16 i_pair = 0;
    DSP_Word16 i_fre = 0;
    DSP_Word16 i_his = 0;
    
    for (i_his = 0; i_his < SSL_EM_N; i_his++)
    {
        for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
        {
            inst->GMM_EM_Euclidean[i_cand][i_his] = 0.1f;
            inst->GMM_EM_PostLik[i_cand][i_his] = 1.0f / SSL_Cand_Coarse_N;
        }
    }
    
    inst->GMM_sigma_Zoom   = 0.1f;
    inst->GMM_eta          = 0.8f;
    inst->GMM_eta_silent   = 0.065f;
    inst->GMM_gamma_Coarse = 0.001f;
    inst->GMM_gamma_Zoom   = 0.0001f;
    
};

void GMM_Reset_Coarse_Silent(SSL_Instance *inst)
{
    DSP_Word16 i_cand = 0;
    
    for (i_cand = 0; i_cand < SSL_Cand_Coarse_N; i_cand++)
    {
        inst->GMM_Weight_Coarse[i_cand] = (1.0f - inst->GMM_eta_silent) * inst->GMM_Weight_Coarse[i_cand] + inst->GMM_eta_silent / SSL_Cand_Coarse_N;
        inst->GMM_sigma_Coarse[i_cand] = 0.1f;
    }
    
    for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
    {
        inst->GMM_Weight_Zoom[i_cand] = (1 - inst->GMM_eta_silent) * inst->GMM_Weight_Zoom[i_cand] + inst->GMM_eta_silent / SSL_Cand_Zoom_N;
    }
};

void GMM_Reset_Zoom_Silent(SSL_Instance *inst)
{
    DSP_Word16 i_cand = 0;
    
    for (i_cand = 0; i_cand < SSL_Cand_Zoom_N; i_cand++)
    {
        inst->GMM_Weight_Zoom[i_cand] = (1 - inst->GMM_eta_silent) * inst->GMM_Weight_Zoom[i_cand] + inst->GMM_eta_silent / SSL_Cand_Zoom_N;
    }
};
