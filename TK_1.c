#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
// #include "dvmh_runtime_api.h"


#define pi 3.14159265358979323846264338327950288
#define LX 0.0
#define RX 2.0
#define BY 0.0
#define TY 1.0
#define eps 1e-5
#define ITER_LIMIT 400


#define M 500
#define N 500
const double h1 = (double)(RX - LX) / (double)M;
const double h2 = (double)(TY - BY) / (double)N;

#pragma dvm array distribute [block][block]
double B[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double Aw[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double w_prev[M + 1][N + 1]; 
#pragma dvm array align([i][j] with B[i][j])
double diff_vec[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double Ar[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double U_true[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j]), shadow[1:1][1:1]
double w[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j]), shadow[1:1][1:1]
double r[M + 1][N + 1];

double u(double x, double y)
{
    return 1.0 + cos(pi * x * y);
}


double q(double x, double y)
{
    return 0;
}


double k(double x, double y)
{
    return 4.0 + x + y;
}

double F(double x, double y)
{
    return (-1.0) * (pi * y * ((-1.0) * sin(pi * x * y) - pi * y * (x + y + 4.0) * cos(pi* x * y)) + pi * x * ((-1.0) * sin(pi * x * y) - pi * x * (x + y + 4.0) * cos(pi * x * y)));
}


double psi_R(double x, double y){
    return (1.0 + cos(pi * x * y)) - (4.0 + x + y) * pi * y * sin(pi * x * y);
}


double psi_L(double x, double y){
    return (4.0 + x + y) * pi * y * sin(pi * x * y);
}

double psi_B(double x, double y){
    return (4.0 + x + y) * pi * x * sin(pi * x * y);
}


double psi_T(double x, double y){
    return (1.0 + cos(pi * x * y)) - (4.0 + x + y) * pi * x * sin(pi * x * y);
}


void Calu_Aw()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Aw[i][j]) shadow_renew(w)
    for (i = 1; i < M; i++){
        for (j = 1; j < N; j++) {

            Aw[i][j] = (-1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + j * h2) * ((w[i + 1][j] - w[i][j]) / h1)
                   - k(LX + (i - 0.5) * h1, BY + j * h2) * (w[i][j] - w[i - 1][j]) / h1) - (1.0 /h2) * (k(LX + i * h1, BY + (j + 0.5) * h2) * (w[i][j + 1] - w[i][j]) / h2
                   - k(LX + i * h1, BY + (j - 0.5) * h2) * (w[i][j] - w[i][j - 1]) / h2);
        }
    }
    // left boundary
    #pragma dvm parallel([j] on Aw[0][j]) shadow_renew(w)
    for (j = 0; j < N; j++) {
            Aw[0][j] = (-2.0 / h1) * (k(LX + (1 - 0.5) * h1, BY + j * h2) * (w[1][j] - w[0][j]) / h1) - 
            (1.0 / h2) * (k(LX + 0.0 * h1, BY + (j + 0.5) * h2) * ((w[0][j + 1] - w[0][j]) / h2)
                   - k(LX + 0.0 * h1, BY + (j - 0.5) * h2) * ((w[0][j] - w[0][j - 1]) / h2));
            if (j == 0){
                Aw[0][0] = (-2.0 / h1) * (k(LX + (1 - 0.5) * h1, BY + 0.0 * h2) * (w[1][0] - w[0][0]) / h1) - 
    (2.0 / h2) * (k(LX + 0.0 * h1, BY + (1 - 0.5) * h2) * (w[0][1] - w[0][0]) / h2);
            }
    }
    
    // right
    #pragma dvm parallel([j] on Aw[M][j]) shadow_renew(w)
    for (j = 1; j < N + 1; j++) {
            Aw[M][j] = (2.0 / h1) * (k(LX + (M - 0.5) * h1, BY + j * h2) * (w[M][j] - w[M - 1][j]) / h1) + (2.0 / h1) * w[M][j] - 
            (1.0 / h2) * (k(LX + M * h1, BY + (j + 0.5) * h2) * (w[M][j + 1] - w[M][j]) / h2
                   - k(LX + M * h1, BY + (j - 0.5) * h2) * (w[M][j] - w[M][j - 1]) / h2);
            if (j == N){
                Aw[M][N] = (2.0 / h1) * (k(LX + (M - 0.5) * h1, BY + N * h2) * (w[M][N] - w[M - 1][N]) / h1) + 
    (2.0 / h2) * (k(LX + M * h1, BY + (N - 0.5) * h2) * (w[M][N] - w[M][N - 1]) / h2) + (2.0 / h1 + 2.0 / h2) * w[M][N];
            }
    }
    
    // top
    #pragma dvm parallel([i] on Aw[i][N]) shadow_renew(w)
    for (i = 0; i < M; i++) {
            Aw[i][N] = (2.0 / h2) * (k(LX + i * h1, BY + (N - 0.5) * h2) * (w[i][N] - w[i][N - 1]) / h2) + 
            (2.0 / h2) * w[i][N] - (1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + N * h2) * (w[i + 1][N] - w[i][N]) / h1
                   - k(LX + (i - 0.5) * h1, BY + N * h2) * (w[i][N] - w[i - 1][N]) / h1);
            if (i == 0){
                Aw[0][N] = (-2.0 / h1) * (k(LX + (1 - 0.5) * h1, BY + N * h2) * (w[1][N] - w[0][N]) / h1) +
    (2.0 / h2) * (k(LX + 0 * h1, BY + (N - 0.5) * h2) * (w[0][N] - w[0][N - 1]) / h2) + 
    (2.0 / h2)* w[0][N];
            }
        }
    
    // bot 
    #pragma dvm parallel([i] on Aw[i][0]) shadow_renew(w)
    for (i = 1; i < M + 1; i++) {
            Aw[i][0] = (-2.0 / h2) * (k(LX + i * h1, BY + (1 - 0.5) * h2) * (w[i][1] - w[i][0]) / h2) - 
                   (1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + 0.0 * h2) * ((w[i + 1][0] - w[i][0])/h1) - 
                   k(LX + (i - 0.5) * h1, BY + 0.0 * h2) * ((w[i][0] - w[i-1][0])/h1));
            if (i == M){
                Aw[M][0] = (2.0 / h1) * (k(LX + (M - 0.5) * h1, BY + 0.0 * h2) * (w[M][0] - w[M - 1][0]) / h1) - 
    (2.0 / h2) * (k(LX + M * h1, BY + (1 - 0.5) * h2) * (w[M][1] - w[M][0]) / h2) + 
    (2.0 / h1) * w[M][0];
            }
        }
    
    }
}

void calu_r_Aw_B(){
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on r[i][j])
    for(i = 0; i < M + 1; i++)
    {
        for (j = 0; j < N + 1; j++)
        {
                r[i][j] = Aw[i][j] - B[i][j];
                w_prev[i][j] = w[i][j];
        }
    }
    }
}

void Calu_Ar()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) shadow_renew(r)
    for (i = 1; i < M; i++){
        for (j = 1; j < N; j++) {

            Ar[i][j] = -(1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + j * h2) * ((r[i + 1][j] - r[i][j]) / h1)
                   - k(LX + (i - 0.5) * h1, BY + j * h2) * ((r[i][j] - r[i - 1][j]) / h1)) - (1.0 /h2) * (k(LX + i * h1,BY + (j + 0.5) * h2) * (r[i][j + 1] - r[i][j]) / h2
                   - k(LX + i * h1, BY + (j - 0.5) * h2) * ((r[i][j] - r[i][j - 1]) / h2));
        }
    }

    #pragma dvm parallel([j] on Ar[0][j]) shadow_renew(r)
    for (j = 0; j < N; j++) {
            Ar[0][j] = (-1) * (2.0 / h1) * (k(LX + (1 - 0.5) * h1, BY + j * h2) * (r[1][j] - r[0][j]) / h1) - 
            ((1.0 /h2) * (k(LX + 0.0 * h1, BY + (j + 0.5) * h2) * ((r[0][j + 1] - r[0][j]) / h2)
                   - k(LX + 0.0 * h1, BY + (j - 0.5) * h2) * ((r[0][j] - r[0][j - 1]) / h2)));
            if (j == 0){
                Ar[0][0] = -(2.0 /h1) * (k(LX + (1 - 0.5) * h1, BY + 0.0 * h2) * (r[1][0] - r[0][0]) / h1) - 
    (2.0 / h2) * (k(LX + 0.0 * h1,BY + (1 - 0.5) * h2) * (r[0][1] - r[0][0]) / h2);
            }
    }
    
    #pragma dvm parallel([j] on Ar[M][j]) shadow_renew(r)
    for (j = 1; j < N + 1; j++) {
            Ar[M][j] = (2.0 / h1) * (k(LX + (M - 0.5) * h1, BY + j * h2) * (r[M][j] - r[M - 1][j]) / h1) + (2.0 /h1) * r[M][j] - 
            ((1.0 / h2) * (k(LX + M * h1, BY + (j + 0.5) * h2) * (r[M][j + 1] - r[M][j]) / h2
                   - k(LX + M * h1, BY + (j - 0.5) * h2) * (r[M][j] - r[M][j - 1]) / h2));
            if (j == N){
                Ar[M][N] = (2.0 /h1) * (k(LX + (M - 0.5) * h1, BY + N * h2) * ((r[M][N] - r[M - 1][N]) / h1)) + 
    (2.0 / h2) * (k(LX + M * h1, BY + (N - 0.5) * h2) * ((r[M][N] - r[M][N - 1]) / h2)) + 
    (2.0 /h1 + 2.0 /h2) * r[M][N];
            }
    }
    
    #pragma dvm parallel([i] on Ar[i][N]) shadow_renew(r)
    for (i = 0; i < M; i++) {
            Ar[i][N] = (2.0 / h2) * (k(LX + i * h1, BY + (N - 0.5) * h2) * (r[i][N] - r[i][N - 1]) / h2) + 
            (2.0 / h2) * r[i][N] - ((1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + N * h2) * (r[i + 1][N] - r[i][N]) / h1
                   - k(LX + (i - 0.5) * h1, BY + N * h2) * (r[i][N] - r[i - 1][N]) / h1));
            if (i == 0){
                Ar[0][N] = -(2.0 / h1) * (k(LX + (1 - 0.5) * h1, BY + N * h2) * (r[1][N] - r[0][N]) / h1) +
    (2.0 / h2) * (k(LX + 0.0 * h1, BY + (N - 0.5) * h2) * (r[0][N] - r[0][N - 1]) / h2) + 
    (2.0 / h2)* r[0][N];
            }
        }
    
    #pragma dvm parallel([i] on Ar[i][0]) shadow_renew(r)
    for (i = 1; i < M + 1; i++) {
            Ar[i][0] = (-2.0 / h2) * (k(LX + i * h1, BY + (1 - 0.5) * h2) * (r[i][1] - r[i][0]) / h2) - 
                   ((1.0 / h1) * (k(LX + (i + 0.5) * h1, BY + 0.0 * h2) * ((r[i + 1][0] - r[i][0])/h1) - 
                   k(LX + (i - 0.5) * h1, BY + 0 * h2) * ((r[i][0] - r[i-1][0])/h1)));
            if (i == M){
                Ar[M][0] = (2.0 / h1) * (k(LX + (M - 0.5) * h1, BY + 0.0 * h2) * (r[M][0] - r[M -1][0]) / h1) - 
    (2.0 / h2) * (k(LX + M * h1, BY + (1 - 0.5) * h2) * (r[M][1] - r[M][0]) / h2) + 
    (2.0 / h1) * r[M][0];
            }
        }
    
    }
}

void Calu_B()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on B[i][j])
    for(i = 1; i < M; i++)
        for (j = 1; j <  N; j++)
            B[i][j] = F(LX + i * h1, BY + j * h2);
    
    #pragma dvm parallel([j] on B[0][j])
    for (j = 0; j < N; j++) {
            B[0][j] = (F(LX, BY + j * h2) +
                    psi_L(LX, BY + j * h2) * 2.0 /h1);
        if (j == 0){
            B[0][0] =  (F(LX, BY)
                + (2.0 /h1 + 2.0 /h2) * (1.0 /(h1 + h2)) * (h2 * psi_L(LX, BY) + h1 * psi_B(LX, BY)));
        }
    }

    #pragma dvm parallel([j] on B[M][j])
    for (j = 1; j < N + 1; j++) {
        B[M][j] = (F(LX + M * h1, BY + j * h2) +
                    psi_R(LX + M * h1, BY + j * h2) * 2.0 /h1);
        if (j == N){
            B[M][N] = (F(LX + M * h1, BY + N * h2) +
                (2.0 /h1 + 2.0 /h2) * (1.0 /(h1 + h2)) * (h2 * psi_R(LX + M * h1, BY + N * h2) +
                                 h1 * psi_T(LX + M * h1, BY + N * h2)));
        }
    }

    #pragma dvm parallel([i] on B[i][N])
    for (i = 0; i < M; i++) {
            B[i][N] = (F(LX + i * h1, BY + N * h2) +
                       psi_T(LX + i * h1, BY + N * h2) * 2.0 /h2);
            if (i == 0){
                B[0][N] = (F(LX, BY + N * h2) +
                (2.0 /h1 + 2.0 /h2) * (1.0 /(h1 + h2)) * (h2 * psi_L(LX, BY + N * h2) +
                                 h1 * psi_T(LX, BY + N * h2)));
            }
        }
    
    #pragma dvm parallel([i] on B[i][0])
    for (i = 1; i < M + 1; i++) {
        B[i][0] = (F(LX + i * h1, BY) +
                    psi_B(LX + i * h1, BY) * 2.0 /h2);
        if (i == M){
            B[M][0] = (F(LX + M * h1, BY) +
                (2.0 /h1 + 2.0 /h2) * (1.0 /(h1 + h2)) * (h2 * psi_R(LX + M * h1, BY) +
                                 h1 * psi_B(LX + M*h1, BY)));
        }
        }
    }
}


void calculate_w_prev_w()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on diff_vec[i][j])
    for(i = 0; i <= M; i++)
    {
        for (j = 0; j <= N; j++)
        {
                diff_vec[i][j] = w[i][j] - w_prev[i][j];
        }
    }
    }
}


double rho_i(int i){
    if (i == 1 || i == M)
        return 0.5;
    return 1.0;
}

double rho_j(int j){
    if (j == 1 || j == N)
        return 0.5;
    return 1.0;
}

double dot_product_for_Ar_r(){
    double inner_product = 0.0;
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(inner_product))
    for (i = 0; i < M + 1; i++){
        for (j = 0; j < N + 1; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * Ar[i][j] * r[i][j]);
        }
    }
    }
    return inner_product * h1 * h2;
}


double dot_product_for_Ar_Ar(){
    double inner_product = 0.0;
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(inner_product))
    for (i = 0; i < M + 1; i++){
        for (j = 0; j < N + 1; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * Ar[i][j] * Ar[i][j]);
        }
    }
    }
    return inner_product * h1 * h2;
}

double dot_product_for_diff_vec(){
    double inner_product = 0.0;
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on diff_vec[i][j]) reduction(sum(inner_product))
    for (i = 0; i < M + 1; i++){
        for (j = 0; j < N + 1; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * diff_vec[i][j] * diff_vec[i][j]);
        }
    }
    }
    return inner_product * h1 * h2;
}

double norm_d_p_Ar_Ar_2(){
    return pow(sqrt(dot_product_for_Ar_Ar()), 2);
}

double norm_diff_vec(){
    return sqrt(dot_product_for_diff_vec());
} 

int main(int argc, char **argv) {
    int i, j;
    double DVMH_start = 0.0, DVMH_end = 0.0;
    FILE *f;
    double tau, diff;
    int flag = 1, count_iter = 0;
    // DVMH_start = dvmh_wtime();
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on w[i][j])
    for(i = 0; i < M + 1; i++){
        for (j = 0; j < N + 1; j++){
            w[i][j] = 1.7;
            
        }
    }
    }
    
    #pragma dvm region
    {
     #pragma dvm parallel([i][j] on U_true[i][j])
    for (i = 0; i < M + 1; ++i){
        for (j = 0; j < N + 1; ++j){
            U_true[i][j] = u(i * h1, j * h2);
    }
    }
    }
    Calu_B();

    while(flag){
        count_iter ++;
        Calu_Aw();

        calu_r_Aw_B();
        Calu_Ar();
        tau = dot_product_for_Ar_r() / norm_d_p_Ar_Ar_2();

        #pragma dvm region
        {
        #pragma dvm parallel([i][j] on w[i][j])
        for (i = 0; i < M + 1; i++){
            for (j = 0; j < N + 1; j++){
                w[i][j] = w[i][j] - tau * r[i][j];
            }
        }
        }

        calculate_w_prev_w();
        diff = norm_diff_vec();
        if (count_iter % 1000 == 0){
            printf("Current diff: %g\n, Current number of iterations: %d\n", diff, count_iter);
        }
        if ((diff < eps)){
            flag = 0;
        }
    }
    // dvmh_barrier();
    // DVMH_end = dvmh_wtime();
    // printf("DVMH TIME:%f\n", DVMH_end - DVMH_start);
    printf("M = %d, N = %d\n", M, N);
    printf("Eps = %g\n", eps);
    printf("Final diff: %f\n", diff);
    printf("Tau: %f\n", tau);
    printf("Total number of iteration: %d\n", count_iter);

    f = fopen("w.csv", "wb");
    fprintf(f, "Numberical_w,    True_u\n");
    #pragma dvm get_actual(w, U_true) 
    for (i = 0; i < M + 1; i++) {
        for (j = 0; j < N + 1; j++) {
            fprintf(f, "%lf,    %lf\n", w[i][j], U_true[i][j]);
        }
    }
    
    fclose(f);
}
