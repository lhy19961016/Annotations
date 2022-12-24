// Reference 
    //1. <<Язык C-DVMH. C-DVMH компилятор. Компиляция, выполнение и отладка CDVMH программ.>>
//2. Распараллеливание алгоритма Якоби на C-DVMH
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
// #include "/polusfs/home/kolganov.a/DVM/dvm_r8114/dvm_sys/include/dvmh_runtime_api.h"


#define pi 3.14159265358979323846264338327950288
#define alpha 1.0
#define LX 0.0
#define RX 2.0
#define BY 0.0
#define TY 1.0
#define eps 1e-5
#define ITER_LIMIT 300

#define M 50
#define N 50
const double h1 = (double)(RX - LX) / (double)M;
const double h2 = (double)(TY - BY) / (double)N;

//Распределить данные

#pragma dvm array distribute [block][block]
double Aw[M + 2][N + 2];
#pragma dvm array align([i][j] with Aw[i][j])
double B[M + 2][N + 2];
#pragma dvm array align([i][j] with Aw[i][j])
double w_prev[M + 2][N + 2]; 
#pragma dvm array align([i][j] with Aw[i][j])
double diff_vec[M + 2][N + 2];
#pragma dvm array align([i][j] with Aw[i][j])
double Ar[M + 2][N + 2];
// #pragma dvm array align([i][j] with Aw[i][j])
// double U_true[M + 2][N + 2];
// Exist like A[i][j] = w[i + 1][j] - w[i][j] || w[i][j] - w[i-1][j]
#pragma dvm array align([i][j] with Aw[i][j]), shadow[1:1][1:1]
double w[M + 2][N + 2];
// Exist like A[i][j] = w[i + 1][j] - w[i][j] || w[i][j] - w[i-1][j]
#pragma dvm array align([i][j] with Aw[i][j]), shadow[1:1][1:1]
double r[M + 2][N + 2];
double u(double x, double y)
{
    return 1 + cos(pi * x * y);
}


double q(double x, double y)
{
    return 0;
}


double k(double x, double y)
{
    return 4 + x + y;
}

double F(double x, double y)
{
    return (-1) * (pi * y * ((-1) * sin(pi * x * y) - pi * y * (x + y + 4) * cos(pi* x * y)) + pi * x * ((-1) * sin(pi * x * y) - pi * x * (x + y + 4) * cos(pi * x * y)));
}


double psi_R(double x, double y){
    return (1 + cos(pi * x * y)) - (4 + x + y) * pi * y * sin(pi * x * y);
}


double psi_L(double x, double y){
    return (4 + x + y) * pi * y * sin(pi * x * y);
}

double psi_B(double x, double y){
    return (4 + x + y) * pi * x * sin(pi * x * y);
}


double psi_T(double x, double y){
    return (1 + cos(pi * x * y)) - (4 + x + y) * pi * x * sin(pi * x * y);
}


void Calu_Aw()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Aw[i][j]) shadow_renew(w)
    for (i = 1; i <= M; i++){
        for (j = 0; j <= N+1; j++) {
            if (i == 0 || i == M + 1 || j == 0 || j == N + 1){
                Aw[i][j] = w[i][j];
            } else {
                Aw[i][j] = -(1/h1) * (k(LX + (i + 0.5 - 1) * h1, BY + (j - 1) * h2) * ((w[i + 1][j] - w[i][j]) / h1)
                   - k(LX + (i - 0.5 - 1) * h1, BY + (j - 1) * h2) * ((w[i][j] - w[i - 1][j]) / h1)) - (1/h2) * (k(LX + (i - 1) * h1,BY + (j + 0.5 - 1) * h2) * (w[i][j + 1] - w[i][j]) / h2
                   - k(LX + (i - 1) * h1, BY + (j - 0.5 - 1) * h2) * ((w[i][j] - w[i][j - 1]) / h2));
            }
        }
    }

    #pragma dvm parallel([j] on Aw[1][j]) shadow_renew(w)
    for (j = 1; j <= N; j++) {
            Aw[1][j] = (-1) * (2 / h1) * ((k(LX + (2 - 0.5 - 1) * h1, BY + (j - 1) * h2) * (w[2][j] - w[2 - 1][j]) / h1))- 
            ((1/h2) * (k(LX + (1 - 1) * h1,BY + (j + 0.5 - 1) * h2) * (w[1][j + 1] - w[1][j]) / h2
                   - k(LX + (1 - 1) * h1,BY + (j - 0.5 - 1) * h2) * (w[1][j] - w[1][j - 1]) / h2));
    }
    
    #pragma dvm parallel([j] on r[M][j]) shadow_renew(w)
    for (j = 1; j <= N; j++) {
            Aw[M][j] = (2 / h1) * (k(LX + (M - 0.5 - 1) * h1,BY + (j - 1) * h2) * (w[M][j] - w[M - 1][j]) / h1) + (2/h1) * w[M][j] - 
            ((1/h2) * (k(LX + (M - 1) * h1, BY + (j + 0.5 - 1) * h2) * (w[M][j + 1] - w[M][j]) / h2
                   - k(LX + (M - 1) * h1, BY + (j - 0.5 - 1) * h2) * (w[M][j] - w[M][j - 1]) / h2));
    }
    

    #pragma dvm parallel([i] on r[i][N]) shadow_renew(w)
    for (i = 1; i <= M; i++) {
            Aw[i][N] = (2 / h2) * (k(LX + (i - 1) * h1,BY + (N - 0.5 - 1) * h2) * (w[i][N] - w[i][N - 1]) / h2) + 
            (2/h2) * w[i][N] - ((1/h1) * (k(LX + (i + 0.5 - 1) * h1,BY + (N - 1) * h2) * (w[i + 1][N] - w[i][N]) / h1
                   - k(LX + (i - 0.5 - 1) * h1,BY + (N - 1) * h2) * (w[i][N] - w[i - 1][N]) / h1));
        }
    

    #pragma dvm parallel([i] on r[i][1]) shadow_renew(w)
    for (i = 1; i <= M; i++) {
            Aw[i][1] = (-1) * (1 / h2) * ((k(LX + (i + 0.5 - 1) * h1,BY + (1 - 1) * h2) * (w[i + 1][1] - w[i][1]) / h1
                   - k(LX + (i - 0.5 - 1) * h1, BY + (1 - 1) * h2) * (w[i][1] - w[i - 1][1]) / h1)) - 
                   (2 / h2) * (k(LX + (i - 1) * h1, BY + (2 - 0.5 - 1) * h2) * (w[i][2] - w[i][1]) / h2);
        }
    
    }
    
    r[1][1] = -(2/h1) * (k(LX + (2 - 0.5 - 1) * h1,BY + (1 - 1) * h2) * (w[2][1] - w[2 - 1][1]) / h1) - 
    (2 / h2) * (k(LX + (1 - 1) * h1,BY + (2 - 0.5 - 1) * h2) * (w[1][2] - w[1][2-1]) / h2);
    

    r[1][N] = -(2/h1) * (k(LX + (2 - 0.5 - 1) * h1, BY + (N - 1) * h2) * (w[2][N] - w[2 - 1][N]) / h1) +
    (2 / h2) * (k(LX + (1 - 1) * h1, BY + (N - 0.5 - 1) * h2) * (w[1][N] - w[1][N-1]) / h2) + 
    (2/h2)* w[1][N];
    

    r[M][1] = (2/h1) * (k(LX + (M - 0.5 - 1) * h1, BY + (1 - 1) * h2) * (w[M][1] - w[M - 1][1]) / h1) - 
    (2 / h2) * (k(LX + (M - 1) * h1, BY + (2 - 0.5 - 1) * h2) * (w[M][2] - w[M][2 - 1]) / h2)+ 
    (2/h1) * w[M][1];
    
    r[M][N] = (2/h1) * (k(LX + (M - 0.5 - 1) * h1, BY + (N - 1) * h2) * ((w[M][N] - w[M - 1][N]) / h1)) + 
    (2 / h2) * (k(LX + (M - 1) * h1, BY + (N - 0.5 - 1) * h2) * ((w[M][N] - w[M][N-1]) / h2)) + 
    (2/h1 + 2/h2) * w[M][N];
    
}


void Calu_Ar()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) shadow_renew(r)
    for (i = 0; i <= M + 1; i++){
        for (j = 0; j <= N + 1; j++) {
            if (i == 0 || i == M + 1 || j == 0 || j == N + 1){
                Ar[i][j] = r[i][j];
            } else {
                Ar[i][j] = -(1/h1) * (k(LX + (i + 0.5 - 1) * h1, BY + (j - 1) * h2) * ((r[i + 1][j] - r[i][j]) / h1)
                   - k(LX + (i - 0.5 - 1) * h1, BY + (j - 1) * h2) * ((r[i][j] - r[i - 1][j]) / h1)) - (1/h2) * (k(LX + (i - 1) * h1,BY + (j + 0.5 - 1) * h2) * (r[i][j + 1] - r[i][j]) / h2
                   - k(LX + (i - 1) * h1, BY + (j - 0.5 - 1) * h2) * ((r[i][j] - r[i][j - 1]) / h2));
            }
        }
    }

    #pragma dvm parallel([j] on Ar[1][j]) shadow_renew(r)
    for (j = 1; j <= N; j++) {
            Ar[1][j] = (-1) * (2 / h1) * ((k(LX + (2 - 0.5 - 1) * h1, BY + (j - 1) * h2) * (r[2][j] - r[2 - 1][j]) / h1)) + 
            ((1/h2) * (k(LX + (1 - 1) * h1,BY + (j + 0.5 - 1) * h2) * (r[1][j + 1] - r[1][j]) / h2
                   - k(LX + (1 - 1) * h1,BY + (j - 0.5 - 1) * h2) * (r[1][j] - r[1][j - 1]) / h2));
    }

    #pragma dvm parallel([j] on Ar[M][j]) shadow_renew(r)
    for (j = 1; j <= N; j++) {
            Ar[M][j] = (2 / h1) * (k(LX + (M - 0.5 - 1) * h1,BY + (j - 1) * h2) * (r[M][j] - r[M - 1][j]) / h1) + (2/h1) * r[M][j] - 
            ((1/h2) * (k(LX + (M - 1) * h1, BY + (j + 0.5 - 1) * h2) * (r[M][j + 1] - r[M][j]) / h2
                   - k(LX + (M - 1) * h1, BY + (j - 0.5 - 1) * h2) * (r[M][j] - r[M][j - 1]) / h2));
    }

    #pragma dvm parallel([i] on Ar[i][N]) shadow_renew(r)
    for (i = 1; i <= M; i++) {
            Ar[i][N] = (2 / h2) * (k(LX + (i - 1) * h1,BY + (N - 0.5 - 1) * h2) * (r[i][N] - r[i][N - 1]) / h2) + 
            (2/h2) * r[i][N] - ((1/h1) * (k(LX + (i + 0.5 - 1) * h1,BY + (N - 1) * h2) * (r[i + 1][N] - r[i][N]) / h1
                   - k(LX + (i - 0.5 - 1) * h1,BY + (N - 1) * h2) * (r[i][N] - r[i - 1][N]) / h1));
        }

    #pragma dvm parallel([i] on Ar[i][1]) shadow_renew(r)
    for (i = 1; i <= M; i++) {
            Ar[i][1] = (-1) * (2 / h2) * ((k(LX + (i + 0.5 - 1) * h1,BY + (1 - 1) * h2) * (r[i + 1][1] - r[i][1]) / h1)
                   - k(LX + (i - 0.5 - 1) * h1, BY + (1 - 1) * h2) * (r[i][1] - r[i - 1][1]) / h1) - 
                   (2 / h2) * (k(LX + (i - 1) * h1, BY + (2 - 0.5 - 1) * h2) * (r[i][2] - r[i][1]) / h2);
        }
    }

    Ar[1][1] = -(2/h1) * (k(LX + (2 - 0.5 - 1) * h1,BY + (1 - 1) * h2) * (r[2][1] - r[2 - 1][1]) / h1) - 
    (2 / h2) * (k(LX + (1 - 1) * h1,BY + (2 - 0.5 - 1) * h2) * (r[1][2] - r[1][2-1]) / h2);
    

    Ar[1][N] = -(2/h1) * (k(LX + (2 - 0.5 - 1) * h1, BY + (N - 1) * h2) * (r[2][N] - r[2 - 1][N]) / h1) +
    (2 / h2) * (k(LX + (1 - 1) * h1, BY + (N - 0.5 - 1) * h2) * (r[1][N] - r[1][N-1]) / h2) + 
    (2/h2)* r[1][N];
    

    Ar[M][1] = (2/h1) * (k(LX + (M - 0.5 - 1) * h1, BY + (1 - 1) * h2) * (r[M][1] - r[M - 1][1]) / h1) - 
    (2 / h2) * (k(LX + (M - 1) * h1, BY + (2 - 0.5 - 1) * h2) * (r[M][2] - r[M][2 - 1]) / h2)+ 
    (2/h1) * r[M][1];
    
    Ar[M][N] = (2/h1) * (k(LX + (M - 0.5 - 1) * h1, BY + (N - 1) * h2) * ((r[M][N] - r[M - 1][N]) / h1)) + 
    (2 / h2) * (k(LX + (M - 1) * h1, BY + (N - 0.5 - 1) * h2) * ((r[M][N] - r[M][N-1]) / h2)) + 
    (2/h1 + 2/h2) * r[M][N];
    
}

void Calu_B()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on B[i][j])
    for(i = 0; i <= M + 1; i++)
        for (j = 0; j <= N + 1; j++)
            B[i][j] = F(LX + (i - 1) * h1, BY + (j - 1) * h2);
    
    #pragma dvm parallel([j] on B[1][j])
    for (j = 1; j <= N; j++) {
            B[1][j] = (F(LX, BY + (j - 1) * h2) +
                    psi_L(LX, BY + (j - 1) * h2) * 2/h1);
    }
    
    #pragma dvm parallel([j] on B[M][j])
    for (j = 1; j <= N; j++) {
        B[M][j] = (F(LX + (M - 1) * h1, BY + (j - 1) * h2) +
                    psi_R(LX + (M - 1)*h1, BY + (j - 1) * h2) * 2/h1);
    }

    #pragma dvm parallel([i] on B[i][N])
    for (i = 1; i <= M; i++) {
            B[i][N] = (F(LX + (i - 1) * h1, BY + (N - 1) * h2) +
                       psi_T(LX + (i - 1) * h1, BY + (N - 1) * h2) * 2/h2);
        }
    
    #pragma dvm parallel([i] on B[i][1])
    for (i = 1; i <= M; i++) {
        B[i][1] = (F(LX + (i - 1)*h1, BY) +
                    psi_B(LX + (i - 1) * h1, BY) * 2/h2);
        }
    }
 
    B[1][1] =  (F(LX, BY)
                + (2/h1 + 2/h2) * (1/(h1 + h2)) * (h2 * psi_L(LX, BY) + h1 * psi_B(LX, BY)));
    


    B[1][N] = (F(LX, BY + (N - 1) * h2) +
                (2/h1 + 2/h2) * (1/(h1 + h2)) * (h2 * psi_L(LX, BY + (N - 1) * h2) +
                                 h1 * psi_T(LX, BY + (N - 1) * h2)));
    
 
    B[M][N] = (F(LX + (M - 1) * h1, BY + (N - 1) * h2) +
                (2/h1 + 2/h2) * (1/(h1 + h2)) * (h2 * psi_R(LX + (M - 1) * h1, BY + (N - 1) * h2) +
                                 h1 * psi_T(LX + (M - 1) * h1, BY + (N - 1) * h2)));
    
    B[M][1] = (F(LX + (M - 1) * h1, BY) +
                (2/h1 + 2/h2) * (1/(h1 + h2)) * (h2 * psi_R(LX + (M - 1) * h1, BY) +
                                 h1 * psi_B(LX + (M - 1)*h1, BY)));
    
}

void calculate_A_B()
{
    int i, j;
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on r[i][j])
    for(i = 1; i <= M; i++)
    {
        for (j = 1; j <= N; j++)
        {
                r[i][j] = Aw[i][j] - B[i][j];
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
    for(i = 1; i <= M; i++)
    {
        for (j = 1; j <= N; j++)
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
    #pragma dvm actual(inner_product)
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(inner_product))
    for (i = 1; i <= M; i++){
        for (j = 1; j <= N; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * Ar[i][j] * r[i][j]);
        }
    }
    }
    return inner_product  * h1 * h2;
}


double dot_product_for_Ar_Ar(){
    double inner_product = 0.0;
    int i, j;
    #pragma dvm actual(inner_product)
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(inner_product))
    for (i = 1; i <= M; i++){
        for (j = 1; j <= N; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * Ar[i][j] * Ar[i][j]);
        }
    }
    }
    return inner_product  * h1 * h2;
}

double dot_product_for_diff_vec(){
    double inner_product = 0.0;
    int i, j;
    #pragma dvm actual(inner_product)
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on diff_vec[i][j]) reduction(sum(inner_product))
    for (i = 1; i <= M; i++){
        for (j = 1; j <= N; j++){
            inner_product += ((rho_i(i) * rho_j(j)) * diff_vec[i][j] * diff_vec[i][j]);
        }
    }
    }
    return inner_product  * h1 * h2;
}

double norm_d_p_Ar_Ar_2(){
    return pow(sqrt(dot_product_for_Ar_Ar()), 2);
}

double norm_diff_vec(){
    return sqrt(dot_product_for_diff_vec());
} 

int main(int argc, char **argv) {
    int i, j;
    double P1, P2;
    double DVMH_start = 0.0, DVMH_end = 0.0;
    FILE *f;
    double tau, diff;
    int flag = 1, count_iter = 0;

    // DVMH_start = dvmh_wtime();
    #pragma dvm region
    {
    #pragma dvm parallel([i][j] on w[i][j])
    for(i = 0; i <= M + 1; i++){
        for (j = 0; j <= N + 1; j++){
            w[i][j] = 1.45;
        }
    }
    }
    Calu_B();

    while(flag){
        count_iter ++;
        #pragma dvm region
        {
        #pragma dvm parallel([i][j] on r[i][j])
        for (i = 0; i <= M + 1; i++){
            for (j = 0; j <= N + 1; j++){
                w_prev[i][j] = w[i][j];
            }
        }
        }

    Calu_Aw(); //Build Aw
    calculate_A_B(); // Build r(diff between Aw and B)
    Calu_Ar(); //Build Ar

    tau = dot_product_for_Ar_r() / norm_d_p_Ar_Ar_2();
    printf("%f\n", tau);
        #pragma dvm region
        {
        #pragma dvm parallel([i][j] on w[i][j])
        for (i = 1; i <= M; i++){
            for (j = 1; j <= N; j++){
                w[i][j] = w[i][j] - tau * r[i][j];
            }
        }
        }

        calculate_w_prev_w();
        diff = norm_diff_vec();

        #pragma dvm get_actual(diff)
        if (count_iter % 1 == 0){
            printf("Current diff: %g\n, Current number of iterations: %d\n", diff, count_iter);
        }
        if (diff < eps){
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
    f = fopen("Whole_w.csv", "wb");
    #pragma dvm get_actual(w) 
    for (i = 0; i < M + 1; i++) {
        for (j = 0; j < N + 1; j++) {
            fprintf(f, "%g\n", w[i][j]);
        }
    }
    fclose(f);
    return 0;
}
