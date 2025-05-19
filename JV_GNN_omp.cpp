
/**
 * @file JV_GNN_omp.cpp
 * @author Malladi Tejasvi (CS23M036), M.Tech CSE, IIT Madras.
 * @date September 8, 2024
 * @brief Graph Neural Networks Suite for Vertex classification
 */

#include<iostream>
#include<iomanip>
#include<fstream>
#include<vector>
#include<set>
#include<unordered_set>
#include<algorithm>
#include<execution>
#include<cassert>
#include<string>
#include<random>
#include<cmath>
#include<algorithm>
#include<typeinfo>
#include<chrono>
#include<sstream>
#include "graph.hpp"

using namespace std;

/*
Class to supply functions that perform Linear Transforms(MUL,ADD,SUB,Hadamard,Outer,MatrixVector Product), Linear Combinations, Non Linear transforms(Square, Square root etc) 
*/
class TensorTransforms
{

    public:
        // Function to multiply A[cur_row:cur_row+batch_size][k] with B[k][m] and return the result.
        template <typename T> //works only for numeric datatypes
        static vector<vector<T>> Mul(vector<vector<T>> &A,vector<vector<T>> &B, int cur_row, int batch_size = 1);

        // Function to multiply A[cur_row:cur_row+batch_size][k] with B[k][m] and STORE the result in the passed param C.
        template <typename T> //works only for numeric datatypes
        static void Mul(vector<vector<T>> &A,vector<vector<T>> &B,vector<vector<T>> &C, int cur_row, int batch_size = 1);

        // Function to add or subtract two matrices [overloaded]
        // Subtraction if multiplier is -1
        template <typename T> //works only for numeric datatypes
        static void Add(vector<T> &A,const vector<T> &B,float multiplier = 1);

        // Function to add or subtract two matrices [overloaded]
        // Subtraction if multiplier is -1
        template <typename T> //works only for numeric datatypes
        static void Add(vector<vector<T>> &A,const vector<vector<T>> &B,int multiplier = 1);

        // Function to add or subtract two tensors [overloaded]
        //if multiplier is -1, it becomes matrix subtraction.
        template <typename T> //works only for numeric datatypes
        static void Add(vector<vector<vector<T>>> &A,const vector<vector<vector<T>>> &B,int multiplier = 1);

        // Function to add a vector V2 to every row of matrix V1
        template <typename T> //works only for numeric datatypes
        static void Add(vector<vector<T>> &v1,vector<T> &v2);

        // Function to add constant to all values of a vector.
        template <typename T> //works only for numeric datatypes
        static void Add(vector<T> &V,T val);

        // Function to add constant to all values of a matrix. [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Add(vector<vector<T>> &matrix,T val);

        // Function to add constant to all values of a Tensor. [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Add(vector<vector<vector<T>>> &tensor,T val);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are vectors
        static vector<float> LinearCombination(vector<float> &v1,float c1,vector<float> &v2,float c2);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are vectors
        //In place LinearCombination : results stored in the first parameter
        static void LinearCombination_(vector<float> &v1,float c1,vector<float> &v2,float c2);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are Matrices [overloaded]
        static vector<vector<float>> LinearCombination(vector<vector<float>> &v1,float c1,vector<vector<float>> &v2,float c2);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are Matrices [overloaded]
        //In place LinearCombination : results stored in the first parameter
        static void LinearCombination_(vector<vector<float>> &v1,float c1,vector<vector<float>> &v2,float c2);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are TENSORS [overloaded]
        static vector<vector<vector<float>>> LinearCombination(vector<vector<vector<float>>> &v1,float c1,vector<vector<vector<float>>> &v2,float c2);

        //Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are TENSORS [overloaded]
        //In place LinearCombination : results stored in the first parameter
        static void LinearCombination_(vector<vector<vector<float>>> &v1,float c1,vector<vector<vector<float>>> &v2,float c2);

        //Function to multiply all values in a matrix by a scalar, inplace. [overloading]
        template <typename T> //works only for numeric datatypes
        static vector<T> Scale(vector<T> &v,T val);

        //Function to multiply all values in a matrix by a scalar, inplace. [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Scale(vector<vector<T>> &matrix,T val);

        //Function to multiply all values in a tensor by a scalar, inplace. [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Scale(vector<vector<vector<T>>> &tensor,T val);

        // Function to compute element wise square Matrix.
        template <typename T> //works only for numeric datatypes
        static void Matrix_sqrt(vector<T> &V);

        // Function to compute element wise square root of a matrix.
        template <typename T> //works only for numeric datatypes
        static void Matrix_sqrt(vector<vector<T>> &matrix);

        // Function to compute element wise square root of a tensor.
        template <typename T> //works only for numeric datatypes
        static void Matrix_sqrt(vector<vector<vector<T>>> &tensor);

        // Function to compute element wise square Matrix.
        template <typename T> //works only for numeric datatypes
        static void Matrix_square(vector<T> &V);

        // Function to compute element wise square root of a matrix.
        template <typename T> //works only for numeric datatypes
        static void Matrix_square(vector<vector<T>> &matrix);

        // Function to compute element wise square root of a tensor.
        template <typename T> //works only for numeric datatypes
        static void Matrix_square(vector<vector<vector<T>>> &tensor);

        //Function to perform element wise division of two vectors
        template <typename T> //works only for numeric datatypes
        static void Matrix_divide(vector<T> &A,vector<T> &B);

        //Function to perform element wise division of two matrices [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Matrix_divide(vector<vector<T>> &A,vector<vector<T>> &B);

        //Function to perform element wise division of two tensors [overloaded]
        template <typename T> //works only for numeric datatypes
        static void Matrix_divide(vector<vector<vector<T>>> &A,vector<vector<vector<T>>> &B);

        // to match types v1 and v2 are vect of vect, but actually they are just 1D vectors
        template <typename T> //works only for numeric datatypes
        static vector<vector<T>> Outer(vector<vector<T>> &v1, vector<T> &v2); //Hi * Ai not Ai * Hi

        // To compute the product of a MxN matrix with Nx1 Vector (vector stored as 1xN)
        template <typename T> //works only for numeric datatypes
        static vector<T> MatrixVectorProduct(vector<vector<T>> &M,vector<T> &V);

        // Function to compute element wise products of two vectors
        // aka Hadamard Product
        template <typename T> //works only for numeric datatypes
        static vector<T> Hadamard(vector<T> &v1, vector<T> &v2);

        // Function to compute element wise products of two vectors
        // aka Hadamard Product
        // In place operation
        template <typename T> //works only for numeric datatypes
        static void Hadamard_(vector<T> &v1, vector<T> &v2);

        //function to overwrite all elements of an existing container with 0s
        template <typename T> //works only for numeric datatypes
        static void fill_with_zeros(vector<T> &M);

        //function to overwrite all elements of an existing container with 0s [overloaded]
        template <typename T> //works only for numeric datatypes
        static void fill_with_zeros(vector<vector<T>> &M);

        //function to overwrite all elements of an existing container with 0s [overloaded]
        template <typename T> //works only for numeric datatypes
        static void fill_with_zeros(vector<vector<vector<T>>> &M);

};

// Fundamental Matrix Operations : Matrix Multiplications, Additions, Addition with scalar and scaling.
/**********************************************************************************/

// Function to multiply A[cur_row:cur_row+batch_size][k] with B[k][m] and return the result.
template <typename T> //works only for numeric datatypes
vector<vector<T>> TensorTransforms::Mul(vector<vector<T>> &A,vector<vector<T>> &B, int cur_row, int batch_size)
/*
Description:
    performs A[cur_row:cur_row+batch_size] * B
*/
{
    assert(cur_row<A.size()); //sanity check

    auto n = cur_row + batch_size;
    auto k = A[0].size();
    auto m = B[0].size();

    if(n>A.size())
        n = A.size();

    vector<vector<T>> C(batch_size,vector<T>(m,0.0));

    #pragma omp parallel for 
    for(auto ii=cur_row;ii<n;ii++)
    {
        for(auto xx=0;xx<k;xx++) 
        {
            #pragma omp simd
            for(auto jj=0;jj<m;jj++)
                C[ii-cur_row][jj] += A[ii][xx]*B[xx][jj];
        }

    }
    return C;
}

// Function to multiply A[cur_row:cur_row+batch_size][k] with B[k][m] and STORE the result in C.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Mul(vector<vector<T>> &A,vector<vector<T>> &B,vector<vector<T>> &C, int cur_row, int batch_size)
/*
Description:
    performs A[cur_row:cur_row+batch_size] * B
*/
{
    assert(cur_row<A.size()); //sanity check

    auto n = cur_row + batch_size;
    auto k = A[0].size();
    auto m = B[0].size();

    if(n>A.size())
        n = A.size();

    #pragma omp parallel for 
    for(auto ii=cur_row;ii<n;ii++)
    {
        for(auto xx=0;xx<k;xx++) 
        {
            #pragma omp simd
            for(auto jj=0;jj<m;jj++)
                C[ii-cur_row][jj] += A[ii][xx]*B[xx][jj];
        }

    }
}



// Function to add or subtract two vectors
// subtraction, if multiplier is -1
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<T> &A,const vector<T> &B,float multiplier)
/*
Description:
    performs A + multiplier*B
*/
{
    assert(A.size() == B.size());
    
    #pragma omp simd
    for(auto ii=0;ii<A.size();ii++)
        A[ii] += multiplier*B[ii];
}

// Function to add or subtract two matrices [overloaded]
// Subtraction if multiplier is -1
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<vector<T>> &A,const vector<vector<T>> &B,int multiplier)
/*
Description:
    performs A + multiplier*B;
*/
{
    assert(A.size() == B.size());//sanity check
    assert(A[0].size() == B[0].size());

    auto n = A.size();
    int m = A[0].size(); //dims(A) = dims(B)
    
    #pragma omp parallel for 
    for(auto ii=0;ii<n;ii++)
    {
        for(auto jj=0;jj<m;jj++)
                A[ii][jj] += multiplier*B[ii][jj];
    }
}

// Function to add or subtract two tensors [overloaded]
//if multiplier is -1, it becomes matrix subtraction.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<vector<vector<T>>> &A,const vector<vector<vector<T>>> &B,int multiplier)
/*
Description:
    performs A + multiplier*B;
*/
{
    //Workload based conditional parallelization
    for(int i=0; i<A.size(); i++) 
    {
        int m = A[i].size();
        int n = A[i][0].size();
        
        if(m>=2500) //overall work dominates
        {
            #pragma omp parallel for schedule(dynamic,16)
            for(int j=0; j<m; j++) 
            {
                #pragma omp simd
                for(int k=0; k<n; k++) {
                    A[i][j][k] += multiplier*B[i][j][k];

                }
            }
        }
        else if(m>500 || n>=1024)
        {
            #pragma omp parallel for
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    A[i][j][k] += multiplier*B[i][j][k];
                }
            }
        }
        else
        {
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    A[i][j][k] += multiplier*B[i][j][k];
                }
            }
        }

    }
}

// Function to add a vector V2 to every row of matrix V1
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<vector<T>> &v1,vector<T> &v2)
/*
Description:
    performs A + multiplier*B;
*/
{   
    #pragma omp parallel for 
    for(auto &row : v1)
        transform(row.begin(),row.end(),v2.begin(),row.begin(), plus<T>());
}

// Function to add constant to all values of a vector.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<T> &V,T val)
{
        // In IITM's Aqua Cluster cache line size is 16 words or 64B and size of float is 4, hence 16 would be a good chunk size to avoid false sharing.
        #pragma omp parallel for schedule(static,16) 
        for(int i=0;i<V.size();i++)
            V[i] += val;
}

// Function to add constant to all values of a matrix. [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<vector<T>> &matrix,T val){

    #pragma omp parallel for 
    for(auto &row : matrix){
        transform(row.begin(),row.end(),row.begin(),
        [val](T & ele) {return ele+ val;});
    }
}

// Function to add constant to all values of a Tensor. [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Add(vector<vector<vector<T>>> &tensor,T val){

    #pragma omp parallel for 
    for(auto &matrix : tensor)
        for(auto &row : matrix)
            transform(row.begin(),row.end(),row.begin(),
            [val](T & ele) {return ele+ val;});
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are vectors
vector<float> TensorTransforms::LinearCombination(vector<float> &v1,float c1,vector<float> &v2,float c2)
{
    assert(v1.size() == v2.size()); //Sanity check

    vector<float> v3(v1.size());

    #pragma omp simd
    for(int i = 0;i< v3.size();i++)
        v3[i] = c1*v1[i] + c2*v2[i];

    return v3;
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are vectors
//In place LinearCombination : results stored in the first parameter
void TensorTransforms::LinearCombination_(vector<float> &v1,float c1,vector<float> &v2,float c2) 
{
    assert(v1.size() == v2.size()); //Sanity check

    #pragma omp simd
    for(int i = 0;i< v1.size();i++)
        v1[i] = c1*v1[i] + c2*v2[i];

}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are Matrices [overloaded]
vector<vector<float>> TensorTransforms::LinearCombination(vector<vector<float>> &v1,float c1,vector<vector<float>> &v2,float c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);
    assert(v1[0].size() == v2[0].size());

    vector<vector<float>> v3 = v1;
    
    #pragma omp parallel for 
    for(int i=0;i<v3.size();i++)
        transform(v3[i].begin(),v3[i].end(),v2[i].begin(),v3[i].begin(),
        [c1,c2](float v3_ele,float v2_ele){
            return c1*v3_ele + c2*v2_ele;
        });

    return v3;
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are Matrices [overloaded]
//In place LinearCombination : results stored in the first parameter
void TensorTransforms::LinearCombination_(vector<vector<float>> &v1,float c1,vector<vector<float>> &v2,float c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);
    assert(v1[0].size() == v2[0].size());

    #pragma omp parallel for 
    for(int i=0;i<v1.size();i++)
        transform(v1[i].begin(),v1[i].end(),v2[i].begin(),v1[i].begin(),
        [c1,c2](float v1_ele,float v2_ele){
            return c1*v1_ele + c2*v2_ele;
        });

}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are TENSORS [overloaded]
vector<vector<vector<float>>> TensorTransforms::LinearCombination(vector<vector<vector<float>>> &v1,float c1,vector<vector<vector<float>>> &v2,float c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);
    
    vector<vector<vector<float>>> v3 = v1;

    //Workload based conditional parallelization
    for(int i=0; i<v1.size(); i++) 
    {
        int m = v1[i].size();
        int n = v1[i][0].size();
        
        if(m>=2500) //overall work dominates
        {
            #pragma omp parallel for schedule(dynamic,16)
            for(int j=0; j<m; j++) 
            {
                #pragma omp simd
                for(int k=0; k<n; k++) {
                    v3[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }
        else if(m>1400 || n>256)
        {
            #pragma omp parallel for
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    v3[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }
        else
        {
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    v3[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }

    }

    return v3;
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are TENSORS [overloaded]
//In place LinearCombination : results stored in the first parameter
void TensorTransforms::LinearCombination_(vector<vector<vector<float>>> &v1,float c1,vector<vector<vector<float>>> &v2,float c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);


    //Workload based conditional parallelization
    for(int i=0; i<v1.size(); i++) 
    {
        int m = v1[i].size();
        int n = v1[i][0].size();
        
        if(m>=2500) //overall work dominates
        {
            #pragma omp parallel for schedule(dynamic,16)
            for(int j=0; j<m; j++) 
            {
                #pragma omp simd
                for(int k=0; k<n; k++) {
                    v1[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }
        else if(m>1400 || n>256)
        {
            #pragma omp parallel for
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    v1[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }
        else
        {
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    v1[i][j][k] = c1 * v1[i][j][k] + c2 * v2[i][j][k];
                }
            }
        }

    }

}


//Function to multiply all values in a matrix by a scalar, inplace. [overloading]
template <typename T> //works only for numeric datatypes
vector<T> TensorTransforms::Scale(vector<T> &v,T val)
{
        vector<T> res(v.size(),0);

        #pragma omp parallel for simd schedule(static,16) 
        for(int i=0;i<v.size();i++)
            res[i] = v[i]*val;

    return res;
}

//Function to multiply all values in a matrix by a scalar, inplace. [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Scale(vector<vector<T>> &matrix,T val){
    
    int n = matrix.size();
    int m = matrix[0].size();
    
    #pragma omp parallel for 
    for(int i=0;i<n;i++)
    {
        #pragma omp simd
        for(int j=0;j<m;j++)
            matrix[i][j] *= val;
    }
}

//Function to multiply all values in a tensor by a scalar, inplace. [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Scale(vector<vector<vector<T>>> &tensor,T val){

    
    //Workload based conditional parallelization
    for(int i=0; i<tensor.size(); i++) 
        {
                int m = tensor[i].size();
                int n = tensor[i][0].size();

                if(m>2500)
                {
                    #pragma omp parallel for schedule(dynamic,16)
                    for(int j=0; j<m; j++) 
                    {
                        #pragma omp simd
                        for(int k=0; k<n; k++) {
                            tensor[i][j][k] *= val;
                        }
                    }
                }
                else if(m>1400)
                {
                    #pragma omp parallel for
                    for(int j=0; j<m; j++) {
                        for(int k=0; k<n; k++) {
                            tensor[i][j][k] *= val;
                        }
                    }
                }
                else if(m>=300 && n>256)
                {
                    #pragma omp parallel for
                    for(int j=0; j<m; j++) {
                        for(int k=0; k<n; k++) {
                            tensor[i][j][k] *= val;
                        }
                    }

                }
                else
                {
                    for(int j=0; j<m; j++) {
                        for(int k=0; k<n; k++) 
                        {
                            tensor[i][j][k] *= val;
                        }
                    }
                }
        }
}

// Function to compute element wise square Matrix.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_sqrt(vector<T> &V)
{
        #pragma omp simd
        for(int i=0;i<V.size();i++)
            V[i] = std::sqrt(V[i]);
}


// Function to compute element wise square root of a matrix.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_sqrt(vector<vector<T>> &matrix)
{
    #pragma omp parallel for 
    for (auto& row : matrix) {
        transform(row.begin(), row.end(), row.begin(), [](T val) {
            return std::sqrt(val);
        });
    }
}

// Function to compute element wise square root of a tensor.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_sqrt(vector<vector<vector<T>>> &tensor)
{    
    //Workload based conditional parallelization
    for(int i=0; i<tensor.size(); i++) 
    {
        int m = tensor[i].size();
        int n = tensor[i][0].size();
        
        if(n*m>=48000) //emperically identified cutoff.
        {
            if(m<2500) //overall work dominates
            {
                #pragma omp parallel for collapse(2)
                for(int j=0; j<m; j++) 
                {
                    for(int k=0; k<n; k++) {
                        tensor[i][j][k] = std::sqrt(tensor[i][j][k]);
                    }
                }
            }

            else //the number of rows dominate
            {

                #pragma omp parallel for schedule(dynamic,16)
                for(int j=0; j<m; j++) 
                {
                    #pragma omp simd
                    for(int k=0; k<n; k++) {
                        tensor[i][j][k] = std::sqrt(tensor[i][j][k]);
                    }
                }

            }
        }
        
        else //not large eough to parallelize
        {
            for(int j=0; j<m; j++) 
            {
                for(int k=0; k<n; k++) {
                    tensor[i][j][k] = std::sqrt(tensor[i][j][k]);
                }
            }
        }
    }
}

// Function to compute element wise square Matrix.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_square(vector<T> &V)
{
        #pragma omp simd
        for(int i=0;i<V.size();i++)
            V[i] *= V[i];
}


// Function to compute element wise root of a matrix.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_square(vector<vector<T>> &matrix)
{

    #pragma omp parallel for 
    for (auto& row : matrix) {
        transform(row.begin(), row.end(), row.begin(), [](T val) {
            return val*val;
        });
    }
}

// Function to compute element wise square root of a tensor.
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_square(vector<vector<vector<T>>> &tensor)
{    
    //Workload based conditional parallelization
    for(int i=0; i<tensor.size(); i++) 
    {
        int m = tensor[i].size();
        int n = tensor[i][0].size();
        
        if(m>=2500) //overall work dominates
        {
            #pragma omp parallel for schedule(dynamic,16)
            for(int j=0; j<m; j++) 
            {
                #pragma omp simd
                for(int k=0; k<n; k++) {
                    tensor[i][j][k] = tensor[i][j][k] * tensor[i][j][k];
                }
            }
        }
        else if(m>1400 || n>256)
        {
            #pragma omp parallel for
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    tensor[i][j][k] = tensor[i][j][k] * tensor[i][j][k];
                }
            }
        }
        else
        {
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    tensor[i][j][k] = tensor[i][j][k] * tensor[i][j][k];
                }
            }
        }

    }
}

//Function to perform element wise division of two vectors
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_divide(vector<T> &A,vector<T> &B)
{
        #pragma omp simd
        for(int i=0;i<A.size();i++)
            A[i] /= B[i];
}

//Function to perform element wise division of two matrices [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_divide(vector<vector<T>> &A,vector<vector<T>> &B)
{
    #pragma omp parallel for 
    for (int i = 0; i < A.size(); i++)
        std::transform(A[i].begin(), A[i].end(), B[i].begin(), A[i].begin(), std::divides<T>());
}

//Function to perform element wise division of two tensors [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::Matrix_divide(vector<vector<vector<T>>> &A,vector<vector<vector<T>>> &B)
{
    //Workload based conditional parallelization
    for(int i=0; i<A.size(); i++) 
    {
        int m = A[i].size();
        int n = A[i][0].size();
        
        if(m>=2500) //overall work dominates
        {
            #pragma omp parallel for schedule(dynamic,16)
            for(int j=0; j<m; j++) 
            {
                #pragma omp simd
                for(int k=0; k<n; k++) {
                    A[i][j][k] /=  B[i][j][k];

                }
            }
        }
        else if(m>1400 || n>256)
        {
            #pragma omp parallel for
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    A[i][j][k] /=  B[i][j][k];
                }
            }
        }
        else
        {
            for(int j=0; j<m; j++) {
                for(int k=0; k<n; k++) {
                    A[i][j][k] /= B[i][j][k];
                }
            }
        }

    }
}

// to match types v1 and v2 are vect of vect, but actually they are just 1D vectors
template <typename T> //works only for numeric datatypes
//Hi * Ai not Ai * Hi
vector<vector<T>> TensorTransforms::Outer(vector<vector<T>> &v1, vector<T> &v2)
{
    vector<vector<T>> result(v1[0].size(),vector<T>(v2.size(),0));
    
    #pragma omp parallel for 
    for(int i=0;i<v1[0].size();i++)
    {
        auto ele = v1[0][i];
        for(int j=0;j<v2.size();j++)
        {
            result[i][j] += ele*v2[j];
        }
    }
    return result;

}

// To compute the product of a MxN matrix with Nx1 Vector (vector stored as 1xN)
template <typename T> //works only for numeric datatypes
vector<T> TensorTransforms::MatrixVectorProduct(vector<vector<T>> &M,vector<T> &V)
{
    assert(M[0].size() == V.size());

    vector<T> res(M.size());

    #pragma omp parallel for 
    for(int i=0;i<M.size();i++)
        for(int j=0;j<M[0].size();j++)
            res[i] += M[i][j]*V[j];

    return res;

}

// Function to compute element wise products of two vectors
// aka Hadamard Product
template <typename T> //works only for numeric datatypes
vector<T> TensorTransforms::Hadamard(vector<T> &v1, vector<T> &v2)
{
    assert(v1.size() == v2.size());

    vector<T> v3(v1);
    
    #pragma omp parallel for simd
    for(int i=0;i<v1.size();i++)
        v3[i] = v1[i]*v2[i];

    return v3;

}

// Function to compute element wise products of two vectors
// aka Hadamard Product
// In place operation
template <typename T> //works only for numeric datatypes
void TensorTransforms::Hadamard_(vector<T> &v1, vector<T> &v2)
{
    assert(v1.size() == v2.size());

    vector<T> v3(v1);
    
    #pragma omp simd
    for(int i=0;i<v1.size();i++)
        v1[i] = v1[i]*v2[i];

}

//function to overwrite all elements of an existing container with 0s
template <typename T> //works only for numeric datatypes
void TensorTransforms::fill_with_zeros(vector<T> &M)
{
    if(M.size()==0)
        return;
    
    #pragma omp simd
    for(int i=0;i<M.size();i++)
        M[i] = 0;
}

//function to overwrite all elements of an existing container with 0s [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::fill_with_zeros(vector<vector<T>> &M)
{
    if(M.size()==0)
        return;
    assert(M[0].size()>0);

    #pragma omp parallel for 
    for(int i=0;i<M.size();i++)
        for(int j=0;j<M[i].size();j++)
            M[i][j] = 0;  
} 

//function to overwrite all elements of an existing container with 0s [overloaded]
template <typename T> //works only for numeric datatypes
void TensorTransforms::fill_with_zeros(vector<vector<vector<T>>> &M)
{
    if(M.size()==0)
        return;

    assert(M[0].size()>0);
    assert(M[0][0].size()>0);

    
    //Workload based conditional parallelization
    for(int i=0; i<M.size(); i++) 
    {
        int m = M[i].size();
        int n = M[i][0].size();
        
        if(n*m>=200000) //emperically identified cutoff.
        {
            if(m<2500) //overall work dominates
            {
                #pragma omp parallel for collapse(2)
                for(int j=0; j<m; j++) 
                {
                    for(int k=0; k<n; k++) {
                        M[i][j][k] = 0;
                    }
                }
            }

            else //the number of rows dominate
            {

                #pragma omp parallel for schedule(dynamic,16)
                for(int j=0; j<m; j++) 
                {
                    #pragma omp simd
                    for(int k=0; k<n; k++) {
                        M[i][j][k] = 0;
                    }
                }

            }
        }
        
        else //not large eough to parallelize
        {
            for(int j=0; j<m; j++) 
            {
                for(int k=0; k<n; k++) {
                    M[i][j][k] = 0;
                }
            }
        }
    }
}

//Function to find and return the index of the largest element in a vector 
template <typename T> //works only for numeric datatypes
T argmax(vector<T> &V)
{
    assert(!V.empty());
    auto max_ele_pos = max_element(V.begin(),V.end());

    return distance(V.begin(),max_ele_pos);
}

//Function to find and return the index of the largest element in a vector [overloaded]
template <typename T> //works only for numeric datatypes
vector<T> argmax(vector<vector<T>> &V)
{
    vector<T> res(V.size());

    #pragma omp parallel for 
    for(int i=0;i<V.size();i++)
    {
        auto max_ele = max_element(V[i].begin(),V[i].end());
        auto amax = distance(V[i].begin(),max_ele);

        res[i] = amax;
    }

    return res;
}

// to compute the total cross entropy loss between the vectors within a vector
template <typename T> //works only for numeric datatypes
float cross_entropy_loss(vector<vector<T>> &pred_probs, vector<int> &true_labels, int vertex, bool par_exe)
{
    float loss=0;
    #pragma omp parallel for reduction(+:loss) 
    for(int i=0;i<pred_probs.size();i++)
        loss = loss + -(log(pred_probs[i][int(true_labels[vertex])]));

    return loss;
}

// to compute the total cross entropy loss between the vectors within a vector
template <typename T> //works only for numeric datatypes
float cross_entropy_loss(vector<T> &pred_probs, vector<int> &true_labels, int vertex)
{
    float loss=0;
    loss = loss + -(log(pred_probs[int(true_labels[vertex])]));

    return loss;
}

// function to compute accuracy, given the predicted classes preds and true_labels
// preds must be predicted classes
template <typename T> //works only for numeric datatypes
T compute_accuracy(vector<T> &preds, vector<T> &true_labels, int cur_row = -1)
{
    int c = (cur_row == -1)?0:cur_row;

    float correct_preds = 0;

    #pragma omp parallel for reduction(+:correct_preds) schedule(static,16) 
    for(int i =0 ;i<preds.size();i++)
    {
        if(int(preds[i]) == int(true_labels[i+c]))
        {
            correct_preds++;
        }
    }

    float accuracy = (correct_preds*100)/(float(preds.size()));

    return accuracy;
}


//function to print elements of a vector
template <typename T>
void print(vector<T> &V)
{
    for(auto &ele : V)
        cout<<ele<<" ";
    cout<<endl;
}

//function to print elements of a matrix of type T [overloaded]
template <typename T> 
void print(vector<vector<T>> &V)
{
    for(auto &row : V)
        print(row);
    cout<<endl;
}

//function to print elements of a tensor of type T [overloaded]
template <typename T> 
void print(vector<vector<vector<T>>> &V)
{
    for(auto &row : V)
        print(row);
    cout<<endl;
}



/**********************************************************************************/

/*
Layers can be vector<vector<vector<float>>>
Inputs (may be to a constructor) : num layers (int), an array of layer sizes, initialization mech.
Activation and Optimisers can be separate classes

Network class has 
    constructor to create network and initialize the weights.
    Forward Pass.
    Bacward Pass.

*/


// Not used anymore
vector<int> randomSample(int N, int k, int seed = 76) // Function to sample k random indices from 0 to N-1
{
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);

    // random_device rd;
    // mt19937 gen(rd());
    mt19937 gen(seed);

    shuffle(indices.begin(), indices.end(), gen);

    indices.resize(k);
    return indices;
}

class Layer
{

    public:

        int size,inp_dim,out_dim;
        vector<vector<float>> weights;
        vector<float> biases;

        Layer(int neurons,int in_dim,int op_dim)
        {
            size = neurons;
            inp_dim = in_dim;
            out_dim = op_dim;
            biases.assign(op_dim,0.01); //Initialize biases to 0.01, a small non zero value.

        }

        //Function to perform xavier_normal initialization.
        void xavier_normal_initialization(int seed = 76)
        {

            int fan_in = inp_dim, fan_out = out_dim;
            float mu = 0; // mean is zero for
            float sigma = sqrt(2.0/(fan_in+fan_out));

            mt19937 gen(seed); //pseudo random number generator, with a seed.
            normal_distribution<float> N(mu,sigma); //mu passed would be 0 and sigma would be as per xavier initialization mech.

            weights.resize(fan_in,vector<float>(fan_out,0)); //Fixing the shape of the weights matrix and initializing with 0s.

            for(int i=0;i<fan_in;i++)
            {
                for(int j=0;j<fan_out;j++)
                    weights[i][j] = N(gen);

            }
// 
        }

};

// Specifies a GCN Layer
class GCNLayer : public Layer
{
    public:

        GCNLayer(int neurons,int in_dim,int op_dim) : Layer(neurons,in_dim,op_dim)
        {
            //Nothing to do here.
        }

};

// Specifies a GraphSAGE Layer
class SAGELayer : public Layer
{
    public:
        
        SAGELayer(int neurons,int in_dim,int op_dim) : Layer(neurons,2*in_dim,op_dim)
        {
            //Input Dimension will be double for GraphSAGE Layer due to concatenation   
        }

};


// Specifies a GraphSAGE Layer
class GINLayer : public Layer
{
    public:

        GINLayer(int neurons,int in_dim,int op_dim) : Layer(neurons,in_dim,op_dim)
        {
            //  Layer specific learnable parameter epsilon must ideally be initialized here

            // Epsilon is part of the GIN class because:
                // GNN class uses a vector of Layer pointers
                // These inturn are used to access the weights and biases of the layers in the optimizer
                // This is is agnostic to the type of layer, because weights and biases are members of the base class Layer
                // TO extend this to GIN, it is necessary to add epsilon to the base class Layer, which causes unncessary confusion for the derived layers where epsilon doesn't exist. 
        }

};

// Class to support activation functions : to apply after neural network layers.
// Supports : Tanh, ReLU, Softmax

class Activation
{
    public:

        //tanh for batch size = 1
        void Tanh(vector<float> &W)
        {
                transform(W.begin(),W.end(),W.begin(),[](float &element)
                {
                    return tanh(element);   
                });
        }

        //tanh for batch size>1 [overloading]
        void Tanh(vector<vector<float>> &W)
        {
                for(auto &row : W)
                    transform(row.begin(),row.end(),row.begin(),[](float &element)
                    {
                        return tanh(element);   
                    });
        }

        //ReLU for batch size=1
        void ReLU(vector<float> &W)
        {

            auto relu = [](float& element) -> float {
                return (element>0)?element:0;
            };

            transform(W.begin(),W.end(),W.begin(),relu);

        }

        //ReLU for batch size>1 [overloading]
        void ReLU(vector<vector<float>> &W)
        {

            auto relu = [](float& element) -> float {
                return (element>0)?element:0;
            };

            for(auto &row : W)
                transform(row.begin(),row.end(),row.begin(),relu);
        }

        //Numerically Stable Softmax
        void Softmax(vector<float> &W)
        {
            // Find the maximum value to stabilize computation
            float max_val = *max_element(W.begin(), W.end());
            float sum = 0;

            // Subtract max and compute exponentials to prevent overflow
            for(int i=0; i<W.size(); i++)
            {
                W[i] = exp(W[i] - max_val);
                sum += W[i];
            }

            // Normalize to get probabilities
            for(int i=0; i<W.size(); i++)
                W[i] /= sum;
        }

        //function to compute the softmax over a vector, with Batch size>1 [overloading]
        void Softmax(vector<vector<float>> &W)
        {
            for(auto &row : W)
            {
                // Find the maximum value in this row
                float max_val = *max_element(row.begin(), row.end());
                float sum = 0;
                
                // Subtract max and compute exponentials to prevent overflow
                for(int i=0; i<row.size(); i++)
                {
                    row[i] = exp(row[i] - max_val);
                    sum += row[i];
                }

                // Normalize to get probabilities
                for(int i=0; i<row.size(); i++)
                    row[i] /= sum;
            }
        }
};

//Class to support functions for the gradient of activation functions
class GradActivation
{
    public:
        //gradient of tanh for batch size = 1
        void Tanh_d(vector<float> &W)
        {
                transform(W.begin(),W.end(),W.begin(),[](float &element)
                {
                    return 1-tanh(element)*tanh(element);
                });
        }

        //gradient of tanh for batch size>1 [overloading]
        void Tanh_d(vector<vector<float>> &W)
        {
                for(auto &row : W)
                    transform(row.begin(),row.end(),row.begin(),[](float &element)
                    {
                        return 1-tanh(element)*tanh(element);
                    });
        }

        //gradient of ReLU for batch size=1
        void ReLU_d(vector<float> &W)
        {

            auto relu = [](float& element) -> float 
            {
                return (element>0)?1:0;
            };

            transform(W.begin(),W.end(),W.begin(),relu);

        }

        //gradient of ReLU for batch size>1 [overloading]
        void ReLU_d(vector<vector<float>> &W)
        {

            auto relu = [](float& element) -> float {
                return (element>0)?1:0;
            };

            for(auto &row : W)
                transform(row.begin(),row.end(),row.begin(),relu);
        }

};

class Dataset
{

    /*
    Expects a folder with the following files:

        ../labels.txt : vector of integers
        ../features.txt : vector of float
    */

    public:

        vector<vector<float>> features;
        vector<int> labels;
        vector<int> train_indices;
        vector<int> val_indices;
        vector<int> test_indices;
        graph Graph;

        int num_classes;
        int input_feature_dim;

        template<typename T>
        vector<vector<T>> readMatrix(string filename, const T& dummy_parm) //dummy parameter to deduce the type of the matrix
        {
            ifstream file(filename);
            string line;
            vector<vector<T>> array;
            vector<T> row;

            if (!file.is_open()) 
            {
                cerr << "Error opening file: " << filename << endl;
                return {};  // Return an empty vector if file cannot be opened
            }

            while (getline(file, line)) 
            {
                stringstream ss(line);
                vector<float> row;
                float value;

                while (ss >> value) 
                    row.push_back(value);

                array.push_back(row);
            }

            return array;
        }

        template<typename T>
        vector<T> readArray(string filename, const T& dummy_parm)
        {
            ifstream file(filename);
            vector<T> array;
            T value;

            if (!file.is_open()) 
            {
                cerr << "Error opening file: " << filename << endl;
                return {};  // Return an empty vector if file cannot be opened
            }

            // Read values from the file and store them in the vector
            while (file >> value) 
                array.push_back(value);
            
            return array;

        }
            
        Dataset(string folder_path,graph &inp_graph, float train_ratio = 0.8) : Graph(inp_graph)
        {
            features = readMatrix(folder_path+"/features.txt",(float)0.0);
            labels = readArray(folder_path+"/labels.txt",(int)0);
            train_indices = readArray(folder_path+"/train_indices.txt",(int)0);
            val_indices = readArray(folder_path+"/val_indices.txt",(int)0);
            test_indices = readArray(folder_path+"/test_indices.txt",(int)0);

            set<int> unique_labels;

            for(auto &l:labels)
                unique_labels.insert(l);

            //assuming labels are 0 indexed and consecutive. 
            num_classes = unique_labels.size();
            input_feature_dim = features[0].size();
            
        }

        void printDataStats()
        {
            cout<<"\n Data Reading Complete...\n";
            cout<<"\n\tNumber of vertices \t\t"<<Graph.num_nodes()<<endl;
            cout<<"\n\tNumber of Edges   \t\t"<<Graph.num_edges()<<endl;
            cout<<"\n\tNumber of Labels \t\t"<<labels.size()<<endl;
            cout<<"\n\tNumber of Train Vertices\t"<<train_indices.size()<<endl;
            cout<<"\n\tNumber of Val Vertices\t\t"<<val_indices.size()<<endl;
            cout<<"\n\tNumber of Test Vertices\t\t"<<test_indices.size()<<endl;
            cout<<"\n\tFeature Matrix Size\t\t"<<features.size()<<"x"<<features[0].size()<<endl<<endl;
        }

};

struct SubGraph
{
    unordered_map<int,vector<int>> adj_list;
    unordered_set<int> vertices;
    unordered_set<int> receptive_field;
    vector<int> target_vertices;
    unordered_map<int, int> vertex_map;
};

class NHSampler
{
    public:

        unordered_set<int> target_vertices;
        Dataset &data;
        int num_layers;
        vector<int> sample_sizes;

        vector<SubGraph> LayerSubgraphs;

        
        NHSampler(Dataset &data, unordered_set<int> target_vertices, vector<int> sample_sizes) : data(data)
        {
            this->target_vertices = target_vertices;
            this->sample_sizes = sample_sizes;
            num_layers = sample_sizes.size();
            LayerSubgraphs.resize(num_layers);
        }

        void sampleLayer(int layer_id, int epoch = 0, int base_seed = 76)
        {
            SubGraph &subgraph = LayerSubgraphs[layer_id];
            
            for (int u : subgraph.target_vertices) 
            {
                unsigned int seed = static_cast<unsigned int>(epoch + layer_id*10 + u + base_seed);
                
                vector<int> sampled_neighbors = data.Graph.RandomSampleNeighbors(u,sample_sizes[layer_id],seed);
                
                for (auto &v: sampled_neighbors)
                {
                    subgraph.vertices.insert(v);
                    subgraph.receptive_field.insert(v);
                }
                    
                subgraph.adj_list[u] = sampled_neighbors;
            }
            
        }

        void sampleAllLayers(int epoch = 0, int base_seed = 76)
        {
            int next_idx = 0;
            
            for (int i = num_layers-1; i >= 0; i--) 
            {
                SubGraph &subgraph = LayerSubgraphs[i];
                
                if (i == num_layers-1)
                {
                    subgraph.target_vertices.assign(target_vertices.begin(), target_vertices.end());
                    for (int v : target_vertices) 
                    {
                        subgraph.vertices.insert(v);
                    }
                    
                    next_idx = 0;
            
                    for (int v : subgraph.target_vertices) 
                    {
                        subgraph.vertex_map[v] = next_idx++;
                    }
                } 
                else 
                {
                    subgraph.target_vertices.assign(LayerSubgraphs[i+1].receptive_field.begin(), LayerSubgraphs[i+1].receptive_field.end());
                    
                    subgraph.vertex_map = LayerSubgraphs[i+1].vertex_map;
                    next_idx = subgraph.vertex_map.size();
                }
                
                sampleLayer(i, epoch, base_seed);
                
                for (int v : subgraph.vertices) 
                {
                    if (subgraph.vertex_map.find(v) == subgraph.vertex_map.end()) 
                    {
                        subgraph.vertex_map[v] = next_idx++;
                    }
                }
            }
        }
};


class Optimiser; //forward declaration

class GNN
{
    public:
            string hidden_activation;
            string output_activation;
            Dataset data;
            vector<int> hidden_sizes;
            string algo;
            
            int count;
            float avg_train_loss;
            float correct_predictions;
            float train_accuracy;
            
            Activation acti;
            
            vector<unique_ptr<Layer>> layers;
            vector<vector<vector<float>>> all_Ais, all_His;
            vector<vector<vector<float>>> dw;
            vector<vector<float>> db;

            vector<vector<float>> eps_state_vectors;
            vector<float> grad_epsilon;
            vector<float> epsilon;

            Optimiser *optimiser;

            GNN(vector<int> &hidden_sizes,int input_size,int output_size,string h_acti,string op_acti, Dataset &ds, string gnn_type) : data(ds)
            {
                assert(!hidden_sizes.empty());

                this->hidden_sizes = hidden_sizes;
                hidden_activation = h_acti;
                output_activation = op_acti;
                algo = gnn_type;

                count = 0;
                avg_train_loss = 0;
                correct_predictions = 0;
                train_accuracy = 0;
            }

            GNN(Dataset &ds, string gnn_type): data(ds)
            {
                algo = gnn_type;
            }

            void processTrainingStats()
            {
                avg_train_loss = avg_train_loss/count;
                train_accuracy = correct_predictions*100/count;
            }

            void recordTrainingStats(vector<float> y_pred_probs,int cur_vertex)
            {                
                int y_true = data.labels[cur_vertex];
                int cur_pred = argmax(y_pred_probs);
                avg_train_loss += cross_entropy_loss(y_pred_probs,data.labels,cur_vertex);
                
                if(cur_pred == y_true)
                    correct_predictions++;
                count++;
            }

            int getNumTrainingVertices()
            {
                return data.train_indices.size();
            }

            void computeTrainingStats()
            {
                avg_train_loss = avg_train_loss/count;
                train_accuracy = correct_predictions*100/count;
            }

            void resetTrainingStats()
            {
                count = 0;
                avg_train_loss = 0;
                correct_predictions = 0;
                train_accuracy = 0;
            }

            virtual void resetGrads()
            {
                TensorTransforms::fill_with_zeros(dw);
                TensorTransforms::fill_with_zeros(db);
            }

            void PrintWeights(int l)
            {
                for(int i=0;i<layers[l]->inp_dim;i++)
                {
                    for(int j=0;j<layers[l]->out_dim;j++)
                        cout<<layers[l]->weights[i][j]<<" ";
                    cout<<"\n";
                }
                cout<<endl;
            }

            void printArchitecture()
            {
                cout << "\n================================================================================" << endl;
                cout << "                             Network Architecture                             " << endl;
                cout << "================================================================================" << endl;

                cout << "Number of Layers : " << layers.size() << endl << endl;

                cout << "Layer : 0(inp)\t Neurons : " << layers[0]->inp_dim << "\t Dim : NA\t Params : NA" << endl << endl;

                long long total_params = 0;
                for(int i = 0; i < layers.size(); i++)
                {
                    long long params = 0;
                    params += layers[i]->inp_dim * layers[i]->out_dim;
                    params += layers[i]->out_dim;
                    total_params += params;

                    cout << "Layer : " << i+1 << "\t Neurons : " << layers[i]->size 
                         << "\t Dim : " << layers[i]->inp_dim << "x" << layers[i]->out_dim 
                         << "\t Params : " << params << endl;
                    
                    if(i < layers.size()-1)
                        cout << hidden_activation << " Activation" << endl << endl;
                    else
                        cout << output_activation << " Activation" << endl << endl;
                }
                cout << "Total Params : " << total_params << endl;
                cout << "================================================================================" << endl;
            }

            void printWeightDims()
            {
                for(int i=0;i<layers.size();i++)
                {
                    cout<<"\t\tCur layer("<<i<<") Weights dim : "<<layers[i]->weights.size()<<"x"<<layers[i]->weights[0].size()<<"\n";
                }
            }

            void printDims(vector<vector<float>> &V)
            {
                for(auto &row : V)
                {
                    cout<<"\t\tdim : "<<row.size()<<"\n";
                }
            }

            void printDims(vector<vector<vector<float>>> &V)
            {
                for(int i=0;i<V.size();i++)
                        cout<<"\t\tdim : "<<V[i].size()<<"x"<<V[i][0].size()<<"\n";
            }

            int getTrainVertexId(int index)
            {
                return data.train_indices[index];
            }

            int getTrainSetSize()
            {
                return data.train_indices.size();
            }

            void displayEpochStats();
            void testModel();
            void optimiser_step();
            void evaluateModel(string);

            void virtual createLayers(vector<int> &hidden_sizes, int input_size, int output_size) = 0;
            void virtual InitializeWeights(int seed=76) = 0;
            vector<float> virtual forward(int current_vertex,int epoch = 0, int batch_size = 1, int test_mode = 0, int base_seed = 76) = 0;
            void virtual backprop(int current_vertex,vector<float> &y_pred) = 0;
};;

//Inductive variant of GCN
// The GCN class is a derived class of the GNN class, inheriting its properties and methods.
class GCN : public GNN
{
    public:
        vector<int> sample_sizes;
        string aggregation_type;

        GCN(vector<int> &hidden_sizes,vector<int> &samp_sizes,int input_size,int output_size,string h_acti,string op_acti, Dataset &ds) : GNN(hidden_sizes,input_size,output_size,h_acti,op_acti, ds, "GCN")
        {
        assert(samp_sizes.size() == hidden_sizes.size()+1);
        sample_sizes = samp_sizes;
        aggregation_type = "mean";
        createLayers(hidden_sizes,input_size,output_size);
        }

        GCN(string filename, Dataset &ds): GNN(ds,"GCN")
        {
        ifstream inFile(filename, ios::binary);
        if (!inFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for reading." << endl;
            return;
        }

        auto readString = [&inFile]() -> string {
            size_t len;
            inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
            string str(len, ' ');
            inFile.read(&str[0], len);
            return str;
        };

        auto readIntVector = [&inFile]() -> vector<int> {
            size_t size;
            inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
            vector<int> vec(size);
            for (size_t i = 0; i < size; i++) {
            inFile.read(reinterpret_cast<char*>(&vec[i]), sizeof(int));
            }
            return vec;
        };

        hidden_activation = readString();
        output_activation = readString();
        aggregation_type = readString();

        int inputSize, outputSize;
        inFile.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        inFile.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

        hidden_sizes = readIntVector();
        sample_sizes = readIntVector();

        int numLayers;
        inFile.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        createLayers(hidden_sizes, inputSize, outputSize);

        dw.clear();
        db.clear();
        for (int i = 0; i < layers.size(); i++) {
            vector<vector<float>> tmp_w(layers[i]->inp_dim, vector<float>(layers[i]->out_dim, 0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim, 0);
            db.push_back(tmp_b);
        }

        count = 0;
        avg_train_loss = 0;
        correct_predictions = 0;
        train_accuracy = 0;

        for (int l = 0; l < numLayers; l++) {
            int rows, cols;
            inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            
            if (rows != layers[l]->weights.size() || cols != layers[l]->weights[0].size()) {
            layers[l]->weights.resize(rows, vector<float>(cols, 0.0));
            layers[l]->inp_dim = rows;
            layers[l]->out_dim = cols;
            }
            
            for (int i = 0; i < rows; i++) {
            inFile.read(reinterpret_cast<char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize;
            inFile.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
            
            if (biasSize != layers[l]->biases.size()) {
            layers[l]->biases.resize(biasSize, 0.0);
            }
            
            inFile.read(reinterpret_cast<char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        inFile.close();
        cout << "Model successfully loaded from binary file " << filename << endl;
        cout << "\nModel Configuration:" << endl;
        cout << "Input Size: " << inputSize << endl;
        cout << "Output Size: " << outputSize << endl;
        cout << "Hidden Activation: " << hidden_activation << endl;
        cout << "Output Activation: " << output_activation << endl;
        cout << "Aggregation Type: " << aggregation_type << endl;
        cout << "Hidden Sizes: ";
        for (int size : hidden_sizes) cout << size << " ";
        cout << endl;
        cout << "Sample Sizes: ";
        for (int size : sample_sizes) cout << size << " ";
        cout << endl;
        }

        bool saveModelTxt(const string& filename)
        {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }
        
        outFile << "HIDDEN_ACTIVATION: " << hidden_activation << endl;
        outFile << "OUTPUT_ACTIVATION: " << output_activation << endl;
        outFile << "AGGREGATION_TYPE: " << aggregation_type << endl;
        outFile << "INPUT_SIZE: " << data.input_feature_dim << endl;
        outFile << "OUTPUT_SIZE: " << data.num_classes << endl;
        
        outFile << "HIDDEN_SIZES: " << hidden_sizes.size() << endl;
        for (int i = 0; i < hidden_sizes.size(); i++) {
            outFile << hidden_sizes[i] << " ";
        }
        outFile << endl;
        
        outFile << "SAMPLE_SIZES: " << sample_sizes.size() << endl;
        for (int i = 0; i < sample_sizes.size(); i++) {
            outFile << sample_sizes[i] << " ";
        }
        outFile << endl;
        
        outFile << "NUM_LAYERS: " << layers.size() << endl;
        
        for (int l = 0; l < layers.size(); l++) {
            outFile << "LAYER: " << l << endl;
            
            outFile << "WEIGHTS_DIMS: " << layers[l]->weights.size() << " " 
                << layers[l]->weights[0].size() << endl;
            
            outFile << "WEIGHTS:" << endl;
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10);
            for (int i = 0; i < layers[l]->weights.size(); i++) {
            for (int j = 0; j < layers[l]->weights[i].size(); j++) {
                outFile << layers[l]->weights[i][j] << " ";
            }
            outFile << endl;
            }
            
            outFile << "BIASES_DIM: " << layers[l]->biases.size() << endl;
            
            outFile << "BIASES:" << endl;
            for (int i = 0; i < layers[l]->biases.size(); i++) {
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10) << layers[l]->biases[i] << " ";
            }
            outFile << endl;
        }
        
        outFile.close();
        cout << "Model successfully saved to " << filename << endl;
        return true;
        }

        bool saveModel(const string& filename)
        {
        ofstream outFile(filename, ios::binary);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }

        auto writeString = [&outFile](const string& str) {
            size_t len = str.size();
            outFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
            outFile.write(str.c_str(), len);
        };

        auto writeIntVector = [&outFile](const vector<int>& vec) {
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
            for (int val : vec) {
            outFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        };

        writeString(hidden_activation);
        writeString(output_activation);
        writeString(aggregation_type);

        int input_size = data.input_feature_dim;
        int output_size = data.num_classes;
        outFile.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        outFile.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        writeIntVector(hidden_sizes);
        writeIntVector(sample_sizes);

        int num_layers = layers.size();
        outFile.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (int l = 0; l < layers.size(); l++) {
            int rows = layers[l]->weights.size();
            int cols = rows > 0 ? layers[l]->weights[0].size() : 0;
            outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            
            for (int i = 0; i < rows; i++) {
            outFile.write(reinterpret_cast<const char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize = layers[l]->biases.size();
            outFile.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
            
            outFile.write(reinterpret_cast<const char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        outFile.close();
        cout << "Model successfully saved to binary file " << filename << endl;
        return true;
        }

        void createLayers(vector<int> &hidden_sizes, int input_size, int output_size) override
        {
        for(int i = 0;i<hidden_sizes.size();i++)
        {
            if(i==0)
            {
            layers.emplace_back(std::make_unique<GCNLayer>(hidden_sizes[i],input_size,hidden_sizes[i]));
            }
            else
            {
            layers.emplace_back(std::make_unique<GCNLayer>(hidden_sizes[i],hidden_sizes[i-1],hidden_sizes[i]));
            }
        }

        layers.emplace_back(std::make_unique<GCNLayer>(output_size,hidden_sizes.back(),output_size));
        }

        void InitializeWeights(int seed = 76) override
        {
        for(auto &layer : layers)
            layer->xavier_normal_initialization(seed);

        for(int i = 0;i<layers.size();i++)
        {
            vector<vector<float>> tmp_w(layers[i]->inp_dim,vector<float>(layers[i]->out_dim,0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim,0);
            db.push_back(tmp_b);
        }
        }

        vector<float> forward(int current_vertex,int epoch = 0, int batch_size = 1, int test_mode = 0, int base_seed = 76) override
        {
        all_Ais.clear();
        all_His.clear();
        
        unordered_set<int> target_vertices = {current_vertex};
        NHSampler sampler(data, target_vertices, sample_sizes); 

        sampler.sampleAllLayers(epoch, base_seed);

        int rows,cols = data.features[0].size(),num_layers = layers.size();

        if(num_layers == 1)
            rows = sampler.LayerSubgraphs[0].target_vertices.size();
        else if(num_layers > 1)
            rows = sampler.LayerSubgraphs[1].vertex_map.size();
        else
            cout<<"Invalid number of layers"<<endl;

        vector<vector<float>> feature_matrix(rows,vector<float>(cols,0));

        int tar_vertex_index;

        if(num_layers == 1)
        {
            tar_vertex_index = 0;

            auto &subgraph = sampler.LayerSubgraphs[0];
            for (int idx = 0; idx<subgraph.target_vertices.size();idx++)
            {
            int tar_node = subgraph.target_vertices[idx];

            std::copy(data.features[tar_node].begin(), data.features[tar_node].end(), 
               feature_matrix[idx].begin());

            auto neighbors = subgraph.adj_list[tar_node];

            vector<float> agg_vector(feature_matrix[idx]);
            
            float factor = 1.0/(neighbors.size()+1);

            TensorTransforms::Scale(agg_vector,factor);

            for(auto &neigh : neighbors)
                TensorTransforms::Add(agg_vector,data.features[neigh],factor);
            
            feature_matrix[idx] = agg_vector;
            }
        }

        if(num_layers > 1)
        {
            auto &subgraph = sampler.LayerSubgraphs[1];

            tar_vertex_index = subgraph.vertex_map[current_vertex];
            
            for (const auto& [node, idx] : subgraph.vertex_map) 
            {
            std::copy(data.features[node].begin(), data.features[node].end(), 
               feature_matrix[idx].begin());
            }

            auto &cur_subgraph = sampler.LayerSubgraphs[0];
            for (const auto& [vertex, neighbors] : cur_subgraph.adj_list)
            {
            int idx = cur_subgraph.vertex_map[vertex];

            vector<float> agg_vector(feature_matrix[idx]);
            
            float factor = 1.0/(neighbors.size()+1);

            TensorTransforms::Scale(agg_vector,factor);

            for(auto &neigh : neighbors) {
                TensorTransforms::Add(agg_vector, data.features[neigh], factor);
            }
            
            feature_matrix[idx] = std::move(agg_vector);
            }
        }
        
        int batch_start_index = 0;
        
        vector<vector<float>> output;

        if(!test_mode)
        {
            vector<vector<float>> cur_data_row;
            cur_data_row.push_back(feature_matrix[tar_vertex_index]);
            all_His.push_back(cur_data_row);
        }

        for(int i=0; i<layers.size(); i++)
        {
            int NH_batch_size = feature_matrix.size();
            
            feature_matrix = TensorTransforms::Mul(feature_matrix,layers[i]->weights,batch_start_index,NH_batch_size);
            
            TensorTransforms::Add(feature_matrix,layers[i]->biases);

            if(!test_mode)
            {
            vector<vector<float>> tmp_Ai;
            tmp_Ai.push_back(feature_matrix[tar_vertex_index]);
            all_Ais.push_back(tmp_Ai);
            }

            if(i<layers.size()-1)
            {
            if(hidden_activation == "tanh")
                acti.Tanh(feature_matrix);
            else if(hidden_activation == "relu")
                acti.ReLU(feature_matrix);

            auto &subgraph = sampler.LayerSubgraphs[i+1];
            for (const auto& [vertex, neighbors] : subgraph.adj_list)
            {
                int idx = subgraph.vertex_map[vertex];

                vector<float> agg_vector(feature_matrix[idx]);
                float factor = factor = 1.0/(neighbors.size()+1);

                TensorTransforms::Scale(agg_vector,factor);

                for(auto &neigh : neighbors)
                TensorTransforms::Add(agg_vector,feature_matrix[subgraph.vertex_map[neigh]],factor);
                
                feature_matrix[idx] = std::move(agg_vector);
            }
            feature_matrix.resize(subgraph.vertex_map.size());
            }
            else
            acti.Softmax(feature_matrix[tar_vertex_index]);
            
            if(!test_mode)
            {
            vector<vector<float>> tmp_Ai;
            tmp_Ai.push_back(feature_matrix[tar_vertex_index]);
            all_His.push_back(tmp_Ai);
            }

            if (i == layers.size()-1)
            output.push_back(feature_matrix[tar_vertex_index]);
        }

        recordTrainingStats(output[0],current_vertex);
        
        return output[0];
        }

        void backprop(int current_vertex,vector<float> &y_pred) override
        {
        GradActivation activ_grad;
        
        int y_true = data.labels[current_vertex];

        auto grad_ai = y_pred;
        
        grad_ai[int(y_true)] = y_pred[int(y_true)]-1;

        for(int i=layers.size()-1;i>=0;i--)
        {
            auto dw_cur = TensorTransforms::Outer(all_His[i], grad_ai);
            
            TensorTransforms::Add(dw[i], dw_cur);
            TensorTransforms::Add(db[i], grad_ai);
            
            if(i>0)
            {
            vector<float> grad_h_prev(layers[i-1]->biases.size());
            
            grad_h_prev = TensorTransforms::MatrixVectorProduct(layers[i]->weights, grad_ai);
            
            auto& prev_layer_ai = all_Ais[i-1][0];
            
            vector<float> prev_layer_ai_grad = prev_layer_ai;
            
            if(hidden_activation == "tanh")
                activ_grad.Tanh_d(prev_layer_ai_grad);
            else if(hidden_activation == "relu")
                activ_grad.ReLU_d(prev_layer_ai_grad);
            
            grad_ai = TensorTransforms::Hadamard(grad_h_prev, prev_layer_ai_grad);
            }
        }
        }
};







//Class to implement the GraphSAGE Algorithm for Node Classification
class GraphSAGE : public GNN
{
    public:

        string aggregation_type; //either "sum" or "mean".
        vector<int> sample_sizes;
        
        GraphSAGE(vector<int> &hidden_sizes,vector<int> &samp_sizes,string aggr_type,int input_size,int output_size,string h_acti,string op_acti, Dataset &ds) : GNN(hidden_sizes,input_size,output_size,h_acti,op_acti, ds, "GraphSAGE")
        {
        
        assert(samp_sizes.size() == hidden_sizes.size()+1);
        sample_sizes = samp_sizes;
        aggregation_type = aggr_type;

        createLayers(hidden_sizes, input_size, output_size);

        }

        GraphSAGE(string filename, Dataset &ds): GNN(ds,"GraphSAGE")
        {
        ifstream inFile(filename, ios::binary);
        if (!inFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for reading." << endl;
            return;
        }

        auto readString = [&inFile]() -> string {
            size_t len;
            inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
            string str(len, ' ');
            inFile.read(&str[0], len);
            return str;
        };

        auto readIntVector = [&inFile]() -> vector<int> {
            size_t size;
            inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
            vector<int> vec(size);
            for (size_t i = 0; i < size; i++) {
            inFile.read(reinterpret_cast<char*>(&vec[i]), sizeof(int));
            }
            return vec;
        };

        hidden_activation = readString();
        output_activation = readString();
        aggregation_type = readString();

        int inputSize, outputSize;
        inFile.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        inFile.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

        hidden_sizes = readIntVector();
        sample_sizes = readIntVector();

        int numLayers;
        inFile.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        createLayers(hidden_sizes, inputSize, outputSize);

        dw.clear();
        db.clear();
        for (int i = 0; i < layers.size(); i++) {
            vector<vector<float>> tmp_w(layers[i]->inp_dim, vector<float>(layers[i]->out_dim, 0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim, 0);
            db.push_back(tmp_b);
        }

        count = 0;
        avg_train_loss = 0;
        correct_predictions = 0;
        train_accuracy = 0;

        for (int l = 0; l < numLayers; l++) {
            int rows, cols;
            inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            
            if (rows != layers[l]->weights.size() || cols != layers[l]->weights[0].size()) {
            layers[l]->weights.resize(rows, vector<float>(cols, 0.0));
            layers[l]->inp_dim = rows;
            layers[l]->out_dim = cols;
            }
            
            for (int i = 0; i < rows; i++) {
            inFile.read(reinterpret_cast<char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize;
            inFile.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
            
            if (biasSize != layers[l]->biases.size()) {
            layers[l]->biases.resize(biasSize, 0.0);
            }
            
            inFile.read(reinterpret_cast<char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        inFile.close();
        cout << "Model successfully loaded from binary file " << filename << endl;
        cout << "\nModel Configuration:" << endl;
        cout << "Input Size: " << inputSize << endl;
        cout << "Output Size: " << outputSize << endl;
        cout << "Hidden Activation: " << hidden_activation << endl;
        cout << "Output Activation: " << output_activation << endl;
        cout << "Aggregation Type: " << aggregation_type << endl;
        cout << "Hidden Sizes: ";
        for (int size : hidden_sizes) cout << size << " ";
        cout << endl;
        cout << "Sample Sizes: ";
        for (int size : sample_sizes) cout << size << " ";
        cout << endl;

        }

        bool saveModelTxt(const string& filename)
        {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }
        
        outFile << "HIDDEN_ACTIVATION: " << hidden_activation << endl;
        outFile << "OUTPUT_ACTIVATION: " << output_activation << endl;
        outFile << "AGGREGATION_TYPE: " << aggregation_type << endl;
        outFile << "INPUT_SIZE: " << data.input_feature_dim << endl;
        outFile << "OUTPUT_SIZE: " << data.num_classes << endl;
        
        outFile << "HIDDEN_SIZES: " << hidden_sizes.size() << endl;
        for (int i = 0; i < hidden_sizes.size(); i++) {
            outFile << hidden_sizes[i] << " ";
        }
        outFile << endl;
        
        outFile << "SAMPLE_SIZES: " << sample_sizes.size() << endl;
        for (int i = 0; i < sample_sizes.size(); i++) {
            outFile << sample_sizes[i] << " ";
        }
        outFile << endl;
        
        outFile << "NUM_LAYERS: " << layers.size() << endl;
        
        for (int l = 0; l < layers.size(); l++) {
            outFile << "LAYER: " << l << endl;
            
            outFile << "WEIGHTS_DIMS: " << layers[l]->weights.size() << " " 
                << layers[l]->weights[0].size() << endl;
            
            outFile << "WEIGHTS:" << endl;
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10);
            for (int i = 0; i < layers[l]->weights.size(); i++) {
            for (int j = 0; j < layers[l]->weights[i].size(); j++) {
                outFile << layers[l]->weights[i][j] << " ";
            }
            outFile << endl;
            }
            
            outFile << "BIASES_DIM: " << layers[l]->biases.size() << endl;
            
            outFile << "BIASES:" << endl;
            for (int i = 0; i < layers[l]->biases.size(); i++) {
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10) << layers[l]->biases[i] << " ";
            }
            outFile << endl;
        }
        
        outFile.close();
        cout << "Model successfully saved to " << filename << endl;
        return true;
        }

        bool saveModel(const string& filename)
        {
        ofstream outFile(filename, ios::binary);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }

        auto writeString = [&outFile](const string& str) {
            size_t len = str.size();
            outFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
            outFile.write(str.c_str(), len);
        };

        auto writeIntVector = [&outFile](const vector<int>& vec) {
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
            for (int val : vec) {
            outFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        };

        writeString(hidden_activation);
        writeString(output_activation);
        writeString(aggregation_type);

        int input_size = data.input_feature_dim;
        int output_size = data.num_classes;
        outFile.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        outFile.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        writeIntVector(hidden_sizes);
        writeIntVector(sample_sizes);

        int num_layers = layers.size();
        outFile.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (int l = 0; l < layers.size(); l++) {
            int rows = layers[l]->weights.size();
            int cols = rows > 0 ? layers[l]->weights[0].size() : 0;
            outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            
            for (int i = 0; i < rows; i++) {
            outFile.write(reinterpret_cast<const char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize = layers[l]->biases.size();
            outFile.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
            
            outFile.write(reinterpret_cast<const char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        outFile.close();
        cout << "Model successfully saved to binary file " << filename << endl;
        return true;
        }

        int getTrainSetSize()
        {
        return data.train_indices.size();
        }
        
        void createLayers(vector<int> &hidden_sizes, int input_size, int output_size) override
        {
        for(int i = 0;i<hidden_sizes.size();i++)
        {

            if(i==0)
            {
            layers.emplace_back(std::make_unique<SAGELayer>(hidden_sizes[i],input_size,hidden_sizes[i]));
            }
            else
            {
            layers.emplace_back(std::make_unique<SAGELayer>(hidden_sizes[i],hidden_sizes[i-1],hidden_sizes[i]));
            }

        }

        layers.emplace_back(std::make_unique<SAGELayer>(output_size,hidden_sizes.back(),output_size));
        }

        void InitializeWeights(int seed = 76) override
        {
        for(auto &layer : layers)
            layer->xavier_normal_initialization(seed);

        for(int i = 0;i<layers.size();i++)
        {
            vector<vector<float>> tmp_w(layers[i]->inp_dim,vector<float>(layers[i]->out_dim,0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim,0);
            db.push_back(tmp_b);
        }
        }

        vector<float> forward(int current_vertex,int epoch = 0, int batch_size = 1, int test_mode = 0, int base_seed = 76) override
        {
        
            all_Ais.clear();
            all_His.clear();

            int cur_vertex_deg = data.Graph.getOutDegree(current_vertex);

            
            unordered_set<int> target_vertices = {current_vertex};
            NHSampler sampler(data, target_vertices, sample_sizes); 

            sampler.sampleAllLayers(epoch, base_seed);

            int rows,cols = data.features[0].size(),num_layers = layers.size();

            if(num_layers == 1)
            rows = sampler.LayerSubgraphs[0].target_vertices.size();
            else if(num_layers > 1)
            rows = sampler.LayerSubgraphs[1].vertex_map.size();
            else
            cout<<"Invalid number of layers"<<endl;

            vector<vector<float>> feature_matrix(rows,vector<float>(cols,0));

            int tar_vertex_index;


            if(num_layers == 1)
            {
            tar_vertex_index = 0;

            auto &subgraph = sampler.LayerSubgraphs[0];
            for (int idx = 0; idx<subgraph.target_vertices.size();idx++)
            {
                int tar_node = subgraph.target_vertices[idx];

                std::copy(data.features[tar_node].begin(), data.features[tar_node].end(), 
                feature_matrix[idx].begin());
                
                auto neighbors = subgraph.adj_list[tar_node];

                vector<float> agg_vector(feature_matrix[idx].size());

                float factor = 1.0;
                
                if(aggregation_type == "mean")
                factor = 1.0/(neighbors.size()+1);

                for(auto &neigh : neighbors)
                TensorTransforms::Add(agg_vector,data.features[neigh],factor);
                
                agg_vector.insert(agg_vector.end(),data.features[tar_node].begin(),data.features[tar_node].end());
                feature_matrix[idx] = std::move(agg_vector);

            }
            }

            if(num_layers > 1)
            {
            auto &subgraph = sampler.LayerSubgraphs[1];

            tar_vertex_index = subgraph.vertex_map[current_vertex];
            
            for (const auto& [node, idx] : subgraph.vertex_map) 
            {
                std::copy(data.features[node].begin(), data.features[node].end(), 
                feature_matrix[idx].begin());
            }

            auto &cur_subgraph = sampler.LayerSubgraphs[0];

            for (const auto& [vertex, neighbors] : cur_subgraph.adj_list)
            {
                int idx = cur_subgraph.vertex_map[vertex];

                
                vector<float> agg_vector(feature_matrix[idx].size());
                
                float factor = 1.0;
                
                if(aggregation_type == "mean")
                factor = 1.0/(neighbors.size()+1);

                for(auto &neigh : neighbors)
                TensorTransforms::Add(agg_vector,data.features[cur_subgraph.vertex_map[neigh]],factor);
                
                agg_vector.insert(agg_vector.end(),data.features[vertex].begin(),data.features[vertex].end());
                feature_matrix[idx] = std::move(agg_vector);
            }

            }

            for (auto& vec : feature_matrix) {
            if (vec.size() == data.features[0].size()) {
                vector<float> doubled_vec(2 * vec.size(), 0.0f);
                
                copy(vec.begin(), vec.end(), doubled_vec.begin());
                
                vec = doubled_vec;
            }
            }

            
            int batch_start_index = 0;
            
            vector<vector<float>> output;

            if(!test_mode)
            {
            vector<vector<float>> cur_data_row;
            cur_data_row.push_back(feature_matrix[tar_vertex_index]);
            all_His.push_back(cur_data_row);
            }

            for(int i=0; i<layers.size(); i++)
            {
            int NH_batch_size = feature_matrix.size();
            
            feature_matrix = TensorTransforms::Mul(feature_matrix,layers[i]->weights,batch_start_index,NH_batch_size);
                
            TensorTransforms::Add(feature_matrix,layers[i]->biases);

            if(!test_mode)
            {
                vector<vector<float>> tmp_Ai;
                tmp_Ai.push_back(feature_matrix[tar_vertex_index]);
                all_Ais.push_back(tmp_Ai);
            }

            if(i<layers.size()-1)
            {
                if(hidden_activation == "tanh")
                acti.Tanh(feature_matrix);
                else if(hidden_activation == "relu")
                acti.ReLU(feature_matrix);

                auto &subgraph = sampler.LayerSubgraphs[i+1];
                for (const auto& [vertex, neighbors] : subgraph.adj_list)
                {
                int idx = subgraph.vertex_map[vertex];

                vector<float> agg_vector(feature_matrix[idx].size());
                float factor = 1.0;

                if(aggregation_type == "mean")
                factor = 1.0/(neighbors.size()+1);

                for(auto &neigh : neighbors)
                    TensorTransforms::Add(agg_vector,feature_matrix[subgraph.vertex_map[neigh]],factor);
                
                agg_vector.insert(agg_vector.end(),feature_matrix[idx].begin(),feature_matrix[idx].end());
                feature_matrix[idx] = std::move(agg_vector);
                }
                feature_matrix.resize(subgraph.vertex_map.size());
            }

            else
                acti.Softmax(feature_matrix[tar_vertex_index]);
            
            if(!test_mode)
            {
                vector<vector<float>> tmp_Ai;
                tmp_Ai.push_back(feature_matrix[tar_vertex_index]);
                all_His.push_back(tmp_Ai);
            }

            if (i == layers.size()-1)
                output.push_back(feature_matrix[tar_vertex_index]);
            }
        
        recordTrainingStats(output[0],current_vertex);
        
        return output[0];
        
        }

        void backprop(int current_vertex,vector<float> &y_pred) override
        {

        GradActivation activ_grad;

        int y_true = data.labels[current_vertex];

        
        auto grad_aL = y_pred;
        
        grad_aL[int(y_true)] = y_pred[int(y_true)]-1;

        auto grad_ai = grad_aL;

        for(int i=layers.size()-1;i>=0;i--)
        {
            if(i>0)
            {
            vector<float> grad_h_prev = TensorTransforms::MatrixVectorProduct(layers[i]->weights,grad_ai);
            
            vector<float> prev_layer_ai = all_Ais[i-1][0];
            
            vector<float> prev_layer_ai_grad = prev_layer_ai;

            if(grad_h_prev.size() == 2*prev_layer_ai_grad.size())
            {
                int mid = grad_h_prev.size()/2;
                for(int j=0;j<mid;j++)
                grad_h_prev[j] += grad_h_prev[j+mid];
                grad_h_prev.resize(mid);
            }

            if(hidden_activation == "tanh")
                activ_grad.Tanh_d(prev_layer_ai_grad);
            else if(hidden_activation == "relu")
                activ_grad.ReLU_d(prev_layer_ai_grad);

            vector<float> grad_a_prev = TensorTransforms::Hadamard(grad_h_prev, prev_layer_ai_grad);

            auto dw_cur = TensorTransforms::Outer(all_His[i],grad_ai);

            TensorTransforms::Add(dw[i],dw_cur);
            TensorTransforms::Add(db[i],grad_ai);

            grad_ai = grad_a_prev;
            }
            else
            {
            auto dw_cur = TensorTransforms::Outer(all_His[i],grad_ai);
            TensorTransforms::Add(dw[i],dw_cur);
            TensorTransforms::Add(db[i],grad_ai);
            }
        }
        }
};




//Class to implement the Graph Isomorphism Network (GIN) Algorithm for Node Classification
class GIN: public GNN
{

    public:

        void resetGrads() override
        {
        TensorTransforms::fill_with_zeros(dw);
        TensorTransforms::fill_with_zeros(db);

        for(auto &e: grad_epsilon)
            e = 0;
        }

        GIN(vector<int> &hidden_sizes,int input_size,int output_size,string h_acti,string op_acti, Dataset &ds) : GNN(hidden_sizes,input_size,output_size,h_acti,op_acti, ds, "GIN")
        {
        
        createLayers(hidden_sizes,input_size,output_size);
        
        grad_epsilon.assign(layers.size(),0);
        epsilon.assign(layers.size(),0.0);

        }
        GIN(string filename, Dataset &ds): GNN(ds,"GIN")
        {
        ifstream inFile(filename, ios::binary);
        if (!inFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for reading." << endl;
            return;
        }

        auto readString = [&inFile]() -> string {
            size_t len;
            inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
            string str(len, ' ');
            inFile.read(&str[0], len);
            return str;
        };

        auto readIntVector = [&inFile]() -> vector<int> {
            size_t size;
            inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
            vector<int> vec(size);
            for (size_t i = 0; i < size; i++) {
            inFile.read(reinterpret_cast<char*>(&vec[i]), sizeof(int));
            }
            return vec;
        };

        auto readFloatVector = [&inFile]() -> vector<float> {
            size_t size;
            inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
            vector<float> vec(size);
            inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
            return vec;
        };

        hidden_activation = readString();
        output_activation = readString();
        
        readString();

        int inputSize, outputSize;
        inFile.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        inFile.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));

        hidden_sizes = readIntVector();
        
        readIntVector();

        int numLayers;
        inFile.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        createLayers(hidden_sizes, inputSize, outputSize);

        dw.clear();
        db.clear();
        for (int i = 0; i < layers.size(); i++) {
            vector<vector<float>> tmp_w(layers[i]->inp_dim, vector<float>(layers[i]->out_dim, 0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim, 0);
            db.push_back(tmp_b);
        }

        count = 0;
        avg_train_loss = 0;
        correct_predictions = 0;
        train_accuracy = 0;

        for (int l = 0; l < numLayers; l++) {
            int rows, cols;
            inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            
            if (rows != layers[l]->weights.size() || cols != layers[l]->weights[0].size()) {
            layers[l]->weights.resize(rows, vector<float>(cols, 0.0));
            layers[l]->inp_dim = rows;
            layers[l]->out_dim = cols;
            }
            
            for (int i = 0; i < rows; i++) {
            inFile.read(reinterpret_cast<char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize;
            inFile.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
            
            if (biasSize != layers[l]->biases.size()) {
            layers[l]->biases.resize(biasSize, 0.0);
            }
            
            inFile.read(reinterpret_cast<char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        epsilon = readFloatVector();
        
        grad_epsilon.assign(epsilon.size(), 0);

        inFile.close();
        cout << "Model successfully loaded from binary file " << filename << endl;
        cout << "\nModel Configuration:" << endl;
        cout << "Input Size: " << inputSize << endl;
        cout << "Output Size: " << outputSize << endl;
        cout << "Hidden Activation: " << hidden_activation << endl;
        cout << "Output Activation: " << output_activation << endl;
        cout << "Hidden Sizes: ";
        for (int size : hidden_sizes) cout << size << " ";
        cout << endl;
        cout << "Epsilon Values: ";
        for (float eps : epsilon) cout << fixed << setprecision(6) << eps << " ";
        cout << endl;
        }

        bool saveModelTxt(const string& filename)
        {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }
        
        outFile << "HIDDEN_ACTIVATION: " << hidden_activation << endl;
        outFile << "OUTPUT_ACTIVATION: " << output_activation << endl;
        outFile << "INPUT_SIZE: " << data.input_feature_dim << endl;
        outFile << "OUTPUT_SIZE: " << data.num_classes << endl;
        
        outFile << "HIDDEN_SIZES: " << hidden_sizes.size() << endl;
        for (int i = 0; i < hidden_sizes.size(); i++) {
            outFile << hidden_sizes[i] << " ";
        }
        outFile << endl;
        
        outFile << "EPSILON_VALUES: " << epsilon.size() << endl;
        for (int i = 0; i < epsilon.size(); i++) {
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10) << epsilon[i] << " ";
        }
        outFile << endl;
        
        outFile << "NUM_LAYERS: " << layers.size() << endl;
        
        for (int l = 0; l < layers.size(); l++) {
            outFile << "LAYER: " << l << endl;
            
            outFile << "WEIGHTS_DIMS: " << layers[l]->weights.size() << " " 
                << layers[l]->weights[0].size() << endl;
            
            outFile << "WEIGHTS:" << endl;
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10);
            for (int i = 0; i < layers[l]->weights.size(); i++) {
            for (int j = 0; j < layers[l]->weights[i].size(); j++) {
                outFile << layers[l]->weights[i][j] << " ";
            }
            outFile << endl;
            }
            
            outFile << "BIASES_DIM: " << layers[l]->biases.size() << endl;
            
            outFile << "BIASES:" << endl;
            for (int i = 0; i < layers[l]->biases.size(); i++) {
            outFile << scientific << setprecision(std::numeric_limits<float>::max_digits10) << layers[l]->biases[i] << " ";
            }
            outFile << endl;
        }
        
        outFile.close();
        cout << "Model successfully saved to " << filename << endl;
        return true;
        }

        bool saveModel(const string& filename)
        {
        ofstream outFile(filename, ios::binary);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return false;
        }

        auto writeString = [&outFile](const string& str) {
            size_t len = str.size();
            outFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
            outFile.write(str.c_str(), len);
        };

        auto writeIntVector = [&outFile](const vector<int>& vec) {
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
            for (int val : vec) {
            outFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        };

        auto writeFloatVector = [&outFile](const vector<float>& vec) {
            size_t size = vec.size();
            outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
            outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        };

        writeString(hidden_activation);
        writeString(output_activation);
        writeString("");

        int input_size = data.input_feature_dim;
        int output_size = data.num_classes;
        outFile.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        outFile.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        writeIntVector(hidden_sizes);
        
        writeIntVector({});

        int num_layers = layers.size();
        outFile.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (int l = 0; l < layers.size(); l++) {
            int rows = layers[l]->weights.size();
            int cols = rows > 0 ? layers[l]->weights[0].size() : 0;
            outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            
            for (int i = 0; i < rows; i++) {
            outFile.write(reinterpret_cast<const char*>(layers[l]->weights[i].data()), cols * sizeof(float));
            }
            
            int biasSize = layers[l]->biases.size();
            outFile.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
            
            outFile.write(reinterpret_cast<const char*>(layers[l]->biases.data()), biasSize * sizeof(float));
        }

        writeFloatVector(epsilon);

        outFile.close();
        cout << "Model successfully saved to binary file " << filename << endl;
        return true;
        }

        

        void createLayers(vector<int> &hidden_sizes, int input_size, int output_size) override
        {
        for(int i = 0;i<hidden_sizes.size();i++)
        {

            if(i==0)
            {

            layers.emplace_back(std::make_unique<GINLayer>(hidden_sizes[i],input_size,hidden_sizes[i]));
            }
            else
            {
            layers.emplace_back(std::make_unique<GINLayer>(hidden_sizes[i],hidden_sizes[i-1],hidden_sizes[i]));
            
            }
        }

        layers.emplace_back(std::make_unique<GINLayer>(output_size,hidden_sizes.back(),output_size));
        
        }

        void InitializeWeights(int seed = 76) override
        {
        for(auto &layer : layers)
            layer->xavier_normal_initialization(seed);

        for(int i = 0;i<layers.size();i++)
        {
            vector<vector<float>> tmp_w(layers[i]->inp_dim,vector<float>(layers[i]->out_dim,0));
            dw.push_back(tmp_w);

            vector<float> tmp_b(layers[i]->out_dim,0);
            db.push_back(tmp_b);
        }
        }

        vector<float> forward(int current_vertex, int epoch = 0, int batch_size = 1, int test_mode = 0, int base_seed = 76) override
        {
        all_Ais.clear();
        all_His.clear();
        eps_state_vectors.clear();

        auto all_neighs = data.Graph.getNeighbors(current_vertex);
        int full_NH_size = all_neighs.size();

        int NH_size = full_NH_size;
        

        vector<vector<float>> NH_data(NH_size+1,vector<float>(data.features[0].size(),0));
        

        for(int i=0;i<all_neighs.size();i++)
            NH_data[i] = data.features[all_neighs[i].destination];
        unordered_map<int,int> vertex_index_map; 

        NH_data[NH_size] = data.features[current_vertex];

        int cur_vertex_index = NH_size;
        int NH_batch_size = NH_size+1;
        int batch_start_index = 0;
        
        vector<vector<float>> output;

        if(!test_mode)
        {
            vector<vector<float>> cur_data_row;
            cur_data_row.push_back(NH_data[cur_vertex_index]);
            all_His.push_back(cur_data_row);
        }

        for(int i=0; i<layers.size(); i++)
        {
            
            vector<float> agg_vector(NH_data[0].size());
            
            float factor = 1.0/NH_size;

            for(int i=1;i<all_neighs.size();i++)
            TensorTransforms::Add(agg_vector,NH_data[i],factor);

            TensorTransforms::Add(agg_vector,NH_data[cur_vertex_index],1+epsilon[i]);

            if(!test_mode)
            {
            eps_state_vectors.push_back(NH_data[cur_vertex_index]);
            }

            NH_data[cur_vertex_index] = agg_vector;
            
            NH_data = TensorTransforms::Mul(NH_data,layers[i]->weights,batch_start_index,NH_batch_size);
            
            TensorTransforms::Add(NH_data,layers[i]->biases);

            if(!test_mode)
            {
            vector<vector<float>> tmp_Ai;
            tmp_Ai.push_back(NH_data[cur_vertex_index]);
            all_Ais.push_back(tmp_Ai);
            }

            if(i<layers.size()-1)
            {
            if(hidden_activation == "tanh")
                acti.Tanh(NH_data);
            else if(hidden_activation == "relu")
                acti.ReLU(NH_data);

            }
            else
            acti.Softmax(NH_data);
            
            if(!test_mode)
            {
            vector<vector<float>> tmp_Ai;
            tmp_Ai.push_back(NH_data[cur_vertex_index]);
            all_His.push_back(tmp_Ai);
            }

            if (i == layers.size()-1)
            output.push_back(NH_data[cur_vertex_index]);
        }

        recordTrainingStats(output[0],current_vertex);

        return output[0];
        
        }

        void backprop(int current_vertex,vector<float> &y_pred) override
        {

        GradActivation activ_grad;

        int y_true = data.labels[current_vertex];

        auto grad_aL = y_pred;
        
        grad_aL[int(y_true)] = y_pred[int(y_true)]-1;

        auto grad_ai = grad_aL;

        for(int i=layers.size()-1;i>=0;i--)
        {
            if(i>0)
            {

            vector<float> grad_h_prev = TensorTransforms::MatrixVectorProduct(layers[i]->weights,grad_ai);
            vector<float> grad_a_prev(grad_h_prev.size());

            auto prev_layer_ai_grad = all_Ais[i-1];

            if(hidden_activation == "tanh")
                activ_grad.Tanh_d(prev_layer_ai_grad);

            else if(hidden_activation == "relu")
                activ_grad.ReLU_d(prev_layer_ai_grad);

            grad_a_prev = TensorTransforms::Hadamard(grad_h_prev , prev_layer_ai_grad[0]);

            auto dw_cur = std::move(TensorTransforms::Outer(all_His[i],grad_ai));

            TensorTransforms::Add(dw[i],dw_cur);
            TensorTransforms::Add(db[i],grad_ai);

            grad_ai = grad_a_prev;

            assert(eps_state_vectors[i].size() == grad_ai.size());

            grad_epsilon[i] += std::inner_product(eps_state_vectors[i].begin(), 
                                 eps_state_vectors[i].end(), 
                                 grad_ai.begin(), 0.0f);

            }

            else
            {
            
            vector<float> grad_input = TensorTransforms::MatrixVectorProduct(layers[i]->weights,grad_ai);

            assert(eps_state_vectors[i].size() == grad_input.size());

            grad_epsilon[i] += std::inner_product(eps_state_vectors[i].begin(), 
                                 eps_state_vectors[i].end(), 
                                 grad_input.begin(), 0.0f);

            auto dw_cur = std::move(TensorTransforms::Outer(all_His[i],grad_ai));
            TensorTransforms::Add(dw[i],dw_cur);
            TensorTransforms::Add(db[i],grad_ai);
            }
        }
        }


};

// Class to implement the optimization algorithms, for training the neural net
// Supports  : Stochastic and Batch GD, Momentum based GD, RMSprop and Adam.
class Optimiser
{
public:

    GNN &model;
    float lr;
    float l2_param;

    Optimiser(GNN &m, float learning_rate, float l2_param = 0) : model(m)
    {
        lr = learning_rate;
        l2_param = l2_param;
    }
    
    void evaluateModel(string dtype, bool print_stats = true, int test_mode = 1)
    {
        float correct_preds = 0;
        float avg_loss = 0;

        Activation activ;

        vector<int> TP(model.data.num_classes, 0);
        vector<int> FP(model.data.num_classes, 0);
        vector<int> FN(model.data.num_classes, 0);

        vector<int> vertex_indices;
        if(dtype == "Val")
            vertex_indices = model.data.val_indices;
        else if(dtype == "Test")
            vertex_indices = model.data.test_indices;
        else
            vertex_indices = model.data.train_indices;

        int epoch = 0;

        int processed_data_count = 0;
        for(auto &cur_vertex : vertex_indices)
        {
            int batch_size = 1;
            auto y_pred_probs = model.forward(cur_vertex, epoch, batch_size, test_mode);
            
            auto cur_prediction = argmax(y_pred_probs);

            if(int(cur_prediction) == int(model.data.labels[cur_vertex]))
            {
                correct_preds++;
                TP[int(cur_prediction)]++;
            }
            else
            {
                FP[int(cur_prediction)]++;
                FN[int(model.data.labels[cur_vertex])]++;
            }

            avg_loss += (cross_entropy_loss(y_pred_probs, model.data.labels, cur_vertex) - avg_loss) / (processed_data_count + 1);
            processed_data_count++;
        }

        float accuracy = correct_preds * 100 / vertex_indices.size();
        float avg_loss_val = avg_loss;
        
        float macro_precision = 0.0, macro_recall = 0.0, macro_f1 = 0.0;
        if (dtype == "Test") {
            vector<float> precision(model.data.num_classes, 0.0);
            vector<float> recall(model.data.num_classes, 0.0);
            vector<float> F1_score(model.data.num_classes, 0.0);

            for (int i = 0; i < model.data.num_classes; i++) {
                if (TP[i] + FP[i] > 0) {
                    precision[i] = static_cast<float>(TP[i]) / (TP[i] + FP[i]);
                }
            }

            for (int i = 0; i < model.data.num_classes; i++) {
                if (TP[i] + FN[i] > 0) {
                    recall[i] = static_cast<float>(TP[i]) / (TP[i] + FN[i]);
                }
            }

            for (int i = 0; i < model.data.num_classes; i++) {
                if (precision[i] + recall[i] > 0) {
                    F1_score[i] = 2.0f * (precision[i] * recall[i]) / (precision[i] + recall[i]);
                }
            }

            const float inv_num_classes = 1.0f / model.data.num_classes;
            for (int i = 0; i < model.data.num_classes; i++) {
                macro_precision += precision[i];
                macro_recall += recall[i];
                macro_f1 += F1_score[i];
            }

            macro_precision *= inv_num_classes;
            macro_recall *= inv_num_classes;
            macro_f1 *= inv_num_classes;
            
            cout << "\n" << dtype << " Detailed Metrics:" << endl;
            cout << "\tAcc.: " << fixed << setprecision(2) << accuracy << "%" << endl;
            cout << "\tAvg. Loss: " << fixed << setprecision(2) << avg_loss_val << endl;
            cout << "\tMacro Precision: " << fixed << setprecision(2) << macro_precision << endl;
            cout << "\tMacro Recall: " << fixed << setprecision(2) << macro_recall << endl; 
            cout << "\tMacro F1: " << fixed << setprecision(2) << macro_f1 << endl;
            
            if (model.data.num_classes <= 100) {
                cout << "\n\tPer-class metrics:" << endl;
                for (int i = 0; i < model.data.num_classes; i++) {
                    cout << "\t\tClass " << i << ": P=" << fixed << setprecision(2) << precision[i]
                          << ", R=" << recall[i] << ", F1=" << F1_score[i] << endl;
                }
            }
        } else if(dtype == "Val") {
            cout << dtype << " Acc. : " << accuracy << fixed << setprecision(2) << "% Avg. " << dtype 
                << " Loss : " << fixed << setprecision(2) << avg_loss_val << " ";
        } else {
            cout << "\n" << dtype << " Metrics:" << endl;
            cout << "\tAcc.: " << fixed << setprecision(2) << accuracy << "%" << endl;
            cout << "\tAvg. Loss: " << fixed << setprecision(2) << avg_loss_val << endl;
        }
    }

    void displayEpochStats()
    {
        model.processTrainingStats();
        cout<<" Train Acc. : "<<model.train_accuracy<<fixed<<setprecision(2)<<"% Avg. Train Loss : "<<fixed<<setprecision(2)<<model.avg_train_loss<<" "<<flush;
        evaluateModel("Val");
        model.resetTrainingStats();
    }

    void virtual step() = 0;
};


class SGD : public Optimiser
{
    public:

        SGD(GNN &m, float learning_rate, float l2_param) : Optimiser(m,learning_rate,l2_param) {}
        
        void step()
        {
            for(int l=0;l<model.layers.size();l++)
            {
                TensorTransforms::Scale(model.dw[l],lr*(1+l2_param));
                TensorTransforms::Add(model.layers[l]->weights,model.dw[l],-1);

                TensorTransforms::Scale(model.db[l],lr*(1+l2_param));
                TensorTransforms::Add(model.layers[l]->biases,model.db[l],-1);

                if(model.algo == "GIN")
                {
                    model.epsilon[l] -= (lr*(1+l2_param))*model.grad_epsilon[l];
                }
            }
        }
};

class MomentumGD : public Optimiser
{
    public:
        float momentum;
        vector<vector<vector<float>>> prev_uw;
        vector<vector<float>> prev_ub;
        vector<float> prev_u_eps;

        MomentumGD(GNN &m, float learning_rate, float l2_param, float Momentum = 0.9) : Optimiser(m,learning_rate,l2_param), prev_uw(model.dw), prev_ub(model.db)
        {
            TensorTransforms::fill_with_zeros(prev_uw);
            TensorTransforms::fill_with_zeros(prev_ub);
            
            if(model.algo == "GIN")
            {
                prev_u_eps = model.grad_epsilon;
                TensorTransforms::fill_with_zeros(prev_u_eps);
            }
            
            momentum = Momentum;
        }

        void step()
        {
            auto uw = prev_uw;
            auto ub = prev_ub;
            auto u_grad_epsilon = prev_u_eps;

            TensorTransforms::LinearCombination_(uw,momentum,model.dw,lr);
            TensorTransforms::LinearCombination_(ub,momentum,model.db,lr);

            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(u_grad_epsilon,momentum,model.grad_epsilon,lr);

            auto temp_uw = uw;
            auto temp_ub = ub;
            auto temp_u_eps = u_grad_epsilon;

            TensorTransforms::LinearCombination_(temp_uw,1,model.dw,lr*l2_param);
            TensorTransforms::LinearCombination_(temp_ub,1,model.db,lr*l2_param);
            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(temp_u_eps,1,model.grad_epsilon,lr*l2_param);

            for(int l=0;l<model.layers.size();l++)
            {
                TensorTransforms::Add(model.layers[l]->weights,temp_uw[l],-1);
                TensorTransforms::Add(model.layers[l]->biases,temp_ub[l],-1);

                if(model.algo == "GIN")
                {
                    model.epsilon[l] -= temp_u_eps[l];
                }
            }

            prev_uw = uw;
            prev_ub = ub;
            prev_u_eps = u_grad_epsilon;
        }
};

class RMSprop : public Optimiser
{
    public:
        float beta;
        float epsilon;
        vector<vector<vector<float>>> v_w;
        vector<vector<float>> v_b;
        vector<float> v_eps;

        RMSprop(GNN &m, float learning_rate, float l2_param, float Beta = 0.5, float epsi = 1e-4) : Optimiser(m,learning_rate,l2_param), v_w(model.dw), v_b(model.db)
        {
            TensorTransforms::fill_with_zeros(v_w);
            TensorTransforms::fill_with_zeros(v_b);

            if(model.algo == "GIN")
            {
                v_eps = model.grad_epsilon;
                TensorTransforms::fill_with_zeros(v_eps);
            }

            beta = Beta;
            epsilon = epsi;
        }

        void step()
        {
            auto dw_tmp = model.dw;
            auto db_tmp = model.db;
            auto de_tmp = model.grad_epsilon;

            TensorTransforms::Matrix_square(dw_tmp);
            TensorTransforms::Matrix_square(db_tmp);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_square(de_tmp);

            TensorTransforms::LinearCombination_(v_w,beta,dw_tmp,1-beta);
            TensorTransforms::LinearCombination_(v_b,beta,db_tmp,1-beta);
            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(v_eps,beta,de_tmp,1-beta);

            auto vw_denominator = v_w;
            auto vb_denominator = v_b;
            auto ve_denominator = v_eps;

            TensorTransforms::Matrix_sqrt(vw_denominator);
            TensorTransforms::Matrix_sqrt(vb_denominator);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_sqrt(ve_denominator);

            TensorTransforms::Add(vw_denominator,epsilon);
            TensorTransforms::Add(vb_denominator,epsilon);
            if(model.algo == "GIN")
                TensorTransforms::Add(ve_denominator,epsilon);

            dw_tmp = model.dw;
            db_tmp = model.db;
            if(model.algo == "GIN")
                de_tmp = model.grad_epsilon;

            TensorTransforms::Matrix_divide(dw_tmp,vw_denominator);
            TensorTransforms::Matrix_divide(db_tmp,vb_denominator);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_divide(de_tmp,ve_denominator);

            TensorTransforms::LinearCombination_(dw_tmp,lr,model.dw,lr*l2_param);
            TensorTransforms::LinearCombination_(db_tmp,lr,model.db,lr*l2_param);
            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(de_tmp,lr,model.grad_epsilon,lr*l2_param);

            for(int l=0;l<model.layers.size();l++)
            {
                TensorTransforms::Add(model.layers[l]->weights,dw_tmp[l],-1);
                TensorTransforms::Add(model.layers[l]->biases,db_tmp[l],-1);
                if(model.algo == "GIN")
                {
                    model.epsilon[l] -= de_tmp[l];
                }
            }
        }
};

class Adam : public Optimiser
{
    public:
        float beta1;
        float beta2;
        float epsilon;
        int update_count;
        vector<vector<vector<float>>> v_w,m_w;
        vector<vector<float>> v_b,m_b;
        vector<float> v_eps,m_eps;

        Adam(GNN &m, float learning_rate, float l2_param,float bet1 = 0.9, float bet2 = 0.999, float epsi = 1e-8) : Optimiser(m,learning_rate,l2_param), v_w(model.dw), v_b(model.db), m_w(model.dw), m_b(model.db)
        {
            beta1 = bet1;
            beta2 = bet2;
            epsilon = epsi;
            update_count = 0;
            
            TensorTransforms::fill_with_zeros(v_w);
            TensorTransforms::fill_with_zeros(v_b);
            TensorTransforms::fill_with_zeros(m_w);
            TensorTransforms::fill_with_zeros(m_b);

            if(model.algo == "GIN")
            {
                v_eps = model.grad_epsilon;
                m_eps = model.grad_epsilon;
                TensorTransforms::fill_with_zeros(v_eps);
                TensorTransforms::fill_with_zeros(m_eps);
            }
        }

        void step()
        {
            update_count++;
            
            auto dw_tmp = model.dw;
            auto db_tmp = model.db;
            auto de_tmp = model.grad_epsilon;

            TensorTransforms::Matrix_square(dw_tmp);
            TensorTransforms::Matrix_square(db_tmp);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_square(de_tmp);
            
            TensorTransforms::LinearCombination_(m_w,beta1,model.dw,1-beta1);
            TensorTransforms::LinearCombination_(m_b,beta1,model.db,1-beta1);
            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(m_eps,beta1,model.grad_epsilon,1-beta1);
            
            TensorTransforms::LinearCombination_(v_w,beta2,dw_tmp,1-beta2);
            TensorTransforms::LinearCombination_(v_b,beta2,db_tmp,1-beta2);
            if(model.algo == "GIN")
                TensorTransforms::LinearCombination_(v_eps,beta2,de_tmp,1-beta2);

            auto mw_hat = m_w;
            auto mb_hat = m_b;
            auto vw_hat = v_w;
            auto vb_hat = v_b;
            auto ve_hat = v_eps;
            auto me_hat = m_eps;

            float correction1 = 1.0f / (1.0f - powf(beta1, update_count));
            float correction2 = 1.0f / (1.0f - powf(beta2, update_count));
            
            TensorTransforms::Scale(mw_hat, correction1);
            TensorTransforms::Scale(mb_hat, correction1);
            TensorTransforms::Scale(vw_hat, correction2);
            TensorTransforms::Scale(vb_hat, correction2);

            if(model.algo == "GIN")
            {
                TensorTransforms::Scale(me_hat, correction1);
                TensorTransforms::Scale(ve_hat, correction2);
            }
            
            auto vw_denominator_hat = vw_hat;
            auto vb_denominator_hat = vb_hat;
            auto ve_denominator_hat = ve_hat;

            TensorTransforms::Matrix_sqrt(vw_denominator_hat);
            TensorTransforms::Matrix_sqrt(vb_denominator_hat);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_sqrt(ve_denominator_hat);

            TensorTransforms::Add(vw_denominator_hat,epsilon);
            TensorTransforms::Add(vb_denominator_hat,epsilon);
            if(model.algo == "GIN")
                TensorTransforms::Add(ve_denominator_hat,epsilon);

            dw_tmp = mw_hat;
            db_tmp = mb_hat;
            if(model.algo == "GIN")
                de_tmp = me_hat;

            TensorTransforms::Matrix_divide(dw_tmp,vw_denominator_hat);
            TensorTransforms::Matrix_divide(db_tmp,vb_denominator_hat);
            if(model.algo == "GIN")
                TensorTransforms::Matrix_divide(de_tmp,ve_denominator_hat);

            TensorTransforms::LinearCombination_(dw_tmp,lr,model.dw,lr*l2_param);
            TensorTransforms::LinearCombination_(db_tmp,lr,model.db,lr*l2_param);

            for(int l=0;l<model.layers.size();l++)
            {
                TensorTransforms::Add(model.layers[l]->weights,dw_tmp[l],-1);
                TensorTransforms::Add(model.layers[l]->biases,db_tmp[l],-1);
                if(model.algo == "GIN")
                {
                    model.epsilon[l] -= de_tmp[l];
                }
            }
        }
};



void GNN::optimiser_step()
{
    optimiser->step();
}

void GNN::displayEpochStats()
{
    optimiser->displayEpochStats();
}

void GNN::testModel()
{
    optimiser->evaluateModel("Test");
}

void GNN::evaluateModel(string mode = "Test")
{
    optimiser->evaluateModel(mode);
}

// Function to compare two models and check their similarity
void compareModels(GNN& model1, GNN& model2) {
    // Check if models have the same architecture
    if (model1.layers.size() != model2.layers.size()) {
        cout << "Models have different architectures, cannot compare." << endl;
        return;
    }
    
    long long total_params = 0;
    long long exact_matches = 0;
    long long soft_matches = 0;
    
    // Compare each layer
    for (int i = 0; i < model1.layers.size(); i++) {
        // Check weights dimensions
        if (model1.layers[i]->weights.size() != model2.layers[i]->weights.size() ||
            model1.layers[i]->weights[0].size() != model2.layers[i]->weights[0].size()) {
            cout << "Layer " << i << " has different weight dimensions." << endl;
            return;
        }
        
        // Check biases dimensions
        if (model1.layers[i]->biases.size() != model2.layers[i]->biases.size()) {
            cout << "Layer " << i << " has different bias dimensions." << endl;
            return;
        }
        
        // Compare weights
        for (int j = 0; j < model1.layers[i]->weights.size(); j++) {
            for (int k = 0; k < model1.layers[i]->weights[j].size(); k++) {
                total_params++;
                float val1 = model1.layers[i]->weights[j][k];
                float val2 = model2.layers[i]->weights[j][k];
                
                // Exact match
                if (val1 == val2) {
                    exact_matches++;
                    soft_matches++;
                }
                // Soft match (within 1% difference)
                else {
                    float threshold;
                    if (abs(val1) < 1e-6) {
                        // For very small values, use absolute difference
                        threshold = 1e-6;
                    } else {
                        threshold = 0.001 * abs(val1);
                    }
                    
                    if (abs(val1 - val2) <= threshold) {
                        soft_matches++;
                    }
                }
            }
        }
        
        // Compare biases
        for (int j = 0; j < model1.layers[i]->biases.size(); j++) {
            total_params++;
            float val1 = model1.layers[i]->biases[j];
            float val2 = model2.layers[i]->biases[j];
            
            // Exact match
            if (val1 == val2) {
                exact_matches++;
                soft_matches++;
            }
            // Soft match (within 1% difference)
            else {
                float threshold;
                if (abs(val1) < 1e-6) {
                    threshold = 1e-6;
                } else {
                    threshold = 0.001 * abs(val1);
                }
                
                if (abs(val1 - val2) <= threshold) {
                    soft_matches++;
                }
            }
        }
    }
    
    // Calculate similarity percentages
    float hard_similarity = (float)exact_matches / total_params * 100.0;
    float soft_similarity = (float)soft_matches / total_params * 100.0;
    
    cout << "\nModel Similarity Comparison:" << endl;
    cout << "\tTotal parameters: " << total_params << endl;
    cout << "\tExact matches: " << exact_matches << " (" << fixed << setprecision(2) << hard_similarity << "%)" << endl;
    cout << "\tParameters within 0.001% range: " << soft_matches << " (" << fixed << setprecision(2) << soft_similarity << "%)" << endl;
}


int main(int argc, char* argv[]) 
{

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    ios::sync_with_stdio(0);
    cin.tie(0);
    
    string dataset = "PubMed";
    if(argc>1)
        dataset = argv[1];
        
    string dataset_dir = "Datasets/"+dataset;
    string graph_file = dataset_dir+"/edgelist.txt";
    char* graph_file_path = strdup(graph_file.c_str());

    graph G(graph_file_path);
    G.parseGraph();

    int num_nodes = G.num_nodes();
    float tot_neighs = 0;

    for(int i=0;i<num_nodes;i++)
        tot_neighs += G.getNeighbors(i).size(); 

    cout<<"Average Neighbours : "<<tot_neighs/num_nodes<<endl;
    

    string base_dir = dataset_dir;

    Dataset data(base_dir,G);
    data.printDataStats();

    vector<int> hidden_sizes = {128};
    
    vector<int> sample_sizes = {30,25};

    int input_size = data.input_feature_dim;
    int output_size = data.num_classes;

    string hidden_activation = "tanh";
    string output_activation = "softmax";
    string aggregation_type = "mean";

    
    //Creating the Model and initializing the weights.
    // GCN model(hidden_sizes,sample_sizes,input_size,output_size,hidden_activation,output_activation,data);
    
    // GraphSAGE model(hidden_sizes,sample_sizes,aggregation_type,input_size,output_size,hidden_activation,output_activation,data);

    GIN model(hidden_sizes,input_size,output_size,hidden_activation,output_activation,data);

    cout<<"\nInitializing Model Parameters with Xavier Initialization...\n\n";
    model.InitializeWeights();

    //Defining the training hyperparmeters and training the model.
    float lr = 1e-3;
    int epochs = 10;
    float weight_decay = 0.5;
    int batch_size = 128;

    Activation activ; //creating an instance of the activation class

    RMSprop optim(model,lr,weight_decay);
    // Adam optim(model,lr,weight_decay);
    model.optimiser = &optim;

    unordered_map<string, string> algo_detail_map;
    
    
    algo_detail_map["GraphSAGE"] = "GraphSAGE as per original paper with concat-aggregation";
    algo_detail_map["GCN"] = "Inductive variant of GCN with mean aggregation";
    algo_detail_map["GIN"] = "Graph Isomorphism Network (GIN) with degree based scaling";

    // Print configuration information
    cout << "================================================================================" << endl;
    cout << "                       GRAPH NEURAL NETWORK CONFIGURATION                       " << endl;
    cout << "================================================================================" << endl;
    cout << endl;
    cout << "[MODEL ARCHITECTURE]" << endl;
    cout << "Type:                     " << model.algo << endl;
    cout << "Specification:            " << algo_detail_map[model.algo] << endl;
    cout << "Hidden dimensions:        ";
    for(int i = 0; i < hidden_sizes.size(); i++) {
        cout << hidden_sizes[i];
        if(i < hidden_sizes.size() - 1) cout << ", ";
    }
    cout << endl;
    cout << "Activation function:      " << hidden_activation << endl;
    cout << endl;

    cout << "[TRAINING PARAMETERS]" << endl;
    cout << "Optimizer:                " << typeid(*model.optimiser).name() << endl;
    cout << "Learning rate:            " << lr << endl;
    cout << "Weight decay:             " << weight_decay << endl;
    cout << "Batch size:               " << batch_size << endl;
    cout << "Total epochs:             " << epochs << endl;
    cout << "Random seed:              " << 76 << endl;
    cout << endl;

    cout << "[SAMPLING CONFIGURATION]" << endl;
    cout << "Neighborhood sizes:       [";
    for(int i = 0; i < sample_sizes.size(); i++) {
        cout << sample_sizes[i];
        if(i < sample_sizes.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << endl;

    cout << "[COMPUTE RESOURCES]" << endl;
    cout << "Device:                   " << "OpenMP Parallel" << endl;
    cout << "Number of OMP Threads:    " << omp_get_max_threads() << endl;
    cout << "Environment:              " << "IITM Aqua Cluster" << endl;
    cout << endl;

    cout << "[DATASET]" << endl;
    cout << "Name:                     " << dataset << endl;
    cout << "Nodes:                    " << data.Graph.num_nodes() << endl;
    cout << "Edges:                    " << data.Graph.num_edges() << endl;
    cout << "Features:                 " << data.input_feature_dim << endl;
    cout << "Classes:                  " << data.num_classes << endl;
    // cout << "================================================================================" << endl;
    cout << endl;

    model.printArchitecture();
    cout << endl;

    cout<<"Starting Training...\n\n";
    for(int epoch = 0;epoch<epochs;epoch++)
    {

        std::chrono::high_resolution_clock::time_point start,end;
        start = chrono::high_resolution_clock::now();
        cout<<"Epoch "<<fixed<<setprecision(3)<<epoch+1<<flush;
        int processed_data_count  = 0;
        
        for(int & cur_vertex: model.data.train_indices) //looping over the train vertices
        {
            
            auto y_pred_probs = model.forward(cur_vertex,epoch); //forward pass works on one sample at time
            model.backprop(cur_vertex,y_pred_probs);
            
            processed_data_count++;

            bool done = (processed_data_count == model.data.train_indices.size());
            if(processed_data_count%batch_size == 0 || done)
            {
                // model.optimiser->step();
                model.optimiser_step();
                
                if(done)
                {
                    end = chrono::high_resolution_clock::now();
                    model.displayEpochStats();
                }
                
                model.resetGrads();
            }
        }

        chrono::duration<float> time = end-start;
        cout<<" Time : "<<fixed<<setprecision(2)<<time.count()<<"s"<<endl;
    }
 
    //Evalute the model over test data.
    cout<<endl;
    model.testModel();

    model.saveModel(dataset + std::string("_model.bin"));
    model.saveModelTxt(dataset + std::string("_model.txt"));
    
    
    // GraphSAGE model2(dataset + std::string("_model.bin"),data);
    
    // model2.printArchitecture();

    // // // Compare the original model with the loaded model
    // compareModels(model, model2);
    
    // RMSprop optim2(model2,lr,weight_decay);
    // model2.optimiser = &optim2;
    // model2.evaluateModel("Train");
    // model2.evaluateModel("Val");
    // model2.evaluateModel("Test");

    // return 0;
}

/*
Important Note :

    When using -O3 flag in compilation, loading the stored model in the same program, it interferes with the other model too.
    However without this flag it works fine.
    With -O3 flag it is still okay to create another model from scratch.

    Careful: Makesure to load a GCN model into a GCN model and so on.

*/
