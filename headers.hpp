#ifndef HEADERS_HPP
#define HEADERS_HPP

#include <iostream>
#include <fstream>
#include <Eigen>
#include <mpi.h>
#include <stdio.h>

using namespace std;

//================================================================================
// SPECIAL TYPES
//================================================================================

// Types for dense storage
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<int,    Eigen::Dynamic, 1> IntVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<int,    Eigen::Dynamic, Eigen::Dynamic> IntMatrix;

// Type for sparse storage
typedef Eigen::SparseMatrix<double> SpMatrix;

// Structure for mesh
struct Mesh
{
  int nbOfNodes;              // number of nodes
  int nbOfTri;                // number of triangles
  Matrix coords;              // coordinates for each node          (Size: nbOfNodes x 3)
  IntMatrix triNodes;         // nodes for each triangle            (Size: nbOfTri x 3)
  IntVector triNum;           // gmsh number for each triangle      (Size: nbOfTri)
  IntVector triPart;          // partition number for each triangle (Size: nbOfTri)
  
  // Infos for parallel computations
  IntVector numNodesToExch;   // number of nodal values to exchanges between the current proc and each other proc  (Size: nbTasks)
  IntMatrix nodesToExch;      // list of nodal values to exchanges between the current proc and each other proc    (Size: nbTasks x max(numNodesToExch) )
  IntVector countShareNodes;  // list counting for each node to how many MPI process its belongs (Size: nbOfNodes)
};

// Structure for problem
struct Problem
{
  SpMatrix K;    // stiffness matrix
  SpMatrix M;    // mass matrix
  SpMatrix A;    // system matrix
  Vector b;      // RHS
};

//================================================================================
// FUNCTIONS
//================================================================================

//==== Functions in 'mesh.cpp'

// Read the mesh from a gmsh-file (.msh) and store in a mesh-structure 'mesh'
void readMsh(Mesh& mesh, string fileName);

// Write a solution 'vec' in a gmsh-file (.msh)
void saveToMsh(Vector& vec, Mesh& mesh, string viewName, string fileName);

//==== Functions in 'parallel.cpp'

// Build the local numbering and list of nodes for MPI communications
void buildListsNodesMPI(Mesh& mesh);

// MPI-parallel exchange/add the interface terms
void exchangeAddInterfMPI(Vector& vec, Mesh& mesh);

//==== Functions in 'problem.cpp'

// Compute the matrices of the linear wgsystem
void buildProblem(Problem& p, Mesh& mesh, double alpha, Vector& f);

//==== Functions in 'solver.cpp'

// Solution of the system Au=b with Jacobi
void jacobi(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit);

// Solution of the system Au=b with Gradient-Conjugue
void gradient_conjugue(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit); 


#endif /* HEADERS_HPP */
