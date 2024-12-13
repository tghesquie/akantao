#include <iostream>
#include "solid_mechanics_model.hh"
#include "tao_solver.hh"

using namespace akantu;

const Int dim = 2;

int main(int argc, char* argv[]) {
    initialize("material.dat", argc, argv);

    Mesh mesh(dim);
    mesh.read("circle_R0.01_P2.msh");

    SolidMechanicsModel model(mesh);
    std::cout << "Model created" << std::endl;    

    

    return 0;
}


/*
model.initDOFManager("petsc");

ModelSolverOptions options;
options.non_linear_solver_type = NonLinearSolverType::_petsc_snes;
options.sparse_solver_type = SparseSolverType::_petsc;

model.initFull(_analysis_method = _static);
*/