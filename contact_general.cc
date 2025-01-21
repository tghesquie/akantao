/* ----------------------------------------------------------------------- */
#include "solid_mechanics_model.hh"
#include "solver_vector_petsc.hh"
#include "sparse_matrix_petsc.hh"
#include "time_step_solver.hh"
/* ----------------------------------------------------------------------- */
#include <iostream>
#include <petscmat.h>
#include <petsctao.h>
#include <typeinfo>
/* ----------------------------------------------------------------------- */

using namespace akantu;
const Int dim = 2;

struct Context {
  DOFManagerPETSc * dof_manager;
  TimeStepSolver * tss;
  SolidMechanicsModel * model;
  Mesh * mesh;
  SolverVectorPETSc initial_positions;
  Array<Idx> constrained_dofs;

  // Constructor to initialize all members
  Context(DOFManagerPETSc * dof_manager, TimeStepSolver * tss,
          SolidMechanicsModel * model, Mesh * mesh)
      : dof_manager(dof_manager), tss(tss), model(model), mesh(mesh),
        initial_positions(*dof_manager, "initial_positions"),
        constrained_dofs() {

    // Init initial positions
    auto & nodes = mesh->getNodes();
    Array<Real> positions;
    Array<Idx> dofs;
    for (std::size_t i = 0; i < static_cast<std::size_t>(nodes.size()); ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        positions.push_back(nodes(i, j));
        dofs.push_back(i * dim + j);
      }
    }
    VecSetValues(initial_positions, dofs.size(), dofs.data(), positions.data(),
                 INSERT_VALUES);

    // Init constrained dofs
    auto & constrained_nodes =
        mesh->getElementGroup("bot").getNodeGroup().getNodes();
    for (auto & node : constrained_nodes) {
      constrained_dofs.push_back(node * dim + 1); // Adds _y dof
    }
  };
};

/* ----------------------------------------------------------------------- */

void corrector(Vec x, void * ctx) {
  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  TimeStepSolver & tss = *context->tss;

  auto & solution = aka::as_type<SolverVectorPETSc>(dof_manager.getSolution());
  if (solution.getVec() != x) {
    VecCopy(x, solution);
  }

  dof_manager.splitSolutionPerDOFs();
  tss.restoreLastConvergedStep();
  tss.corrector();
}

/* ----------------------------------------------------------------------- */

void assembleResidual(Vec x, Vec f, void * ctx) {
  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  TimeStepSolver & tss = *context->tss;

  corrector(x, ctx);
  auto & residual =
      dynamic_cast<SolverVectorPETSc &>(dof_manager.getResidual());

  tss.assembleResidual();

  const auto & blocked_dofs = dof_manager.getGlobalBlockedDOFsIndexes();
  std::vector<Real> zeros_to_set(blocked_dofs.size());
  VecSetValuesLocal(residual, blocked_dofs.size(), blocked_dofs.data(),
                    zeros_to_set.data(), INSERT_VALUES);

  VecScale(residual, -1);
  if (residual.getVec() != f) {
    VecCopy(residual, f);
  }
}

/* ----------------------------------------------------------------------- */

void assembleJacobian(Vec x, Mat J, void * ctx) {
  // Extract the context
  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  TimeStepSolver & tss = *context->tss;

  corrector(x, ctx);
  tss.assembleMatrix("J"); // callback
  auto & _J = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));
  if (_J.getMat() != J) {
    MatCopy(_J, J, SAME_NONZERO_PATTERN);
  }
}

/* ----------------------------------------------------------------------- */

PetscErrorCode FormFunctionGradient(Tao /*tao*/, Vec x, PetscReal * obj,
                                    Vec grad, void * ctx) {
  // Extract the context
  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;

  // Compute the objective and gradient
  auto & K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));
  auto & rhs = aka::as_type<SolverVectorPETSc>(dof_manager.getResidual());

  // Assemble the residual
  assembleResidual(x, rhs, ctx);
  assembleJacobian(x, K, ctx);
  SolverVectorPETSc Kx(x, aka::as_type<DOFManagerPETSc>(dof_manager), "Kx");

  Real fx;
  Real xKx;

  MatMult(K, x, Kx);
  VecWAXPY(grad, 1, Kx, rhs);
  VecDot(rhs, x, &fx);
  VecDot(x, Kx, &xKx);
  *obj = 0.5 * xKx + fx;

  return 0;
}

/* ----------------------------------------------------------------------- */

PetscErrorCode FormHessian(Tao /*tao*/, Vec x, Mat H, Mat /*pre*/, void * ctx) {
  // Assemble the Jacobian
  assembleJacobian(x, H, ctx);

  return 0;
}

/* ----------------------------------------------------------------------- */

PetscErrorCode EvaluateInequalityConstraints(Tao /*tao*/, Vec x, Vec c,
                                             void * ctx) {
  // Extract the context
  Context * context = static_cast<Context *>(ctx);

  VecSet(c, PETSC_NINFINITY);

  Array<Real> initial_pos(context->constrained_dofs.size());
  Array<Real> current_displ(context->constrained_dofs.size());

  VecGetValues(context->initial_positions, context->constrained_dofs.size(),
               context->constrained_dofs.data(), initial_pos.data());
  VecGetValues(x, context->constrained_dofs.size(),
               context->constrained_dofs.data(), current_displ.data());

  // Add initial positions to the constraint vector (c = x0)
  VecSetValues(c, context->constrained_dofs.size(),
               context->constrained_dofs.data(), initial_pos.data(),
               INSERT_VALUES);

  // Add displacement to the constraint vector (c = x0 + x)
  VecSetValues(c, context->constrained_dofs.size(),
               context->constrained_dofs.data(), current_displ.data(),
               ADD_VALUES);

  // Scale so that c >= 0 becomes c <= 0
  VecScale(c, -1.0);

  //// Set the constraint to zero (c = 0)
  // VecSet(c, 0.0);
  //// Add initial positions to c (c = x0)
  // VecCopy(initial_positions, c);
  //// Add displacement to c (c = x0 + x)
  // VecAXPY(c, 1.0, x);
  //// Scale so that c >= 0 becomes c <= 0
  // VecScale(c, -1.0);

  VecView(c, PETSC_VIEWER_STDOUT_WORLD); // OPTIONAL: Print the vector

  return 0;
}

/* ----------------------------------------------------------------------- */

PetscErrorCode EvaluateInequalityJacobian(Tao /*tao*/, Vec x, Mat J,
                                          Mat /*pre*/, void * ctx) {
  PetscInt size;
  VecGetSize(x, &size);

  MatZeroEntries(J); // Clear the matrix before setting it to identity
  for (PetscInt i = 0; i < size; ++i) {
    MatSetValue(J, i, i, 1.0, INSERT_VALUES);
  }
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  MatScale(J, -1.0);

  return 0;
}

/* ----------------------------------------------------------------------- */

void solve(Tao & tao, Context * ctx) {

  DOFManagerPETSc & dof_manager = *ctx->dof_manager;
  TimeStepSolver & tss = *ctx->tss;

  tss.beforeSolveStep();
  dof_manager.updateGlobalBlockedDofs();

  tss.assembleMatrix("J");
  auto & x = dynamic_cast<SolverVectorPETSc &>(dof_manager.getSolution());
  x.zero();

  auto & rhs = aka::as_type<SolverVectorPETSc>(dof_manager.getResidual());
  rhs.zero();

  auto & J = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));

  TaoSetSolution(tao, x);
  TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, ctx);
  TaoSetHessian(tao, J, J, FormHessian, ctx);

  // Set the constraint routine and its Jacobian
  SolverVectorPETSc c(dof_manager, "c");
  c.resize();
  SparseMatrixPETSc Ji(dof_manager, _mt_not_defined, "Ji");
  Ji.resize();
  TaoSetInequalityConstraintsRoutine(tao, c, EvaluateInequalityConstraints,
                                     ctx);
  TaoSetJacobianInequalityRoutine(tao, Ji, Ji, EvaluateInequalityJacobian, ctx);

  tss.predictor();

  TaoSolve(tao);

  // Declare the variables
  TaoConvergedReason reason;
  PetscInt n_iter;

  TaoGetConvergedReason(tao, &reason);
  TaoGetIterationNumber(tao, &n_iter);

  dof_manager.splitSolutionPerDOFs();
  tss.restoreLastConvergedStep();
  tss.corrector();

  bool converged = reason > 0;
  tss.afterSolveStep(converged);

  if (!converged) {
    PetscReal atol;
    PetscReal rtol;
    PetscReal ttol;
    PetscInt maxit;

    TaoGetTolerances(tao, &atol, &rtol, &ttol);
    TaoGetMaximumIterations(tao, &maxit);

    std::cerr << "Tao solver did not converge after " << n_iter
              << " iterations. Reason: " << reason << std::endl;
  }
}

/* ----------------------------------------------------------------------- */

int main(int argc, char * argv[]) {

  initialize("material.dat", argc, argv);

  Mesh mesh(dim);
  mesh.read("square_L0.01_P1_dy0.002.msh");

  SolidMechanicsModel model(mesh);

  model.initDOFManager("petsc");
  DOFManagerPETSc & dof_manager =
      aka::as_type<DOFManagerPETSc>(model.getDOFManager());
  model.initFull(_analysis_method = _static);

  model.setBaseName("contact");
  model.addDumpFieldVector("displacement");
  model.addDumpField("external_force");
  model.addDumpField("internal_force");
  model.addDumpField("stress");
  model.dump();

  // Boundary conditions
  model.applyBC(BC::Dirichlet::FixedValue(-4e-3, _y), "top");
  model.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "left");

  // Initialize the context
  TimeStepSolver & tss = dof_manager.getTimeStepSolver("static");
  Context ctx(&dof_manager, &tss, &model, &mesh);

  // TAO solver
  auto && mpi_comm = dof_manager.getMPIComm();
  Tao tao;
  TaoCreate(mpi_comm, &tao);

  TaoSetType(tao, TAOALMM);
  TaoSetFromOptions(tao);

  // Set solver tolerances and maximum iterations
  TaoSetTolerances(tao, 1e-4, 1e-4, 1e-4);
  TaoSetMaximumIterations(tao, 5e3);

  // Enable debugging output
  PetscOptionsSetValue(NULL, "-tao_monitor", NULL);
  PetscOptionsSetValue(NULL, "-tao_view", NULL);

  // Solve the problem
  solve(tao, &ctx);

  // Dump the results
  model.dump();

  // Destroy the TAO solver
  TaoDestroy(&tao);
}