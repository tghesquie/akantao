/* ----------------------------------------------------------------------- */
#include "solid_mechanics_model.hh"
#include "solver_vector_petsc.hh"
#include "sparse_matrix_petsc.hh"
#include "time_step_solver.hh"
/* ----------------------------------------------------------------------- */
#include <iostream>
#include <petscmat.h>
#include <petsctao.h>
/* ----------------------------------------------------------------------- */

using namespace akantu;
const Int dim = 2;

struct Context {
  DOFManagerPETSc * dof_manager;
  TimeStepSolver * tss;
  SolidMechanicsModel * model;
  Mesh * mesh;
  SolverVectorPETSc X0;

  // Constructor to initialize all members
  Context(DOFManagerPETSc * dof_manager, TimeStepSolver * tss,
          SolidMechanicsModel * model, Mesh * mesh)
      : dof_manager(dof_manager), tss(tss), model(model), mesh(mesh),
        X0(*dof_manager, "X0") {}
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

void setBounds(Tao & tao, void * ctx) {
  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  Mesh & mesh = *context->mesh;

  SolverVectorPETSc lower_bound(dof_manager, "lower_bound");
  lower_bound.resize();
  lower_bound.set(PETSC_NINFINITY);

  // Set to 0 the correct dofs in lower_bound
  Array<Idx> & constrained_nodes =
      mesh.getElementGroup("bot").getNodeGroup().getNodes();
  Array<Idx> constrained_dofs;
  for (auto & node : constrained_nodes) {
    constrained_dofs.push_back(node * dim + 1); // Adds _y dof
  }

  Array<Real> values(constrained_dofs.size(), 0.0);
  auto to_add = values.data();
  VecSetValues(lower_bound, constrained_dofs.size(), constrained_dofs.data(),
               to_add, INSERT_VALUES);

  SolverVectorPETSc upper_bound(dof_manager, "upper_bound");
  upper_bound.resize();
  upper_bound.set(PETSC_INFINITY);

  TaoSetVariableBounds(tao, lower_bound, upper_bound);
}

/* ----------------------------------------------------------------------- */

void solve(Tao & tao, void * ctx) {

  Context * context = static_cast<Context *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  TimeStepSolver & tss = *context->tss;

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
  mesh.read("square_L0.01_P1.msh");

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
  model.applyBC(BC::Dirichlet::FixedValue(-1e-3, _y), "top");
  model.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "left");

  // Initialize the context
  TimeStepSolver & tss = dof_manager.getTimeStepSolver("static");
  Context ctx(&dof_manager, &tss, &model, &mesh);
  // ctx.X0 = dynamic_cast<SolverVectorPETSc &>(mesh.getNodes());

  // TAO solver
  auto && mpi_comm = dof_manager.getMPIComm();
  Tao tao;
  TaoCreate(mpi_comm, &tao);

  TaoSetType(tao, TAOBQPIP);
  TaoSetFromOptions(tao);

  // Set Variable Bounds
  setBounds(tao, &ctx);

  // Set solver tolerances and maximum iterations
  TaoSetTolerances(tao, 1e-6, 1e-6, 1e-6);
  TaoSetMaximumIterations(tao, 500);

  // Enable debugging output
  PetscOptionsSetValue(NULL, "-tao_monitor", NULL);
  PetscOptionsSetValue(NULL, "-tao_view", NULL);

  // Solve the problem
  solve(tao, &ctx);

  model.dump();

  TaoDestroy(&tao);
}