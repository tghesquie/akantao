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
  PetscInt n, ni;
  Vec ci;
  Mat Ai;

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
      // constrained_dofs.push_back(node * dim);     // Adds _x dof
      constrained_dofs.push_back(node * dim + 1); // Adds _y dof
      std::cout << "Constrained node: " << node << std::endl;
      std::cout << "Constrained DOF: " << node * dim + 1 << std::endl;
    }

    // Set the number of DOFs and constraints
    n = dof_manager->getPureLocalSystemSize();
    ni = constrained_dofs.size();
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

PetscErrorCode FormHessian(Tao /*tao*/, Vec x, Mat H, Mat /*Hpre*/,
                           void * ctx) {
  // Assemble the Jacobian
  assembleJacobian(x, H, ctx);

  return 0;
}

/* ----------------------------------------------------------------------- */

PetscErrorCode EvaluateInequalityConstraints(Tao /*tao*/, Vec x, Vec ci,
                                             void * ctx) {

  PetscFunctionBegin;

  // Extract the context
  Context * context = static_cast<Context *>(ctx);
  PetscInt ni = context->ni;

  VecSet(ci, 0.0);

  Array<Real> initial_pos(ni);
  Array<Real> current_displ(ni);

  VecGetValues(context->initial_positions, ni, context->constrained_dofs.data(),
               initial_pos.data());
  VecGetValues(x, ni, context->constrained_dofs.data(), current_displ.data());

  // Print the current displacements and the initial positions
  std::cout << "Initial positions: ";
  for (PetscInt i = 0; i < ni; ++i) {
    std::cout << initial_pos[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Current displacements: ";
  for (PetscInt i = 0; i < ni; ++i) {
    std::cout << current_displ[i] << " ";
  }
  std::cout << std::endl;

  for (PetscInt i = 0; i < ni; ++i) {
    VecSetValue(ci, i, initial_pos[i] + current_displ[i], INSERT_VALUES);
  }

  // VecScale(c, -1.0);

  VecAssemblyBegin(ci);
  VecAssemblyEnd(ci);

  VecView(ci, PETSC_VIEWER_STDOUT_WORLD); // OPTIONAL: Print the vector

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----------------------------------------------------------------------- */

PetscErrorCode EvaluateInequalityJacobian(Tao /*tao*/, Vec x, Mat Ai,
                                          Mat /*Apre*/, void * ctx) {
  PetscFunctionBegin;

  Context * context = static_cast<Context *>(ctx);
  PetscInt ni = context->ni;

  MatZeroEntries(Ai);

  for (PetscInt i = 0; i < ni; ++i) {
    PetscInt dof = context->constrained_dofs[i];
    MatSetValue(Ai, i, dof, 1.0, ADD_VALUES);
  }

  MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY);

  MatView(Ai, PETSC_VIEWER_STDOUT_WORLD); // OPTIONAL: Print the matrix

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----------------------------------------------------------------------- */

void solve(Tao & tao, Context * ctx) {

  DOFManagerPETSc & dof_manager = *ctx->dof_manager;
  TimeStepSolver & tss = *ctx->tss;

  tss.beforeSolveStep();
  dof_manager.updateGlobalBlockedDofs();

  tss.assembleMatrix("J");
  auto & x = dynamic_cast<SolverVectorPETSc &>(dof_manager.getSolution());
  // x.zero();

  auto & rhs = aka::as_type<SolverVectorPETSc>(dof_manager.getResidual());
  rhs.zero();

  auto & K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("K"));

  TaoSetSolution(tao, x);
  TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, ctx);
  TaoSetHessian(tao, K, K, FormHessian, ctx);

  auto && mpi_comm = dof_manager.getMPIComm();
  auto n = ctx->n;
  auto ni = ctx->ni;
  std::cout << "Number of DOFs: " << n << std::endl;
  std::cout << "Number of inequality constraints: " << ni << std::endl;

  VecCreate(mpi_comm, &ctx->ci);
  VecSetSizes(ctx->ci, ni, ni);
  VecSetFromOptions(ctx->ci);
  VecSetUp(ctx->ci);

  MatCreate(mpi_comm, &ctx->Ai);
  MatSetSizes(ctx->Ai, ni, n, ni, n);
  MatSetFromOptions(ctx->Ai);
  MatSetUp(ctx->Ai);

  TaoSetInequalityConstraintsRoutine(tao, ctx->ci,
                                     EvaluateInequalityConstraints, ctx);
  TaoSetJacobianInequalityRoutine(tao, ctx->Ai, ctx->Ai,
                                  EvaluateInequalityJacobian, ctx);

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
  model.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "top");

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
  TaoSetTolerances(tao, 1e-1, 0, 0);
  // TaoSetMaximumIterations(tao, 1e3);

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