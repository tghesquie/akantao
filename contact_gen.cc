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

/* ----------------------------------------------------------------------- */
typedef struct {
  PetscInt n, ni;
  Vec ci;
  Mat Ai;
  DOFManagerPETSc * dof_manager;
  TimeStepSolver * tss;
  SolidMechanicsModel * model;
  Mesh * mesh;
  Vec initial_positions;
  Array<Idx> constrained_dofs;
} Ctx;

/* ----------------------------------------------------------------------- */

void corrector(Vec, Ctx *);
void assembleResidual(Vec, Vec, Ctx *);
void assembleJacobian(Vec, Mat, Ctx *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormInequalityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormInequalityJacobian(Tao, Vec, Mat, Mat, void *);
void initializeProblem(Ctx *);
void destroyProblem(Ctx *);
void solve(Tao &, Ctx *);

/* ----------------------------------------------------------------------- */

void corrector(Vec x, Ctx * ctx) {
  auto & dof_manager = *ctx->dof_manager;
  auto & tss = *ctx->tss;

  auto & solution =
      dynamic_cast<SolverVectorPETSc &>(dof_manager.getSolution());
  if (solution.getVec() != x) {
    VecCopy(x, solution);
  }

  dof_manager.splitSolutionPerDOFs();
  tss.restoreLastConvergedStep();
  tss.corrector();
};

/* ----------------------------------------------------------------------- */

void assembleResidual(Vec x, Vec f, Ctx * ctx) {
  auto & dof_manager = *ctx->dof_manager;
  auto & tss = *ctx->tss;

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
};

/* ----------------------------------------------------------------------- */

void assembleJacobian(Vec x, Mat K, Ctx * ctx) {
  auto & dof_manager = *ctx->dof_manager;
  auto & tss = *ctx->tss;

  corrector(x, ctx);
  tss.assembleMatrix("J");
  auto & _K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));
  if (_K.getMat() != K) {
    MatCopy(_K, K, SAME_NONZERO_PATTERN);
  }
};

/* ----------------------------------------------------------------------- */

// PetscErrorCode FormFunctionGradient(Tao /*tao*/, Vec x, PetscReal * obj,
//                                     Vec grad, void * ctx) {
//   Ctx * context = static_cast<Ctx *>(ctx);
//   auto & dof_manager = *context->dof_manager;
//
//   auto & K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));
//   auto & rhs = aka::as_type<SolverVectorPETSc>(dof_manager.getResidual());
//
//   assembleResidual(x, rhs, context);
//   assembleJacobian(x, K, context);
//   SolverVectorPETSc Kx(x, aka::as_type<DOFManagerPETSc>(dof_manager), "Kx");
//
//   Real fx;
//   Real xKx;
//
//   MatMult(K, x, Kx);
//   VecWAXPY(grad, 1, Kx, rhs);
//   VecDot(rhs, x, &fx);
//   VecDot(x, Kx, &xKx);
//   *obj = 0.5 * xKx + fx;
//
//   return 0;
// };

/* ----------------------------------------------------------------------- */

PetscErrorCode FormFunctionGradient(Tao /*tao*/, Vec x, PetscReal * obj,
                                    Vec grad, void * ctx) {
  // Extract the context
  Ctx * context = static_cast<Ctx *>(ctx);
  DOFManagerPETSc & dof_manager = *context->dof_manager;
  SolidMechanicsModel & model = *context->model;

  // Compute the objective and gradient
  auto & K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));
  auto & f = model.getExternalForce();
  SolverVectorPETSc rhs(dof_manager, "rhs");
  rhs.resize();
  // Fill the rhs vector
  for (auto i = 0; i < f.size(); i++) {
    VecSetValue(rhs, 2 * i, f(i, 0), INSERT_VALUES);     // First column
    VecSetValue(rhs, 2 * i + 1, f(i, 1), INSERT_VALUES); // Second column
  }
  VecAssemblyBegin(rhs);
  VecAssemblyEnd(rhs);

  assembleJacobian(x, K, context);
  SolverVectorPETSc Kx(x, aka::as_type<DOFManagerPETSc>(dof_manager), "Kx");

  Real fx;
  Real xKx;

  MatMult(K, x, Kx);
  VecWAXPY(grad, -1, rhs, Kx);
  VecDot(rhs, x, &fx);
  VecDot(x, Kx, &xKx);
  *obj = 0.5 * xKx - fx;

  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(rhs, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(K, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(Kx, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(grad, PETSC_VIEWER_STDOUT_WORLD);
  // std::cout << "fx: " << fx << std::endl;
  // std::cout << "xKx: " << xKx << std::endl;
  // std::cout << "obj: " << *obj << std::endl;
  // std::cout << "--------------------------------" << std::endl;

  return 0;
}

/* ----------------------------------------------------------------------- */

PetscErrorCode FormHessian(Tao /*tao*/, Vec x, Mat H, Mat /*Hp*/, void * ctx) {
  Ctx * context = static_cast<Ctx *>(ctx);
  assembleJacobian(x, H, context);

  return 0;
};

/* ----------------------------------------------------------------------- */

PetscErrorCode FormInequalityConstraints(Tao /*tao*/, Vec x, Vec ci,
                                         void * ctx) {
  Ctx * context = static_cast<Ctx *>(ctx);
  auto ni = context->ni;
  auto & constrained_dofs = context->constrained_dofs;
  auto & initial_positions = context->initial_positions;

  VecSet(ci, 0.0);

  Array<Real> initial_pos(ni);
  Array<Real> current_displ(ni);

  VecGetValues(initial_positions, ni, constrained_dofs.data(),
               initial_pos.data());
  VecGetValues(x, ni, constrained_dofs.data(), current_displ.data());
  for (PetscInt i = 0; i < ni; ++i) {
    VecSetValue(ci, i, initial_pos[i] + current_displ[i], INSERT_VALUES);
  }

  // VecScale(ci, -1);

  VecAssemblyBegin(ci);
  VecAssemblyEnd(ci);

  // VecView(ci, PETSC_VIEWER_STDOUT_WORLD); // OPTIONAL: Print the vector

  return 0;
};

/* ----------------------------------------------------------------------- */

PetscErrorCode FormInequalityJacobian(Tao /*tao*/, Vec /*x*/, Mat Ai,
                                      Mat /*Aip*/, void * ctx) {
  Ctx * context = static_cast<Ctx *>(ctx);
  auto ni = context->ni;
  auto & constrained_dofs = context->constrained_dofs;

  MatZeroEntries(Ai);

  for (PetscInt i = 0; i < ni; ++i) {
    PetscInt dof = constrained_dofs[i];
    MatSetValue(Ai, i, dof, 1.0, ADD_VALUES);
  }

  MatAssemblyBegin(Ai, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ai, MAT_FINAL_ASSEMBLY);

  // MatView(Ai, PETSC_VIEWER_STDOUT_WORLD); // OPTIONAL: Print the matrix

  return 0;
};

/* ----------------------------------------------------------------------- */

void initializeProblem(Ctx * ctx) {
  auto & dof_manager = *ctx->dof_manager;
  auto & mesh = *ctx->mesh;

  // Init constrained dofs
  auto & constrained_nodes =
      mesh.getElementGroup("bot").getNodeGroup().getNodes();
  for (auto & node : constrained_nodes) {
    ctx->constrained_dofs.push_back(node * dim + 1); // Adds _y dof
    // std::cout << "Constrained node: " << node << std::endl;
    // std::cout << "Constrained DOF: " << node * dim + 1 << std::endl;
  }

  // Set the number of DOFs and constraints
  ctx->n = dof_manager.getPureLocalSystemSize();
  ctx->ni = ctx->constrained_dofs.size();

  // Create the vectors and matrices
  auto && mpi_comm = dof_manager.getMPIComm();
  VecCreate(mpi_comm, &ctx->initial_positions);
  VecSetSizes(ctx->initial_positions, ctx->n, ctx->n);
  VecSetFromOptions(ctx->initial_positions);
  VecSetUp(ctx->initial_positions);

  VecCreate(mpi_comm, &ctx->ci);
  VecSetSizes(ctx->ci, ctx->ni, ctx->ni);
  VecSetFromOptions(ctx->ci);
  VecSetUp(ctx->ci);

  MatCreate(mpi_comm, &ctx->Ai);
  MatSetSizes(ctx->Ai, ctx->ni, ctx->n, ctx->ni, ctx->n);
  MatSetFromOptions(ctx->Ai);
  MatSetUp(ctx->Ai);

  // Init initial positions
  auto & nodes = mesh.getNodes();
  Array<Real> positions;
  Array<Idx> dofs;
  for (std::size_t i = 0; i < static_cast<std::size_t>(nodes.size()); ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
      positions.push_back(nodes(i, j));
      dofs.push_back(i * dim + j);
    }
  }
  VecSetValues(ctx->initial_positions, dofs.size(), dofs.data(),
               positions.data(), INSERT_VALUES);
};

/* ----------------------------------------------------------------------- */

void destroyProblem(Ctx * ctx) {
  VecDestroy(&ctx->ci);
  MatDestroy(&ctx->Ai);
  VecDestroy(&ctx->initial_positions);
};

/* ----------------------------------------------------------------------- */

void solve(Tao & tao, Ctx * ctx) {
  auto & dof_manager = *ctx->dof_manager;
  auto & tss = *ctx->tss;

  tss.beforeSolveStep();
  dof_manager.updateGlobalBlockedDofs();

  tss.assembleMatrix("J");
  auto & x = dynamic_cast<SolverVectorPETSc &>(dof_manager.getSolution());
  // x.zero();
  // auto & rhs = aka::as_type<SolverVectorPETSc>(dof_manager.getResidual());
  // rhs.zero();
  auto & K = aka::as_type<SparseMatrixPETSc>(dof_manager.getMatrix("J"));

  TaoSetSolution(tao, x);
  TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, ctx);
  TaoSetHessian(tao, K, K, FormHessian, ctx);
  TaoSetTolerances(tao, 1e-12, 1e-12, 1e-12);
  TaoSetConstraintTolerances(tao, 1e-12, 0);
  TaoSetMaximumIterations(tao, 1e4);
  TaoSetFromOptions(tao);

  TaoSetInequalityConstraintsRoutine(tao, ctx->ci, FormInequalityConstraints,
                                     ctx);
  TaoSetJacobianInequalityRoutine(tao, ctx->Ai, ctx->Ai, FormInequalityJacobian,
                                  ctx);

  tss.predictor();

  TaoSolve(tao);

  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

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
};

/* ----------------------------------------------------------------------- */

int main(int argc, char * argv[]) {
  Tao tao;
  Ctx ctx;

  initialize("material.dat", argc, argv);

  Mesh mesh(dim);
  mesh.read("square_L0.01_P1_dy0.002_lc0.002.msh");

  SolidMechanicsModel model(mesh);

  model.initDOFManager("petsc");
  DOFManagerPETSc & dof_manager =
      aka::as_type<DOFManagerPETSc>(model.getDOFManager());
  model.initFull(_analysis_method = _static);
  TimeStepSolver & tss = dof_manager.getTimeStepSolver("static");

  model.setBaseName("contact");
  model.addDumpFieldVector("displacement");
  model.addDumpField("external_force");
  model.addDumpField("internal_force");
  model.addDumpField("stress");
  model.dump();

  // Boundary conditions
  Vector<Real> traction(dim);
  traction.zero();
  traction(_y) = -2e2;
  // model.applyBC(BC::Dirichlet::FixedValue(2e-3, _y), "top");
  model.applyBC(BC::Neumann::FromTraction(traction), "top");
  model.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "left");
  // model.applyBC(BC::Dirichlet::FixedValue(0.0, _y), "bot_left");
  //  model.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "left");

  // Initialize the context
  ctx.dof_manager = &dof_manager;
  ctx.tss = &tss;
  ctx.model = &model;
  ctx.mesh = &mesh;

  initializeProblem(&ctx);

  auto && mpi_comm = dof_manager.getMPIComm();
  TaoCreate(mpi_comm, &tao);
  TaoSetType(tao, TAOALMM);

  PetscOptionsSetValue(NULL, "-tao_monitor", NULL);
  PetscOptionsSetValue(NULL, "-tao_smonitor", NULL);
  PetscOptionsSetValue(NULL, "-tao_view", NULL);
  // PetscOptionsSetValue(NULL, "-tao_almm_subsolver_tao_monitor", NULL);
  // PetscOptionsSetValue(NULL, "-tao_almm_subsolver_tao_ls_monitor", NULL);

  // PetscOptionsSetValue(NULL, "-tao_almm_subsolver_tao_type", "bqnls");
  // PetscOptionsSetValue(NULL, "-tao_almm_subsolver_ksp_type", "preonly");
  // PetscOptionsSetValue(NULL, "-tao_almm_subsolver_pc_type", "lu");

  // Optionally set penalty parameters
  // PetscOptionsSetValue(NULL, "-tao_almm_penalty_initial", "10.0");
  // PetscOptionsSetValue(NULL, "-tao_almm_penalty_multiplier", "2.0");
  solve(tao, &ctx);

  model.dump();

  destroyProblem(&ctx);
  TaoDestroy(&tao);
};