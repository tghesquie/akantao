/* -------------------------------------------------------------------------- */
#include "non_linear_solver.hh"
#include "solver_vector_petsc.hh"
/* -------------------------------------------------------------------------- */
#include <petsctao.h>
/* -------------------------------------------------------------------------- */

#ifndef TAO_SOLVER_HH
#define TAO_SOLVER_HH

namespace akantu {
class DOFManagerPETSc;
class SolverVectorPETSc;
} // namespace akantu

namespace akantu {

class TAOSolver : public NonLinearSolver {
  /* ------------------------------------------------------------------------ */
  /* Constructors/Destructors                                                 */
  /* ------------------------------------------------------------------------ */
public:
  TAOSolver(DOFManagerPETSc & dof_manager,
            const ModelSolverOptions & solver_options,
            const ID & id = "tao_solver");

  ~TAOSolver() override;

  /* ------------------------------------------------------------------------ */
  /* Methods                                                                  */
  /* ------------------------------------------------------------------------ */
};

} // namespace akantu

#endif // TAO_SOLVER_HH