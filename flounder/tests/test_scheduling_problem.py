import numpy as np
import unittest
import os

from flounder.flounder import SchedulingProblem, SchedulingProblemType, \
    TaskLoadType, TaskRelationType, MachineLoadType


class NonuniformTaskCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        p = [0, 1, 2]
        target_dir = os.path.dirname(os.path.realpath(__file__))

        cls.t_sample = np.loadtxt(target_dir + "/data/t_sample.csv", delimiter=',')
        cls.delta_funs = np.loadtxt(target_dir + "/data/random_5_fun.csv", delimiter=',')
        cls.fits = np.loadtxt(target_dir + "/data/fit_5_fun.csv", delimiter=',')
        cls.scheduling_problem_type = SchedulingProblemType(TaskLoadType.NONUNIFORM,
                                                            TaskRelationType.UNRELATED
                                                            )

        cls.scheduling_problem = SchedulingProblem(cls.scheduling_problem_type,
                                                   N=len(p),
                                                   W=10,
                                                   delta_sample=cls.delta_funs[p, :],
                                                   t_sample=cls.t_sample)
        cls.scheduling_problem.compute_schedule()
        cls.scheduling_problem.exact_compute_schedule()
        cls.scheduling_problem.WCPT_compute_schedule()

    def test_exact_schedule(self):
        self.assertEqual(self.scheduling_problem.exact_schedule,
                         [(1.8181818181818379, 0), (0.8080808080808168, 0), (0.0, 0)],
                         "Incorrect Exact Schedule")

    def test_exact_makespan(self):
        self.assertEqual(self.scheduling_problem.exact_objective,
                         2.929292929292961,
                         "Incorrect Exact Makespan")

    def test_schedule(self):
        self.assertEqual(self.scheduling_problem.schedule,
                         [(3.2455924754830625, 0), (1.559905452082737, 0), (0.0, 0)],
                         "Incorrect Schedule")

    def test_makespan(self):
        self.assertEqual(self.scheduling_problem.objective,
                         5.030984073673951,
                         "Incorrect Makespan")

    def test_WCPT_schedule(self):
        self.assertEqual(self.scheduling_problem.WCPT_schedule,
                         [(6.706743685520727, 0), (4.402587281021963, 0), (0.0, 0)],
                         "Incorrect WCPT Schedule")

    def test_WCPT_makespan(self):
        self.assertEqual(self.scheduling_problem.WCPT_objective,
                         8.492135283711617,
                         "Incorrect WCPT Makespan")


class PrecedenceTaskCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = [2, 1, 0]
        A = np.zeros((len(p), len(p)))
        A[1, 0] = 1
        A[2, 0] = 1

        target_dir = os.path.dirname(os.path.realpath(__file__))

        cls.t_sample = np.loadtxt(target_dir + "/data/t_sample.csv", delimiter=',')
        cls.delta_funs = np.loadtxt(target_dir + "/data/random_5_fun.csv", delimiter=',')
        cls.fits = np.loadtxt(target_dir + "/data/fit_5_fun.csv", delimiter=',')
        cls.scheduling_problem_type = SchedulingProblemType(TaskLoadType.NONUNIFORM,
                                                            TaskRelationType.PRECEDENCE
                                                            )
        cls.scheduling_problem = SchedulingProblem(cls.scheduling_problem_type,
                                                   N=len(p),
                                                   W=10,
                                                   A=A,
                                                   delta_sample=cls.delta_funs[p, :],
                                                   t_sample=cls.t_sample)
        cls.scheduling_problem.compute_schedule()
        cls.scheduling_problem.exact_compute_schedule()
        cls.scheduling_problem.WCPT_compute_schedule()
        # cls.scheduling_problem.

    def test_exact_schedule(self):
        self.assertEqual(self.scheduling_problem.exact_schedule,
                         [(1.7171717171717358, 0), (0.0, 0), (0.8080808080808168, 0)],
                         "Incorrect Exact Schedule")

    def test_exact_makespan(self):
        self.assertEqual(self.scheduling_problem.exact_objective,
                         4.040404040404084,
                         "Incorrect Exact Makespan")

    def test_schedule(self):
        self.assertEqual(self.scheduling_problem.schedule,
                         [(3.274096863657885, 0), (0.0, 0), (1.4887052654669963, 0)],
                         "Incorrect Schedule")

    def test_makespan(self):
        self.assertEqual(self.scheduling_problem.objective,
                         6.622045283981137,
                         "Incorrect Makespan")

    def test_WCPT_schedule(self):
        self.assertEqual(self.scheduling_problem.WCPT_schedule,
                         [(4.089548002689651, 0), (1.7853915981908863, 0), (0.0, 0)],
                         "Incorrect WCPT Schedule")

    def test_WCPT_makespan(self):
        self.assertEqual(self.scheduling_problem.WCPT_objective,
                         8.492135283711615,
                         "Incorrect WCPT Makespan")


class MultimachineTaskCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        p = [0, 1, 2, 3, 4]
        m = 2

        target_dir = os.path.dirname(os.path.realpath(__file__))

        cls.t_sample = np.loadtxt(target_dir + "/data/t_sample.csv", delimiter=',')
        cls.delta_funs = np.loadtxt(target_dir + "/data/random_5_fun.csv", delimiter=',')
        cls.fits = np.loadtxt(target_dir + "/data/fit_5_fun.csv", delimiter=',')
        cls.scheduling_problem_type = SchedulingProblemType(TaskLoadType.NONUNIFORM,
                                                            TaskRelationType.UNRELATED,
                                                            machine_load_type=MachineLoadType.UNIFORM)

        cls.scheduling_problem = SchedulingProblem(cls.scheduling_problem_type,
                                                   N=len(p),
                                                   W=10,
                                                   M=m,
                                                   delta_sample=cls.delta_funs[p, :],
                                                   t_sample=cls.t_sample)
        cls.scheduling_problem.compute_schedule()
        cls.scheduling_problem.exact_compute_schedule()
        cls.scheduling_problem.WCPT_compute_schedule()
        # cls.scheduling_problem.

    def test_exact_schedule(self):
        self.assertEqual(self.scheduling_problem.exact_schedule,
                         [(1.5151515151515316, 1), (0.6060606060606126, 1), (0.6060606060606126, 0), (0.0, 0),
                          (0.0, 1)],
                         "Incorrect Exact Schedule")

    def test_exact_makespan(self):
        self.assertEqual(self.scheduling_problem.exact_objective,
                         2.5252525252525526,
                         "Incorrect Exact Makespan")

    def test_schedule(self):
        self.assertEqual(self.scheduling_problem.schedule,
                         [(2.467813083200846, 0), (2.454544405638826, 1), (0.0, 1), (0.8930288668682778, 0), (0.0, 0)],
                         "Incorrect Schedule")

    def test_makespan(self):
        self.assertEqual(self.scheduling_problem.objective,
                         4.253204681391735,
                         "Incorrect Makespan")

    def test_WCPT_schedule(self):
        self.assertEqual(self.scheduling_problem.WCPT_schedule,
                         [(2.304156404498764, 0), (0.0, 0), (0.0, 1), (4.402587281021963, 1), (4.089548002689651, 0)],
                         "Incorrect WCPT Schedule")

    def test_WCPT_makespan(self):
        self.assertEqual(self.scheduling_problem.WCPT_objective,
                         9.435998743015091,
                         "Incorrect WCPT Makespan")


class NonuniformMultimachineTaskCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        N = 4
        s = [0, 1, 2]

        target_dir = os.path.dirname(os.path.realpath(__file__))

        cls.t_sample = np.loadtxt(target_dir + "/data/t_sample.csv", delimiter=',')
        cls.delta_funs = np.loadtxt(target_dir + "/data/random_5_fun.csv", delimiter=',')
        cls.fits = np.loadtxt(target_dir + "/data/fit_5_fun.csv", delimiter=',')

        new_delta_funs = np.zeros((N, cls.delta_funs.shape[0], cls.delta_funs.shape[1]))
        for i in range(N):
            new_delta_funs[:, i, :] = cls.delta_funs[i, :]
        cls.delta_funs = new_delta_funs

        cls.scheduling_problem_type = SchedulingProblemType(TaskLoadType.NONUNIFORM,
                                                            TaskRelationType.UNRELATED,
                                                            machine_load_type=MachineLoadType.NONUNIFORM)

        cls.scheduling_problem_0 = SchedulingProblem(cls.scheduling_problem_type,
                                                     N=N,
                                                     W=10,
                                                     M=len(s),
                                                     delta_sample=cls.delta_funs[:, s, :],
                                                     t_sample=cls.t_sample,
                                                     het_method_hyperplane=0)

        cls.scheduling_problem_1 = SchedulingProblem(cls.scheduling_problem_type,
                                                     N=N,
                                                     W=10,
                                                     M=len(s),
                                                     delta_sample=cls.delta_funs[:, s, :],
                                                     t_sample=cls.t_sample,
                                                     het_method_hyperplane=1)

        cls.scheduling_problem_2 = SchedulingProblem(cls.scheduling_problem_type,
                                                     N=N,
                                                     W=10,
                                                     M=len(s),
                                                     delta_sample=cls.delta_funs[:, s, :],
                                                     t_sample=cls.t_sample,
                                                     het_method_hyperplane=2)

        cls.scheduling_problem_0.compute_schedule()
        cls.scheduling_problem_1.compute_schedule()
        cls.scheduling_problem_2.compute_schedule()

        cls.scheduling_problem_0.exact_compute_schedule()
        cls.scheduling_problem_0.WCPT_compute_schedule()
        # cls.scheduling_problem.

    def test_exact_schedule(self):
        self.assertEqual(self.scheduling_problem_0.exact_schedule,
                         [(0.0, 2), (0.0, 0), (0.0, 1), (0.8080808080808168, 0)],
                         "Incorrect Exact Schedule")

    def test_exact_makespan(self):
        self.assertEqual(self.scheduling_problem_0.exact_objective,
                         1.7171717171717358,
                         "Incorrect Exact Makespan")

    def test_schedule_0(self):
        self.assertEqual(self.scheduling_problem_0.schedule,
                         [(1.7645078332619106, 0), (0.6335105672368196, 1), (0.7767498981904866, 0), (0.0, 0)],
                         "Incorrect Schedule for Method 1")

    def test_makespan_0(self):
        self.assertEqual(self.scheduling_problem_0.objective,
                         3.0205952052325733,
                         "Incorrect Makespan for Method 1")

    def test_schedule_1(self):
        self.assertEqual(self.scheduling_problem_1.schedule,
                         [(0.0, 1), (1.0906123698361174, 1), (0.19935223355203197, 0), (1.0906123698361174, 0)],
                         "Incorrect Schedule for Method 2")

    def test_makespan_1(self):
        self.assertEqual(self.scheduling_problem_1.objective,
                         2.4491946114701117,
                         "Incorrect Makespan for Method 2")

    def test_schedule_2(self):
        self.assertEqual(self.scheduling_problem_2.schedule,
                         [(0.0, 0), (0.0, 1), (1.4887052654669968, 1), (0.0, 2)],
                         "Incorrect Schedule for Method 3")

    def test_makespan_2(self):
        self.assertEqual(self.scheduling_problem_2.objective,
                         3.165401270914264,
                         "Incorrect Makespan for Method 3")

    def test_WCPT_schedule(self):
        self.assertEqual(self.scheduling_problem_0.WCPT_schedule,
                         [(0.8318040846401877, 0), (0.0, 1), (0.0, 2), (2.6171956828310754, 0)],
                         "Incorrect WCPT Schedule")

    def test_WCPT_makespan(self):
        self.assertEqual(self.scheduling_problem_0.WCPT_objective,
                         4.4025872810219635,
                         "Incorrect WCPT Makespan")


if __name__ == '__main__':
    unittest.main()
