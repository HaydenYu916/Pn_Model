
import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from scipy.ndimage import gaussian_filter

# 直接使用标准RankAndCrowding

class iNSGA2(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=None,
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        # 创建选择算子
        if selection is None:
            selection = TournamentSelection(func_comp=self.binary_tournament)
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    @staticmethod
    def binary_tournament(pop, P, algorithm, **kwargs):
        S = np.full(P.shape[0], np.nan)
        for i in range(P.shape[0]):
            a, b = int(P[i, 0]), int(P[i, 1])
            a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
            rank_a = getattr(pop[a], "rank", 0)
            cd_a = getattr(pop[a], "crowding", 0.0)
            rank_b = getattr(pop[b], "rank", 0)
            cd_b = getattr(pop[b], "crowding", 0.0)

            if a_cv > 0 or b_cv > 0:
                S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
            else:
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b
                elif rank_a != rank_b:
                    S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')
                else:
                    S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

        return S[:, None].astype(int, copy=False)

    @staticmethod
    def compute_nis(pop):
        F = pop.get("F")
        crowding = pop.get("crowding")
        importance = np.sum(F, axis=1) + 0.001 * crowding
        importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + 1e-32)
        pop.set("nis", importance)
        return importance

    @staticmethod
    def find_knee_point(pop):
        F = pop.get("F")
        if F.shape[1] != 2:
            raise ValueError("Knee point detection only implemented for 2D objective space")
        f1, f2 = F[:, 0], F[:, 1]
        x = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1))
        y = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2))
        z = gaussian_filter(x ** 2 + y ** 2, sigma=1)
        knee_idx = np.argmin(z)
        return pop[knee_idx]
