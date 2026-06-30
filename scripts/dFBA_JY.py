import cobra
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Callable, Optional
# fva
from cobra.flux_analysis import flux_variability_analysis


def _fva_worker(model, reaction_list, fraction_of_optimum, loopless, queue):
    """
    Runs in a forked child process. Inherits the parent's model via
    copy-on-write (Linux fork), so no pickling of the cobra.Model is needed.
    """
    try:
        result = flux_variability_analysis(
            model, reaction_list=reaction_list,
            fraction_of_optimum=fraction_of_optimum, loopless=loopless
        )
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", str(e)))


def run_fva_with_hard_timeout(model, reaction_list, fraction_of_optimum=0.995,
                               loopless=False, timeout=15):
    """
    Runs flux_variability_analysis in a forked subprocess and hard-kills it
    if it exceeds `timeout` seconds. Necessary because a stalled GLPK simplex
    call is stuck in C code and won't respond to SIGALRM/SIGINT.
    Returns (result_df_or_None, error_str_or_None). error_str is "timeout"
    if it stalled, otherwise the exception message, or None on success.
    """
    ctx = mp.get_context("fork")  # Linux-only; fine on erishpc
    queue = ctx.Queue()
    p = ctx.Process(target=_fva_worker, args=(model, reaction_list, fraction_of_optimum, loopless, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, "timeout"

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            return payload, None
        else:
            return None, payload
    return None, "unknown failure"


class MetaboliteConstraint:
    """
    Represents a time-dependent constraint on a metabolite.
    Should provide lower/upper bounds for uptake or production.
    """
    def __init__(self, met_id: str, constraint_fn: Callable[[float], tuple]):
        self.met_id = met_id  # e.g., 'glc'
        self.constraint_fn = constraint_fn  # e.g., lambda t: (-10, 0)

    def get_bounds(self, time: float):
        lb, ub = self.constraint_fn(time)
        if lb > ub:
            print(f"[Warning] Invalid bounds at t={time:.2f}: lb={lb}, ub={ub}")
        return (min(lb, ub), max(lb, ub))


class dFBA:
    def __init__(
        self,
        model: cobra.Model,
        objective: str,
        constraints: Dict[str, MetaboliteConstraint],
        time_range: tuple = (0, 48),
        steps_per_hour: int = 1,
        fba_method: Callable = cobra.flux_analysis.pfba,
        fva: bool = False,
        tracked_reactions: Optional[List[str]] = None,
        fva_exclude: Optional[List[str]] = None,
        fva_timeout: int = 15,
    ):
        self.model = model
        self.objective = objective
        self.constraints = constraints  # { 'Ex_glc': MetaboliteConstraint(...) }
        self.timecourse = np.linspace(*time_range, int((time_range[1]-time_range[0])*steps_per_hour)+1)
        self.fba_method = fba_method
        self.fva = fva
        self.tracked_reactions = tracked_reactions or []
        self.fva_exclude = set(fva_exclude or [])
        self.fva_timeout = fva_timeout

        # track all nonzero fluxes
        self.all_fluxes = {}  # time -> full flux Series

        # Output structures
        self.solution_fluxes = pd.DataFrame(index=self.timecourse, columns=self.tracked_reactions)
        if self.fva:
            self.fva_bounds = {rxn: {"min": [], "max": []} for rxn in self.tracked_reactions}

    def apply_constraints(self, t: float):
        """
        Applies time-dependent metabolite constraints.
        This modifies the model's exchange bounds directly.
        """
        for met_id, constraint in self.constraints.items():
            lb, ub = constraint.get_bounds(t)
            exch_rxn_id = f"{met_id}"
            if exch_rxn_id in self.model.reactions:
                rxn = self.model.reactions.get_by_id(exch_rxn_id)

                if exch_rxn_id == "Ex_glc":
                    print(f"[DEBUG Ex_glc] t={t:.2f} get_bounds=({lb:.6f},{ub:.6f}) model=({rxn.lower_bound:.6f},{rxn.upper_bound:.6f})")

                # Update upper bound before lower bound to avoid conflict
                if ub < rxn.lower_bound:
                    rxn.lower_bound = lb
                    rxn.upper_bound = ub
                else:
                    rxn.upper_bound = ub
                    rxn.lower_bound = lb

                print(f"[t={t:.2f}] {rxn} bounds set to ({lb}, {ub})")
            else:
                print(f"[WARN] Exchange reaction {exch_rxn_id} not found in model.")

    def run(self):
        """
        Runs the dFBA simulation over the timecourse.
        Stores results in self.solution_fluxes (and FVA bounds if enabled).
        """
        self.model.objective = self.model.reactions.get_by_id(self.objective)
        for t in self.timecourse:
            self.apply_constraints(t)
            sol = self.fba_method(self.model)

            # Save tracked reaction fluxes
            for rxn_id in self.tracked_reactions:
                self.solution_fluxes.at[t, rxn_id] = sol.fluxes.get(rxn_id, np.nan)

            self.all_fluxes[t] = sol.fluxes.copy()

            # Optionally perform FVA
            if self.fva:
                fva_reactions = [r for r in self.tracked_reactions if r not in self.fva_exclude]

                fva_result, err = run_fva_with_hard_timeout(
                    self.model, fva_reactions,
                    fraction_of_optimum=0.995, loopless=False,
                    timeout=self.fva_timeout
                )
                if err == "timeout":
                    print(f"[t={t:.2f}] FVA stalled (> {self.fva_timeout}s), filling NaN for this step.")
                elif err is not None:
                    print(f"[t={t:.2f}] FVA failed: {err}, filling NaN for this step.")

                for rxn_id in self.tracked_reactions:
                    if fva_result is None or rxn_id in self.fva_exclude:
                        self.fva_bounds[rxn_id]["min"].append(np.nan)
                        self.fva_bounds[rxn_id]["max"].append(np.nan)
                    else:
                        self.fva_bounds[rxn_id]["min"].append(fva_result.loc[rxn_id, "minimum"])
                        self.fva_bounds[rxn_id]["max"].append(fva_result.loc[rxn_id, "maximum"])

        print("dFBA simulation complete.")

    def diagnose_fva_stalls(self, t: float, per_rxn_timeout: int = 15):
        """
        One-off diagnostic: applies constraints for time t, then runs FVA on
        tracked_reactions one at a time (each in its own forked subprocess
        with a hard timeout) to find which reaction(s) stall.
        Call this manually (not inside run()) when you suspect a stall.
        """
        self.apply_constraints(t)
        self.fba_method(self.model)  # warm solve, mirrors run()

        for rxn_id in self.tracked_reactions:
            result, err = run_fva_with_hard_timeout(
                self.model, [rxn_id],
                fraction_of_optimum=0.995, loopless=False,
                timeout=per_rxn_timeout
            )
            if err == "timeout":
                print(f"{rxn_id}: STALLED (> {per_rxn_timeout}s)")
            elif err is not None:
                print(f"{rxn_id}: failed -> {err}")
            else:
                print(f"{rxn_id}: -> {result.values.tolist()}")

    def export_results(self, prefix="dfba_output"):
        self.solution_fluxes.dropna(how='all').to_csv(f"{prefix}_fluxes.csv")
        if self.fva:
            completed_times = self.timecourse[:len(list(self.fva_bounds.values())[0]["min"])]
            pd.DataFrame({
                rxn: self.fva_bounds[rxn]["min"] for rxn in self.tracked_reactions
            }, index=completed_times).to_csv(f"{prefix}_fva_min.csv")
            pd.DataFrame({
                rxn: self.fva_bounds[rxn]["max"] for rxn in self.tracked_reactions
            }, index=completed_times).to_csv(f"{prefix}_fva_max.csv")