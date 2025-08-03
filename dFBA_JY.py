import cobra
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional


class MetaboliteConstraint:
    """
    Represents a time-dependent constraint on a metabolite.
    Should provide lower/upper bounds for uptake or production.
    """
    def __init__(self, met_id: str, constraint_fn: Callable[[float], tuple]):
        self.met_id = met_id  # e.g., 'glc'
        self.constraint_fn = constraint_fn  # e.g., lambda t: (-10, 0)

    # def get_bounds(self, time: float):
    #     return self.constraint_fn(time)
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
        tracked_reactions: Optional[List[str]] = None
    ):
        self.model = model
        self.objective = objective
        self.constraints = constraints  # { 'Ex_glc': MetaboliteConstraint(...) }
        self.timecourse = np.linspace(*time_range, int((time_range[1]-time_range[0])*steps_per_hour)+1)
        self.fba_method = fba_method
        self.fva = fva
        self.tracked_reactions = tracked_reactions or []

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
            print(f"[t={t:.2f}] Applying constraint for {met_id}")
            lb, ub = constraint.get_bounds(t)

            print(f"[DEBUG] At time {t:.2f}, constraint returned: lb={lb}, ub={ub}")

            # exch_rxn_id = f"Ex_{met_id}"  # Assumes BiGG-style naming
            exch_rxn_id = f"{met_id}"  # Assumes BiGG-style naming
            if exch_rxn_id in self.model.reactions:
                rxn = self.model.reactions.get_by_id(exch_rxn_id)

                # Update upper bound before lower bound to avoid conflict
                if ub < rxn.lower_bound:
                    rxn.lower_bound = lb  # safe since lb <= ub
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
                from cobra.flux_analysis import flux_variability_analysis
                fva_result = flux_variability_analysis(self.model, reaction_list=self.tracked_reactions)
                for rxn_id in self.tracked_reactions:
                    self.fva_bounds[rxn_id]["min"].append(fva_result.loc[rxn_id, "minimum"])
                    self.fva_bounds[rxn_id]["max"].append(fva_result.loc[rxn_id, "maximum"])

        print("dFBA simulation complete.")

    def export_results(self, prefix="dfba_output"):
        """
        Writes results to disk.
        """
        self.solution_fluxes.to_csv(f"{prefix}_fluxes.csv")
        if self.fva:
            # Save FVA min/max as DataFrames
            pd.DataFrame({
                rxn: self.fva_bounds[rxn]["min"] for rxn in self.tracked_reactions
            }, index=self.timecourse).to_csv(f"{prefix}_fva_min.csv")
            pd.DataFrame({
                rxn: self.fva_bounds[rxn]["max"] for rxn in self.tracked_reactions
            }, index=self.timecourse).to_csv(f"{prefix}_fva_max.csv")

