import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cobra.flux_analysis import production_envelope, pfba
from gsmmutils.model import MyModel


def production_env_plot(model, reactions_map, x, y, z, prefix):
    prod_env = production_envelope(model, [x, y], objective=z, carbon_sources="EX_C00011__dra",
                                   points=10)
    df = prod_env.rename(columns={"flux_maximum": z})
    df = df.fillna(0)
    df[x] = df[x].abs()
    df[y] = df[y].abs()

    pivot_table = df.round(3).pivot_table(index=y, columns=x, values=z)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, cmap='viridis', annot=False, xticklabels=2, yticklabels=2)
    ax.collections[0].colorbar.set_label(fr"{reactions_map[z]} $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.yticks(rotation=0)
    plt.xlabel(reactions_map[x] + " $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.ylabel(reactions_map[y] + r" $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.savefig(f'../results/ppp/ppp_{prefix}_{x.split("_")[1], y.split("_")[1], z.split("_")[1]}.png')


def set_mmol_g_h(model):
    for reaction in model.reactions:
        new_bounds = list(reaction.bounds)
        if reaction.upper_bound < 10000:
            new_bounds[1] = reaction.upper_bound / 24
        if reaction.lower_bound > -10000:
            new_bounds[0] = reaction.lower_bound / 24
        if new_bounds != list(reaction.bounds):
            print(reaction, new_bounds)
        reaction.bounds = new_bounds
    return model


def adjust_biomass(model, metabolite, macro_reaction, macromolecule, g_gdw):
    biomass_copy = model.reactions.e_Biomass__cytop.copy()
    biomass_reactions = [reaction for reaction in model.reactions if "Biomass" in reaction.id and "EX_" not in reaction.id and "_v" in reaction.id]
    last_version = max([int(reaction.id.split("__")[-2].split("_")[-1].lstrip("v")) for reaction in biomass_reactions] + [0])
    biomass_copy.id = biomass_copy.id.replace("__cytop", f"_v{last_version + 1}__cytop")
    model.add_reactions([biomass_copy])
    biomass_copy.backtrace = f"Changed experimental reaction to account for {metabolite.id} content of {g_gdw} g/gDW"
    new_st = g_gdw * 1000 / metabolite.formula_weight / biomass_copy.metabolites[macromolecule]
    model.set_stoichiometry(macro_reaction, metabolite.id, new_st)
    old_g_gDW_sum = abs(sum([biomass_copy.metabolites[reactant] for reactant in biomass_copy.reactants if biomass_copy.metabolites[reactant] > -1 and reactant != macromolecule]))
    g_gDW_sum = 0
    for reactant in macro_reaction.reactants:
        st = macro_reaction.metabolites[reactant]
        g_gDW_sum += st * reactant.formula_weight / 1000 * biomass_copy.metabolites[macromolecule]
    other_macros_sum = 1 - g_gDW_sum
    model.set_stoichiometry(model.reactions.get_by_id(biomass_copy.id), macromolecule.id, -g_gDW_sum)
    for reactant in biomass_copy.reactants:
        if biomass_copy.metabolites[reactant] > -1 and reactant != macromolecule:
            tmp = biomass_copy.metabolites[reactant] * other_macros_sum / old_g_gDW_sum
            model.set_stoichiometry(biomass_copy, reactant.id, tmp)
    assert round(abs(sum([biomass_copy.metabolites[reactant] for reactant in biomass_copy.reactants if biomass_copy.metabolites[reactant] > -1])), 4) == 1
    return model, biomass_copy


def get_ps_params(model):
    sol = pfba(model)
    photons_absorbed = sum(sol.fluxes[r.id] for r in model.reactions if r.id.startswith("PHOA"))
    # print("using RUBISCO")
    # print(sol['R00024__chlo']/photons_absorbed)
    # print("using exchange co2")
    # print("Quantum Yield")
    qy = round(abs(sol['EX_C00011__dra']/photons_absorbed), 3)
    # print(qy)
    # print("Photosynthetic Quotient")
    pq = round(abs(sol['EX_C00007__dra']/sol['EX_C00011__dra']), 3)
    # print(pq)
    # print("using PSII RUBISCO")
    # print(sol["PSII__lum"]/sol['R00024__chlo'])
    # print("using PSII exchange CO2 ")
    # print(sol["PSII__lum"]/sol['EX_C00011__dra'])
    return qy, pq


def get_ps_params_multi(models: dict[str, MyModel]):
    res = {}
    for key, model in models.items():
        res[key] = get_ps_params(model)
    as_df = pd.DataFrame.from_dict(res, orient="index", columns=["QY", "PQ"])
    return as_df

def evaluate_light_sources(model):
    light_sources = [reaction for reaction in model.reactions if reaction.id.startswith("PRISM")]
    results = {}
    max_growth = model.slim_optimize()
    for light_reaction in light_sources:
        with model as temp_model:
            temp_model.reactions.e_Biomass__cytop.bounds = (max_growth*0.50, max_growth*0.50)
            temp_model.objective = "EX_C00205__dra"
            light_reaction.bounds = (0, 10000)
            for other_reaction in light_sources:
                if other_reaction.id != light_reaction.id:
                    other_reaction.bounds = (0, 0)
            sol = temp_model.slim_optimize()
            results[light_reaction.id] = -sol
    return results