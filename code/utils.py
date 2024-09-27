from gsmmutils.experimental.ExpMatrix import ExpMatrix
from gsmmutils.io import write_simulation
from tqdm import tqdm
from cobra.flux_analysis import production_envelope
import seaborn as sns
import matplotlib.pyplot as plt


def experimental_data_processing(filename, model):
    matrix = ExpMatrix(filename, model)
    matrix.conditions = "Resume"
    matrix.remove_trials(["Resume", "area", "19", "21", "R21", "25", "PC1"])
    matrix.set_exponential_phases({"1": (2, 8), "2": (2, 8), "3": (2, 16), "4": (2, 10), "5": (2, 8), "6": (2, 8), "7": (2, 16), "8": (2, 16), "9": (2, 8), "10": (2, 8), "11": (2, 12),
                                   "12": (2, 10), "13": (2, 8), "14": (2, 8), "15": (2, 14), "16": (2, 14), "17": (2, 12), "18": (2, 12), "20": (2, 14),
                                   "22": (2, 14), "23": (2, 10), "24": (2, 10), "PC1": (2, 10), "PC2": (2, 14), "PC3": (2, 10), "PC4": (2, 10), "RPC1": (2, 10), "RPC2": (2, 10),
                                   "RPC3": (2, 10), "N1": (0, 12), "N2": (0, 18), "N3": (0, 14), "N4": (0, 16), "N5": (0, 12), "N6": (0, 12), "N7": (0, 12), "N8": (0, 16), "N9": (0, 16)})
    matrix.get_experimental_data(parameter='all')
    matrix.get_substrate_uptake_from_biomass("C", "CO2", header="C00011")
    matrix.get_substrate_uptake_from_biomass("P", "C00009")
    matrix.get_substrate_uptake_from_biomass("N", "C00244")
    matrix.get_substrate_uptake("[P] mmol", header='HPO4')
    matrix.get_substrate_uptake("[N] mmol", header='NO3')
    matrix.save()
    return matrix


def simulation_for_conditions(model, conditions_df, growth_rate_df, save_in_file=False, filename=None, objective=None):
    as_dict = conditions_df.to_dict(orient='index')
    growth_rate = growth_rate_df.to_dict(orient='index')
    complete_results = {}
    error_sum = 0
    values_for_plot = {}
    model.exchanges.EX_C00011__dra.bounds = (-1000, 1000)
    for index, condition in tqdm(as_dict.items(), colour="blue"):
        if {f"e_Biomass_trial{index}__cytop"}.issubset(model.reaction_ids):
            model_copy = model.copy()
            for reaction in model_copy.reactions:
                if ("Biomass" in reaction.id and "EX_" not in reaction.id
                        and reaction.id != f"e_Biomass_trial{index}__cytop"):
                    reaction.bounds = (0, 0)
            model_copy.reactions.get_by_id(f"e_Biomass_trial{index}__cytop").bounds = (0, 1000)
            if objective:
                [setattr(x, 'objective_coefficient', 0) for x in model.reactions if x.objective_coefficient != 0]
                model_copy.reactions.get_by_id(f"e_Biomass_trial{index}__cytop").objective_coefficient = 1
                for key, value in objective.items():
                    model_copy.reactions.get_by_id(key).objective_coefficient = value
            else:
                model_copy.objective = f"e_Biomass_trial{index}__cytop"
            for met, lb in condition.items():
                lb = -lb if lb < 0 else lb
                model_copy.reactions.get_by_id("EX_" + met + "__dra").bounds = (round(-lb, 4), 1000)
            sol = model_copy.optimize()
            biomass = round(sol[f"e_Biomass_trial{index}__cytop"], 3)
            error_sum += abs(growth_rate[index]['growth_rate'] - biomass)
            complete_results[index] = sol
            values_for_plot[index] = (growth_rate[index]['growth_rate'], biomass)
    if save_in_file:
        write_simulation(complete_results, filename)
    return complete_results, values_for_plot, round(error_sum, 6)


def simulations(matrix, model):
    complete_results, values_for_plot_carbon_limited, error1 = simulation_for_conditions(model, matrix.conditions[["C00011"]], matrix.conditions[["growth_rate"]], save_in_file=True, filename="../results/trial_simulations/carbon_limited")
    _, values_for_plot_p_limited, error2 = simulation_for_conditions(model, matrix.conditions[["C00009"]], matrix.conditions[["growth_rate"]], save_in_file=True, filename="../results/trial_simulations/p_limited")
    _, values_for_plot_carbon_and_p_limited, error3 = simulation_for_conditions(model, matrix.conditions[["C00011", "C00009"]], matrix.conditions[["growth_rate"]], save_in_file=True, filename="../results/trial_simulations/carbon_and_p_limited")
    _, values_for_plot_n_limited, error4 = simulation_for_conditions(model, matrix.conditions[["C00244"]], matrix.conditions[["growth_rate"]], save_in_file=True, filename="../results/trial_simulations/n_limited")
    _, values_for_plot_carbon_and_p_and_n_limited, error5 = simulation_for_conditions(model, matrix.conditions[["C00011", "C00009", "C00244"]], matrix.conditions[["growth_rate"]], save_in_file=True, filename="../results/trial_simulations/carbon_and_p_and_n_limited")
    res = []
    print("Error1: ", error1)
    print("Error2: ", error2)
    print("Error3: ", error3)
    print("Error4: ", error4)
    print("Error5: ", error5)
    print(sum([error1, error2, error3, error4]))
    for index, element in values_for_plot_carbon_limited.items():
        new_value = values_for_plot_carbon_and_p_limited[index][1]
        new_value2 = values_for_plot_p_limited[index][1]
        new_value3 = values_for_plot_carbon_and_p_and_n_limited[index][1]
        new_value4 = values_for_plot_n_limited[index][1]
        res.append((element[0], element[1], new_value, new_value2, new_value3, new_value4))
    df = pd.DataFrame(data=res, index=complete_results.keys(), columns=["Exp", "in silico (C limited)", "in silico (C & P limited)", "in silico (P limited)", "in silico (C, P, and N limited)", "in silico (N limited)"])
    df.to_excel("../results/trial_simulations/simulation_summary.xlsx")
    barplot(df, to_show=False, path="../results/trial_simulations/simulation_summary.png")
    
    
    

def production_env_plot(model, reactions_map, x, y, z, prefix):   
    prod_env = production_envelope(model, [x, y], objective=z, carbon_sources="EX_C00011__dra",
                                  points = 10)
    df = prod_env.rename(columns={"flux_maximum": z})
    df = df.fillna(0)
    df[x] = df[x].abs()
    df[y] = df[y].abs()    

    pivot_table = df.round(3).pivot_table(index=y, columns=x, values=z)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, cmap='viridis', annot=False, xticklabels=2, yticklabels=2)
    ax.collections[0].colorbar.set_label(f"{reactions_map[z]} $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.yticks(rotation=0)
    plt.xlabel(reactions_map[x] + " $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.ylabel(reactions_map[y] + r" $(mmol \cdot gDW^{-1} \cdot d^{-1})$")
    plt.savefig(f'../data/ppp_{prefix}_{x.split("_")[1],y.split("_")[1],z.split("_")[1]}.png')
   

def set_mmol_g_h(model):
    for reaction in model.reactions:
        new_bounds = list(reaction.bounds)
        if reaction.upper_bound < 10000:
            new_bounds[1] = reaction.upper_bound/24
        if reaction.lower_bound > -10000:
            new_bounds[0] = reaction.lower_bound/24
        if new_bounds != list(reaction.bounds):
            print(reaction, new_bounds)
        reaction.bounds = new_bounds
    return model
            
                
                
         
