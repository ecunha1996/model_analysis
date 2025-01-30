import math
from os.path import join

import numpy as np
import pandas as pd
from gsmmutils.model import MyModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import copy

sns.set(rc={'figure.figsize': (60, 50)})
DATA_PATH = r"../data"
from os.path import join
from cobra.flux_analysis import pfba, flux_variability_analysis as fva
from utils.utils import get_ps_params_multi, evaluate_light_sources

WAVE_LENGTHS = {"298": [281, 306],
                "437": [406, 454],
                "438": [378, 482],
                "450": [417, 472],
                "490": [451, 526],
                "646": [608, 666],
                "673": [659, 684],
                "680": [662, 691]}


def load_data(paths: list[str]) -> pd.DataFrame:
    """
    Load data from a list of paths
    Parameters
    ----------
    paths

    Returns
    -------

    """
    final_df = pd.DataFrame()
    for path in paths:
        data = pd.read_csv(path, sep="\t", index_col=0)
        final_df = pd.concat([final_df, data], axis=1)
    return final_df


def integrate_for_wave_length(df: pd.DataFrame, wave_range: list, pigments: set[str], biomass_mass) -> tuple[dict, float]:
    """
    Integrate the absorption values for a given wave length
    Parameters
    ----------
    df
    wave_length
    pigments

    Returns
    -------

    """
    minimum = max(wave_range[0], min(df.index.tolist()))
    df = df.loc[(df.index > minimum) & (df.index < wave_range[1])]
    area_of_pigments = {}
    for pigment in pigments:
        vals = df[pigment] * 150
        area_of_pigments[pigment] = abs(np.trapz(vals, df.index)) / 10000 * biomass_mass[pigment]
    # normalize
    total_area = sum(area_of_pigments.values())
    if total_area > 0:
        for pigment in pigments:
            area_of_pigments[pigment] = round(area_of_pigments[pigment] / total_area, 4)
    return area_of_pigments, total_area


def main(organism, light_intensity):
    """
    Main function to calculate the absorption of light by pigments
    Data to determine the xanthophyll percentage at each light uptake was inferred from  10.1073/PNAS.2214119120/FORMAT/EPUB.

    Parameters
    ----------
    organism
    light_intensity

    Returns
    -------

    """
    original_biomass_mass = {"ngaditana": {"Chlorophyll-a": 14.418, "B,B-carotene": 0.173, "Zeaxanthin": 0.174, "Violaxanthin": 1.652, "Antheraxanthin": 0.388},
                             "dsalina": {"Chlorophyll-a": 5.962, "Chlorophyll-b": 4.3195, "B,B-carotene": 0.9732, "Lutein": 2.6201, "Zeaxanthin": 0.1909,
                                         "Violaxanthin": 0.4540, "Neoxanthin": 0.4970, "cis-B-carotene": 0.9732, "Antheraxanthin": 0.4522},
                             "plutheri": {"Chlorophyll-a": 14.328, "chlorophyll c1": 1.453, "chlorophyll c2": 1.453, "B,B-carotene": 3.992, "Diatoxanthin": 6.328, "Fucoxanthin": 8.546, "Diadinoxanthin": 4.815}}

    xantophyls_total = sum([val for key, val in original_biomass_mass[organism].items() if key in ["Zeaxanthin", "Violaxanthin", "Antheraxanthin", "Diatoxanthin", "Diadinoxanthin"]])

    if "Antheraxanthin" in original_biomass_mass[organism]:
        zeaxanthin = (0.0116 * light_intensity + 3.5151) / 100
        violaxanthin = (-10.13 * math.log(light_intensity) + 135.53) / 100 if light_intensity > 0 else 0.95
        antheraxanthin = (1 - zeaxanthin - violaxanthin)
        biomass_mass = original_biomass_mass[organism]
        biomass_mass["Zeaxanthin"] = zeaxanthin * xantophyls_total
        biomass_mass["Violaxanthin"] = violaxanthin * xantophyls_total
        biomass_mass["Antheraxanthin"] = antheraxanthin * xantophyls_total
    else:
        diatoxanthin = (0.0116 * light_intensity + 3.5151) / 100
        diadinoxanthin = (-10.13 * math.log(light_intensity + 1e-30) + 135.53) / 100
        # normalize to 1
        diatoxanthin = diatoxanthin / (diatoxanthin + diadinoxanthin)
        diadinoxanthin = 1 - diatoxanthin
        biomass_mass = original_biomass_mass[organism]
        biomass_mass["Diatoxanthin"] = diatoxanthin * xantophyls_total
        biomass_mass["Diadinoxanthin"] = diadinoxanthin * xantophyls_total

    df = load_data(["../data/absorption/mmc2.txt", "../data/absorption/mmc3.txt"])  # from https://doi.org/10.1016%2Fj.dib.2019.103875
    # df['alpha-carotene'] = df['B,B-carotene']  # https://envi.geoscene.cn/help/Subsystems/envi/Content/Vegetation%20Analysis/PlantFoliage.htm
    df['cis-B-carotene'] = df['B,B-carotene']
    df['chlorophyll c1'] = df['chlorophyll c2']
    total_area = 0
    area_by_carotene = {}
    total_area_by_wave_length = {}
    for wave_length, wave_range in WAVE_LENGTHS.items():
        area, total = integrate_for_wave_length(df, wave_range, set(biomass_mass.keys()), biomass_mass)
        total_area += total
        # sort area by key
        area = {key: val for key, val in sorted(area.items(), key=lambda item: item[0])}
        total_area_by_wave_length[wave_length] = area
        for key, val in area.items():
            if key not in area_by_carotene:
                area_by_carotene[key] = 0
            area_by_carotene[key] += val
    return total_area, area_by_carotene, total_area_by_wave_length


def update_model(model, organism, light_intensity):
    """
    Update the model with the absorption of light by pigments.
    Parameters
    ----------
    model
    organism
    light_intensity

    Returns
    -------

    """
    metabolites_map = {"Chlorophyll-a": ("C05306", "chla_soret_exc", "chla_qy2_exc"),
                       "Chlorophyll-b": ("C05307", "chlb_soret_exc", "chlb_qy2_exc"),
                       "chlorophyll c1": ("CPD_10336", "chlc1_soret_exc", "chlc1_qy2_exc"),
                       "chlorophyll c2": ("CPD_10337", "chlc2_soret_exc", "chlc2_qy2_exc"),
                       "B,B-carotene": ("C02094", "bcaro_exc"),
                       "Zeaxanthin": ("C06098", "zeax_exc"),
                       "Violaxanthin": ("C08614", "vioxan_exc"),
                       "Antheraxanthin": ("C08579", "anthxan_exc"),
                       "Lutein": ("C08601", "lut_exc"),
                       "cis-B-carotene": ("C20484", "cbcaro_exc"),
                       "Neoxanthin": ("C08606", "neoxan_exc"),
                       "Diatoxanthin": ("C19920", "diatox_exc"),
                       "Diadinoxanthin": ("C19921", "diadinx_exc"),
                       "Fucoxanthin": ("C08596", "fxanth_exc"), }
    total_area, area_by_carotene, total_area_by_wave_length = main(organism, light_intensity)
    for wave_length, values in total_area_by_wave_length.items():
        if f"PHOA{wave_length}__chlo" in model.reaction_ids:
            reaction = model.reactions.get_by_id(f"PHOA{wave_length}__chlo")
            for pigment, area in values.items():
                reactant = model.metabolites.get_by_id(metabolites_map[pigment][0] + "__chlo")
                if int(wave_length) > 600 and "chlorophyll" in pigment.lower():
                    product = model.metabolites.get_by_id(metabolites_map[pigment][2] + "__chlo")
                else:
                    product = model.metabolites.get_by_id(metabolites_map[pigment][1] + "__chlo")
                model.set_stoichiometry(reaction, reactant.id, -area)
                model.set_stoichiometry(reaction, product.id, area)
    return model


def evaluate_photosynthesis(model, conversion_factor, organism):
    """
    Evaluate the photosynthesis process, including growth rate, and the flux of photosystems and ROS reactions.
    Parameters
    ----------
    model: MyModel
        GSM model
    conversion_factor: float
        Conversion factor from uE.m-2.s-1 to mmol.gDW-1.d-1
    organism: str
        Organism name

    Returns
    -------

    """
    model.set_prism_reaction("PRISM_solar_litho__extr")
    coeff = sum(e for e in model.reactions.PRISM_solar_litho__extr.metabolites.values() if e > 0)
    for rxn in model.reactions:
        if rxn.lower_bound <= -1000:
            rxn.lower_bound = -20000
        if rxn.upper_bound >= 1000:
            rxn.upper_bound = 20000
    same_flux = model.problem.Constraint(
        model.reactions.R00024__chlo.flux_expression * 0.025 - model.reactions.R03140__chlo.flux_expression,
        lb=-1000,
        ub=0)

    max_cef_lef_ratio = {'ngaditana': 0.1, 'dsalina': 0.9, 'plutheri': 0.90}  # sol.fluxes['CEF__chlo'] / (sol.fluxes['R01195__chlo'] + sol.fluxes['CEF__chlo'])
    cef_constrain = model.problem.Constraint(
        model.reactions.CEF__chlo.flux_expression * (1 - max_cef_lef_ratio[organism]) - model.reactions.R01195__chlo.flux_expression * max_cef_lef_ratio[organism],
        lb=-1000,
        ub=0)
    model.add_cons_vars([same_flux, cef_constrain])
    plt.rcParams['figure.figsize'] = (40, 20)
    max_photosynthesis = {"dsalina": 177.8, "ngaditana": 136.5, "plutheri": 44.7}
    step_size = {"dsalina": 20, "ngaditana": 20, "plutheri": 20}
    model.reactions.PSII__lum.bounds = (0, max_photosynthesis[organism])
    with model as tmp:
        old_co2 = pfba(tmp).fluxes["EX_C00011__dra"]
        max_growth = tmp.slim_optimize()
        factor = 10 ** 2
        tmp.reactions.e_Biomass__cytop.bounds = (math.floor(max_growth * factor)/factor, 1000)
        tmp.exchanges.EX_C00205__dra.bounds = (-10000, 10000)
        tmp.objective = "EX_C00205__dra"
        print(tmp.slim_optimize()/conversion_factor)
        # tmp.exchanges.EX_C00205__dra.bounds = (-10000, 10000)
        # tmp.objective = "EX_C00205__dra"
        # tmp.objective_direction = "max"
        # min_val = tmp.slim_optimize()
        # tmp.objective_direction = "min"
        # max_val = min(abs(tmp.slim_optimize()), 2100*conversion_factor)
    model.exchanges.EX_C00011__dra.lower_bound = old_co2
    npq = {}
    # uptake = range(int(round(abs(min_val), 0)), int(round(max_val, 0)), 100)
    uptake = range(0, int(round(1020)), step_size[organism])
    ros_reactions = {"FLV__chlo", "CEF__chlo",
                     # "NGAM_D1__chlo",
                     "R12570__chlo", "R09540__chlo", "R12571__chlo", "R12571__mito", "R00275__chlo", "R00275__mito", "R00009__pero", "R08360__cytop", "R09540__cytop",
                     "R00274__chlo", "R00017__mito",
                     "PSII__lum", "R01195__chlo", "PSI__lum", "PSIc6__lum"
                     }
    photosystems = {"PSII__lum", "R01195__chlo", "PSI__lum", "PSIc6__lum"}
    ros_reactions = {key: [0, 0, 0] for key in ros_reactions if key in model.reaction_ids}
    ros_results_by_light = {}
    ros_results_by_reaction = {}
    for ue_m2s in tqdm(uptake):
        with model as tmp:
            mmol_gdwd = conversion_factor * ue_m2s
            tmp = update_model(tmp, organism, ue_m2s)
            tmp.exchanges.EX_C00205__dra.bounds = (-mmol_gdwd, -mmol_gdwd)
            ngam = ue_m2s * 0.022 + 6.85
            tmp.reactions.NGAM__lum.bounds = (ngam, ngam)
            npq_tmp = (ue_m2s * 0.0574+33.556) / 100
            max_npq = {"dsalina": 0.37, "ngaditana": 0.68, "plutheri": 0.68}[organism]
            if npq_tmp > max_npq:
                npq_tmp = max_npq
            npq_constrain_ps2 = tmp.problem.Constraint(
                tmp.reactions.CHLAPSIIdex__chlo.flux_expression * (1 - npq_tmp) - tmp.reactions.PSIICSa__chlo.flux_expression * npq_tmp,
                lb=-1000,
                ub=0)
            tmp.add_cons_vars([npq_constrain_ps2])
            sol = tmp.maximize(value=False)
            # sol = model.optimize()
            if not isinstance(sol, int) and sol.status == "optimal":
                # fva_sol = fva(model, list(ros_reactions.keys()), fraction_of_optimum=0.99, processes=6)
                # print(sol.fluxes['DM_pho_loss__chlo'] / abs(sol.fluxes['EX_C00205__dra'])* coeff)
                heat = sol.fluxes['DM_pho_loss__chlo'] / (abs(sol.fluxes['EX_C00205__dra']) * coeff)
                print(f"HEAT: {heat}")
                # print(f"CEF: {sol.fluxes['CEF__chlo'] / (sol.fluxes['R01195__chlo'] + sol.fluxes['CEF__chlo'])}")
                # print(f"NPQ: {sol.fluxes['CHLAPSIIdex__chlo'] / (sol.fluxes['flux_expression'] + sol.fluxes['CHLAPSIIdex__chlo'])}")
                # print(sol.fluxes["CEF_2__chlo"]/(sol.fluxes["CEF_2__chlo"]+ sol.fluxes["R01195__chlo"]))
                for r in ros_reactions:
                    ros_reactions[r][0] = sol.fluxes[r] / (abs(sol.fluxes['EX_C00205__dra']) * coeff)
                    ros_reactions[r][1] = 0 #fva_sol.loc[r, 'minimum'] / (abs(sol.fluxes['EX_C00205__dra']) * coeff)
                    ros_reactions[r][2] =0# fva_sol.loc[r, 'maximum'] / (abs(sol.fluxes['EX_C00205__dra']) * coeff)
                ros = sum([abs(v[0]) for k, v in ros_reactions.items() if k not in photosystems])
                npq[ue_m2s] = (heat + ros, heat,
                               ros,
                               sol.fluxes["e_Biomass__cytop"])
                ros_results_by_light[ue_m2s] = copy.deepcopy(ros_reactions)
                for r, val in ros_reactions.items():
                    if r not in ros_results_by_reaction:
                        ros_results_by_reaction[r] = {}
                    ros_results_by_reaction[r][ue_m2s] = copy.deepcopy(val)

    # create fig with two plots
    n_rows = int(1 + round(len(ros_results_by_reaction) / 3 + 0.50, 0))
    fig, ax = plt.subplots(n_rows, 3)
    plt.subplots_adjust(hspace=1.2)

    tmp_axis = sns.lineplot(x=list(npq.keys()), y=[v[0] for v in npq.values()], label="Total NPQ", ax=ax[0][0])
    sns.lineplot(x=list(npq.keys()), y=[v[1] for v in npq.values()], label="Heat NPQ", ax=ax[0][0])
    sns.lineplot(x=list(npq.keys()), y=[v[2] for v in npq.values()], label="ROS NPQ", ax=ax[0][0])

    for reaction, fluxes in ros_results_by_reaction.items():
        if not all(v[0] == 0 for v in fluxes.values()):
            sns.lineplot(x=list(npq.keys()), y=[v[0] for v in fluxes.values()], label=reaction, ax=ax[0][1])

    sns.lineplot(x=list(npq.keys()), y=[v[-1] for v in npq.values()], ax=ax[0][2], color='red', label="Growth rate")

    dataframes = {}

    for key, value in ros_results_by_light.items():
        for mmol_gdwd, (reaction, values) in enumerate(value.items()):
            if reaction not in dataframes:
                dataframes[reaction] = pd.DataFrame()
            dataframes[reaction][key] = values
    dataframes = {key: value.T for key, value in dataframes.items()}
    for df in dataframes.values():
        df.columns = ["Flux", "Minimum", "Maximum"]

    npq_df = pd.DataFrame(npq.values(), columns=["Total NPQ", "Heat NPQ", "ROS NPQ", "Growth rate"], index=npq.keys())
    npq_df.to_hdf(f"../results/npq_{organism}.h5", key="npq")

    for key, value in dataframes.items():
        value.to_hdf(f"../results/npq_{organism}.h5", key=key)

    row = 1
    col = 0
    for mmol_gdwd, (reaction, dataframe) in enumerate(dataframes.items()):
        if col > 2:
            col = 0
            row += 1
        tmp_axis = dataframe.plot(y="Flux", ax=ax[row][col], color='black', linestyle='-', marker='o', use_index=False)
        dataframe['Height'] = dataframe['Maximum'] - dataframe['Minimum']
        dataframe.plot(kind="bar", y="Height", bottom=dataframe["Minimum"], ax=tmp_axis, title=reaction)
        tmp_axis.set_xticks(range(len(dataframe)))
        tmp_axis.set_xticklabels([int(round(e, 0)) for e in dataframe.index], rotation=45)
        # tmp_axis.set_xlabel("Light intensity")
        tmp_axis.legend(['pFBA', "FVA"])
        # bars = tmp_axis.patches
        # for bar in bars:
        #     if bar.get_height() == dataframe["Minimum"].max():
        #         bar.set_facecolor('none')
        col += 1
    for j, tmp in enumerate(ax):
        for i, axis in enumerate(tmp):
            axis.set_xlabel("Light intensity (umol/m2/s)")
            axis.set_ylabel("Flux (mmol / mmol photon)")
            if i == 2 and j == 0:
                axis.set_ylabel("Growth rate (h-1)")
    plt.savefig(f"../results/figures/npq_{organism}.png", bbox_inches='tight', dpi=600)
    # plt.show()
    return npq


def merge_and_plot_results():
    """
    Reads the results from the evaluate_photosynthesis function and plots the results.
    Returns
    -------

    """
    # plt.style.use('seaborn-whitegrid')
    sns.set_theme(context='paper', style='ticks', palette="colorblind",  font='Arial')
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
    })
    organisms = {"dsalina": "D. salina", "ngaditana": "N. gaditana", "plutheri": "P. lutheri"}
    photosystems = {"PSII__lum", "R01195__chlo", "PSI__lum", "PSIc6__lum"}
    ros_reactions = {"CEF__chlo",
                     "R12570__chlo", "R09540__chlo",
                     "R00274__chlo","R12571__chlo", "R12571__mito",
                     # "R00275__chlo", "R00275__mito",
                     # "R00009__pero",
                     "R08360__cytop", "R09540__cytop"
                     }
    index_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    legend_map = {"PSII__lum": "PSII", "R01195__chlo": "FNR", "PSI__lum": "PSI", "PSIc6__lum": "PSI", "R12570__chlo": "PRDX", "R09540__chlo": "APX", "R00274__chlo": "GPX",
                  "CEF__chlo": "CEF", "R00275__chlo": "SOD", "R00009__pero": "CAT", "R00275__mito": "SOD"}
    fig, axs = plt.subplots(4, 3, figsize=(7.08, 6.2))
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    for i, (abb, organism) in enumerate(organisms.items()):
        df = pd.read_hdf(f"../results/npq_{abb}.h5", key="npq")
        for reaction in ros_reactions.union(photosystems):
            try:
                tmp = pd.read_hdf(f"../results/npq_{abb}.h5", key=reaction)
                df[reaction] = tmp["Flux"]
            except:
                pass
        light_intensity = df.index.tolist()
        ax = axs[i]

        sns.lineplot(x=light_intensity, y=df["Growth rate"], ax=axs[0][i], color='red', label="Growth rate")
        axs[0][i].set_xticks(range(0, max(light_intensity) + 50, 200))
        axs[0][i].set_title(organism, style='italic')

        for reaction in sorted(list(photosystems.intersection(set(df.columns)))):
            if not all(v == 0 for v in df[reaction]):
                sns.lineplot(x=light_intensity, y=df[reaction], label=legend_map.get(reaction, reaction), ax=axs[1][i])
                # set xticks
                axs[1][i].set_xticks(range(0, max(light_intensity) + 50, 200))

        for reaction in sorted(list(ros_reactions.intersection(set(df.columns)))):
            if not all(v == 0 for v in df[reaction]):
                sns.lineplot(x=light_intensity, y=df[reaction], label=legend_map.get(reaction, reaction), ax=axs[2][i])
                axs[2][i].set_xticks(range(0, max(light_intensity) + 50, 200))
        # add heat to axs[2][i] in a secondary axis
        df["Heat NPQ"] = df["Heat NPQ"].round(2)
        sns.lineplot(x=light_intensity, y=df["Heat NPQ"], ax=axs[3][i], label="Y(NPQ)")
        axs[3][i].set_xticks(range(0, max(light_intensity) + 50, 200))
        for lab in axs[3][i].get_yticklabels():
            lab.set_fontsize(6)
        for lab in axs[3][i].get_xticklabels():
            lab.set_fontsize(6)
        # set fontsize
        # merge legends
        for axis in ax:
            for lab in axis.get_yticklabels():
                lab.set_fontsize(6)
            for lab in axis.get_xticklabels():
                lab.set_fontsize(6)
            # axis.set_xlabel(r"Light intensity ($\mu \mathit{mol} \cdot \mathit{m}^{-2} \cdot \mathit{s}^{-1}$)")
            axis.set_xlabel(r"Light intensity ($\mu mol \cdot m^{-2} \cdot s^{-1}$)")

        axs[0][i].set_ylabel("Growth rate ($d^{-1}$)")
        axs[1][i].set_ylabel("Flux ($mol / mol_{hn}$)")
        axs[2][i].set_ylabel("Flux ($mol / mol_{hn}$)")
        axs[3][i].set_ylabel("Flux ($mol / mol_{hn}$)")
        axs[3][i].set_xlabel(r"Light intensity ($\mu mol \cdot m^{-2} \cdot s^{-1}$)")

    counter = 0
    for row in axs:
        for axis in row:
            axis.legend(fontsize=6)
            axis.text(-0.25, 1.1, index_chars[counter], transform=axis.transAxes, fontsize=10, fontweight='bold', va='top')
            counter += 1

    plt.savefig(f"../results/figures/photoprotection_all.pdf", bbox_inches='tight', format="pdf", dpi=1200)
    # plt.show()


if __name__ == '__main__':
    # ng = MyModel(join(DATA_PATH, 'models/model_ng.xml'), 'e_Biomass__cytop')
    # with ng as tmp:
    #     evaluate_photosynthesis(tmp,  3.81, "ngaditana")
    # # #
    # ds = MyModel(join(DATA_PATH, 'models/model_ds.xml'), 'e_Biomass__cytop')
    # with ds as tmp:
    #     evaluate_photosynthesis(tmp, 2.99, "dsalina")
    # #
    # pl = MyModel(join(DATA_PATH, 'models/model_pl.xml'), 'e_Biomass__cytop')
    # with pl as tmp:
    #     evaluate_photosynthesis(tmp, 4.50, "plutheri")
    #
    merge_and_plot_results()

    # wave_lenghts = {"298": [281, 306],
    #                 "437": [406, 454],
    #                 "438": [378, 482],
    #                 "450": [417, 472],
    #                 "490": [451, 526],
    #                 "646": [608, 666],
    #                 "673": [659, 684],
    #                 "680": [662, 691]}
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin", "Antheraxanthin", "Violaxanthin", "alpha-carotene", "Neoxanthin", "cis-B-carotene"}
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin", "Antheraxanthin", "Violaxanthin",  "Neoxanthin"}  # Fluorescence quenching in four unicellular algae with different light-harvesting and xanthophyll-cycle pigments
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin",  "Violaxanthin", "Neoxanthin", "cis-B-carotene"}
    # biomass_mass = {"Chlorophyll-a": 5.962, "Chlorophyll-b": 4.3195, "B,B-carotene":0.9732, "Lutein": 2.6201, "Zeaxanthin": 0.1909,
    #                 "Violaxanthin": 0.4540, "Neoxanthin": 0.4970, "cis-B-carotene":0.9732, "Antheraxanthin": 0.4522}

    #n gaditana
    # pigments = {"Chlorophyll-a", "B,B-carotene", "Zeaxanthin", "Violaxanthin", "Antheraxanthin"}
    # biomass_mass = {"Chlorophyll-a": 14.418,  "B,B-carotene": 0.173, "Zeaxanthin": 0.174, "Violaxanthin": 1.652, "Antheraxanthin": 0.388}

    # p lutheri
    # pigments = {"Chlorophyll-a", "chlorophyll c1", "chlorophyll c2", "B,B-carotene", "Diatoxanthin", "Fucoxanthin", "Diadinoxanthin"}
    # biomass_mass = {"Chlorophyll-a": 14.328, "chlorophyll c1": 1.453, "chlorophyll c2": 1.453, "B,B-carotene": 3.992, "Diatoxanthin": 6.328, "Fucoxanthin": 8.546, "Diadinoxanthin": 4.815}

    # df = load_data(["../data/absorption/mmc2.txt", "../data/absorption/mmc3.txt"])  # from https://doi.org/10.1016%2Fj.dib.2019.103875
    # # # df['alpha-carotene'] = df['B,B-carotene']  # https://envi.geoscene.cn/help/Subsystems/envi/Content/Vegetation%20Analysis/PlantFoliage.htm
    # df['cis-B-carotene'] = df['B,B-carotene']
    # # df['chlorophyll c1'] = df['chlorophyll c2']
    # total_area = 0
    # area_by_carotene = {}
    # for wave_length, wave_range in WAVE_LENGTHS.items():
    #     area, total =  integrate_for_wave_length(df, wave_range, set(biomass_mass.keys()), biomass_mass)
    #     total_area += total
    #     #sort area by key
    #     area = {key: val for key, val in sorted(area.items(), key=lambda item: item[0])}
    #     print(wave_length, area, total)
    #     for key, val in area.items():
    #         if key not in area_by_carotene:
    #             area_by_carotene[key] = 0
    #         area_by_carotene[key] += val
    # print(total_area)
    # print(area_by_carotene)
