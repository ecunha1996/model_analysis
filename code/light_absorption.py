import numpy as np
import pandas as pd


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


if __name__ == '__main__':
    wave_lenghts = {"298": [281, 306],
                    "437": [406, 454],
                    "438": [378, 482],
                    "450": [417, 472],
                    "490": [451, 526],
                    "646": [608, 666],
                    "673": [659, 684],
                    "680": [662, 691]}
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin", "Antheraxanthin", "Violaxanthin", "alpha-carotene", "Neoxanthin", "cis-B-carotene"}
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin", "Antheraxanthin", "Violaxanthin",  "Neoxanthin"}  # Fluorescence quenching in four unicellular algae with different light-harvesting and xanthophyll-cycle pigments
    # pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin",  "Violaxanthin", "Neoxanthin", "cis-B-carotene"}
    # biomass_mass = {"Chlorophyll-a": 5.962, "Chlorophyll-b": 4.3195, "B,B-carotene":0.9732, "Lutein": 2.6201, "Zeaxanthin": 0.1909, "Violaxanthin": 0.4540, "Neoxanthin": 0.4970, "cis-B-carotene":0.9732}

    #n gaditana
    # pigments = {"Chlorophyll-a", "B,B-carotene", "Zeaxanthin", "Violaxanthin"}
    # biomass_mass = {"Chlorophyll-a": 14.758,  "B,B-carotene": 0.177, "Zeaxanthin": 0.178, "Violaxanthin": 1.691}

    # p lutheri
    pigments = {"Chlorophyll-a", "chlorophyll c1", "chlorophyll c2", "B,B-carotene", "Diatoxanthin", "Fucoxanthin", "Diadinoxanthin"}
    biomass_mass = {"Chlorophyll-a": 14.328, "chlorophyll c1": 1.453, "chlorophyll c2": 1.453, "B,B-carotene": 3.992, "Diatoxanthin": 6.328, "Fucoxanthin": 8.546, "Diadinoxanthin": 4.815}

    df = load_data(["../data/absorption/mmc2.txt", "../data/absorption/mmc3.txt"])  # from https://doi.org/10.1016%2Fj.dib.2019.103875
    # df['alpha-carotene'] = df['B,B-carotene']  # https://envi.geoscene.cn/help/Subsystems/envi/Content/Vegetation%20Analysis/PlantFoliage.htm
    # df['cis-B-carotene'] = df['B,B-carotene']
    df['chlorophyll c1'] = df['chlorophyll c2']
    total_area = 0
    area_by_carotene = {}
    for wave_length, wave_range in wave_lenghts.items():
        area, total =  integrate_for_wave_length(df, wave_range, pigments, biomass_mass)
        total_area += total
        #sort area by key
        area = {key: val for key, val in sorted(area.items(), key=lambda item: item[0])}
        print(wave_length, area, total)
        for key, val in area.items():
            if key not in area_by_carotene:
                area_by_carotene[key] = 0
            area_by_carotene[key] += val
    print(total_area)
    print(area_by_carotene)