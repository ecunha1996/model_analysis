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


def integrate_for_wave_length(df: pd.DataFrame, wave_range: list, pigments: set[str]) -> pd.DataFrame:
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
        area_of_pigments[pigment] = abs(np.trapz(df[pigment], df.index))
    # normalize
    total_area = sum(area_of_pigments.values())
    if total_area > 0:
        for pigment in pigments:
            area_of_pigments[pigment] = round(area_of_pigments[pigment] / total_area, 4)
    return area_of_pigments


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
    pigments = {"Chlorophyll-a", "Chlorophyll-b", "B,B-carotene", "Lutein", "Zeaxanthin", "Antheraxanthin", "Violaxanthin",  "Neoxanthin"}  # Fluorescence quenching in four unicellular algae with different light-harvesting and xanthophyll-cycle pigments
    df = load_data(["../data/absorption/mmc2.txt", "../data/absorption/mmc3.txt"])  # from https://doi.org/10.1016%2Fj.dib.2019.103875
    # df['alpha-carotene'] = df['B,B-carotene']  # https://envi.geoscene.cn/help/Subsystems/envi/Content/Vegetation%20Analysis/PlantFoliage.htm
    # df['cis-B-carotene'] = df['B,B-carotene']
    for wave_length, wave_range in wave_lenghts.items():
        print(wave_length, integrate_for_wave_length(df, wave_range, pigments))
