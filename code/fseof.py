from os.path import join
import sys

from gsmmutils.model import MyModel, FSEOF

def main():
    model = MyModel("../data/model_with_media.xml", 'e_Biomass__cytop')
    model.set_prism_reaction("PRISM_white_LED__extr")
    constraint = model.problem.Constraint(
        model.reactions.R09503_hn438__lum.flux_expression + model.reactions.R09503_hn673__lum.flux_expression,
        lb=0,
        ub=199.44)
    model.add_cons_vars(constraint)
    fseof = FSEOF(model, targets=["C02094__chlo", "C08601__chlo"], workdir=join("../results", "fseof"))
    fseof.run()


if __name__ == '__main__':
    main()
