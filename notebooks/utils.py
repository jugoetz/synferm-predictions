import seaborn as sns
import wandb


class BodeColorPalette:
    def __init__(self):
        super().__init__()

    # colors as rgb values
    _blues = [
        (22 / 256, 22 / 256, 56 / 256),
        (74 / 256, 74 / 256, 104 / 256),
        (125 / 256, 126 / 256, 151 / 256),
        (177 / 256, 177 / 256, 191 / 256),
        (228 / 256, 228 / 256, 241 / 256),
    ]

    _oranges = [
        (147 / 256, 60 / 256, 16 / 256),
        (175 / 256, 85 / 256, 38 / 256),
        (188 / 256, 117 / 256, 77 / 256),
        (202 / 256, 149 / 256, 112 / 256),
        (227 / 256, 210 / 256, 183 / 256),
        (241 / 256, 241 / 256, 220 / 256),
    ]
    _blacks = [(0.0, 0.0, 0.0)]

    _all_colors = _blacks + _blues + _oranges[::-1]

    # primary colors
    blue = _blues[0]
    orange = _oranges[0]
    black = _blacks[0]

    # color palettes
    full = sns.color_palette(palette=_all_colors, n_colors=12)
    blues = sns.color_palette(palette=_blues, n_colors=5)
    oranges = sns.color_palette(palette=_oranges, n_colors=6)
    dark = sns.color_palette(palette=[_blues[0], _oranges[0], _blacks[0]], n_colors=3)
    light = sns.color_palette(palette=[_blues[4], _oranges[5]], n_colors=2)


def get_runs_as_list(project="jugoetz/synferm-predictions", filters={}):
    api = wandb.Api()
    runs = api.runs(project, filters=filters)
    summary_list, config_list, name_list, tag_list = [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        tag_list.append(tuple(run.tags))  # tuple b/c we will need to hash it later

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    return summary_list, config_list, tag_list, name_list
