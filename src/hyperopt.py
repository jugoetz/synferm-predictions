from copy import deepcopy

from ax.service.managed_loop import optimize

from src.cross_validation import cross_validate, cross_validate_sklearn
from src.util.io import get_hparam_bounds


def update_hyperparameters(hparams, update_params):
    """
    This function updates the hparams dict with the parameters as understood by Ax.
    Can be used with an input parametrization for a trail or with the best parameters.

    Args:
        hparams (dict): Original hyperparameters
        update_params (dict): Input parameters used in or best hyperparameters returned from Ax's optimize()

    Returns:
        dict: Updated hyperparameters
    """
    hparams_local = deepcopy(hparams)

    if hparams_local["name"] in ["D-MPNN", "GCN", "GraphAgnostic", "FFN"]:
        if hparams_local["name"] in ["D-MPNN", "GCN"]:
            hparams_local["encoder"]["hidden_size"] = update_params.get(
                "encoder_hidden_size", hparams["encoder"]["hidden_size"]
            )
            hparams_local["encoder"]["depth"] = update_params.get(
                "encoder_depth", hparams["encoder"]["depth"]
            )
            hparams_local["encoder"]["dropout_ratio"] = update_params.get(
                "dropout", hparams["encoder"]["dropout_ratio"]
            )
            hparams_local["encoder"]["aggregation"] = update_params.get(
                "aggregation", hparams["encoder"]["aggregation"]
            )
        hparams_local["decoder"]["hidden_size"] = update_params.get(
            "decoder_hidden_size", hparams["decoder"]["hidden_size"]
        )
        hparams_local["decoder"]["depth"] = update_params.get(
            "decoder_depth", hparams["decoder"]["depth"]
        )
        hparams_local["decoder"]["dropout_ratio"] = update_params.get(
            "dropout", hparams["decoder"]["dropout_ratio"]
        )
        hparams_local["optimizer"]["lr"] = update_params.get(
            "learning_rate", hparams["optimizer"]["lr"]
        )
        hparams_local["optimizer"]["lr_scheduler"]["lr_min"] = (
            hparams_local["optimizer"]["lr"] / 10
        )  # we enforce this in the optimization

    elif hparams_local["name"] == "LogisticRegression":
        hparams_local["decoder"]["C"] = update_params.get("C", hparams["decoder"]["C"])
        hparams_local["decoder"]["penalty"] = update_params.get(
            "penalty", hparams["decoder"]["penalty"]
        )

    elif hparams_local["name"] == "XGB":
        hparams_local["decoder"]["reg_lambda"] = update_params.get(
            "reg_lambda", hparams["decoder"]["reg_lambda"]
        )

        hparams_local["decoder"]["reg_alpha"] = update_params.get(
            "reg_alpha", hparams["decoder"]["reg_alpha"]
        )

        hparams_local["decoder"]["max_depth"] = update_params.get(
            "max_depth", hparams["decoder"]["max_depth"]
        )

        hparams_local["decoder"]["learning_rate"] = update_params.get(
            "learning_rate", hparams["decoder"]["learning_rate"]
        )

        hparams_local["decoder"]["gamma"] = update_params.get(
            "gamma", hparams["decoder"]["gamma"]
        )

        hparams_local["decoder"]["colsample_bytree"] = update_params.get(
            "colsample_bytree", hparams["decoder"]["colsample_bytree"]
        )

    else:
        raise ValueError("Unknown decoder type")
    return hparams_local


def optimize_hyperparameters_bayes(
    data, hparams, hparam_config_path, cv_parameters, n_iter=50
):
    def objective_function(parameterization):
        """
        Wrapper around cross_validate() for Bayesian optimization with Ax.

        Ax requires that this function takes only one argument, parameterization.
        Consequently, variables `hparams` and `data` need to be defined in outside the function.

        Args:
            parameterization (dict): Parameters to be probed

        Returns:
            tuple: Score and STD for the probed parameters
        """
        hparams_local = update_hyperparameters(hparams, parameterization)

        metrics = cross_validate(
            hparams=hparams_local,
            data=data,
            **cv_parameters,
            return_fold_metrics=False,
        )
        # note that metrics returned from cross_validate are tensors, which ax cannot handle
        # thus we convert to float
        return {k: v.item() for k, v in metrics.items()}

    def objective_function_sklearn(parameterization):
        """
        Wrapper around cross_validate_sklearn() for Bayesian optimization with Ax.

        Ax requires that this function takes only one argument, parameterization.
        Consequently, variables `hparams` and `data` need to be defined in outside the function.

        Args:
            parameterization (dict): Parameters to be probed

        Returns:
            tuple: Score and STD for the probed parameters
        """
        hparams_local = update_hyperparameters(hparams, parameterization)

        metrics = cross_validate_sklearn(
            data=data,
            hparams=hparams_local,
            **cv_parameters,
            return_fold_metrics=False,
        )
        # note that metrics returned from cross_validate_predefined are tensors, which ax cannot handle
        # thus we convert to float
        return metrics

    bounds = get_hparam_bounds(hparam_config_path)

    if hparams["name"] in ["D-MPNN", "GCN", "GraphAgnostic", "FFN"]:
        obj_func = objective_function
    else:
        obj_func = objective_function_sklearn

    best_parameters, values, experiment, model = optimize(
        parameters=bounds,
        evaluation_function=obj_func,
        objective_name="val/loss_mean",
        total_trials=n_iter,
        minimize=True,
    )

    best_hparams = update_hyperparameters(hparams, best_parameters)

    return best_hparams
