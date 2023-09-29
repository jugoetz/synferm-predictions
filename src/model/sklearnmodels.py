from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier, XGBModel


def load_sklearn_model(hparams):
    if hparams["name"] == "LogisticRegression":
        # for LogReg, we have to filter the hparams as the constructor does not accept **kwargs
        model = LogisticRegression(
            **{
                k: v
                for k, v in hparams["decoder"].items()
                if k in LogisticRegression.__init__.__code__.co_varnames
            }
        )
    elif hparams["name"] == "XGB":
        # for XGBClassifier, it largely inherits __init__ from XGBModel, so we filter for the latter's kwargs
        model = XGBClassifier(
            eval_metric="logloss",  # suppresses a warning about changing the default value
            use_label_encoder=False,  # suppresses a deprecation warning
            **{
                k: v
                for k, v in hparams["decoder"].items()
                if k in XGBModel.__init__.__code__.co_varnames
            },
        )
    else:
        raise ValueError(f"Model type {hparams['name']} not supported.")

    if hparams["training"]["task"] == "multilabel":
        # for multilabel, we need to wrap the model (n.b. this will train independent models for each label)
        model = MultiOutputClassifier(model)

    return model
