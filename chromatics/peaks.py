import numpy as np
import re, copy, math

import lmfit
import lmfit.models as models

import matplotlib.pyplot as plt

from scipy.special import erf, wofz
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

GRID_NUM = int(1e4)
GRID_WINDOW = 5

class Peak():
    """
    "center" and "amplitude" are assumed for ``params``.


    Attributes:
        model (lmfit.Model):
        params (dict):
        param_settings (dict):
        label (str):
        prefix (str):
    """
    def __init__(self, prefix=None, model=None, params=None, param_settings=None, label=None, **kwargs):
        """
        Passing ``max_X`` will limit the range of ``center``.
        """

        assert isinstance(model, lmfit.Model)
        self.model = model

        assert isinstance(prefix, str), "prefix must be string"
        self.prefix = prefix

        assert isinstance(params, dict), "params must be dict"
        self.params = params

        assert isinstance(param_settings, dict), "param_settings must be dict"
        self.param_settings = param_settings

        if "max_X" in kwargs:
            self.param_settings["center"] = {"min": 0, "max": kwargs["max_X"]}

        assert isinstance(label, (type(None), str))
        self.label = label

    def area(self):
        model, params, _ = self.export()
        grid = np.linspace(params["center"] - GRID_WINDOW, params["center"] + GRID_WINDOW, GRID_NUM)
        pred_Y = model.eval(params, x=grid)
        return np.trapz(pred_Y, x=grid)

    def unit_area(self):
        old_amplitude = self.params["amplitude"]
        self.params["amplitude"] = 1
        unit_area = self.area()
        self.params["amplitude"] - old_amplitude
        return unit_area

    def set_center(self, new_center):
        self.params["center"] = new_center

    def height(self):
        model, params, _ = self.export()
        grid = np.linspace(params["center"] - GRID_WINDOW, params["center"] + GRID_WINDOW, GRID_NUM)
        pred_Y = model.eval(params, x=grid)
        return np.max(pred_Y)

    def width(self, threshold=0.1, return_bounds=False):
        """
        May not work well if peak is multimodal.
        """
        model, params, _ = self.export()
        grid = np.linspace(params["center"] - GRID_WINDOW, params["center"] + GRID_WINDOW, GRID_NUM)
        pred_Y = model.eval(params, x=grid)

        height = np.max(pred_Y)
        cutoff = height * threshold

        if height == 0:
            if return_bounds:
                return None, None # this peak is not actually a peak.
            else:
                return None

        delta = pred_Y - cutoff
        if len(np.argwhere(delta > 0)):
            try:
                x1 = grid[min(np.argwhere(delta > 0))][0]
                x2 = grid[max(np.argwhere(delta > 0))][0]
                if return_bounds:
                    return x1, x2
                else:
                    return x2 - x1
            except Exception as e:
                #### usually peak is messed up
                print(f"width calculation error: {e}\n{delta}")
                if return_bounds:
                    return None, None
                else:
                    return None
        else:
            # peak too tiny. return tiny value but probably meaningless.
            return np.finfo(np.float64).eps

    def set_height(self, new_height):
        assert isinstance(new_height, (int, float)), f"{new_height} is not numeric"
        if np.isnan(new_height):
            raise ValueError(f"new_height is nan!")
        current_height = self.height()
        if current_height > 0:
            self.params["amplitude"] *= new_height/current_height
        else:
            self.params["amplitude"] = new_height # give up on proper scaling

    def set_sigma(self, new_sigma):
        assert isinstance(new_sigma, (int, float)), f"{new_sigma} is not numeric"
        if np.isnan(new_sigma):
            raise ValueError(f"new_sigma is nan!")
        self.params["sigma"] = new_sigma

    def set_gamma(self, new_gamma):
        assert isinstance(new_gamma, (int, float)), f"{new_gamma} is not numeric"
        if np.isnan(new_gamma):
            raise ValueError(f"new_gamma is nan!")
        self.params["gamma"] = new_gamma

    def build_model(self, include_prefix=True):
        if include_prefix:
            self.model.prefix = self.prefix
        return self.model

    def build_parameters(self, params=None, amplitude_only=False, include_prefix=True):
        if params is None:
            params = lmfit.Parameters()
        else:
            assert isinstance(params, lmfit.Parameters)

        for key, value in self.params.items():
            kwargs = dict()

            for param, constraints in self.param_settings.items():
                if re.search(param, key):
                    kwargs.update(constraints)

            if amplitude_only and not re.search(key, "amplitude"):
                kwargs["vary"] = False

            if include_prefix:
                params.add(f"{self.prefix}{key}", value, **kwargs)
            else:
                params.add(f"{key}", value, **kwargs)
        return params

    def export(self):
        """
        Return tuple of lmfit.Model, lmfit.Parameters, and label.
        """
        return self.build_model(include_prefix=False), self.build_parameters(include_prefix=False), self.label

    def __str__(self):
        return f"{self.__class__.__name__}(center={self.params['center']:.2f}, label='{self.label}', prefix='{self.prefix}')"

    def __repr__(self):
        return f"chromatics.peaks.{self.__class__.__name__}(params={self.params}, param_settings={self.param_settings}, label='{self.label}', prefix='{self.prefix}')"

class BaselinePeak(Peak):
    def build_parameters(self, freeze_baseline=False, **kwargs):
        params = super().build_parameters(**kwargs)
        if freeze_baseline:
            for key, value in self.params.items():
                params[f"{self.prefix}{key}"].set(vary=False)
        return params

def frankenstein(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.0):
    """
    Piecewise function that works well for modelling tailed SFC peaks.
    """
    tiny = np.finfo(np.float64).eps

    def sg(x, amplitude=amplitude, center=center, sigma=sigma, gamma=gamma):
        return ((amplitude/(max(tiny, np.sqrt(2*np.pi)*sigma)))
            * np.exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))

    def sv(x, amplitude=amplitude, center=center, sigma=sigma, gamma=gamma):
        if gamma is None:
            gamma = sigma
        z = (x-center + 1j*gamma) / max(tiny, (sigma*np.sqrt(2.0)))
        return amplitude*wofz(z).real / max(tiny, (sigma*np.sqrt(2*np.pi)))

    voigt_multiplier = sg(center) / max(tiny, sv(center))
    return np.piecewise(
        x,
        [x < center, x >= center],
        [sg, lambda x: voigt_multiplier * sv(x)]
    )


class FrankensteinPeak(Peak):
    def __init__(self, **kwargs):
        model = lmfit.Model(frankenstein)

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.01,
                "gamma": 0.001,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 0.01},
                "amplitude": {"min": 0, "max": 100},
                "sigma": {"min": 0, "max": 1},
#                "sigma": {"min": 0.005},
            }

        super().__init__(model=model, **kwargs)


class SkewedGaussianPeak(Peak):
    def __init__(self, **kwargs):
        model = models.SkewedGaussianModel()

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.01,
                "gamma": 0.001,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 100},
                "amplitude": {"min": 0},
#                "sigma": {"min": 0.005},
            }


        super().__init__(model=model, **kwargs)


class ConstantBaseline(BaselinePeak):
    def __init__(self, **kwargs):
        model = models.ConstantModel()
        if "params" not in kwargs:
            kwargs["params"] = {"c": 0}
        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {}
        super().__init__(model=model, **kwargs)

class GaussianPeak(Peak):
    def __init__(self, **kwargs):
        model = models.GaussianModel()

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.01,
                "gamma": 0.001,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 100},
                "amplitude": {"min": 0},
#                "sigma": {"min": 0.005},
            }

        super().__init__(model=model, **kwargs)

class VoigtPeak(Peak):
    def __init__(self, **kwargs):
        model = models.VoigtModel()

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.01,
                "gamma": 0.001,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 100},
                "amplitude": {"min": 0},
#                "sigma": {"min": 0.005},
            }

        super().__init__(model=model, **kwargs)

class SkewedVoigtPeak(Peak):
    def __init__(self, **kwargs):
        model = models.SkewedVoigtModel()

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.01,
                "gamma": 0.001,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 100},
                "amplitude": {"min": 0},
#                "sigma": {"min": 0.005},
            }

        super().__init__(model=model, **kwargs)


