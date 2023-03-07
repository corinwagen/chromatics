import numpy as np
import re, copy, math

import lmfit
import lmfit.models as models

import matplotlib.pyplot as plt

from scipy.special import erf, wofz
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# some preset values for numerical integration of peaks (e.g. to get areas, widths).
# GRID_WINDOW is how far on either side of a peak to integrate
# GRID_NUM refers to how many integration points to place within the window
GRID_NUM = int(1e4)
GRID_WINDOW = 5

class Peak():
    """
    This object represents a single chromatographic peak.

    Attributes:
        model (lmfit.Model): ``lmfit.Model`` representing the peak
        params (dict): dictionary of params, ``center`` and ``amplitude`` must be included
        param_settings (dict): various settings for the ``lmfit`` optimization
        label (str): optional, a label to accompany this peak
        prefix (str): optional, distinguishes the peak from other peaks
    """
    def __init__(self, prefix=None, model=None, params=None, param_settings=None, label=None, **kwargs):
        """
        Passing ``max_X`` in ``kwargs`` will limit the range of ``center``.
        """

        assert isinstance(model, lmfit.Model)
        self.model = model

        assert isinstance(prefix, str), "prefix must be string"
        self.prefix = prefix

        assert isinstance(params, dict), "params must be dict"

        if not isinstance(self, BaselinePeak):
            assert "center" in params, "center must be in params"
            assert "amplitude" in params, "amplitude must be in params"

        self.params = params

        assert isinstance(param_settings, dict), "param_settings must be dict"
        self.param_settings = param_settings

        if "max_X" in kwargs:
            self.param_settings["center"] = {"min": 0, "max": kwargs["max_X"]}

        assert isinstance(label, (type(None), str))
        self.label = label

    def __str__(self):
        return f"{self.__class__.__name__}(center={self.params['center']:.2f}, label='{self.label}', prefix='{self.prefix}')"

    def __repr__(self):
        return f"chromatics.peaks.{self.__class__.__name__}(params={self.params}, param_settings={self.param_settings}, label='{self.label}', prefix='{self.prefix}')"

    def area(self):
        """ Evaluate peak area via trapezoidal numerical integration. """
        model, params, _ = self.export()
        grid = np.linspace(params["center"] - GRID_WINDOW, params["center"] + GRID_WINDOW, GRID_NUM)
        pred_Y = model.eval(params, x=grid)
        return np.trapz(pred_Y, x=grid)

    def height(self):
        """ Evaluate maximum height of the peak. """
        model, params, _ = self.export()
        grid = np.linspace(params["center"] - GRID_WINDOW, params["center"] + GRID_WINDOW, GRID_NUM)
        pred_Y = model.eval(params, x=grid)
        return np.max(pred_Y)

    def width(self, threshold=0.1, return_bounds=False):
        """
        Compute width of the peak, estimated as the distance where the peak is greater than ``threshold * self.height()``.
        This may not work well if peak is multimodal.

        (``threshold`` of 0.5 corresponds to full-width at half-max)

        This definition is necessary because many peak models, e.g. Gaussian functions, never reach zero.
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

    def set_center(self, new_center):
        """ Update peak center. """
        assert isinstance(new_center, (int, float)), f"{new_center} is not numeric"
        if np.isnan(new_center):
            raise ValueError(f"new_center is nan!")
        self.params["center"] = new_center

    def set_amplitude(self, new_amplitude):
        """ Update peak amplitude. """
        assert isinstance(new_amplitude, (int, float)), f"{new_amplitude} is not numeric"
        if np.isnan(new_amplitude):
            raise ValueError(f"new_amplitude is nan!")
        self.params["amplitude"] = new_amplitude

    def set_sigma(self, new_sigma):
        """ Update sigma. """
        assert isinstance(new_sigma, (int, float)), f"{new_sigma} is not numeric"
        if np.isnan(new_sigma):
            raise ValueError(f"new_sigma is nan!")
        self.params["sigma"] = new_sigma

    def set_gamma(self, new_gamma):
        """ Update gamma. """
        assert isinstance(new_gamma, (int, float)), f"{new_gamma} is not numeric"
        if np.isnan(new_gamma):
            raise ValueError(f"new_gamma is nan!")
        self.params["gamma"] = new_gamma

    def build_model(self, include_prefix=True):
        """ Return ``lmfit.Model`` object corresponding to ``self``. """
        if include_prefix:
            self.model.prefix = self.prefix
        return self.model

    def build_parameters(self, params=None, amplitude_only=False, include_prefix=True):
        """
        Return ``lmfit.Parameters`` object corresponding to ``self``.

        If ``params`` has been passed, parameters from the current peak will be added to the existing object.
        """
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
        Return tuple of ``lmfit.Model``, ``lmfit.Parameters``, and label corresponding to self.
        """
        return self.build_model(include_prefix=False), self.build_parameters(include_prefix=False), self.label

class BaselinePeak(Peak):
    def build_parameters(self, freeze_baseline=False, **kwargs):
        params = super().build_parameters(**kwargs)
        if freeze_baseline:
            for key, value in self.params.items():
                params[f"{self.prefix}{key}"].set(vary=False)
        return params

# The rest of this file is devoted to specific instantiations of the above generic Peak class.

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
            }


        super().__init__(model=model, **kwargs)

class ExponentialGaussianPeak(Peak):
    def __init__(self, **kwargs):
        model = models.ExponentialGaussianModel()

        if "params" not in kwargs:
            kwargs["params"] = {
                "amplitude": 0.1,
                "center": 0,
                "sigma": 0.1,
                "gamma": 10,
            }

        if "param_settings" not in kwargs:
            kwargs["param_settings"] = {
                "gamma": {"min": 0, "max": 100},
                "amplitude": {"min": 0},
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
            }

        super().__init__(model=model, **kwargs)


