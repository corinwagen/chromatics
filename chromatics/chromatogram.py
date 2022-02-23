import numpy as np
import re, copy, math, sys, os

import lmfit
import lmfit.models as models

import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from sklearn.metrics import mean_squared_error, r2_score

import chromatics

class Chromatogram():
    """
    Object which represents data and models for a single chromatographic run, perhaps with multiple response variables (wavelengths/masses).

    Can be saved using ``self.save(filename)`` and loaded using ``Chromatogram.load(filename)``. Can also be initiated from ``.csv`` files, perhaps from OpenChrom.

    Attributes:
        X (np.ndarray): X values (1D), like times
        Y (np.ndarray): Y values (1D or 2D), like UV absorbance or ion count
        Y_labels (np.ndarray): labels associated with 2D Y values, like wavelength or m/z
        scale_factor (float): multiply by this to regain original Y data! (Y data usually scaled to [0, 1])
        peaks (list of chromatics.Peak): the component peaks to be fit to the model
        baseline_models (list of chromatics.BaselinePeak): the baseline model to be fit to the model
        freeze_baseline (Bool): whether to optimize baseline or not.
    """

    def __init__(self, X, Y, Y_labels=None, peaks=2, scale_factor=1, baseline_corrections=True, model=chromatics.FrankensteinPeak, freeze_baseline=False, frozen=False, **kwargs):
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        # is the 2nd dimension spurious??
        if len(Y) == 1 and Y.ndim == 2 and Y_labels is None:
            Y = Y[0]

        # no negatives!
        if np.min(Y) < 0:
            Y += -1 * np.min(Y)

        if Y.ndim == 1:
            assert len(X) == len(Y)
        elif Y.ndim == 2:
            assert isinstance(Y_labels, np.ndarray)
            assert len(X) == len(Y[0])
            assert len(Y_labels) == len(Y)
        else:
            raise ValueError("Y must have 1 or 2 dimensions")

        self.X = X
        self.Y = Y
        self.Y_labels = Y_labels

        if isinstance(peaks, list):
            if all(isinstance(peak, chromatics.Peak) for peak in peaks):
                self.peaks = peaks
            elif all(isinstance(peak, str) for peak in peaks):
                self.peaks = [eval(p) for p in peaks]
            else:
                raise ValueError(f"can't make sense of peaks - {peaks}")
        elif isinstance(peaks, int):
            self.peaks = []
            self.add_peaks(peaks, model=model)
        else:
            self.peaks = []

        if isinstance(baseline_corrections, list):
            if all(isinstance(peak, chromatics.Peak) for peak in baseline_corrections):
                self.baseline_corrections = baseline_corrections
            elif all(isinstance(peak, str) for peak in baseline_corrections):
                self.baseline_corrections = [eval(p) for p in baseline_corrections]
            else:
                raise ValueError(f"can't make sense of baseline corrections - {baseline_corrections}")
        elif baseline_corrections:
            self.baseline_corrections = []
            self.add_baseline()
        else:
            self.baseline_corrections = []

        assert isinstance(scale_factor, (int, float))
        self.scale_factor = scale_factor
        if scale_factor == 1:
            self.scale_Y()

        self.freeze_baseline = freeze_baseline
        self.frozen = frozen

    def __str__(self):
        return f"Chromatogram({len(self.X)} time points, peaks={self.peaks})"

    def __repr__(self):
        with np.printoptions(threshold=sys.maxsize):
            return f"chromatics.chromatogram.{self.__class__.__name__}(X={repr(self.X)}, Y={repr(self.Y)}, Y_labels={repr(self.Y_labels)}, peaks={[repr(p) for p in self.peaks]}, baseline_corrections={[repr(b) for b in self.baseline_corrections]}, scale_factor={self.scale_factor}, freeze_baseline={self.freeze_baseline}, frozen={self.frozen})"

    def save(self, filename, label=None):
        """ Save full Chromatogram object to a file."""
        save_obj = copy.deepcopy(self)

        if label is not None:
            save_idx = np.where(self.Y_labels == label)
            save_obj.Y = save_obj.Y[save_idx]
            save_obj.Y_labels = save_obj.Y_labels[save_idx]

        with open(filename, "w") as savefile:
            savefile.write("# chsavefile\n")
            savefile.write(repr(save_obj))

    @classmethod
    def load(cls, filename):
        """ Load full chromatogram object from a file. """
        from numpy import array # needed for eval() to work on numpy arrays. blame numpy, not me!

        assert os.path.exists(filename), f"can't load chromatogram from {filename} -- no file!"
        try:
            with open(filename, "r") as savefile:
                return eval(savefile.read())
        except Exception as e:
            raise ValueError(f"error reading file {filename}!\n{e}")

    def Y_from_label(self, label):
        """
        Given label ``label``, returns a 1D ``np.ndarray`` corresponding to that label.
        """
        if self.Y_labels is None or label is None:
            return self.Y
        assert label in self.Y_labels, f"label {label} not found in self.Y_labels"
        return self.Y[np.where(self.Y_labels == label)][0]

    def add_peaks(self, num, model):
        """
        Add ``num`` peaks to ``self.peaks``, populated with default parameters.
        """
        assert isinstance(num, int)
        if num == 0:
            return

        max_X = np.max(self.X)

        for idx in range(len(self.peaks), len(self.peaks) + num):
            new_peak = model(prefix=f"p{idx}_", max_X=max_X)
            assert isinstance(new_peak, chromatics.Peak)
            self.peaks.append(new_peak)

        # make sure indices didn't get scrambled at any point.
        for idx, peak in enumerate(self.peaks):
            if peak.prefix != f"p{idx}_":
                peak.prefix = f"p{idx}_"

    def add_baseline(self, num=1, model=chromatics.ConstantBaseline):
        """
        Add ``num`` baseline models to ``self.baseline``, populated with default parameters.
        """
        assert isinstance(num, int)
        if num == 0:
            return

        for idx in range(len(self.baseline_corrections), len(self.baseline_corrections) + num):
            new_peak = model(prefix=f"b{idx}_")
            assert isinstance(new_peak, chromatics.BaselinePeak)
            self.baseline_corrections.append(new_peak)

    def detect_peaks(self, label=None, **kwargs):
        """
        Auto-detect indices of peaks from ``self.Y``.
        """
        Y = self.Y_from_label(label)
        peaks, params = find_peaks(Y, height=np.max(Y)/10, **kwargs)
        if "height" in params:
            ordered_peaks = np.argsort(params['height']) # biggest last
            return peaks[ordered_peaks]
        elif "prominence" in params:
            ordered_peaks = np.argsort(params['prominences']) # biggest last
            return peaks[ordered_peaks]
        else:
            return peaks

    def build_parameters(self, params=None, prefix="", baseline_only=False, amplitude_only=False, same_sigma_gamma=True):
        """
        Create ``lmfit.Parameters`` object from ``self.params``.

        Args:
            params (lmfit.Parameters or None): existing parameter object to append to (if desired)
            prefix (string): prefix for all parameters, if desired
            baseline_only (bool): only return parameters for baseline
            amplitude_only (bool): freeze all parameters but peak amplitudes, for multi-chromatogram fit.
            same_sigma_gamma (bool):
        """
        if params is None:
            params = lmfit.Parameters()
        else:
            assert isinstance(params, lmfit.Parameters)

        if not baseline_only:
            for peak in self.peaks:
                params = peak.build_parameters(params=params, amplitude_only=amplitude_only)

            if same_sigma_gamma:
                names = list(params.valuesdict().keys())

                sigmas = [p for p in names if "sigma" in p]
                gammas = [p for p in names if "gamma" in p]

                for sigma in sigmas[1:]:
                    params[sigma].set(expr=sigmas[0])

                for gamma in gammas[1:]:
                    params[gamma].set(expr=gammas[0])

        for peak in self.baseline_corrections:
            params = peak.build_parameters(params=params, freeze_baseline=self.freeze_baseline)

        return params

    def build_model(self, baseline_only=False):
        """
        Create ``lmfit.CompositeModel`` from ``self.peaks``.
        """
        model = None
        if not baseline_only:
            model = self.peaks[0].build_model()
            if len(self.peaks) > 1:
                for peak in self.peaks[1:]:
                    model = model + peak.build_model()
            for peak in self.baseline_corrections:
                model = model + peak.build_model()
        else:
            model = self.baseline_corrections[0].build_model()
            if len(self.baseline_corrections) > 1:
                for peak in self.baseline_corrections[1:]:
                    model = model + peak.build_model()
        return model

    def unpack_parameters(self, params):
        """
        Updates ``self.params`` from an ``lmfit.Parameters`` object.

        Args:
            params (lmfit.Parameters):
        """
        assert isinstance(params, lmfit.Parameters)
        param_dict = params.valuesdict()

        for peak in self.peaks + self.baseline_corrections:
            for parameter in peak.params.keys():
                prefix_parameter = f"{peak.prefix}{parameter}"
                if prefix_parameter in param_dict:
                    peak.params[parameter] = param_dict[prefix_parameter]

    def build_peaks(self, include_baseline=False):
        """
        From ``self.peaks`` and ``self.params`` and ``self.peak_labels``, build list of peaks with corresponding ``lmfit.Parameters`` object.

        Args:
            include_baseline (Bool): whether or not the baseline peaks should be included

        Returns:
            list of (``lmfit.Model``, ``lmfit.Parameters``, label str) tuples
        """
        peaks = self.peaks
        if include_baseline:
            peaks += self.baseline_corrections
        return [p.export() for p in peaks]

    def fit_peaks(
        self,
        label=None,
        opt=True,
        detect_params={"width": 5,},
        times=None,
        print_fit_report=False,
        prune=False,
        res_cutoff=1.0,
        amp_cutoff=1e-6,
        fit_args=None,
        time_window=None,
        same_sigma_gamma=True,
    ):
        """
        Fit ``self.peaks`` to ``self.X`` and ``self.Y``. This is the big method.

        Args:
            label (float):
            opt (bool): whether or not to actually perform optimization
            detect_params (dict): params to pass to ``self.detect_peaks()``
            times (list): list of retention times to use for peak centers
                or "keep" which means that the current times are good
                or None which means to autodetect
            print_fit_report(bool): whether or not to print ``lmfit`` fit report
            prune (bool): whether to check if fewer peaks are more appropriate with res_cutoff
            res_cutoff (float): R_s value below which peaks are viewed as "the same"
            amp_cutoff (float): amplitude value below which peaks are viewed as spurious/noise
            fit_args (dict): kwargs to pass to lmfit.Minimizer.minimize(), e.g. {"method": "differential_evolution"}
            time_window (2-tuple): times to consider for fitting.
            same_sigma_gamma (bool): whether sigma/gamma should be constrained to be the same or not.
        """
        Y = self.Y_from_label(label)

        if fit_args is None:
            fit_args = dict()

        relevant_idxs = slice(None)
        if time_window is not None:
            assert len(time_window) == 2, "time_window must be 2-tuple"
            if time_window[0] > time_window[1]:
                time_window[0], time_window[1] = time_window[1], time_window[0]

            relevant_idxs = slice(np.abs(self.X-time_window[0]).argmin(), np.abs(self.X-time_window[1]).argmin())

        if isinstance(times, list):
            assert len(times) == len(self.peaks)
            for peak, time in zip(self.peaks, times):
                idx = np.abs(self.X-time).argmin()

                # get sane defaults
                peak.set_center(time)
                peak.set_sigma(0.005)
                peak.set_gamma(0)

                try:
                    peak.set_height(np.mean(Y[idx-2:idx+2]))
                except: # can get nan on boundaries
                    peak.set_height(Y[idx])
        elif times == "keep":
            pass
        else:
            #### find the retention times
            peak_idxs = self.detect_peaks(label, **detect_params)
            if len(self.peaks) > len(peak_idxs):
                peak_idxs = np.tile(peak_idxs, 2)

            #### initialize guess models
            times = []
            for peak, center_idx in zip(self.peaks, peak_idxs):
                peak.params["center"] = self.X[center_idx]
                times.append(self.X[center_idx])

                # get sane defaults
                peak.set_sigma(0.005)
                peak.set_gamma(0)

                try:
                    peak.set_height(np.mean(Y[center_idx-2:center_idx+2]))
                except: # can get nan on boundaries
                    peak.set_height(Y[center_idx])

        params = self.build_parameters(same_sigma_gamma=same_sigma_gamma)
        if opt:
            model = self.build_model()

            def minimize_trace(params, model, Y, X):
                resid = Y - model.eval(params, x=X)
                return resid * (Y != 0) # exclude zeros from fit - too messy with the cutoff

            #### perform fit
            opt = lmfit.Minimizer(minimize_trace, params, fcn_args=(model, Y[relevant_idxs], self.X[relevant_idxs]))
            out = opt.minimize(**fit_args)
            self.unpack_parameters(out.params)

            if len(self.peaks) == 2:
                if prune:
                    # which one we're keeping
                    peak_idx = None

                    # check for None
                    if self.peaks[0].width() is None:
                        self.peaks = [self.peaks[0]]
                        peak_idx = 0
                    elif self.peaks[1].width() is None:
                        self.peaks = [self.peaks[1]]
                        peak_idx = 1
                    else:
                        assert isinstance(self.R(), float), f"despite having peaks {self.peaks}, R is not numeric but is instead {self.R()}"
                        if self.R() < res_cutoff:
                            # we can't really distinguish two peaks
                            self.peaks = [self.peaks[0]]
                            peak_idx = 0

                    if len(self.peaks) == 1:
                        if len(times) == 0:
                            times = [0]

                        return self.fit_peaks(
                            label=label,
                            opt=opt,
                            detect_params=detect_params,
                            times=times[peak_idx],
                            print_fit_report=print_fit_report,
                            res_cutoff=res_cutoff,
                            amp_cutoff=amp_cutoff
                        )

                if len(self.peaks) > 1 and prune:
                    for idx, peak in enumerate(self.peaks):
                        if peak.params["amplitude"] < amp_cutoff:
                            del self.peaks[idx]
                            times = times[:idx] + times[idx+1:]

                            return self.fit_peaks(
                                label=label,
                                opt=opt,
                                detect_params=detect_params,
                                times=times,
                                print_fit_report=print_fit_report,
                                res_cutoff=res_cutoff,
                                amp_cutoff=amp_cutoff
                            )

            if print_fit_report:
                print(lmfit.fit_report(out))
        else:
            self.unpack_parameters(params)

    def plot(self, ax, title=None, label=None, color=None, alpha=0.3, legend=True, scale=1.0):
        """
        Plots the chromatogram on a given ``matplotlib.pyplot.Axes`` object (``ax``).

        Args:
            title (str): plot tile
            label (str): Y label
            color (str): desired color of peaks (valid matplotlib color name)
            alpha (float): transparency of peak shading
            legend (bool): whether or not to have a legend
            scale (float): scale Y values by constant multiplier
        """
        if label is None and self.Y.ndim == 2:
            if self.Y.shape[0] == 1:
                label = self.Y_labels[0]
            else:
                raise ValueError("need a label to plot 2D Y values!")

        Y = self.Y_from_label(label) * self.scale_factor * scale

        colors = ["darkorange", "royalblue"]
        if color:
            colors = [color, color]
        else:
            areas = self.areas()
            if len(areas) == 2:
                if areas[0] < areas[1]:
                    colors = ["royalblue", "darkorange"]

        for idx, (peak, params, label) in enumerate(self.build_peaks()):
            pred_Y = peak.eval(params, x=self.X)
            ax.plot(self.X, pred_Y * self.scale_factor * scale, color=colors[idx])
            ax.fill_between(self.X, 0, pred_Y * self.scale_factor * scale, facecolor=colors[idx], alpha=alpha, label=label)

        ax.plot(self.X, self.build_model().eval(self.build_parameters(), x=self.X) * self.scale_factor * scale, color="red", alpha=0.4, label='fit')
        ax.plot(self.X, Y, "o", markersize=2, color="black", label="data", alpha=0.5)

        ax.set_ylim(top=(1.1*np.max(Y) - 0.1*np.min(Y)), bottom=(1.1*np.min(Y) - 0.1*np.max(Y))) # keep bounds from exploding on bad fit

        ax.set_xlabel("retention time (min)")
        ax.set_ylabel("counts")
        if title:
            ax.set_title(title, fontweight="bold")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        if legend:
            ax.legend(loc="best")

    def areas(self):
        """
        Returns an ordered list of normalized peak areas.
        """
        areas = [p.area() for p in self.peaks]
        return areas / np.sum(areas)

    def ee(self):
        """
        Returns the enantiomeric excess of the chromatogram, defined as A(peak #2) – A(peak #1).

        Only defined for 2-peak chromatograms.
        """
        areas = self.areas()
        if len(areas) == 2:
            if self.peaks[0].params["center"] < self.peaks[1].params["center"]:
                return 100 * (areas[1] - areas[0])
            else: # peak RT flipped
                return 100 * (areas[0] - areas[1])
        else:
            return None

    def er(self):
        """
        Returns the enantiomeric ratio of the chromatogram, defined as A(peak #1)/A(total).

        Only defined for 2-peak chromatograms.
        """
        areas = self.areas()
        if len(areas) == 2:
            return areas[0]*100.0/(areas[0]+areas[1])
        else:
            return None

    def R(self, threshold=0.5, peak_idxs=None):
        """
        Returns the chromatographic resolution of the chromatogram.
        A threshold of 0.5 is the USP standard.

        If the chromatogram has more than 2 peaks, the desired peaks can be specified with ``peak_idxs.``
        """

        assert isinstance(threshold, float), "threshold must be float"
        assert 0 < threshold < 1, "threshold must be between 0 and 1"

        peak1, peak2 = None, None

        if len(self.peaks) == 2:
            peak1 = self.peaks[0]
            peak2 = self.peaks[1]
        elif len(self.peaks) == 1:
            return None
        else:
            assert isinstance(peak_idxs, (list, tuple)), "peak_idxs must be defined if there aren't 2 peaks exactly"
            assert len(peak_idxs, 2), "R is only defined for 2 peaks"
            peak1 = self.peaks[peak_idxs[0]]
            peak1 = self.peaks[peak_idxs[1]]

        assert isinstance(peak1.params["center"], (int, float)), f"peak #1 has no center defined: params={peak1.params}"
        assert isinstance(peak2.params["center"], (int, float)), f"peak #2 has no center defined: params={peak2.params}"
        try:
            return 2 * abs(peak1.params["center"] - peak2.params["center"]) / (peak1.width(threshold=threshold) + peak2.width(threshold=threshold))
        except Exception as e:
            print(f"R calculation error - {e}")
            return None

    def total_area(self):
        """ Returns the total area of all peaks. """
        return sum([p.area() for p in self.peaks]) * self.scale_factor

    @classmethod
    def new_from_csv(cls, file, x_start, y_start, label_start=None, encoding="utf16", max_len=None, transpose=False, **kwargs):
        """
        Creates a new ``Chromatogram`` object from a ``.csv`` file. Assumption is that rows are masses and columns are times.
        If this is wrong, set ``transpose`` to ``True``.

        Args:
            file (str): path to file
            x_start (2-tuple): start of row containing X values (time, usually)
            y_start (2-tuple): start of row or block containing Y values
            label_start (2-tuple): start of column containing Y labels (for 2D Y-datasets; ``None`` for 1D Y-datasets)
            encoding (str): text encoding (e.g. "utf8")
            max_len (int): max number of X and Y values (ignore after this many)
            transpose (bool): transpose rows and columns
        """

        with open(file, encoding=encoding) as f:
            if not transpose:
                x_row = f.readlines()[x_start[0]].strip()
                X = x_row.split(",")[x_start[1]:max_len]
                X = np.array([float(re.sub(r'_min$', '', x)) for x in X])

                if label_start:
                    assert y_start[0] == label_start[0], "labels should match up to rows"
                    data = np.genfromtxt(file, delimiter=",", encoding=encoding, skip_header=y_start[0])

                    assert data.ndim == 2
                    Y = np.nan_to_num(data[0:, y_start[1]:max_len])
                    labels = data[0:, label_start[1]]

                    assert len(Y[0]) == len(X), f"can't have X and Y different lengths! ({len(X)} and {len(Y[1])}, respectively)"
                    assert len(Y) == len(labels), f"can't have labels and Y different lengths! ({len(labels)} and {len(Y[0])}, respectively)"
                    return cls(X, Y, Y_labels=labels, **kwargs)
                else:
                    data = np.genfromtxt(file, delimiter=",", encoding=encoding, skip_header=y_start[0])
                    if data.ndim == 2:
                        data = data[0]
                    Y = np.nan_to_num(data[y_start[1]:])

                    assert len(Y) == len(X), f"can't have X and Y different lengths! ({len(X)} and {len(Y)}, respectively)"
                    return cls(X, Y, **kwargs)
            else:
                # for this format we expect labels
                label_row = f.readlines()[label_start[0]].strip()
                labels = label_row.split(",")[label_start[1]:max_len]
                labels = np.array([float(l) for l in labels])

                assert x_start[0] == y_start[0], "labels should match up to rows"
                assert label_start[1] == y_start[1], "labels should match up to rows"

                data = np.genfromtxt(file, delimiter=",", encoding=encoding, skip_header=y_start[0])
                assert data.ndim == 2
                Y = np.nan_to_num(data[0:, y_start[1]:max_len]).T
                X = data[0:, x_start[0]]

                assert len(Y[0]) == len(X), f"can't have X and Y different lengths! ({len(X)} and {len(Y[1])}, respectively)"
                assert len(Y) == len(labels), f"can't have labels and Y different lengths! ({len(labels)} and {len(Y[0])}, respectively)"
                return cls(X, Y, Y_labels=labels, **kwargs)

    @classmethod
    def new_from_openchrom(cls, file, **kwargs):
        """ Convenient alias to simplify importing from the OpenChrom .csv format. """
        return cls.new_from_csv(file, x_start=(1,1), y_start=(1,3), label_start=(0,3), transpose=True, encoding="utf8", **kwargs)

    def fit_metrics(self, label=None, window_size=0.2):
        """
        Compute common fit metrics (R**2 and mean squared error) within ``window_size`` of the peaks.
        """
        centers = [p.params["center"] for p in self.peaks]
        max_X = np.max(self.X)
        time_window_min = max(0, min(centers) - window_size)
        time_window_max = min(max_X, max(centers) + window_size)

        relevant_idxs = slice(np.abs(self.X-time_window_min).argmin(), np.abs(self.X-time_window_max).argmin())

        expected = self.Y_from_label(label)[relevant_idxs]
        predicted = self.build_model().eval(self.build_parameters(), x=self.X)[relevant_idxs]

        assert expected.shape == predicted.shape, "wrong shapes for fit_metrics - did you forget to specify a label?"

        metrics = dict()
        metrics["R2"] = r2_score(expected, predicted)
        metrics["MSE"] = mean_squared_error(expected, predicted)

        return metrics

    def scale_Y(self):
        """
        Scale all Y values down by dividing by the maxiumum Y value.
        The original scale can be recovered by multiplying by ``self.scale_factor``.
        """
        self.scale_factor = self.scale_factor * np.max(self.Y)
        self.Y = self.Y / self.scale_factor

    def fit_baseline(self, label, X_min, X_max):
        """
        Fit the baseline correction on a specific region of the chromatogram.

        Args:
            label (float): which Y-value to use
            X_min (int): minimum X value to consider
            X_max (int): maxiumum X value to consider
        """
        Y = self.Y_from_label(label)

        min_idx = np.abs(self.X-X_min).argmin()
        max_idx = np.abs(self.X-X_max).argmin()

        assert min_idx < max_idx, "min must come before max!"

        params = self.build_parameters(baseline_only=True)
        model = self.build_model(baseline_only=True)

        def minimize_trace(params, model, Y, X):
            resid = Y - model.eval(params, x=X)
            return resid * (Y != 0) # exclude zeros from fit - too messy with the cutoff

        opt = lmfit.Minimizer(minimize_trace, params, fcn_args=(model, Y[min_idx:max_idx], self.X[min_idx:max_idx]))
        out = opt.minimize()
        self.unpack_parameters(out.params)
        self.freeze_baseline = True

    def top_labels(self):
        """ Return an ordered list of which Y-labels are most abundant. """
        avg_tic = np.mean(self.Y, axis=1)
        top_idx = np.argsort(avg_tic)
        top_labels = self.Y_labels[top_idx[::-1]]
        return top_labels

    def set_peak_labels(self, labels):
        """ Assign labels to peaks. """
        assert isinstance(labels, (list, tuple)), "labels must be list of peak labels"
        assert len(labels) == len(self.peaks), "must be one for every peak"
        for l, p in zip(labels, self.peaks):
            p.label = l
        return self

    def refit(self, label=None, num=20, fit_args=None, print_fit_report=False, time_window=None, same_sigma_gamma=True):
        """
        Rejiggle the parameters a few times to see if the fit can be improved.

        Args:
            label (str): which label to use
            num (int): how many times to try, the best one will be kept
            fit_args (dict): as for ``fit_peaks()``
            print_fit_report (bool): whether to print final fit report or not
            time_window (2-tuple): times to consider for fitting.
            same_sigma_gamma (bool): whether sigma/gamma should be constrained to be the same or not.
        """
        if fit_args is None:
            fit_args = dict()

        relevant_idxs = slice(None)
        if time_window is not None:
            assert len(time_window) == 2, "time_window must be 2-tuple"
            if time_window[0] > time_window[1]:
                time_window[0], time_window[1] = time_window[1], time_window[0]

            relevant_idxs = slice(np.abs(self.X-time_window[0]).argmin(), np.abs(self.X-time_window[1]).argmin())

        Y = self.Y_from_label(label)

        best_chi2 = None
        best_out = None

        rng = np.random.default_rng()

        for n in range(num+1):
            params = self.build_parameters(same_sigma_gamma=same_sigma_gamma)

            # add normally-distributed noise to parameters. stdev is 5%
            if n > 0:
                for key in params.keys():
                    params[key].value *= rng.normal(loc=1.0, scale=0.01)

            model = self.build_model()

            def minimize_trace(params, model, Y, X):
                resid = Y - model.eval(params, x=X)
                return resid * (Y != 0) # exclude zeros from fit - too messy with the cutoff

            #### perform fit
            opt = lmfit.Minimizer(minimize_trace, params, fcn_args=(model, Y[relevant_idxs], self.X[relevant_idxs]))
            out = opt.minimize(**fit_args)

            if best_chi2 is None or out.chisqr < best_chi2:
                best_chi2 = out.chisqr
                best_out = out

        if best_out is not None:
            if print_fit_report:
                print(lmfit.fit_report(best_out))
            self.unpack_parameters(best_out.params)


    def manual_integration(self, peak1, peak2, label=None, ax=None):
        """
        Manually integrate two peaks using ``np.trapz`` trapezoid-rule integration.

        Args:
            label:
            peak1 (2-tuple of times)
            peak2 (2-tuple of times)
            ax (matplotlib Axes): axes to plot on, optional

        Returns:
            area1, area2
        """

        assert len(peak1) == 2
        assert len(peak2) == 2

        if peak1[0] > peak2[0]:
            peak2, peak1 = peak1, peak2

        #assert peak1[1] <= peak2[0], "peak time ranges cannot overlap"

        # convert times to peak idxs
        idx1 = (np.abs(self.X-peak1[0]).argmin(), np.abs(self.X-peak1[1]).argmin())
        idx2 = (np.abs(self.X-peak2[0]).argmin(), np.abs(self.X-peak2[1]).argmin())

        # apply baseline correction
        Y = self.Y_from_label(label)
        if len(self.baseline_corrections):
            baseline, params, _ = self.baseline_corrections[0].export()
            Y += -1 * baseline.eval(params, x=self.X)

        # the big moment...
        area1 = np.trapz(Y[idx1[0]:idx1[1]], self.X[idx1[0]:idx1[1]])
        area2 = np.trapz(Y[idx2[0]:idx2[1]], self.X[idx2[0]:idx2[1]])
        areasum = area1 + area2

        area1 *= 1/areasum
        area2 *= 1/areasum

        # optional plotting
        if ax is not None:
            Y = self.Y_from_label(label) * self.scale_factor

            filter1 = np.zeros_like(self.X, dtype=np.int8)
            filter2 = np.zeros_like(self.X, dtype=np.int8)
            filter1[list(range(*idx1))] = True
            filter2[list(range(*idx2))] = True

            ax.fill_between(self.X, 0, Y, facecolor="darkorange", alpha=0.3, where=filter1)
            ax.fill_between(self.X, 0, Y, facecolor="royalblue", alpha=0.3, where=filter2)

            ax.plot(self.X, Y, "o", markersize=2, color="black", label="data", alpha=0.5)

            ax.set_xlabel("retention time (min)")
            ax.set_xlabel("counts")

        return area1, area2

