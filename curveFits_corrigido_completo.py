

import collections
import math
import warnings
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import scipy.optimize
from scipy.stats import chi2


class HillCurve:

    def __init__(self,
                 cs,
                 fs,
                 *,
                 inhibitory_or_mortality ='mortality',
                 fs_stderr=None,
                 fixbottom=0,
                 fixtop=1,
                 fitlogc=False,
                 use_stderr_for_fit=False,
                 ):
        """See main class docstring."""
        # get data into arrays sorted by concentration
        self._infectivity_or_neutralized = None
        self.cs = numpy.array(cs)
        self.fs = numpy.array(fs)
        if fs_stderr is not None:
            self.fs_stderr = numpy.array(fs_stderr)
            self.fs_stderr = self.fs_stderr[self.cs.argsort()]
        else:
            self.fs_stderr = None
        self.fs = self.fs[self.cs.argsort()]
        self.cs = self.cs[self.cs.argsort()]

        if inhibitory_or_mortality == 'mortality':
            if self.fs[0] < self.fs[-1] and (self.fs[0] < 0.3 and
                                             self.fs[-1] > 0.7):
                warnings.warn('`f` increases with concentration, consider `inhibitory_or_mortality="mortality"')
        elif inhibitory_or_mortality == 'inhibitory':
            if self.fs[0] > self.fs[-1] and (self.fs[0] > 0.7 and self.fs[-1] < 0.3):
                warnings.warn('`f` decreases with concentration, consider '
                              '`inhibitory_or_mortality="inhibitory"')
        else:
            raise ValueError('invalid `inhibitory_or_mortality`')
        self._inhibitory_or_mortality = inhibitory_or_mortality

        if any(self.cs <= 0):
            raise ValueError('concentrations in `cs` must all be > 0')

        # first try to fit using curve_fit
        try:
            fit_tup, self.params_stdev = self._fit_curve(
                                      fixtop=fixtop,
                                      fixbottom=fixbottom,
                                      fitlogc=fitlogc,
                                      use_stderr_for_fit=use_stderr_for_fit)
        except RuntimeError:
            # curve_fit failed, try using minimize
            for method in ['TNC', 'L-BFGS-B', 'SLSQP', 'Powell']:
                fit_tup = self._minimize_fit(
                                    fixtop=fixtop,
                                    fixbottom=fixbottom,
                                    fitlogc=fitlogc,
                                    use_stderr_for_fit=use_stderr_for_fit,
                                    method=method,
                                    )
                self.params_stdev = None  # can't estimate errors
                if fit_tup is not False:
                    break
            else:
                raise RuntimeError(f"fit failed:\ncs={self.cs}\nfs={self.fs}")

        for i, param in enumerate(['midpoint', 'slope', 'bottom', 'top']):
            setattr(self, param, fit_tup[i])
            
            

    def _fit_curve(self, *, fixtop, fixbottom, fitlogc, use_stderr_for_fit):
        """curve_fit, return `(midpoint, slope, bottom, top), params_stdev`."""
        # make initial guess for slope to have the right sign
        slope = 1.5

        # make initial guess for top and bottom
        if fixtop is False:
            top = max(1, self.fs.max())
        else:
            if not isinstance(fixtop, (int, float)):
                raise ValueError('`fixtop` is not `False` or a number')
            top = fixtop
        if fixbottom is False:
            bottom = min(0, self.fs.min())
        else:
            if not isinstance(fixbottom, (int, float)):
                raise ValueError('`fixbottom` is not `False` or a number')
            bottom = fixbottom

        # make initial guess for midpoint
        # if midpoint guess outside range, guess outside range by amount
        # equal to spacing of last two points
        midval = (top - bottom) / 2.0
        if (self.fs > midval).all():
            midpoint = {'inhibitory': self.cs[-1]**2 / self.cs[-2],
                        'mortality': self.cs[0] / (self.cs[-1] / self.cs[-2])
                        }[self._inhibitory_or_mortality]
        elif (self.fs <= midval).all():
            midpoint = {'mortality': self.cs[-1]**2 / self.cs[-2],
                        'inhibitory': self.cs[0] / (self.cs[-1] / self.cs[-2])
                        }[self._inhibitory_or_mortality]
        else:
            # get first index where f crosses midpoint
            i = numpy.argmax((self.fs > midval)[:-1] !=
                             (self.fs > midval)[1:])
            assert (self.fs[i] > midval) != (self.fs[i + 1] > midval)
            midpoint = (self.cs[i] + self.cs[i + 1]) / 2.0

        # set up function and initial guesses
        if fitlogc:
            evalfunc = self._evaluate_log
            xdata = numpy.log(self.cs)
            midpoint = numpy.log(midpoint)
        else:
            evalfunc = self.evaluate
            xdata = self.cs

        if fixtop is False and fixbottom is False:
            initguess = [midpoint, slope, bottom, top]

            def func(c, m, s, b, t):
                return evalfunc(c, m, s, b, t,
                                self._inhibitory_or_mortality)

        elif fixtop is False:
            initguess = [midpoint, slope, top]

            def func(c, m, s, t):
                return evalfunc(c, m, s, bottom, t,
                                self._inhibitory_or_mortality)

        elif fixbottom is False:
            initguess = [midpoint, slope, bottom]

            def func(c, m, s, b):
                return evalfunc(c, m, s, b, top,
                                self._inhibitory_or_mortality)
        else:
            initguess = [midpoint, slope]

            def func(c, m, s):
                return evalfunc(c, m, s, bottom, top,
                                self._inhibitory_or_mortality)

        (popt, pcov) = scipy.optimize.curve_fit(
                f=func,
                xdata=xdata,
                ydata=self.fs,
                p0=initguess,
                sigma=self.fs_stderr if use_stderr_for_fit else None,
                absolute_sigma=True,
                maxfev=1000,
                )

        perr = numpy.sqrt(numpy.diag(pcov))

        if fitlogc:
            midpoint = numpy.exp(midpoint)

        midpoint = popt[0]
        slope = popt[1]
        params_stderr = {'midpoint': perr[0],
                         'slope': perr[1],
                         'top': 0,
                         'bottom': 0}
        if fixbottom is False and fixtop is False:
            bottom = popt[2]
            params_stderr['bottom'] = perr[2]
            top = popt[3]
            params_stderr['top'] = perr[3]
        elif fixbottom is False:
            bottom = popt[2]
            params_stderr['bottom'] = perr[2]
        elif fixtop is False:
            top = popt[2]
            params_stderr['top'] = perr[2]

        return (midpoint, slope, bottom, top), params_stderr

    

    def _minimize_fit(self, *, fixtop, fixbottom, fitlogc, use_stderr_for_fit,
                      method):
        """Fit via minimization, return `(midpoint, slope, bottom, top)`."""
        # make initial guess for slope to have the right sign
        slope = 1.5

        # make initial guess for top and bottom
        if fixtop is False:
            top = max(1, self.fs.max())
        else:
            if not isinstance(fixtop, (int, float)):
                raise ValueError('`fixtop` is not `False` or a number')
            top = fixtop
        if fixbottom is False:
            bottom = min(0, self.fs.min())
        else:
            if not isinstance(fixbottom, (int, float)):
                raise ValueError('`fixbottom` is not `False` or a number')
            bottom = fixbottom

        # make initial guess for midpoint
        # if midpoint guess outside range, guess outside range by amount
        # equal to spacing of last two points
        midval = (top - bottom) / 2.0
        if (self.fs > midval).all():
            midpoint = {'inhibitory': self.cs[-1]**2 / self.cs[-2],
                        'mortality': self.cs[0] / (self.cs[-1] / self.cs[-2])
                        }[self._infectivity_or_neutralized]
        elif (self.fs <= midval).all():
            midpoint = {'mortality': self.cs[-1]**2 / self.cs[-2],
                        'inhibitory': self.cs[0] / (self.cs[-1] / self.cs[-2])
                        }[self._inhibitory_or_mortality]
        else:
            # get first index where f crosses midpoint
            i = numpy.argmax((self.fs > midval)[:-1] !=
                             (self.fs > midval)[1:])
            assert (self.fs[i] > midval) != (self.fs[i + 1] > midval)
            midpoint = (self.cs[i] + self.cs[i + 1]) / 2.0

        # set up function and initial guesses
        if fitlogc:
            evalfunc = self._evaluate_log
            xdata = numpy.log(self.cs)
            midpoint = numpy.log(midpoint)
            bounds = [(None, None), (0, None)]
        else:
            evalfunc = self.evaluate
            xdata = self.cs
            bounds = [(0, None), (0, None)]

        if fixtop is False and fixbottom is False:
            initguess = [midpoint, slope, bottom, top]
            bounds = bounds + [(None, None), (None, None)]

            def func(c, m, s, b, t):
                return evalfunc(c, m, s, b, t,
                                self._inhibitory_or_mortality)

        elif fixtop is False:
            initguess = [midpoint, slope, top]
            bounds.append((bottom, None))

            def func(c, m, s, t):
                return evalfunc(c, m, s, bottom, t,
                                self._inhibitory_or_mortality)

        elif fixbottom is False:
            initguess = [midpoint, slope, bottom]
            bounds.append((None, top))

            def func(c, m, s, b):
                return evalfunc(c, m, s, b, top,
                                self._inhibitory_or_mortality)
        else:
            initguess = [midpoint, slope]

            def func(c, m, s):
                return evalfunc(c, m, s, bottom, top,
                                self._inhibitory_or_mortality)

        def min_func(p):
            """Evaluate to zero when perfect fit."""
            if (use_stderr_for_fit is None) or (self.fs_stderr is None):
                return sum((func(xdata, *p) - self.fs)**2)
            else:
                return sum((func(xdata, *p) - self.fs / self.fs_stderr)**2)

        initguess = numpy.array(initguess, dtype='float')
        res = scipy.optimize.minimize(min_func,
                                      initguess,
                                      bounds=bounds,
                                      method=method)

        if not res.success:
            return False

        if fitlogc:
            midpoint = numpy.exp(midpoint)

        midpoint = res.x[0]
        slope = res.x[1]
        if fixbottom is False and fixtop is False:
            bottom = res.x[2]
            top = res.x[3]
        elif fixbottom is False:
            bottom = res.x[2]
        elif fixtop is False:
            top = res.x[2]

        return (midpoint, slope, bottom, top)
            
    
    def icXX(self, fracmort, *, method='interpolate'):
        """Generalizes :meth:`HillCurve.ic50` to arbitrary frac mortality.

        For instance, set `fracmort` to 0.95 if you want the IC95, the
        concentration where 95% are dead.

        Args:
            `fracmort` (float)
                Compute concentration at which `fracmort` of the population
                is expected to be killed. Note that this is the
                expected fraction of mortality, not the fraction
                ineffective.
            `method` (str)
                Can have following values:

                  - 'interpolate': only return a number for ICXX if it
                    is in range of concentrations, otherwise return `None`.

                  - 'bound': if ICXX is out of range of concentrations,
                    return upper or lower measured concentration depending
                    on if ICXX is above or below range of concentrations.
                    Assumes infectivity decreases with concentration.

        Returns:
            Number giving ICXX or `None` (depending on value of `method`).

        """
        fracinf = 1 - fracmort
        if self.top < fracinf and self.bottom < fracinf:
            bound = 'lower'
        elif self.top >= fracinf and self.bottom >= fracinf:
            bound = 'upper'
        else:
            icXX = (self.midpoint * ((self.top - fracinf) /
                    (fracinf - self.bottom))**(1.0 / self.slope))
            if (self.cs[0] <= icXX <= self.cs[-1]):
                return icXX
            elif icXX < self.cs[0]:
                bound = 'lower'
            else:
                bound = 'upper'

        if method == 'bound':
            if bound == 'upper':
                return self.cs[-1]
            elif bound == 'lower':
                return self.cs[0]
            else:
                raise ValueError(f"invalid `bound` {bound}")
        elif method == 'interpolate':
            return None
        else:
            raise ValueError(f"invalid `method` of {method}")
    
    
    def ic50(self, method='interpolate'):
        return self.icXX(0.5, method=method)


    def ic50_stdev(self):
        r"""Get standard deviation of fit IC50 parameter.

        Calculated just from estimated standard deviation on `midpoint`.
        Note if you have replicates, we recommend fitting separately
        and calculating standard error from those fits rather than
        using this value.

        Returns:
            A number giving the standard deviation, or `None` if cannot
            be estimated or if IC50 is at bound.

        """
        ic50 = self.ic50()
        if ic50 is None:
            return None
        midpoint_stdev = self.params_stdev['midpoint']
        if midpoint_stdev is None:
            return None
        else:
            return midpoint_stdev * ic50 / self.midpoint
    
    
    def icXX_bound(self, fracmort):
        """Like :meth:`HillCurve.ic50_bound` for arbitrary frac neutralized."""
        if self.icXX(fracmort, method='interpolate') is not None:
            return 'interpolated'
        else:
            icXX = self.icXX(fracmort, method='bound')
            if icXX == self.cs[0]:
                return 'upper'
            elif icXX == self.cs[-1]:
                return 'lower'
            else:
                raise RuntimeError(f"icXX not bound for {fracmort}")

    def ic50_bound(self):
        """Is IC50 'interpolated', or an 'upper' or 'lower' bound."""
        return self.icXX_bound(0.5)    
    
    
    def icXX_str(self, fracmort, *, precision=3):
        """Like :meth:`HillCurve.ic50_str` for arbitrary frac mortality."""
        icXX = f"{{:.{precision}g}}".format(self.icXX(fracmort,
                                                      method='bound'))
        prefix = {'interpolated': '',
                  'upper': '<',
                  'lower': '>'}[self.icXX_bound(fracmort)]
        return f"{prefix}{icXX}"


    def ic50_str(self, precision=3):
        """LC50 as string indicating upper / lower bounds with > or <.

        Args:
            Number of significant digits in returned string.

        """
        return self.icXX_str(0.5, precision=precision)
    
 
    def fracmortality(self, c):
        """mortality at `c` for fitted parameters."""
        return self.evaluate(c, self.midpoint, self.slope,
                             self.bottom, self.top,
                             self._inhibitory_or_mortality)  

    @staticmethod
    def evaluate(c, m, s, b, t,
                 inhibitory_or_mortality ='inhibitory'):
        r""":math:`f\left(c\right) = b + \frac{t-b}{1+\left(c/m\right)^s}`.

        If `inhibitory_or_mortality` is 'neutralized' rather than
        'infectivity', instead return
        :math:`f\left(c\right) = t + \frac{b-t}{1+\left(c/m\right)^s}`.

        """
        if inhibitory_or_mortality == 'inhibitory':
            return b + (t - b) / (1 + (c / m)**s)
        elif inhibitory_or_mortality == 'mortality':
            return t + (b - t) / (1 + (c / m)**s)
        else:
            raise ValueError('invalid `inhibitory_or_mortality`')

    @staticmethod
    def _evaluate_log(logc, logm, s, b, t,
                      inhibitory_or_mortality='inhibitory'):
        """Like :class:`HillCurve.evaluate` but on log concentration scale."""
        if inhibitory_or_mortality == 'inhibitory':
            return b + (t - b) / (1 + numpy.exp(s * (logc - logm)))
        elif inhibitory_or_mortality == 'mortality':
            return t + (b - t) / (1 + numpy.exp(s * (logc - logm)))
        else:
            raise ValueError('invalid `inhibitory_or_mortality`')          
            
            
    def plot(self,
             *,
             concentrations='auto',
             ax=None,
             xlabel='concentration',
             ylabel='Mortality',
             color='black',
             marker='o',
             markersize=6,
             linewidth=1,
             linestyle='-',
             yticklocs=None,
             ):
        """Plot the neutralization curve.

        Args:
            `concentrations`
                Concentrations to plot, same meaning as for
                :meth:`HillCurve.dataframe`.
            `ax` (`None` or matplotlib axes.Axes object)
                Use to plot on an existing axis. If using an existing
                axis, do **not** re-scale the axis limits to the data.
            `xlabel` (str or `None`)
                Label for x-axis.
            `ylabel` (str or `None`)
                Label for y-axis.
            `color` (str)
                Color of line and point.
            `marker` (str)
                Marker shape: https://matplotlib.org/api/markers_api.html
            `markersize` (float)
                Size of point marker.
            `linewidth` (float)
                Width of line.
            `linestyle` (str)
                Line style.
            `yticklocs` (`None` or list)
                Exact locations to place yticks; `None` means auto-locate.

        Returns:
            The 2-tuple `(fig, ax)` giving the matplotlib figure and axis.

        """
        data = self.dataframe(concentrations)
        
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches((4, 3))
            check_ybounds = True
            ylowerbound = -0.05
            yupperbound = 1.05
            ax.autoscale(True, 'both')
        else:
            fig = ax.get_figure()
            ax.autoscale(False, 'both')
            check_ybounds = False
            
        maxcon = max(data['fit']) 
        
        if maxcon < 0.3:
            ax.plot()
        else:
            ax.plot('concentration',
                    'fit',
                    data=data,
                    linestyle=None,
                    linewidth=None,
                    color=color,
                    )
            
        ax.errorbar(x='concentration',
                    y='measurement',
        #            yerr='stderr',
                    data=data,
                    fmt=marker,
                    color=color,
        #            markersize=markersize,
        #            capsize=markersize / 1.5,
                    )

        ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params('both', labelsize=12, length=5, width=1)
        ax.minorticks_off()
        if yticklocs is not None:
            ax.set_yticks(yticklocs)

        if check_ybounds:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, ylowerbound), max(ymax, yupperbound))

        return fig, ax

    def dataframe(self, concentrations='auto'):
        """Get data frame with curve data for plotting.

        Useful if you want to get both the points and the fit
        curve to plot.

        Args:
            `concentrations` (array-like or 'auto' or 'measured')
                Concentrations for which we compute the fit values.
                If 'auto' the automatically computed from data
                range using :func:`concentrationRange`. If
                'measured' then only include measured values.

        Returns:
            A pandas DataFrame with the following columns:

              - 'concentration': concentration
              - 'fit': curve fit value at this point
              - 'measurement': value of measurement at this point,
                or numpy.nan if no measurement here.
              - 'stderr': standard error of measurement if provided.

        """
        if concentrations == 'auto':
            concentrations = concentrationRange(self.cs[0], self.cs[-1])
        elif concentrations == 'measured':
            concentrations = []
        concentrations = numpy.concatenate([self.cs, concentrations])
        n = len(concentrations)

        points = numpy.concatenate([self.fs,
                                    numpy.full(n - len(self.fs), numpy.nan)
                                    ])

        if self.fs_stderr is None:
            stderr = numpy.full(n, numpy.nan)
        else:
            stderr = numpy.concatenate([self.fs_stderr,
                                        numpy.full(n - len(self.fs), numpy.nan)
                                        ])

        fit = numpy.array([self.fracmortality(c) for c in concentrations])

        return (pd.DataFrame.from_dict(
                    collections.OrderedDict(
                        [('concentration', concentrations),
                         ('measurement', points),
                         ('fit', fit),
                         ('stderr', stderr),
                         ])
                    )
                .sort_values('concentration')
                .reset_index(drop=True)
                )
    

    def ic50_confidence_interval(self, confidence_level=0.95):
        """Calculate the confidence interval for the IC50 (midpoint) parameter.
        Args:
        confidence_level (float): The desired confidence level (default is 0.95).
        Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval for IC50.
        """   
        
        ic50 = self.ic50()
        if ic50 is None:
            raise ValueError("IC50 cannot be calculated; it may be out of bounds.")

        # Get the standard deviation of the midpoint
        midpoint_stdev = self.params_stdev['midpoint']
        if midpoint_stdev is None:
            raise ValueError("Standard deviation for the midpoint is not available.")

        # Calculate the Z value for the desired confidence level
        z_value = 1.96  # for 95% confidence interval

        # Calculate the confidence interval
        lower_bound = ic50 - z_value * midpoint_stdev
        upper_bound = ic50 + z_value * midpoint_stdev

        return lower_bound, upper_bound
    
    def chi_squared_test(self):
        observed = self.fs
        expected = self.fitted_values

        if numpy.any(expected == 0):
            raise ValueError("Expected values contain zero - cannot compute chi-squared.")
        
        chi2_stat = numpy.sum((observed - expected)**2 /expected)

        n_params = 2 # midpoint and slope
        if not isinstance(self.bottom, (int, float)):
            n_param += 1
        if not isinstance(self.top, (int, float)):
            n_param += 1

        dof = len(observed) - n_params
        if dof <= 0:
            return {'chi2': numpy.nan, 'dof': dof, 'p_value': numpy.nan}
        
        p_value = 1 - chi2.cdf(chi2_stat, dof)

        return{
            'chi2': chi2_stat,
            'dof': dof,
            'p_value': p_value
        }

    @property
    def fitted_values(self):
        return self.evaluate(self.cs, self.midpoint, self.slope, self.bottom, self.top, self._inhibitory_or_mortality)
    
    @property
    def residuals(self):
        return self.fs - self.fitted_values
    
    @property
    def coefficients(self):
        return numpy.array([self.midpoint, self.slope, self.bottom, self.top])
    
    @property
    def r_squared(self):
        y_true = self.fs
        y_pred = self.fitted_values
        ss_res = numpy.sum((y_true - y_pred)**2)
        ss_tot = numpy.sum((y_true - numpy.mean(y_true))**2)
        if ss_tot == 0:
            return numpy.nan
        return 1 - (ss_res/ ss_tot)

    @property
    def rmse(self):
        residual = self.fs - self.fitted_values
        return numpy.sqrt(numpy.mean(residual **2))


def concentrationRange(bottom, top, npoints=200, extend=0.1):
    """Logarithmically spaced concentrations for plotting.

    Useful if you want to plot a curve by fitting values to densely
    sampled points and need the concentrations at which to compute
    these points.

    Args:
        `bottom` (float)
            Lowest concentration.
        `top` (float)
            Highest concentration.
        `npoints` (int)
            Number of points.
        `extend` (float)
            After transforming to log space, extend range of points
            by this much below and above `bottom` and `top`.

    Returns:
        A numpy array of `npoints` concentrations.

    >>> numpy.allclose(concentrationRange(0.1, 100, 10, extend=0),
    ...                [0.1, 0.22, 0.46, 1, 2.15, 4.64, 10, 21.54, 46.42, 100],
    ...                atol=1e-2)
    True
    >>> numpy.allclose(concentrationRange(0.1, 100, 10),
    ...                [0.05, 0.13, 0.32, 0.79, 2.00, 5.01,
    ...                 12.59, 31.62, 79.43, 199.53],
    ...                atol=1e-2)
    True

    """
    if top <= bottom:
        raise ValueError('`bottom` must be less than `top`')
    if bottom <= 0:
        raise ValueError('`bottom` must be greater than zero')
    if extend < 0:
        raise ValueError('`extend` must be >= 0')

    logbottom = math.log10(bottom)
    logtop = math.log10(top)
    logrange = logtop - logbottom
    assert logrange > 0
    bottom = logbottom - logrange * extend
    top = logtop + logrange * extend

    return numpy.logspace(bottom, top, npoints)
