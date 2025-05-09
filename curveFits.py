import itertools
from matplotlib.lines import Line2D
import collections
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from hillcurve_test import HillCurve


"""
============
colorschemes
============

Color schemes.
"""

#: color-blind safe palette with gray, from
#: http://bconnelly.net/2013/10/creating-colorblind-friendly-figures
CBPALETTE = ('#999999', '#E69F00', '#56B4E9', '#009E73',
             '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

#: color-blind safe palette with black, from
#: http://bconnelly.net/2013/10/creating-colorblind-friendly-figures
CBBPALETTE = ('#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

#: list of matplotlib markers same length as `CBPALETTE`, from
#: https://matplotlib.org/api/markers_api.html
CBMARKERS = ('o', '^', 's', 'D', 'v', '<', '>', 'p')
assert len(CBPALETTE) == len(CBBPALETTE) == len(CBMARKERS)


class CurveFits:
    _SUS_NAMES = (['sus', 'susceptible'])

    def __init__(self,
                 data,
                 *,
                 conc_col='concentration',
                 fracmort_col='mortality',
                 btProtein_col='btProtein',
                 population_col='population',
                 replicate_col='replicate',
                 inhibitory_or_mortality='inhibitory',
                 fixbottom=0,
                 fixtop=1,
                 ):
        """See main class docstring."""
        # make args into attributes
        self.conc_col = conc_col
        self.fracmort_col = fracmort_col
        self.btProtein_col = btProtein_col
        self.population_col = population_col
        self.replicate_col = replicate_col
        self.fixbottom = fixbottom
        self.fixtop = fixtop
        self._inhibitory_or_mortality = inhibitory_or_mortality

        # check for required columns
        cols = [self.btProtein_col, self.population_col, self.replicate_col,
                self.conc_col, self.fracmort_col]
        if len(cols) != len(set(cols)):
            raise ValueError('duplicate column names:\n\t' + '\n\t'.join(cols))
        if not (set(cols) <= set(data.columns)):
            raise ValueError('`data` lacks required columns, which are:\n\t' +
                             '\n\t'.join(cols))

        # create `self.df`, ensure that replicates are str rather than number
        self.df = (data[cols]
                   .assign(**{replicate_col: lambda x: (x[replicate_col]
                                                        .astype(str))
                              })
                   )

        # create bioassay / populations / replicates attributes, error check them
        self.bioassay = self.df[self.btProtein_col].unique().tolist()
        self.populations = {}
        self.replicates = {}
        for btProtein in self.bioassay:
            btProtein_data = self.df.query(f"{self.btProtein_col} == @btProtein")
            btProtein_populations = btProtein_data[self.population_col].unique().tolist()
            self.populations[btProtein] = btProtein_populations
            for population in btProtein_populations:
                population_data = btProtein_data.query(f"{self.population_col} == @population")
                population_reps = population_data[self.replicate_col].unique().tolist()
                if 'average' in population_reps:
                    raise ValueError('A replicate is named "average". This is '
                                     'not allowed as that name is used for '
                                     'replicate averages.')
                self.replicates[(btProtein, population)] = population_reps + ['average']
                for i, rep1 in enumerate(population_reps):
                    conc1 = (population_data
                             .query(f"{self.replicate_col} == @rep1")
                             [self.conc_col]
                             .sort_values()
                             .tolist()
                             )
                    if len(conc1) != len(set(conc1)):
                        raise ValueError('duplicate concentrations for '
                                         f"{btProtein}, {population}, {rep1}")
                    for rep2 in population_reps[i + 1:]:
                        conc2 = (population_data
                                 .query(f"{self.replicate_col} == @rep1")
                                 [self.conc_col]
                                 .sort_values()
                                 .tolist()
                                 )
                        if conc1 != conc2:
                            raise ValueError(f"replicates {rep1} and {rep2} "
                                             'have different concentrations '
                                             f"for {btProtein}, {population}")
        self.allpopulations = collections.OrderedDict()
        for btProtein in self.bioassay:
            for population in self.populations[btProtein]:
                self.allpopulations[population] = True
        self.allpopulations = list(self.allpopulations.keys())

        if pd.isnull(self.allpopulations).any():
            raise ValueError(f"a population has name NaN:\n{self.allpopulations}")
        if pd.isnull(self.bioassay).any():
            raise ValueError(f"a btProtein has name NaN:\n{self.bioassay}")

        # compute replicate average and add 'stderr'
        if 'stderr' in self.df.columns:
            raise ValueError('`data` has column "stderr"')
        avg_df = (self.df
                  .groupby([self.btProtein_col, self.population_col, self.conc_col],
                           observed=True)
                  [self.fracmort_col]
                  # sem is sample stderr, evaluates to NaN when just 1 rep
                  .aggregate(['mean', 'sem', 'count'])
                  .rename(columns={'mean': self.fracmort_col,
                                   'sem': 'stderr',
                                   })
                  .reset_index()
                  .assign(**{replicate_col: 'average'})
                  )
        self.df = pd.concat([self.df, avg_df],
                            ignore_index=True,
                            sort=False,
                            )

        self._hillcurves = {}  # curves computed by `getCurve` cached here
        self._fitparams = {}  # cache data frame computed by `fitParams`

    def getCurve(self, *, btProtein, population, replicate):
        key = (btProtein, population, replicate)

        if key not in self._hillcurves:
            if btProtein not in self.bioassay:
                raise ValueError(f"invalid `btProtein` of {btProtein}")
            if population not in self.populations[btProtein]:
                raise ValueError(f"invalid `population` of {population} for "
                                 f"`btProtein` of {btProtein}")
            if replicate not in self.replicates[(btProtein, population)]:
                raise ValueError(f"invalid `replicate` of {replicate} for "
                                 f"`btProtein` of {btProtein} and `population` of {population}")

            idata = self.df.query(f"({self.btProtein_col} == @btProtein) & "
                                  f"({self.population_col} == @population) & "
                                  f"({self.replicate_col} == @replicate)")
            if len(idata) < 1:
                raise RuntimeError(f"no data for btProtein {btProtein} population {population}")

            if idata['stderr'].isna().all():
                fs_stderr = None
            elif idata['stderr'].isna().any():
                raise RuntimeError('`stderr` has only some entries NaN\n' + str(idata))
            else:
                fs_stderr = idata['stderr']

            try:
                curve = HillCurve(
                    cs=idata[self.conc_col],
                    fs=idata[self.fracmort_col],
                    fs_stderr=fs_stderr,
                    fixbottom=self.fixbottom,
                    fixtop=self.fixtop,
                    inhibitory_or_mortality=(
                        self._inhibitory_or_mortality)
                )
            except RuntimeError as e:
                idata.to_csv('temp.csv', index=False)
                # following here: https://stackoverflow.com/a/46091127
                raise RuntimeError('Error while fitting HillCurve for '
                                   f"btProtein {btProtein}, population {population}, "
                                   f"replicate {replicate}.\nData are:\n" +
                                   str(idata)) from e

            self._hillcurves[key] = curve

        return self._hillcurves[key]

    def get_lc50_and_ci(self, btProtein, population, replicate):
        # Get the curve using the fits module
        curve = self.getCurve(btProtein=btProtein, population=population, replicate=replicate)

        # Calculate the LC50
        lc50 = curve.ic50(method='interpolate')  # Use appropriate method

        if lc50 is None:
            return None, None, None  # Return None for all if IC50 cannot be calculated

        # Get the confidence intervals
        lower_ci, upper_ci = curve.ic50_confidence_interval()

        return lc50, lower_ci, upper_ci

    def fitParams(self,
              *,
              average_only=True,
              ics=(50,),
              ics_precision=0,
              ic50_error=None,
              bootstrap_ci=True,
              numBoot=None,
              ciLevs=(0.025, 0.975),
              ):
        if ic50_error not in {None, 'fit_stdev'}:
            raise ValueError(f"invalid ic50_error of {ic50_error}")

        ics = tuple(ics)
        ic_colprefixes = [f"ic{{:.{ics_precision}f}}".format(ic) for ic in ics]
        if len(ic_colprefixes) != len(set(ic_colprefixes)):
            raise ValueError('column names for ICXX not unique.\n'
                             'Either you have duplicate entries in `ics` '
                             'or you need to increase `ics_precision`.')

        key = (average_only, ics, ics_precision, ic50_error, bootstrap_ci)

        if key not in self._fitparams:
            d = collections.defaultdict(list)
            params = ['midpoint', 'slope', 'top', 'bottom']
            for btProtein in self.bioassay:
                for population in self.populations[btProtein]:
                    replicates = self.replicates[(btProtein, population)]
                    nreplicates = sum(r != 'average' for r in replicates)
                    assert nreplicates == len(replicates) - 1
                    if average_only:
                        replicates = ['average']
                    for replicate in replicates:
                        curve = self.getCurve(btProtein=btProtein,
                                              population=population,
                                              replicate=replicate
                                              )
                        d['btProtein'].append(btProtein)
                        d['population'].append(population)
                        d['replicate'].append(replicate)
                        if replicate == 'average':
                            d['nreplicates'].append(nreplicates)
                        else:
                            d['nreplicates'].append(float('nan'))

                        # Reforço de segurança: skip se curva sem ponto
                        if len(curve.cs) < 2:
                            d['lc50'].append(None)
                            d['lower_ci'].append(None)
                            d['upper_ci'].append(None)
                            continue

                        # --- Cálculo do LC50 e CI ---
                        try:
                            if bootstrap_ci:
                                self.calc_hill_bootstrap(curve, numBoot=numBoot, ciLevs=ciLevs)
                                lc50_ci = self.calc_hill_conf_int(
                                    curve,
                                    parfunc=lambda coefs: numpy.array([coefs[0]]),
                                    civals=ciLevs
                                )
                                d['lc50'].append(lc50_ci[0, 1])
                                d['lower_ci'].append(lc50_ci[0, 0])
                                d['upper_ci'].append(lc50_ci[0, 2])
                            else:
                                lc50 = curve.ic50(method='interpolate')
                                if lc50 is None:
                                    d['lc50'].append(None)
                                    d['lower_ci'].append(None)
                                    d['upper_ci'].append(None)
                                else:
                                    d['lc50'].append(lc50)
                                    low, high = curve.ic50_confidence_interval()
                                    d['lower_ci'].append(low)
                                    d['upper_ci'].append(high)
                        except Exception as e:
                            d['lc50'].append(None)
                            d['lower_ci'].append(None)
                            d['upper_ci'].append(None)

                        # ICXXs
                        for ic, colprefix in zip(ics, ic_colprefixes):
                            f = ic / 100
                            d[colprefix].append(curve.icXX(f, method='bound'))
                            d[f"{colprefix}_bound"].append(curve.icXX_bound(f))
                            d[f"{colprefix}_str"].append(curve.icXX_str(f))

                        if ic50_error == 'fit_stdev':
                            d['ic50_error'].append(curve.ic50_stdev())

                        for param in params:
                            d[param].append(getattr(curve, param))

            ic_cols = []
            for prefix in ic_colprefixes:
                ic_cols += [prefix, f"{prefix}_bound", f"{prefix}_str"]
            if ic50_error == 'fit_stdev':
                ic_cols.append('ic50_error')

            self._fitparams[key] = (
                pd.DataFrame(d)
                [['btProtein', 'population', 'replicate', 'nreplicates', 'lc50', 'lower_ci', 'upper_ci']
                + ic_cols + params]
                .assign(nreplicates=lambda x: (x['nreplicates'].astype('Int64')))
            )
            return self._fitparams[key]


    def plotPop(self,
                *,
                ncol=4,
                nrow=None,
                bioassay='all',
                populations='all',
                ignore_btProtein_population=None,
                colors=CBPALETTE,
                markers=CBMARKERS,
                population_to_color_marker=None,
                max_populations_per_subplot=5,
                multi_btProtein_subplots=True,
                all_subplots=_SUS_NAMES,
                titles=None,
                vlines=None,
                **kwargs,
                ):
        """Plot grid with replicate-average of populations for each btProtein.

        Args:
            `ncol`, `nrow` (int or `None`)
                Specify one of these to set number of columns or rows,
                other should be `None`.
            `bioassay` ('all' or list)
                bioassay to include on plot, in this order.
            `populations` ('all' or list)
                populations to include on plot, in this order unless one
                is specified in `all_subplots`.
            `ignore_btProtein_population` (`None` or dict)
                Specific btProtein / population combinations to ignore (not plot). Key
                by btProtein, and then list populations to ignore.
            `colors` (iterable)
                List of colors for different populations.
            `markers` (iterable)
                List of markers for different populations.
            `population_to_color_marker` (dict or `None`)
                Optionally specify a specific color and for each population as
                2-tuples `(color, marker)`. If you use this option, `colors`
                and `markers` are ignored.
            `max_populations_per_subplot` (int)
                Maximum number of populations to show on any subplot.
            `multi_btProtein_subplots` (bool)
                If a btProtein has more than `max_population_per_subplot` populations,
                do we make multiple subplots for it or raise an error?
            `all_subplots` (iterable)
                If making multiple subplots for btProtein, which populations
                do we show on all subplots? These are also shown first.
            `titles` (`None` or list)
                Specify custom titles for each subplot different than `bioassay`.
            `vlines` (`None` or dict)
                Add vertical lines to plots. Keyed by btProtein name, values
                are lists of dicts with a key 'x' giving x-location of vertical
                line, and optional keys 'linewidth', 'color', and 'linestyle'.
            `**kwargs`
                Other keyword arguments that can be passed to
                :meth:`CurveFits.plotGrid`.

        Returns:
            The 2-tuple `(fig, axes)` of matplotlib figure and 2D axes array.

        """
        bioassay, populations = self._bioassay_populations_lists(bioassay, populations)
        populations = list(collections.OrderedDict.fromkeys(populations))

        if titles is None:
            titles = bioassay
        elif len(bioassay) != len(titles):
            raise ValueError(f"`titles`, `bioassay` != length:\n{titles}\n{bioassay}")

        if max_populations_per_subplot < 1:
            raise ValueError('`max_populations_per_subplot` must be at least 1')

        # get color scheme for populations
        if population_to_color_marker:
            extra_populations = set(populations) - set(population_to_color_marker.keys())
            if extra_populations:
                raise ValueError('populations not in `population_to_color_marker`: ' +
                                 str(extra_populations))
        elif len(populations) <= min(len(colors), len(markers)):
            # can share scheme among subplots
            ordered_populations = ([v for v in populations if v in all_subplots] +
                                   [v for v in populations if v not in all_subplots])
            population_to_color_marker = {v: (c, m) for (v, c, m) in
                                          zip(ordered_populations, colors, markers)}
        elif min(len(colors), len(markers)) < max_populations_per_subplot:
            raise ValueError('`max_populations_per_subplot` larger than '
                             'number of colors or markers')
        else:
            population_to_color_marker = None

        # Build a list of plots appropriate for `plotGrid`.
        # Code is complicated because we could have several curve
        # per btProtein, and in that case need to share populations in
        # `all_subplots` among curves.
        plotlist = []
        vlines_list = []
        for btProtein, title in zip(bioassay, titles):
            if ignore_btProtein_population and btProtein in ignore_btProtein_population:
                ignore_population = ignore_btProtein_population[btProtein]
            else:
                ignore_population = {}
            curvelist = []
            ipopulation = 0
            btProtein_shared_populations = [v for v in self.populations[btProtein] if
                                            (v in populations) and (v in all_subplots) and
                                            (v not in ignore_population)]
            btProtein_unshared_populations = [v for v in self.populations[btProtein] if
                                              (v in populations) and
                                              (v not in all_subplots) and
                                              (v not in ignore_population)]
            unshared = int(bool(len(btProtein_unshared_populations)))
            if len(btProtein_shared_populations) > max_populations_per_subplot - unshared:
                raise ValueError(f"btProtein {btProtein} has too many subplot-shared "
                                 'populations (in `all_subplots`) relative to '
                                 'value of `max_populations_per_subplot`:\n'
                                 f"{btProtein_shared_populations} is more than "
                                 f"{max_populations_per_subplot} populations.")
            shared_curvelist = []
            for population in btProtein_shared_populations + btProtein_unshared_populations:
                if ipopulation >= max_populations_per_subplot:
                    if multi_btProtein_subplots:
                        plotlist.append((title, curvelist))
                        if vlines and (btProtein in vlines):
                            vlines_list.append(vlines[btProtein])
                        else:
                            vlines_list.append(None)
                        curvelist = list(shared_curvelist)
                        ipopulation = len(curvelist)
                        assert ipopulation < max_populations_per_subplot
                    else:
                        raise ValueError(f"btProtein {btProtein} has more than "
                                         '`max_populations_per_subplot` populations '
                                         'and `multi_btProtein_subplots` is False')
                if population_to_color_marker:
                    color, marker = population_to_color_marker[population]
                else:
                    color = colors[ipopulation]
                    marker = markers[ipopulation]
                curvelist.append({'btProtein': btProtein,
                                  'population': population,
                                  'replicate': 'average',
                                  'label': population,
                                  'color': color,
                                  'marker': marker,
                                  })
                if population in btProtein_shared_populations:
                    shared_curvelist.append(curvelist[-1])
                ipopulation += 1
            if curvelist:
                plotlist.append((title, curvelist))
                if vlines and (btProtein in vlines):
                    vlines_list.append(vlines[btProtein])
                else:
                    vlines_list.append(None)
        if not plotlist:
            raise ValueError('no curves for these bioassay / populations')

        # get number of columns
        if (nrow is not None) and (ncol is not None):
            raise ValueError('either `ncol` or `nrow` must be `None`')
        elif isinstance(nrow, int) and nrow > 0:
            ncol = math.ceil(len(plotlist) / nrow)
        elif not (isinstance(ncol, int) and ncol > 0):
            raise ValueError('`nrow` or `ncol` must be integer > 0')

        # convert plotlist to plots dict for `plotGrid`
        plots = {}
        vlines_axkey = {}
        assert len(plotlist) == len(vlines_list)
        for iplot, (plot, ivline) in enumerate(zip(plotlist, vlines_list)):
            irow = iplot // ncol
            icol = iplot % ncol
            plots[(irow, icol)] = plot
            if ivline:
                vlines_axkey[(irow, icol)] = ivline

        if population_to_color_marker and 'orderlegend' not in kwargs:
            orderlegend = population_to_color_marker.keys()
            kwargs['orderlegend'] = orderlegend

        return self.plotGrid(plots,
                             vlines=vlines_axkey,
                             **kwargs,
                             )

    def plotpopulations(self,
                        *,
                        ncol=4,
                        nrow=None,
                        bioassay='all',
                        populations='all',
                        ignore_population_btProtein=None,
                        colors=CBPALETTE,
                        markers=CBMARKERS,
                        btProtein_to_color_marker=None,
                        max_bioassay_per_subplot=5,
                        multi_population_subplots=True,
                        all_subplots=(),
                        titles=None,
                        vlines=None,
                        **kwargs,
                        ):
        """Plot grid with replicate-average of bioassay for each population.

        Args:
            `ncol`, `nrow` (int or `None`)
                Specify one of these to set number of columns or rows,
                other should be `None`.
            `bioassay` ('all' or list)
                bioassay to include on plot, in this order, unless one is
                specified in `all_subplots`.
            `populations` ('all' or list)
                populations to include on plot, in this order.
            `ignore_population_btProtein` (`None` or dict)
                Specific population / btProtein combinations to ignore (not plot). Key
                by population, and then list bioassay to ignore.
            `colors` (iterable)
                List of colors for different bioassay.
            `markers` (iterable)
                List of markers for different bioassay.
            `btProtein_to_color_marker` (dict or `None`)
                Optionally specify a specific color and for each btProtein as
                2-tuples `(color, marker)`. If you use this option, `colors`
                and `markers` are ignored.
            `max_bioassay_per_subplot` (int)
                Maximum number of bioassay to show on any subplot.
            `multi_population_subplots` (bool)
                If a population has more than `max_bioassay_per_subplot` bioassay,
                do we make multiple subplots for it or raise an error?
            `all_subplots` (iterable)
                If making multiple subplots for population, which bioassay
                do we show on all subplots? These are also shown first.
            `titles` (`None` or list)
                Specify custom titles for each subplot different than
                `populations`.
            `vlines` (`None` or dict)
                Add vertical lines to plots. Keyed by population name, values
                are lists of dicts with a key 'x' giving x-location of vertical
                line, and optional keys 'linewidth', 'color', and 'linestyle'.
            `**kwargs`
                Other keyword arguments that can be passed to
                :meth:`CurveFits.plotGrid`.

        Returns:
            The 2-tuple `(fig, axes)` of matplotlib figure and 2D axes array.

        """
        bioassay, populations = self._bioassay_populations_lists(bioassay, populations)
        populations = list(collections.OrderedDict.fromkeys(populations))

        if titles is None:
            titles = populations
        elif len(populations) != len(titles):
            raise ValueError(f"`titles`, `populations` != length:\n"
                             f"{titles}\n{populations}")

        if max_bioassay_per_subplot < 1:
            raise ValueError('`max_bioassay_per_subplot` must be at least 1')

        # get color scheme for bioassay
        if btProtein_to_color_marker:
            extra_bioassay = set(bioassay) - set(btProtein_to_color_marker.keys())
            if extra_bioassay:
                raise ValueError('bioassay not in `btProtein_to_color_marker`: ' +
                                 str(extra_bioassay))
        elif len(bioassay) <= min(len(colors), len(markers)):
            # can share scheme among subplots
            ordered_bioassay = ([s for s in bioassay if s in all_subplots] +
                                [s for s in bioassay if s not in all_subplots])
            btProtein_to_color_marker = {s: (c, m) for (s, c, m) in
                                         zip(ordered_bioassay, colors, markers)}
        elif min(len(colors), len(markers)) < max_bioassay_per_subplot:
            raise ValueError('`max_bioassay_per_subplot` larger than '
                             'number of colors or markers')
        else:
            btProtein_to_color_marker = None

        # Build a list of plots appropriate for `plotGrid`.
        # Code is complicated because we could have several curve
        # per population, and in that case need to share bioassay in
        # `all_subplots` among curves.
        population_bioassay = {v: [s for s in self.bioassay if v in self.populations[s]]
                               for v in self.allpopulations}
        plotlist = []
        vlines_list = []
        for population, title in zip(populations, titles):
            if ignore_population_btProtein and population in ignore_population_btProtein:
                ignore_btProtein = ignore_population_btProtein[population]
            else:
                ignore_btProtein = {}
            curvelist = []
            ibtProtein = 0
            population_shared_bioassay = [s for s in population_bioassay[population] if
                                          (s in bioassay) and (s in all_subplots) and
                                          (s not in ignore_btProtein)]
            population_unshared_bioassay = [s for s in population_bioassay[population] if
                                            (s in bioassay) and
                                            (s not in all_subplots) and
                                            (s not in ignore_btProtein)]
            unshared = int(bool(len(population_unshared_bioassay)))
            if len(population_shared_bioassay) > max_bioassay_per_subplot - unshared:
                raise ValueError(f"population {population} has too many subplot-shared "
                                 'bioassay (in `all_subplots`) relative to '
                                 'value of `max_bioassay_per_subplot`:\n'
                                 f"{population_shared_bioassay} is more than "
                                 f"{max_bioassay_per_subplot} populations.")
            shared_curvelist = []
            for btProtein in population_shared_bioassay + population_unshared_bioassay:
                if ibtProtein >= max_bioassay_per_subplot:
                    if multi_population_subplots:
                        plotlist.append((title, curvelist))
                        if vlines and (population in vlines):
                            vlines_list.append(vlines[population])
                        else:
                            vlines_list.append(None)
                        curvelist = list(shared_curvelist)
                        ibtProtein = len(curvelist)
                        assert ibtProtein < max_bioassay_per_subplot
                    else:
                        raise ValueError(f"population {population} has more than "
                                         '`max_bioassay_per_subplot` populations '
                                         'and `multi_population_subplots` is False')
                if btProtein_to_color_marker:
                    color, marker = btProtein_to_color_marker[btProtein]
                else:
                    color = colors[ibtProtein]
                    marker = markers[ibtProtein]
                curvelist.append({'btProtein': btProtein,
                                  'population': population,
                                  'replicate': 'average',
                                  'label': btProtein,
                                  'color': color,
                                  'marker': marker,
                                  })
                if btProtein in population_shared_bioassay:
                    shared_curvelist.append(curvelist[-1])
                ibtProtein += 1
            if curvelist:
                plotlist.append((title, curvelist))
                if vlines and (population in vlines):
                    vlines_list.append(vlines[population])
                else:
                    vlines_list.append(None)
        if not plotlist:
            raise ValueError('no curves for these populations / bioassay')

        # get number of columns
        if (nrow is not None) and (ncol is not None):
            raise ValueError('either `ncol` or `nrow` must be `None`')
        elif isinstance(nrow, int) and nrow > 0:
            ncol = math.ceil(len(plotlist) / nrow)
        elif not (isinstance(ncol, int) and ncol > 0):
            raise ValueError('`nrow` or `ncol` must be integer > 0')

        # convert plotlist to plots dict for `plotGrid`
        plots = {}
        vlines_axkey = {}
        assert len(plotlist) == len(vlines_list)
        for iplot, (plot, ivline) in enumerate(zip(plotlist, vlines_list)):
            irow = iplot // ncol
            icol = iplot % ncol
            plots[(irow, icol)] = plot
            if ivline:
                vlines_axkey[(irow, icol)] = ivline

        if btProtein_to_color_marker and 'orderlegend' not in kwargs:
            orderlegend = btProtein_to_color_marker.keys()
        else:
            orderlegend = None

        return self.plotGrid(plots,
                             orderlegend=orderlegend,
                             vlines=vlines_axkey,
                             **kwargs,
                             )

    def plotAverages(self,
                     *,
                     color='black',
                     marker='o',
                     **kwargs,
                     ):
        """Plot grid with a curve for each btProtein / population pair.

        Args:
            `color` (str)
                Color the curves.
            `marker` (str)
                Marker for the curves.
            `**kwargs`
                Other keyword arguments that can be passed to
                :meth:`CurveFits.plotReplicates`.

        Returns:
            The 2-tuple `(fig, axes)` of matplotlib figure and 2D axes array.

        """
        return self.plotReplicates(average_only=True,
                                   colors=[color],
                                   markers=[marker],
                                   **kwargs)

    def plotReplicates(self,
                       *,
                       ncol=4,
                       nrow=None,
                       bioassay='all',
                       populations='all',
                       colors=CBPALETTE,
                       markers=CBMARKERS,
                       subplot_titles='{btProtein} vs {population}',
                       show_average=False,
                       average_only=False,
                       **kwargs,
                       ):
        """Plot grid with replicates for each btProtein / population on same plot.

        Args:
            `ncol`, `nrow` (int or `None`)
                Specify one of these to set number of columns or rows.
            `bioassay` ('all' or list)
                bioassay to include on plot, in this order.
            `populations` ('all' or list)
                populations to include on plot, in this order.
            `colors` (iterable)
                List of colors for different replicates.
            `markers` (iterable)
                List of markers for different replicates.
            `subplot_titles` (str)
                Format string to build subplot titles from *btProtein* and *population*.
            `show_average` (bool)
                Include the replicate-average as a "replicate" in plots.
            `average_only` (bool)
                Show **only** the replicate-average on each plot. No
                legend in this case.
            `**kwargs`
                Other keyword arguments that can be passed to
                :meth:`CurveFits.plotGrid`.

        Returns:
            The 2-tuple `(fig, axes)` of matplotlib figure and 2D axes array.

        """
        try:
            subplot_titles.format(population='dummy', btProtein='dummy')
        except KeyError:
            raise ValueError(f"`subplot_titles` {subplot_titles} invalid. "
                             'Should have format keys only for population '
                             'and btProtein')

        bioassay, populations = self._bioassay_populations_lists(bioassay, populations)

        # get replicates and make sure there aren't too many
        nplottable = max(len(colors), len(markers))
        if average_only:
            replicates = ['average']
        else:
            replicates = collections.OrderedDict()
            if show_average:
                replicates['average'] = True
            for bioassay, population in itertools.product(bioassay, populations):
                if population in self.populations[btProtein]:
                    for replicate in self.replicates[(btProtein, population)]:
                        if replicate != 'average':
                            replicates[replicate] = True
            replicates = list(collections.OrderedDict(replicates).keys())
        if len(replicates) > nplottable:
            raise ValueError('Too many unique replicates. There are'
                             f"{len(replicates)} ({', '.join(replicates)}) "
                             f"but only {nplottable} `colors` or `markers`.")

        # build list of plots appropriate for `plotGrid`
        plotlist = []
        for btProtein, population in itertools.product(bioassay, populations):
            if population in self.populations[btProtein]:
                title = subplot_titles.format(btProtein=btProtein, population=population)
                curvelist = []
                for i, replicate in enumerate(replicates):
                    if replicate in self.replicates[(btProtein, population)]:
                        curvelist.append({'btProtein': btProtein,
                                          'population': population,
                                          'replicate': replicate,
                                          'label': {False: replicate,
                                                    True: None}[average_only],
                                          'color': colors[i],
                                          'marker': markers[i],
                                          })
                if curvelist:
                    plotlist.append((title, curvelist))
        if not plotlist:
            raise ValueError('no curves for these bioassay / populations')

        # get number of columns
        if (nrow is not None) and (ncol is not None):
            raise ValueError('either `ncol` or `nrow` must be `None`')
        elif isinstance(nrow, int) and nrow > 0:
            ncol = math.ceil(len(plotlist) / nrow)
        elif not (isinstance(ncol, int) and ncol > 0):
            raise ValueError('`nrow` or `ncol` must be integer > 0')

        # convert plotlist to plots dict for `plotGrid`
        plots = {}
        for iplot, plot in enumerate(plotlist):
            plots[(iplot // ncol, iplot % ncol)] = plot

        return self.plotGrid(plots, **kwargs)

    def _bioassay_populations_lists(self, bioassay, populations):
        """Check and build lists of `bioassay` and their `populations`.

        Args:
            `bioassay` ('all' or list)
            `populations` ('all' or list)

        Returns:
            The 2-tuple `(bioassay, populations)` which are checked lists.

        """
        if isinstance(bioassay, str) and bioassay == 'all':
            bioassay = self.bioassay
        else:
            extra_bioassay = set(bioassay) - set(self.bioassay)
            if extra_bioassay:
                raise ValueError(f"unrecognized bioassay: {extra_bioassay}")

        allpopulations = collections.OrderedDict()
        for btProtein in bioassay:
            for population in self.populations[btProtein]:
                allpopulations[population] = True
        allpopulations = list(allpopulations.keys())

        if isinstance(populations, str) and populations == 'all':
            populations = allpopulations
        else:
            extra_populations = set(populations) - set(allpopulations)
            if extra_populations:
                raise ValueError('unrecognized populations for specified '
                                 f"bioassay: {extra_populations}")

        return bioassay, populations

    def plotGrid(self,
                 plots,
                 *,
                 xlabel=None,
                 ylabel=None,
                 widthscale=1,
                 heightscale=1,
                 attempt_shared_legend=True,
                 fix_lims=None,
                 bound_ymin=0,
                 bound_ymax=1,
                 extend_lim=0.07,
                 markersize=6,
                 linewidth=1,
                 linestyle='-',
                 legendtitle=None,
                 orderlegend=None,
                 titlesize=14,
                 labelsize=15,
                 ticksize=12,
                 legendfontsize=12,
                 align_to_dmslogo_facet=False,
                 despine=False,
                 yticklocs=None,
                 sharex=True,
                 sharey=True,
                 vlines=None,
                 ):
        """Plot arbitrary grid of curves.

        Args:
            `plots` (dict)
                Plots to draw on grid. Keyed by 2-tuples `(irow, icol)`, which
                give row and column (0, 1, ... numbering) where plot should be
                drawn. Values are the 2-tuples `(title, curvelist)` where
                `title` is title for this plot (or `None`) and `curvelist`
                is a list of dicts keyed by:

                  - 'btProtein'
                  - 'population'
                  - 'replicate'
                  - 'label': label for this curve in legend, or `None`
                  - 'color'
                  - 'marker': https://matplotlib.org/api/markers_api.html

            `xlabel`, `ylabel` (`None`, str, or list)
                Labels for x- and y-axes. If `None`, use `conc_col`
                and `fracinf_col`, respectively. If str, use this shared
                for all axes. If list, should be same length as `plots`
                and gives axis label for each subplot.
            `widthscale`, `heightscale` (float)
                Scale width or height of figure by this much.
            `attempt_shared_legend` (bool)
                Share a single legend among plots if they all share
                in common the same label assigned to the same color / marker.
            `fix_lims` (dict or `None`)
                To fix axis limits, specify any of 'xmin', 'xmax', 'ymin',
                or 'ymax' with specified limit.
            `bound_ymin`, `bound_ymax` (float or `None`)
                Make y-axis min and max at least this small / large.
                Ignored if using `fix_lims` for that axis limit.
            `extend_lim` (float)
                For all axis limits not in `fix_lims`, extend this fraction
                of range above and below bounds / data limits.
            `markersize` (float)
                Size of point marker.
            `linewidth` (float)
                Width of line.
            `linestyle` (str)
                Line style.
            `legendtitle` (str or `None`)
                Title of legend.
            `orderlegend` (`None` or list)
                If specified, place legend labels in this order.
            `titlesize` (float)
                Size of subplot title font.
            `labelsize` (float)
                Size of axis label font.
            `ticksize` (float)
                Size of axis tick fonts.
            `legendfontsize` (float)
                Size of legend fonts.
            `align_to_dmslogo_facet` (`False` or dict)
                Make plot vertically alignable to ``dmslogo.facet_plot``
                with same number of rows; dict should have keys for
                `height_per_ax`, `hspace`, `tmargin`, and `bmargin` with
                same meaning as ``dmslogo.facet_plot``. Also
                `right` and `left` for passing to ``subplots_adjust``.
            `despine` (bool)
                Remove top and right spines from plots.
            `yticklocs` (`None` or list)
                Same meaning as for :meth:`neutcurve.hillcurve.HillCurve.plot`.
            `sharex` (bool)
                Share x-axis scale among plots.
            `sharey` (bool)
                Share y-axis scale among plots.
            `vlines` (dict or `None`)
                Vertical lines to draw. Keyed by 2-tuples `(irow, icol)`, which
                give row and column of plot in grid (0, 1, ... numbering).
                Values are lists of dicts with a key 'x' giving the x-location
                of the vertical line, and optionally keys 'linewidth',
                'color', and 'linestyle'.

        Returns:
            The 2-tuple `(fig, axes)` of matplotlib figure and 2D axes array.

        """
        vline_defaults = {'linewidth': 1.5,
                          'color': 'gray',
                          'linestyle': '--',
                          }

        if not plots:
            raise ValueError('empty `plots`')

        # get number of rows / cols, curves, and data limits
        nrows = ncols = None
        if fix_lims is None:
            fix_lims = {}
        lims = {key: {} for key in plots.keys()}
        for (irow, icol), (_title, curvelist) in plots.items():
            if irow < 0:
                raise ValueError('invalid row index `irow` < 0')
            if icol < 0:
                raise ValueError('invalid row index `icol` < 0')
            if nrows is None:
                nrows = irow + 1
            else:
                nrows = max(nrows, irow + 1)
            if ncols is None:
                ncols = icol + 1
            else:
                ncols = max(ncols, icol + 1)
            for curvedict in curvelist:
                curve = self.getCurve(btProtein=curvedict['btProtein'],
                                      population=curvedict['population'],
                                      replicate=curvedict['replicate']
                                      )
                # XXXXXXXXXXXXXXXX
                curvedict['curve'] = curve
                for lim, attr, f in [('xmin', 'cs', min), ('xmax', 'cs', max),
                                     ('ymin', 'fs', min), ('ymax', 'fs', max)]:
                    if lim in fix_lims:
                        lims[(irow, icol)][lim] = fix_lims[lim]
                    else:
                        val = f(getattr(curve, attr))
                        if lim in lims[(irow, icol)]:
                            val = f(val, lims[(irow, icol)][lim])
                        if lim == 'ymin' and (bound_ymin is not None):
                            lims[(irow, icol)][lim] = min(val, bound_ymin)
                        elif lim == 'ymax' and (bound_ymax is not None):
                            lims[(irow, icol)][lim] = max(val, bound_ymax)
                        else:
                            lims[(irow, icol)][lim] = val

        for share, axtype in [(sharex, 'x'), (sharey, 'y')]:
            if share:
                for limtype, limfunc in [('min', min), ('max', max)]:
                    lim = limfunc(lims[key][axtype + limtype] for key in lims)
                    for key in lims.keys():
                        lims[key][axtype + limtype] = lim

        # check and then extend limits
        for key in plots.keys():
            if lims[key]['xmin'] <= 0:
                raise ValueError('xmin <= 0, which is not allowed')
            yextent = lims[key]['ymax'] - lims[key]['ymin']
            if yextent <= 0:
                raise ValueError('no positive extent for y-axis')
            if 'ymin' not in fix_lims:
                lims[key]['ymin'] -= yextent * extend_lim
            if 'ymax' not in fix_lims:
                lims[key]['ymax'] += yextent * extend_lim
            xextent = math.log(lims[key]['xmax']) - math.log(lims[key]['xmin'])
            if xextent <= 0:
                raise ValueError('no positive extent for x-axis')
            if 'xmin' not in fix_lims:
                lims[key]['xmin'] = math.exp(math.log(lims[key]['xmin']) -
                                             xextent * extend_lim)
            if 'xmax' not in fix_lims:
                lims[key]['xmax'] = math.exp(math.log(lims[key]['xmax']) +
                                             xextent * extend_lim)

        if align_to_dmslogo_facet:
            import dmslogo.facet
            hparams = dmslogo.facet.height_params(
                nrows,
                align_to_dmslogo_facet['height_per_ax'],
                align_to_dmslogo_facet['hspace'],
                align_to_dmslogo_facet['tmargin'],
                align_to_dmslogo_facet['bmargin'],
            )
            height = hparams['height']
        else:
            height = (1 + 2.25 * nrows) * heightscale

        width = (1 + 3 * ncols) * widthscale
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex=sharex,
                                 sharey=sharey,
                                 squeeze=False,
                                 figsize=(width, height),
                                 )

        # set limits for share axes
        for irow, icol in plots.keys():
            axes[irow, icol].set_xlim(lims[irow, icol]['xmin'],
                                      lims[irow, icol]['xmax'])
            axes[irow, icol].set_ylim(lims[irow, icol]['ymin'],
                                      lims[irow, icol]['ymax'])

        # make plots
        shared_legend = attempt_shared_legend
        kwargs_tup_to_label = {}  # used to determine if shared legend
        legend_handles = collections.defaultdict(list)
        shared_legend_handles = []  # handles if using shared legend
        for i, ((irow, icol), (title, curvelist)) in enumerate(plots.items()):
            ax = axes[irow, icol]
            ax.set_title(title, fontsize=titlesize)
            for curvedict in curvelist:
                kwargs = {'color': curvedict['color'],
                          'marker': curvedict['marker'],
                          'linestyle': linestyle,
                          'linewidth': linewidth,
                          'markersize': markersize,
                          }
                if isinstance(xlabel, list):
                    ixlabel = xlabel[i]
                else:
                    ixlabel = None
                if isinstance(ylabel, list):
                    iylabel = ylabel[i]
                else:
                    iylabel = None
                curvedict['curve'].plot(ax=ax,
                                        xlabel=ixlabel,
                                        ylabel=iylabel,
                                        yticklocs=yticklocs,
                                        **kwargs,
                                        )
                label = curvedict['label']
                if label:
                    handle = Line2D(xdata=[],
                                    ydata=[],
                                    label=label,
                                    **kwargs,
                                    )
                    legend_handles[(irow, icol)].append(handle)
                    if shared_legend:
                        kwargs_tup = tuple(sorted(kwargs.items()))
                        if kwargs_tup in kwargs_tup_to_label:
                            if kwargs_tup_to_label[kwargs_tup] != label:
                                shared_legend = False
                        else:
                            kwargs_tup_to_label[kwargs_tup] = label
                            shared_legend_handles.append(handle)
            ax.tick_params('both', labelsize=ticksize, bottom=True, left=True,
                           right=False, top=False)
            if despine:
                import dmslogo.utils
                dmslogo.utils.despine(ax=ax)
            if vlines and ((irow, icol) in vlines):
                for vline in vlines[(irow, icol)]:
                    vline_d = vline_defaults.copy()
                    for key, val in vline.items():
                        vline_d[key] = val
                    ax.axvline(vline_d['x'],
                               linestyle=vline_d['linestyle'],
                               linewidth=vline_d['linewidth'],
                               color=vline_d['color'])

        # draw legend(s)
        legend_kwargs = {'fontsize': legendfontsize,
                         'numpoints': 1,
                         'markerscale': 1,
                         'handlelength': 1,
                         'labelspacing': 0.1,
                         'handletextpad': 0.4,
                         'frameon': True,
                         'borderaxespad': 0.1,
                         'borderpad': 0.2,
                         'title': legendtitle,
                         'title_fontsize': legendfontsize + 1,
                         'framealpha': 0.6,
                         }

        def _ordered_legend(hs):
            """Get ordered legend handles."""
            if not orderlegend:
                return hs
            else:
                order_dict = {h: i for i, h in enumerate(orderlegend)}
                h_labels = [h.get_label() for h in hs]
                extra_hs = set(h_labels) - set(orderlegend)
                if extra_hs:
                    raise ValueError('there are legend handles not in '
                                     f"`orderlegend`: {extra_hs}")
                return [h for _, h in sorted(zip(h_labels, hs),
                                             key=lambda x: order_dict[x[0]])]

        if shared_legend and shared_legend_handles:
            if align_to_dmslogo_facet:
                right = align_to_dmslogo_facet['right']
                ranchor = right + 0.15 * (1 - right)
            else:
                ranchor = 1
            shared_legend_handles = _ordered_legend(shared_legend_handles)
            # shared legend as here: https://stackoverflow.com/a/17328230
            fig.legend(handles=shared_legend_handles,
                       labels=[h.get_label() for h in shared_legend_handles],
                       loc='center left',
                       bbox_to_anchor=(ranchor, 0.5),
                       bbox_transform=fig.transFigure,
                       **legend_kwargs,
                       )
        elif legend_handles:
            for (irow, icol), handles in legend_handles.items():
                ax = axes[irow, icol]
                handles = _ordered_legend(handles)
                ax.legend(handles=handles,
                          labels=[h.get_label() for h in handles],
                          loc='lower left',
                          **legend_kwargs,
                          )

        # hide unused axes
        for irow, icol in itertools.product(range(nrows), range(ncols)):
            if (irow, icol) not in plots:
                axes[irow, icol].set_axis_off()

        # common axis labels as here: https://stackoverflow.com/a/53172335
        bigax = fig.add_subplot(111, frameon=False)
        bigax.grid(False)
        bigax.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False, which='both')
        if xlabel is None:
            bigax.set_xlabel(self.conc_col, fontsize=labelsize, labelpad=10)
        elif not isinstance(xlabel, list):
            bigax.set_xlabel(xlabel, fontsize=labelsize, labelpad=10)
        if ylabel is None:
            bigax.set_ylabel(self.fracinf_col, fontsize=labelsize, labelpad=10)
        elif not isinstance(ylabel, list):
            bigax.set_ylabel(ylabel, fontsize=labelsize, labelpad=10)

        if align_to_dmslogo_facet:
            fig.subplots_adjust(hspace=hparams['hspace'],
                                top=hparams['top'],
                                bottom=hparams['bottom'],
                                left=align_to_dmslogo_facet['left'],
                                right=align_to_dmslogo_facet['right'],
                                )
        else:
            fig.tight_layout()

        return fig, axes
    
    def calc_hill_bootstrap(self, hfit, ciLevs=(0.025, 0.975), numBoot=None):
        """Estimate Bootstrapped Confidence Intervals on Hill Model Parameters."""
        if not hasattr(hfit, 'fitted_values') or not hasattr(hfit, 'residuals'):
            raise ValueError("Object 'hfit' must have 'fitted_values' and 'residuals' attributes.")

        if hasattr(hfit, 'ciLevs'):
            print("Warning: Existing confidence intervals will be replaced.")
            hfit.ciLevs = None
            hfit.ciCoefs = None
            hfit.ciMat = None

        if numBoot is None:
            numBoot = int(max(min(10 / (1 - ciLevs[1] + ciLevs[0]), 1000), 100))

        bcoefs = numpy.empty((numBoot, 4))
        for i in range(numBoot):
            # Generate bootstrap sample
            bact = hfit.fitted_values + numpy.random.choice(hfit.residuals, size=len(hfit.residuals), replace=True)

            # Refit the Hill model with the bootstrap sample
            try:
                tfit = HillCurve(
                    hfit.cs,
                    bact,
                    inhibitory_or_mortality=hfit._inhibitory_or_mortality,
                    fixbottom=hfit.bottom,
                    fixtop=hfit.top
                )
                bcoefs[i, :] = tfit.coefficients
            except Exception as e:
                bcoefs[i, :] = numpy.nan  # Flag this row as invalid

        bcoefs = bcoefs[~numpy.isnan(bcoefs).any(axis=1)]
        if bcoefs.shape[0] < 10:
            raise RuntimeError("Too few successful bootstrap fits to calculate confidence intervals.")

        qmat = numpy.quantile(bcoefs, ciLevs, axis=0)

        hfit.ciLevs = ciLevs
        hfit.ciCoefs = bcoefs
        hfit.ciMat = qmat

        return hfit


    def calc_hill_conf_int(self, hfit, parfunc, civals=None):
        """Estimate a confidence interval on a Hill model property using bootstrapped coefficients."""
        if not hasattr(hfit, 'ciCoefs'):
            raise ValueError("Input 'hfit' must have bootstrapped coefficients. Run `calc_hill_bootstrap` first.")
        if not callable(parfunc):
            raise ValueError("Input 'parfunc' must be a callable function.")
    
        if civals is None:
            civals = hfit.ciLevs
    
        outval = parfunc(hfit.coefficients)
        outmat = numpy.empty((len(outval), len(hfit.ciCoefs)))
    
        for b in range(hfit.ciCoefs.shape[0]):
            outmat[:, b] = parfunc(hfit.ciCoefs[b, :])
    
        outci = numpy.quantile(outmat, civals, axis=1)
        fullout = numpy.column_stack((outci[0, :], outval, outci[1, :]))
    
        return fullout



    def fit_hill_model(self, cs, fs, *args, **kwargs):
        return HillCurve(cs, fs, *args, **kwargs)

