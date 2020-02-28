# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import sys
import json
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


class DistributionFitter(object):
    """An object for fitting empirical distributions to theoretical distributions.
    Includes functionality for displaying fit empirical distributions as well as
    testing new data.
    
    Init Arguments:
        bins {int} -- Number of bins to group histograms by. (default: {200})
        ax {axis object} -- matplotlib axis object to place plot. (default: {None})
    """

    def __init__(self, bins=200, ax=None):
        self.bins=bins
        self.ax=ax
        self.best_distribution=st.norm
        self.best_params=(0.0, 1.0)
        self.best_sse=np.inf
        self.high_performance=[        
        st.alpha,st.anglit,st.arcsine,st.bradford,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.erlang,st.expon,st.f,st.fatiguelife,st.foldcauchy,st.foldnorm,
        st.genlogistic,st.gamma,st.gilbrat,st.gompertz,st.gumbel_r,st.gumbel_l,
        st.halfcauchy,st.halflogistic,st.halfnorm,st.hypsecant,st.invgamma,st.invgauss,
        st.johnsonsb,st.laplace,st.levy,st.levy_l,st.logistic,st.loggamma,st.lognorm,
        st.lomax,st.maxwell,st.pareto,st.powerlaw,st.rdist,st.reciprocal,st.rayleigh,
        st.semicircular,st.t,st.truncexpon,st.truncnorm,st.vonmises_line,st.wald,
        st.wrapcauchy,st.kstwobign,st.uniform]
        self.comprehensive_fit=[        
        st.alpha,st.anglit,st.arcsine,st.betaprime,st.bradford,st.cauchy,st.chi,
        st.chi2,st.cosine,st.dgamma,st.erlang,st.expon,st.exponnorm,st.f,st.fatiguelife,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.genlogistic,st.genpareto,st.gamma,
        st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,st.gumbel_l,st.halfcauchy,
        st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.nakagami,
        st.norm,st.pareto,st.pearson3,st.powerlaw,st.rdist,st.reciprocal,st.rayleigh,
        st.rice,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.vonmises,
        st.vonmises_line,st.wald,st.weibull_min,st.wrapcauchy,st.kstwobign,st.uniform]
        self.full_fit=[        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,
        st.cauchy,st.chi,st.chi2,st.cosine,st.dgamma,st.dweibull,st.erlang,st.expon,
        st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,st.foldcauchy,
        st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,
        st.genexpon,st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,
        st.gilbrat,st.gompertz,st.gumbel_r,st.gumbel_l,st.halfcauchy,st.halflogistic,
        st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,st.invweibull,
        st.johnsonsb,st.johnsonsu,st.ksone,st.laplace,st.levy,st.levy_l,st.logistic,
        st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,
        st.ncx2,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,
        st.rdist,st.reciprocal,st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,
        st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,st.vonmises,st.vonmises_line,
        st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy,st.kstwobign,st.ncf,st.nct,st.uniform]
        
        '''Slow: burr(4s), beta(2s), dweibull(2s), exponweib(3s), exponpow(2s),
        fisk(3.5s), frechet_r (1.7s), frechet_l (2s), gennorm(1.5s), genexpon(2.6s), genextreme(2.6s),
        gausshyper(6.2s), gengamma(2.4s), johnsonsu(1.5s), ksone (40s), loglaplace(1.5s), mielke(5s),
        nakagami(1.7s), ncx2(38s), pearson3(1.7s), powerlognorm (4.73s), powernorm(2.3s), rice(1.6s),
        recipinvgauss(2.4s), tukeylambda(2m17s), weibull_min(1.6s), weibull_max(2s), ncf(1m42s), nct(15s), betaprime(1.9),


        fast: gompertz, gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm, hypsecant, invgamma,
        invgauss, johnsonsb, laplace, levy, levy_l, logistic, loggamma, lognorm, lomax, maxwell, norm, pareto, 
        powerlaw, rdist, reciprocal, rayleigh, semicircular, t, truncexpon, truncnorm, vonmises_line, wald, 
        wrapcauchy, kstwobign, uniform, alpha, anglit, arcsine, bradford, cauchy, chi, chi2, cosine, dgamma,
        erlang, expon, f, fatiguelife, foldcauchy, foldnorm, genlogistic, gamma, gilbrat
        '''


    def load_data(self, data):
        """Takes JSON from file_path into dataframe and formats data. Saves 
        relevant data points. Finally, creates histogram values to be used in the
        fit_distribution_to_data function.
        
        Arguments:
            data {JSON} -- A JSON file of the data to be used.
        
        Returns:
            None
        """

        df = pd.DataFrame(data)
            
        # Save the data
        self.data = df['value']

        # Save Histogram Y values and buckets (self.x)
        self.y, self.x = np.histogram(self.data, bins=self.bins, density=True)
        # Change self.x to bucket centers
        self.x = (self.x + np.roll(self.x, -1))[:-1] / 2.0

        return None

    
    def fit_distribution_to_data(self, performance='high_performance'):
        """Iterates through a list of theoretical distributions, fitting each to
        self.data and calculates sum of squared errors. Finally, checks which 
        theoretical distribution has lowest SSE and saves it and the fit parameters.
        
        Arguments:
            performance {str} -- setting for number of distributions to test when fitting
            the data (default: {'fast'})
        Returns:
            Tuple -- Name of best fit distribution and relevant parameters.
        """

        # Use performance setting to choose distribution list to fit
        settings = {'high_performance', 'comprehensive_fit', 'full_fit'}
        if performance not in settings:
            raise ValueError('performance accepts only "high_performance", "comprehensive_fit", or\
            "full_fit" settings.')
        if performance == 'high_performance':
            self.distributions_list = self.high_performance
        elif performance == 'comprehensive_fit':
            self.distributions_list = self.comprehensive_fit
        elif performance == 'full_fit':
            self.distributions_list = self.full_fit

        # Estimate distribution parameters from data
        for distribution in self.distributions_list:
            
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(self.data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(self.x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(self.y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if self.ax:
                            pd.Series(pdf, self.x).plot(ax=self.ax)

                    except Exception:
                        continue

                    # identify if this distribution is better
                    if self.best_sse > sse > 0:
                        self.best_distribution = distribution
                        self.best_params = params
                        self.best_sse = sse
        
            except Exception:
                continue
    
        self.arg = self.best_params[:-2]
        self.loc = self.best_params[-2]
        self.scale = self.best_params[-1]

        return None

    
    def make_pdf(self, size=10000):
        """Takes parameters and the best theoretical distribution probability 
        density function and creates a distribution with which to test against.
        
        Keyword Arguments:
            size {int} -- Number of points in pdf. (default: {10000})
        
        Returns:
            None
        """

        # Get sane start and end points of distribution
        start = self.best_distribution.ppf(0.01, *self.arg, loc=self.loc, scale=self.scale)\
            if self.arg else self.best_distribution.ppf(0.01, loc=self.loc, scale=self.scale)

        end = self.best_distribution.ppf(0.99, *self.arg, loc=self.loc, scale=self.scale)\
            if self.arg else self.best_distribution.ppf(0.99, loc=self.loc, scale=self.scale)

        # Build PDF for best distribution
        x = np.linspace(start, end, size)
        y = self.best_distribution.pdf(x, loc=self.loc, scale=self.scale, *self.arg)

        # Save PDF as numpy array
        self.pdf = np.concatenate([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))], axis=1)

        return None

    
    def display_pdf(self, bins=500, density=False):
        '''Plots the distribution created using the best fit theoretical 
        distribution's probability density function.'''

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        ax1.hist((self.data * 1000), bins=bins, density=density, color='b')
        ax1.set_title(f'Energy Generation Histogram')
        ax1.set_xlabel('Amount of Energy Generated in KWH')
        ax1.set_ylabel('Frequency')
        ax2.plot(self.pdf[:,0], self.pdf[:,1])
        ax2.set_title(f'Fitted {self.best_distribution.name} Distribution')
        ax2.set_xlabel('Amount of Energy Generated in MWH')
        ax2.set_ylabel('Relative Likelihood')

        return None

    
    def test_common_cause_variation(self, file_path, confidence_interval=0.999):
        """[Checks values of data against common cause variation. Returns an
        array of booleans with index corresponding to input data signifying
        whether the datapoint is within the confidence interval or not (True
        within, False without).]
        
        Arguments:
            file_path {string} -- the path to the JSON file of data points to test
        
        Keyword Arguments:
            confidence_interval {float} -- the test interval (default: {0.997})
        
        Returns:
            Results {array-like} -- True if within confidence interval else False
        """

        # Check file_path input is valid str type
        assert isinstance(file_path, str), "ERROR: file path provided in not string"
        # Check for JSON files
        assert file_path.endswith('.json'), "ERROR: file is not of type JSON"

        # Load data from JSON at file_path
        data = pd.read_json('test.json')['value'].values

        # Ensure data is an array of floats
        test_array = 1.0 * data

        # Define upper confidence limit and lower confidence limit
        lower_limit, upper_limit = self.best_distribution.interval(confidence_interval, *self.arg, loc=self.loc, scale=self.scale)
        
        return (test_array > lower_limit) & (test_array < upper_limit)


if __name__ == '__main__':
    # initialize model
    # find where data is coming from
    # test data against model
    model = DistributionFitter()
    model.load_data(sys.argv[1])
    model.fit_distribution_to_data()
    model.make_pdf()
    model.display_pdf()

    