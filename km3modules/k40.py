# coding=utf-8
# Filename: k40.py
# pylint: disable=locally-disabled
"""
A collection of k40 related functions and modules.

"""
from __future__ import division, absolute_import, print_function

import os
from itertools import combinations
from collections import defaultdict
from functools import partial
from datetime import datetime

from scipy import optimize
import numpy as np
import h5py
import pickle

import km3pipe as kp

__author__ = "Jonas Reubelt"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

log = kp.logger.logging.getLogger(__name__)  # pylint: disable=C0103
# log.setLevel(logging.DEBUG)


class K40BackgroundSubtractor(kp.Module):
    """Subtracts random coincidence background from K40 data

    Required Services
    -----------------
    'MeanPMTRates': dict (key=dom_id, value=list of pmt rates)
    
    Input Keys
    ----------
    'K40Counts': dict (key=dom_id, value=matrix of k40 counts 465x(dt*2+1))

    Output Keys
    -----------
    'K40Counts': dict, Corrected K40 counts
    
    """
    def configure(self):
        self.combs = list(combinations(range(31), 2))

    def process(self, blob):
        dom_ids = list(blob['K40Counts'].keys())
        mean_rates = self.services['MeanPMTRates']
        corrected_counts = {}
        for dom_id in dom_ids:
            pmt_rates = mean_rates[dom_id]
            k40_counts = blob['K40Counts'][dom_id]

            bg_rates = []
            for c in self.combs:
                bg_rates.append(pmt_rates[c[0]]*pmt_rates[c[1]]*1e-9)
            corrected_counts[dom_id] = (k40_counts.T - np.array(bg_rates)).T
        blob["K40Counts"] = corrected_counts
        return blob


class IntraDOMCalibrator(kp.Module):
    """Intra DOM calibrator which performs the calibration from K40Counts.

    Input Keys
    ----------
    'K40Counts': dict (key=dom_id, value=matrix of k40 counts 465x(dt*2+1))

    Output Keys
    -----------
    'IntraDOMCalibration': dict (key=dom_id, value=calibration)

    """
    def configure(self):
        det_id = self.get("det_id") or 14
        self.input_key = self.get("input_key") or 'K40Counts'
        self.detector = kp.hardware.Detector(det_id=det_id)

    def process(self, blob):
        print("Starting calibration:")
        blob["IntraDOMCalibration"] = {}
        for dom_id, data in blob[self.input_key].items():
            print(" calibrating DOM '{0}'".format(dom_id))
            try:
                calib = calibrate_dom(dom_id, data,
                                      self.detector,
                                      livetime=blob['Livetime'])
            except RuntimeError:
                log.error(" skipping DOM '{0}'.".format(dom_id))
            else:
                blob["IntraDOMCalibration"][dom_id] = calib
        return blob


class CoincidenceFinder(kp.Module):
    """Finds K40 coincidences in TimesliceFrames

    Input Keys
    ----------
    'TimesliceFrames': dict (key=dom_id, value=list of hits as tuples)

    Output Keys
    -----------
    'K40Counts': dict (key=dom_id, value=matrix (465,(dt*2+1)))
    'Livetime': float

    """
    def configure(self):
        self.accumulate = self.get("accumulate") or 100
        self.tmax = self.get("tmax") or 20
        self.counts = defaultdict(partial(np.zeros, (465, self.tmax * 2 + 1)))
        self.n_timeslices = 0
        self.start_time = datetime.utcnow()

    def process(self, blob):
        for dom_id, hits in blob['TimesliceFrames'].items():
            hits.sort(key=lambda x: x[1])
            coinces = self.mongincidence([t for (_,t,_) in hits],
                                         [t for (t,_,_) in hits])

            combs = list(combinations(range(31), 2))
            for pmt_pair, t in coinces:
                if pmt_pair[0] > pmt_pair[1]:
                    pmt_pair = (pmt_pair[1], pmt_pair[0])
                    t = -t
                self.counts[dom_id][combs.index(pmt_pair), t+self.tmax] += 1

        self.n_timeslices += 1
        if self.n_timeslices == self.accumulate:
            print("Calibrating DOMs")
            blob["K40Counts"] = self.counts
            blob["Livetime"] = self.n_timeslices / 10
            pickle.dump({'data': self.counts, 'livetime': self.n_timeslices / 10}, open("coinc_counts.p", "wb"))
            self.n_timeslices = 0
            self.counts = defaultdict(partial(np.zeros, (465, 41)))
        return blob

    def mongincidence(self, times, tdcs):
        coincidences = []
        cur_t = 0
        las_t = 0
        for t_idx, t in enumerate(times):
            cur_t = t
            diff = cur_t - las_t
            if diff <= self.tmax and t_idx > 0 and tdcs[t_idx - 1] != tdcs[t_idx]:
                coincidences.append(((tdcs[t_idx - 1], tdcs[t_idx]), diff))
            las_t = cur_t
        return coincidences


def calibrate_dom(dom_id, data, detector, livetime=None, fixed_ang_dist=None,
                  auto_scale=False, ad_fit_shape='pexp', fit_background=True):
    """Calibrate intra DOM PMT time offsets, efficiencies and sigmas

        Parameters
        ----------
        dom_id: DOM ID
        data: dict of coincidences or root or hdf5 file
        detector: instance of detector class
        livetime: data-taking duration [s]
        fixed_ang_dist: fixing angular distribution e.g. for data mc comparison
        auto_scale: auto scales the fixed angular distribution to the data

        Returns
        -------
        return_data: dictionary with fit results
    """

    if isinstance(data, str):
        filename = data
        loaders = {'.h5': load_k40_coincidences_from_hdf5,
                   '.root': load_k40_coincidences_from_rootfile}
        try:
            loader = loaders[os.path.splitext(filename)[1]]
        except KeyError:
            log.critical('File format not supported.')
            raise IOError
        else:
            data, livetime = loader(filename, dom_id)

    try:
        rates, means, sigmas, popts, pcovs = fit_delta_ts(data, livetime,
                                                fit_background=fit_background)
    except:
        return 0
    angles = calculate_angles(detector)
    rate_errors = np.array([np.diag(pc)[2] for pc in pcovs])
    mean_errors = np.array([np.diag(pc)[0] for pc in pcovs])
    scale_factor = None
    if type(fixed_ang_dist) in (list, np.ndarray):
        if auto_scale:
            scale_factor = np.mean(rates[angles<1.5]) / np.mean(fixed_ang_dist[angles<1.5])
            fitted_rates = fixed_ang_dist * scale_factor
        else:
            fitted_rates = fixed_ang_dist
        exp_popts = []
        print('Using fixed angular distribution')
    else:
        fitted_rates, exp_popts, exp_pcov = fit_angular_distribution(angles,
                                                           rates,
                                                           rate_errors,
                                                           shape=ad_fit_shape)
    #t0_weights = np.array([0. if a>1. else 1. for a in angles])
    opt_t0s = minimize_t0s(means, fitted_rates)
    opt_sigmas = minimize_sigmas(sigmas, fitted_rates)
    opt_qes = minimize_qes(fitted_rates, rates, fitted_rates)
    corrected_means = correct_means(means, opt_t0s.x)
    corrected_rates = correct_rates(rates, opt_qes.x)
    rms_means, rms_corrected_means = calculate_rms_means(means,
                                                         corrected_means)
    rms_rates, rms_corrected_rates = calculate_rms_rates(rates, fitted_rates,
                                                         corrected_rates)
    return_data = {'opt_t0s': opt_t0s, 'opt_qes': opt_qes, 'data': data,
                   'means': means, 'rates': rates, 'fitted_rates': fitted_rates,
                   'angles': angles, 'corrected_means': corrected_means,
                   'corrected_rates': corrected_rates,
                   'rms_means': rms_means,
                   'rms_corrected_means': rms_corrected_means,
                   'rms_rates': rms_rates,
                   'rms_corrected_rates': rms_corrected_rates,
                   'gaussian_popts': popts,
                   'livetime': livetime,
                   'exp_popts': exp_popts,
                   'exp_pcov': exp_pcov,
                   'scale_factor': scale_factor,
                   'opt_sigmas': opt_sigmas,
                   'sigmas': sigmas}
    return return_data





def load_k40_coincidences_from_hdf5(filename, dom_id):
    """Load k40 coincidences from hdf5 file

    Parameters
    ----------
    filename: filename of hdf5 file
    dom_id: DOM ID

    Returns
    -------
    data: numpy array of coincidences
    livetime: duration of data-taking
    """

    with h5py.File(filename, 'r') as h5f:
        data = h5f['/k40counts/{0}'.format(dom_id)]
        livetime = data.attrs['livetime']
        data = np.array(data)

    return data, livetime


def load_k40_coincidences_from_rootfile(filename, dom_id):
    """Load k40 coincidences from JMonitorK40 ROOT file

    Parameters
    ----------
    filename: root file produced by JMonitorK40
    dom_id: DOM ID

    Returns
    -------
    data: numpy array of coincidences
    dom_weight: weight to apply to coincidences to get rate in Hz
    """

    from ROOT import TFile
    root_file_monitor = TFile( filename, "READ" )
    dom_name = str(dom_id) + ".2S"
    histo_2d_monitor = root_file_monitor.Get(dom_name)
    data = []
    #for c in jmonitork40_comb_indices:
    for c in range(1, histo_2d_monitor.GetNbinsX() + 1):
        combination = []
        for b in range(1, histo_2d_monitor.GetNbinsY() + 1):
            combination.append(histo_2d_monitor.GetBinContent(c, b))
        data.append(combination)

    weights = {}
    weights_histo = root_file_monitor.Get('weights_hist')
    try:
        for i in range(1, weights_histo.GetNbinsX() + 1):
            # we have to read all the entries, unfortunately
            weight = weights_histo.GetBinContent(i)
            label = weights_histo.GetXaxis().GetBinLabel(i)
            weights[label[3:]] = weight
        dom_weight = weights[str(dom_id)]
    except AttributeError:
        log.info("Weights histogram broken or not found, setting weight to 1.")
        dom_weight = 1.
    return np.array(data), dom_weight


def gaussian(x, mean, sigma, rate, offset):
    return rate / np.sqrt(2 * np.pi) /  \
           sigma * np.exp(-0.5*(x-mean)**2 / sigma**2) + offset


def fit_delta_ts(data, livetime, fit_background=True):
    """Fits gaussians to delta t for each PMT pair.

    Parameters
    ----------
    data: 2d np.array: x = PMT combinations (465), y = time, entry = frequency
    livetime: length of data taking in seconds

    Returns
    -------
    numpy arrays with rates and means for all PMT combinations
    """

    data = data / livetime
    start = -(data.shape[1] - 1) / 2
    end = -start + 1
    xs = np.arange(start, end)
    print (start, end)

    rates = []
    sigmas = []
    means = []
    popts = []
    pcovs = []
    if fit_background:
        offset_start = 0.1
        offset_bounds = (0, 5)
    else:
        offset_start = 0
        offset_bounds = (0, 0)
    for combination in data:
        mean0 = np.argmax(combination) + start
        try:
            popt, pcov = optimize.curve_fit(gaussian, xs, combination,
                                    p0=[mean0, 4., 5., offset_start],
                                    bounds = ([-20, 0, 0, offset_bounds[0]],
                                              [20, 10, 10, offset_bounds[1]]))
        except RuntimeError:
            popt = (0, 0, 0, 0)
        rates.append(popt[2])
        means.append(popt[0])
        sigmas.append(popt[1])
        popts.append(popt)
        pcovs.append(pcov)
    return np.array(rates), np.array(means), np.array(sigmas), np.array(popts), np.array(pcovs)


def calculate_angles(detector):
    """Calculates angles between PMT combinations according to positions in
    detector_file

    Parameters
    ----------
    detector_file: file from which to read the PMT positions (.detx)

    Returns
    -------
    angles: numpy array of angles between all PMT combinations

    """
    angles = []
    pmt_combinations = list(combinations(range(31), 2))
    pmt_angles = detector.pmt_angles
    for first, second in pmt_combinations:
        angles.append(kp.math.angle_between(np.array(pmt_angles[first]),
                                            np.array(pmt_angles[second])))
    return np.array(angles)


def exponential_polinomial(x, p1, p2, p3, p4):
    return  1 * np.exp(p1 + x * (p2 + x * (p3 + x * p4)))

def exponential(x, a, b):
    return a * np.exp(b * x)

def fit_angular_distribution(angles, rates, rate_errors, shape='pexp'):
    """Fits angular distribution of rates.

    Parameters
    ----------
    rates: numpy array with rates for all PMT combinations
    angles: numpy array with angles for all PMT combinations
    shape: which function to fit exp for exponential or pexp for exponential_polinomial

    Returns
    -------
    fitted_rates: numpy array of fitted rates (fit_function(angles, *popt))

    """
    if shape == 'exp':
        fit_function = exponential
        p0 = [-0.91871169,  2.72224241, -1.19065965,  1.48054122]
    if shape == 'pexp':
        fit_function = exponential_polinomial
        p0 = [ 0.34921202,  2.8629577 ]

    cos_angles = np.cos(angles)
    popt, pcov = optimize.curve_fit(fit_function, cos_angles, rates, sigma=rate_errors)
    fitted_rates = fit_function(cos_angles, *popt)
    return fitted_rates, popt, pcov


def minimize_t0s(means, weights):
    """Varies t0s to minimize the deviation of the gaussian means from zero.

    Parameters
    ----------
    fitted_rates: numpy array of fitted rates from fit_angular_distribution
    means: numpy array of means of all PMT combinations
    weights: numpy array of weights for the squared sum

    Returns
    -------
    opt_t0s: optimal t0 values for all PMTs

    """
    def make_quality_function(means, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(t0s):
            sq_sum = 0
            for mean, comb, weight in zip(means, combs, weights):
                sq_sum += ((mean - (t0s[comb[1]] - t0s[comb[0]])) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(means, weights)
    #t0s = np.zeros(31)
    t0s = np.random.rand(31)
    bounds = [(0, 0)]+[(-10., 10.)] * 30
    opt_t0s = optimize.minimize(qfunc, t0s, bounds=bounds)
    return opt_t0s

def minimize_sigmas(sigmas, weights):
    """Varies sigmas to minimize gaussian sigma12 - sqrt(sigma1² + sigma2²).

    Parameters
    ----------
    sigmas: numpy array of fitted sigmas of gaussians
    weights: numpy array of weights for the squared sum

    Returns
    -------
    opt_sigmas: optimal sigma values for all PMTs

    """
    def make_quality_function(sigmas, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(s):
            sq_sum = 0
            for sigma, comb, weight in zip(sigmas, combs, weights):
                sq_sum += ((sigma - np.sqrt(s[comb[1]]**2 + s[comb[0]]**2)) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(sigmas, weights)
    s = np.ones(31) * 2.5
    #s = np.random.rand(31)
    bounds =[(0., 5.)] * 31
    opt_sigmas = optimize.minimize(qfunc, s, bounds=bounds)
    return opt_sigmas

def minimize_qes(fitted_rates, rates, weights):
    """Varies QEs to minimize the deviation of the rates from the fitted_rates.

    Parameters
    ----------
    fitted_rates: numpy array of fitted rates from fit_angular_distribution
    rates: numpy array of rates of all PMT combinations
    weights: numpy array of weights for the squared sum

    Returns
    -------
    opt_qes: optimal qe values for all PMTs

    """
    def make_quality_function(fitted_rates, rates, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(qes):
            sq_sum = 0
            for fitted_rate, comb, rate, weight  \
                    in zip(fitted_rates, combs, rates, weights):
                sq_sum += ((rate / qes[comb[0]] / qes[comb[1]]
                            - fitted_rate) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(fitted_rates, rates, weights)
    qes = np.ones(31)
    bounds = [(0.1, 2.)] * 31
    opt_qes = optimize.minimize(qfunc, qes, bounds=bounds)
    return opt_qes


def correct_means(means, opt_t0s):
    """Applies optimal t0s to gaussians means.

    Should be around zero afterwards.

    Parameters
    ----------
    means: numpy array of means of gaussians of all PMT combinations
    opt_t0s: numpy array of optimal t0 values for all PMTs

    Returns
    -------
    corrected_means: numpy array of corrected gaussian means for all PMT combs

    """
    combs = list(combinations(range(31), 2))
    corrected_means = np.array([(opt_t0s[comb[1]] - opt_t0s[comb[0]])
                                - mean for mean, comb in zip(means, combs)])
    return corrected_means


def correct_rates(rates, opt_qes):
    """Applies optimal qes to rates.

    Should be closer to fitted_rates afterwards.

    Parameters
    ----------
    rates: numpy array of rates of all PMT combinations
    opt_qes: numpy array of optimal qe values for all PMTs

    Returns
    -------
    corrected_rates: numpy array of corrected rates for all PMT combinations
    """

    combs = list(combinations(range(31), 2))
    corrected_rates = np.array([rate / opt_qes[comb[0]] / opt_qes[comb[1]]  \
                            for rate, comb in zip(rates, combs)])
    return corrected_rates


def calculate_rms_means(means, corrected_means):
    """Calculates RMS of means from zero before and after correction

    Parameters
    ----------
    means: numpy array of means of gaussians of all PMT combinations
    corrected_means: numpy array of corrected gaussian means for all PMT combs

    Returns
    -------
    rms_means: RMS of means from zero
    rms_corrected_means: RMS of corrected_means from zero
    """
    rms_means = np.sqrt(np.mean((means - 0)**2))
    rms_corrected_means = np.sqrt(np.mean((corrected_means - 0)**2))
    return rms_means, rms_corrected_means


def calculate_rms_rates(rates, fitted_rates, corrected_rates):
    """Calculates RMS of rates from fitted_rates before and after correction

    Parameters
    ----------
    rates: numpy array of rates of all PMT combinations
    corrected_rates: numpy array of corrected rates for all PMT combinations

    Returns
    -------
    rms_rates: RMS of rates from fitted_rates
    rms_corrected_rates: RMS of corrected_ratesrates from fitted_rates
    """
    rms_rates = np.sqrt(np.mean((rates - fitted_rates)**2))
    rms_corrected_rates = np.sqrt(np.mean((corrected_rates - fitted_rates)**2))
    return rms_rates, rms_corrected_rates


jmonitork40_comb_indices =  \
    np.array((254, 423, 424, 391, 392, 255, 204, 205, 126, 120, 121, 0,
    22, 12, 80, 81, 23, 48, 49, 148, 150, 96, 296, 221, 190, 191, 297, 312,
    313, 386, 355, 132, 110, 431, 42, 433, 113, 256, 134, 358, 192, 74,
    176, 36, 402, 301, 270, 69, 384, 2, 156, 38, 178, 70, 273, 404, 302,
    77, 202, 351, 246, 440, 133, 262, 103, 118, 44, 141, 34, 4, 64, 30,
    196, 91, 172, 61, 292, 84, 157, 198, 276, 182, 281, 410, 381, 289,
    405, 439, 247, 356, 102, 263, 119, 140, 45, 35, 88, 65, 194, 31,
    7, 60, 173, 82, 294, 158, 409, 277, 280, 183, 200, 288, 382, 406,
    212, 432, 128, 388, 206, 264, 105, 72, 144, 52, 283, 6, 19, 14,
    169, 24, 310, 97, 379, 186, 218, 59, 93, 152, 317, 304, 111, 387,
    129, 207, 104, 265, 73, 18, 53, 5, 284, 146, 168, 15, 308, 26,
    98, 92, 187, 58, 219, 380, 316, 154, 305, 112, 434, 257, 357, 135,
    193, 300, 177, 401, 37, 75, 68, 271, 1, 385, 159, 403, 179, 272,
    71, 39, 76, 303, 203, 213, 393, 248, 442, 298, 145, 184, 89, 377,
    315, 216, 57, 309, 27, 99, 8, 54, 16, 171, 287, 153, 21, 78,
    394, 441, 249, 299, 314, 185, 376, 90, 147, 56, 217, 25, 311, 100,
    286, 55, 170, 17, 9, 20, 155, 79, 425, 426, 383, 306, 220, 290,
    291, 307, 188, 189, 149, 151, 101, 86, 13, 50, 51, 87, 28, 29,
    3, 352, 399, 375, 274, 407, 197, 285, 180, 279, 83, 295, 160, 199,
    66, 174, 63, 33, 10, 95, 40, 400, 282, 275, 195, 408, 378, 278,
    181, 293, 85, 161, 32, 67, 62, 175, 201, 94, 11, 41, 435, 415,
    359, 360, 436, 347, 348, 258, 259, 318, 136, 162, 222, 223, 137, 114,
    115, 43, 451, 443, 266, 389, 335, 456, 208, 396, 363, 250, 238, 327,
    235, 107, 130, 215, 116, 343, 344, 452, 461, 462, 331, 332, 417, 226,
    324, 371, 372, 229, 240, 241, 163, 142, 267, 230, 412, 122, 428, 319,
    353, 227, 340, 166, 47, 108, 253, 138, 444, 411, 231, 427, 123, 320,
    46, 228, 165, 341, 354, 252, 109, 139, 455, 336, 395, 209, 364, 106,
    239, 234, 328, 251, 214, 131, 117, 373, 447, 243, 418, 164, 369, 325,
    460, 342, 329, 237, 224, 242, 448, 419, 339, 370, 459, 326, 167, 236,
    330, 225, 127, 365, 124, 333, 244, 450, 430, 397, 211, 260, 366, 429,
    334, 449, 245, 125, 210, 398, 261, 321, 420, 421, 422, 322, 367, 368,
    323, 345, 413, 232, 143, 268, 446, 361, 463, 464, 346, 453, 454, 416,
    374, 233, 337, 458, 349, 414, 457, 338, 350, 445, 269, 362, 390, 437, 438))

"""
jmonitork40_comb_indices =  \
    np.array((417, 418, 419, 420, 421, 422, 363, 364, 365, 366, 367, 368,
    318, 319, 320, 321, 322, 323, 156, 157, 158, 159, 160, 161, 96, 97, 98,
    99, 100, 101, 461, 369, 324, 371, 464, 427, 331, 237, 238, 333, 434, 415,
    339, 231, 175, 232, 342, 278, 184, 61, 62, 186, 281, 220, 162, 54, 13,
    56, 168, 459, 370, 325, 374, 423, 428, 328, 244, 239, 338, 343, 411, 346,
    226, 178, 229, 270, 271, 181, 68, 69, 191, 170, 216, 164, 50, 16, 58,
    462, 373, 326, 327, 429, 424, 337, 240, 245, 222, 345, 412, 347, 228, 179,
    180, 272, 273, 190, 70, 71, 48, 163, 217, 172, 57, 17, 463, 372, 234,
    332, 430, 431, 334, 241, 174, 230, 340, 416, 341, 233, 60, 185, 279, 280,
    187, 63, 12, 52, 165, 221, 166, 59, 460, 242, 235, 336, 425, 432, 330,
    223, 176, 225, 348, 413, 350, 64, 65, 189, 274, 275, 183, 49, 14, 55,
    173, 218, 169, 335, 236, 243, 329, 433, 426, 344, 224, 177, 227, 349, 414,
    188, 66, 67, 182, 276, 277, 171, 53, 15, 51, 167, 219, 387, 204, 128,
    209, 396, 443, 435, 263, 112, 120, 249, 375, 283, 73, 6, 85, 301, 310,
    306, 148, 22, 28, 154, 388, 208, 126, 211, 254, 451, 452, 256, 104, 105,
    282, 383, 285, 84, 1, 87, 144, 312, 313, 150, 24, 25, 395, 210, 129,
    110, 262, 436, 445, 248, 121, 72, 284, 376, 300, 86, 7, 18, 146, 307,
    314, 152, 29, 389, 205, 111, 118, 247, 446, 437, 265, 4, 81, 299, 377,
    287, 75, 19, 26, 149, 315, 308, 155, 390, 255, 102, 103, 257, 453, 454,
    80, 0, 83, 286, 384, 289, 145, 20, 21, 151, 316, 317, 444, 246, 119,
    113, 264, 438, 298, 82, 5, 74, 288, 378, 311, 147, 27, 23, 153, 309,
    351, 136, 42, 138, 354, 403, 194, 33, 34, 200, 410, 385, 292, 91, 3,
    94, 297, 359, 137, 44, 133, 399, 404, 196, 40, 35, 202, 290, 379, 303,
    92, 10, 79, 352, 132, 45, 192, 405, 400, 198, 36, 41, 88, 302, 380,
    294, 78, 11, 353, 139, 30, 195, 406, 407, 201, 37, 2, 90, 293, 386,
    296, 95, 360, 38, 31, 197, 401, 408, 203, 89, 8, 77, 295, 381, 305,
    193, 32, 39, 199, 409, 402, 291, 76, 9, 93, 304, 382, 355, 134, 46,
    141, 362, 455, 439, 251, 108, 124, 269, 356, 140, 43, 143, 258, 447, 448,
    260, 116, 117, 361, 142, 47, 106, 250, 440, 457, 268, 125, 357, 135, 107,
    122, 267, 458, 441, 253, 358, 259, 114, 115, 261, 449, 450, 456, 266, 123,
    109, 252, 442, 391, 212, 127, 214, 394, 397, 213, 130, 207, 392, 206, 131,
    393, 215, 398))
"""



