# Filename: mc.py
# pylint: disable=C0103
"""
Monte Carlo related things.

"""
from __future__ import absolute_import, print_function, division

import particle
import numpy as np

from .logger import get_logger

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103


def geant2pdg(geant_code):
    """Convert GEANT particle ID to PDG"""
    try:
        return particle.Geant3ID(geant_code).to_pdgid()
    except KeyError:
        return 0


def pdg2name(pdg_id):
    """Convert PDG ID to human readable names"""
    try:
        return particle.Particle.from_pdgid(pdg_id).name
    except particle.particle.particle.InvalidParticle:
        log.debug("No particle with PDG ID %d found!" % pdg_id)
        return "N/A"


def name2pdg(name):
    """Return best match of a PDG ID for the given name"""
    return particle.Particle.from_string(name)


def most_energetic(df):
    """Grab most energetic particle from mc_tracks dataframe."""
    idx = df.groupby(['event_id'])['energy'].transform(max) == df['energy']
    return df[idx].reindex()


def leading_particle(df):
    """Grab leading particle (neutrino, most energetic bundle muon).

    Note: selecting the most energetic mc particle does not always select
    the neutrino! In some sub-percent cases, the post-interaction
    secondaries can have more energy than the incoming neutrino!

    aanet convention: mc_tracks[0] = neutrino
    so grab the first row

    if the first row is not unique (neutrinos are unique), it's a muon bundle
    grab the most energetic then
    """
    leading = df.groupby('event_id', as_index=False).first()
    unique = leading.type.unique()

    if len(unique) == 1 and unique[0] == 0:
        leading = most_energetic(df)
    return leading


def get_flavor(pdg_types):
    """Build a 'flavor' from the 'type' column."""
    import pandas as pd
    return pd.Series(pdg_types).apply(pdg2name)


def _p_eq_nu(pdg_type):
    return np.abs(pdg_type) in {12, 14, 16}


def _p_eq_mu(pdg_type):
    return pdg_type == -13


def is_neutrino(pdg_types):
    """flavor string -> is_neutrino"""
    import pandas as pd
    return pd.Series(pdg_types).apply(_p_eq_nu)


def is_muon(pdg_types):
    """flavor string -> is_neutrino"""
    import pandas as pd
    return pd.Series(pdg_types).apply(_p_eq_mu)
