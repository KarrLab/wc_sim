""" Classes to represent the metadata of a simulation run

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-18
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import datetime
import os
import socket
import wc_sim.sim_config
import wc_utils.util.git


class Metadata(object):
    """ Represents the metadata of a simulation run

    Attributes:
        model (:obj:`ModelMetadata`): Information about the simulated model (e.g. revision)
        simulation (:obj:`wc_sim.sim_config.SimulationConfig`): Information about the simulation's
            configuration (e.g. perturbations, random seed)
        run (:obj:`RunMetadata`): Information about the simulation's run (e.g. start time, duration)
        author (:obj:`AuthorMetadata`): Information about the person who ran the simulation
            (e.g. name, email)
    """

    def __init__(self, model, simulation, run, author):
        self.model = model
        self.simulation = simulation
        self.run = run
        self.author = author


class ModelMetadata(object):
    """ Represents the simulated model (repository, branch, revision)

    Attributes:
        url (:obj:`str`): URL of the repository of the simulated model
        branch (:obj:`str`): repository branch
        revision (:obj:`str`): repository revision
    """

    def __init__(self, url, branch, revision):
        """ Construct a representation of a simulation model

        Args:
            url (:obj:`str`): URL of the repository of the simulated model
            branch (:obj:`str`): repository branch
            revision (:obj:`str`): repository revision
        """

        self.url = url
        self.branch = branch
        self.revision = revision

    @staticmethod
    def create_from_repository(repo_path='.'):
        """ Collect a model's metadata from its repository and construct a Model
        representation of this metadata

        Args:
            repo_path (:obj:`str`): path to Git repository
        """

        md = wc_utils.util.git.get_repo_metadata(repo_path)
        return ModelMetadata(md.url, md.branch, md.revision)


Simulation = wc_sim.sim_config.SimulationConfig
""" Alias for :obj:`wc_sim.sim_config.SimulationConfig` """


class RunMetadata(object):
    """ Represent a simulation's run

    Attributes:
        start_time (:obj:`datetime.datetime`): simulation start time
        run_time (:obj:`float`): simulation run time in seconds
        ip_address (:obj:`str`): ip address of the machine that ran the simulation
    """

    def __init__(self, start_time=None, run_time=None, ip_address=None):
        """ Construct a representation of simulation run

        Args:
            start_time (:obj:`datetime.datetime`): simulation start time
            run_time (:obj:`float`): simulation run time in seconds
            ip_address (:obj:`str`): ip address of the machine that ran the simulation
        """

        self.start_time = start_time
        self.run_time = run_time
        self.ip_address = ip_address

    def record_start(self):
        self.start_time = datetime.datetime.now()

    def record_end(self):
        self.run_time = (datetime.datetime.now() - self.start_time).total_seconds()

    def record_ip_address(self):
        self.ip_address = socket.gethostbyname(socket.gethostname())


class AuthorMetadata(object):
    """ Represents a simulation's author

    Attributes:
        name (:obj:`str`): authors' name
        email (:obj:`str`): author's email address
        organization (:obj:`str`): author's organization
        ip_address (:obj:`str`): author's ip address
    """

    def __init__(self, name, email, organization, ip_address):
        """ Construct a representation of the author of a simulation run

        Args:
            name (:obj:`str`): authors' name
            email (:obj:`str`): author's email address
            organization (:obj:`str`): author's organization
            ip_address (:obj:`str`): author's ip address
        """

        self.name = name
        self.email = email
        self.organization = organization
        self.ip_address = ip_address
