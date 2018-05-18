""" Classes to represent the metadata of a simulation run

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-08-18
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import datetime
import os
import socket
import wc_sim.sim_config
import wc_utils.util.git


class SimulationMetadata(object):
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

    def __eq__(self, other):
        """ Compare two simulation metadata objects

        Args:
            other (:obj:`SimulationMetadata`): other simulation metadata object

        Returns:
            :obj:`bool`: true if simulation metadata objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        attrs = 'model simulation run author'.split()
        for attr in attrs:
            if getattr(other, attr) != getattr(self, attr):
                return False

        return True

    def __ne__(self, other):
        """ Compare two simulation metadata objects

        Args:
            other (:obj:`SimulationMetadata`): other simulation metadata object

        Returns:
            :obj:`bool`: true if simulation metadata objects are semantically unequal
        """
        return not self.__eq__(other)


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

    def __eq__(self, other):
        """ Compare two model metadata objects

        Args:
            other (:obj:`ModelMetadata`): other model metadata object

        Returns:
            :obj:`bool`: true if model metadata objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        attrs = 'url branch revision'.split()
        for attr in attrs:
            if getattr(other, attr) != getattr(self, attr):
                return False

        return True

    def __ne__(self, other):
        """ Compare two model metadata objects

        Args:
            other (:obj:`ModelMetadata`): other model metadata object

        Returns:
            :obj:`bool`: true if model metadata objects are semantically unequal
        """
        return not self.__eq__(other)


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

    def __eq__(self, other):
        """ Compare two run metadata objects

        Args:
            other (:obj:`RunMetadata`): other run metadata object

        Returns:
            :obj:`bool`: true if run metadata objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        attrs = 'start_time run_time ip_address'.split()
        for attr in attrs:
            if getattr(other, attr) != getattr(self, attr):
                return False

        return True

    def __ne__(self, other):
        """ Compare two run metadata objects

        Args:
            other (:obj:`RunMetadata`): other run metadata object

        Returns:
            :obj:`bool`: true if run metadata objects are semantically unequal
        """
        return not self.__eq__(other)


class AuthorMetadata(object):
    """ Represents a simulation's author

    Attributes:
        name (:obj:`str`): authors' name
        email (:obj:`str`): author's email address
        username (:obj:`str`): authors' username
        organization (:obj:`str`): author's organization
        ip_address (:obj:`str`): author's ip address
    """

    def __init__(self, name, email, username, organization, ip_address):
        """ Construct a representation of the author of a simulation run

        Args:
            name (:obj:`str`): authors' name
            email (:obj:`str`): author's email address
            username (:obj:`str`): authors' username
            organization (:obj:`str`): author's organization
            ip_address (:obj:`str`): author's ip address
        """
        self.name = name
        self.email = email
        self.username = username
        self.organization = organization
        self.ip_address = ip_address

    def __eq__(self, other):
        """ Compare two author metadata objects

        Args:
            other (:obj:`AuthorMetadata`): other author metadata object

        Returns:
            :obj:`bool`: true if author metadata objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        attrs = ['name', 'email', 'username', 'organization', 'ip_address']
        for attr in attrs:
            if getattr(other, attr) != getattr(self, attr):
                return False

        return True

    def __ne__(self, other):
        """ Compare two author metadata objects

        Args:
            other (:obj:`AuthorMetadata`): other author metadata object

        Returns:
            :obj:`bool`: true if author metadata objects are semantically unequal
        """
        return not self.__eq__(other)
