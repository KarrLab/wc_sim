""" Classes to represent WC simulation configurations and import/export configurations to/from SED-ML

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-08-19
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

# .. todo:: comprehensively test the __eq__ methods
# .. todo:: support add, remove changes
# .. todo:: represent other stopping conditions e.g. cell divided
# .. todo:: represent observables in SED-ML

from dataclasses import dataclass
from enum import Enum
import dataclasses
import libsedml
import math
import numpy
import warnings

from de_sim.simulation_config import SimulationConfig
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_utils.util.misc import EnhancedDataClass
import wc_lang.transform


@dataclass
class WCSimulationConfig(EnhancedDataClass):
    """ Whole-cell simulation configuration

    - A simulation configuration for DE-Sim
    - Random number generator seed
    - ODE and dFBA timesteps
    - Checkpoint period
    - Submodels to skip
    - Whether to produce verbose output
    - Model changes: Instructions to change parameter values and add or remove model
      components. These instructions are executed before the initial conditions are
      calculated
    - External perturbations: Instructions to set parameter values or state at specific
      time points or time ranges

    Attributes:
        de_simulation_config (:obj:`SimulationConfig`): a simulation configuration for DE-Sim, needed
            for validation
        random_seed (:obj:`int`): random number generator seed
        ode_time_step (:obj:`float`, optional): ODE submodel timestep (s)
        dfba_time_step (:obj:`float`, optional): dFBA submodel timestep (s)
        checkpoint_period (:obj:`float`, optional): checkpointing timestep (s)
        submodels_to_skip (:obj:`list` of :obj:`str`, optional): submodels that should not be run,
            identified by their ids
        verbose (:obj:`bool`, optional): whether to produce verbose output
        changes (:obj:`list`, optional): list of desired model changes (e.g. modified parameter values,
            additional species/reactions, removed species/reactions)
        perturbations (:obj:`list`, optional): list of desired simulated perturbations (e.g. set state
            to a value at a specified time or time range)
    """

    de_simulation_config: object
    random_seed: int = None
    ode_time_step: float = None
    dfba_time_step: float = None
    checkpoint_period: float = None
    submodels_to_skip: list = None
    verbose: bool = False
    changes: list = None
    perturbations: list = None

    def __post_init__(self):

        # if lists aren't set, make them empty
        if self.changes is None:
            self.changes = []
        if self.perturbations is None:
            self.perturbations = []
        if self.submodels_to_skip is None:
            self.submodels_to_skip = []

    def __setattr__(self, name, value):
        """ Validate an attribute when it is changed """
        try:
            super().__setattr__(name, value)
        except TypeError as e:
            raise MultialgorithmError(e)

    def validate_individual_fields(self):
        """ Validate individual fields in a `WCSimulationConfig` instance

        Returns:
            :obj:`None`: if no error is found

        Raises:
            :obj:`MultialgorithmError`: if an attribute fails validation
        """

        # additional validation
        if self.ode_time_step is not None and self.ode_time_step <= 0:
            raise MultialgorithmError(f'ode_time_step ({self.ode_time_step}) must be positive')

        if self.dfba_time_step is not None and self.dfba_time_step <= 0:
            raise MultialgorithmError(f'dfba_time_step ({self.dfba_time_step}) must be positive')

        if self.checkpoint_period is not None and self.checkpoint_period <= 0:
            raise MultialgorithmError(f'checkpoint_period ({self.checkpoint_period}) must be positive')

    def validate(self):
        """ Fully validate a `WCSimulationConfig` instance

        Returns:
            :obj:`None`: if no error is found

        Raises:
            :obj:`MultialgorithmError`: if validation fails
        """

        self.validate_individual_fields()

        de_sim_config = self.de_simulation_config

        # Max time must be positive
        if de_sim_config.time_max <= 0:
            raise MultialgorithmError("Maximum time ({de_sim_config.time_max}) must be positive")

        if de_sim_config.output_dir is None and self.checkpoint_period is not None:
            raise MultialgorithmError(f"a data directory (self.de_simulation_config.output_dir) must be "
                                      f"provided when a checkpoint_period ({self.checkpoint_period}) is provided")

        # Check that timesteps divide evenly into the simulation duration
        self.check_periodic_timestep('ode_time_step')
        self.check_periodic_timestep('dfba_time_step')
        self.check_periodic_timestep('checkpoint_period')

    def check_periodic_timestep(self, periodic_attr):
        """ Check that simulation duration is an integral multiple of a periodic activity's timestep

        Args:
            periodic_attr (:obj:`str`): name of an attribute storing the duration of a periodic activity

        Returns:
            :obj:`None`: if no error is found

        Raises:
            :obj:`MultialgorithmError`: if the simulation duration is not an integral multiple of the
                periodic activity's timestep
        """
        de_sim_cfg = self.de_simulation_config
        if getattr(self, periodic_attr) is not None and \
            (de_sim_cfg.time_max - de_sim_cfg.time_init) / getattr(self, periodic_attr) % 1 != 0:
            raise MultialgorithmError(f'(time_max - time_init) ({de_sim_cfg.time_max} - {de_sim_cfg.time_init}) '
                                      f'must be a multiple of {periodic_attr} ({getattr(self, periodic_attr)})')

    def apply_changes(self, model):
        """ Applies changes to model

        Args:
            model (:obj:`wc.sim.model.Model`): model
        """

        for change in self.changes:
            change.run(model)

    def apply_perturbations(self, model):
        """ Applies perturbations to model

        Args:
            model (:obj:`wc.sim.model.Model`): model
        """

        for perturbation in self.perturbations:
            # .. todo:: implement
            pass  # pragma: no cover

    def get_num_time_steps(self):
        """ Calculate number of simulation timesteps

        Returns:
            :obj:`int`: number of simulation timesteps
        """
        return int((self.de_simulation_config.time_max - self.de_simulation_config.time_init) / self.ode_time_step)

    def semantically_equal(self, other):
        """ Are two instances semantically equal with respect to a simulation's predictions?

        Overrides `semantically_equal` in :obj:`EnhancedDataClass`.
        Ignore `verbose`, `changes`, and `perturbations`.
        `verbose` does not influence a simulation's predictions and `changes` and `perturbations`
        are not currently used.

        Args:
            other (:obj:`Object`): other object

        Returns:
            :obj:`bool`: :obj:`True` if `other` is semantically equal to `self`, :obj:`False` otherwise
        """
        return self.de_simulation_config.semantically_equal(other.de_simulation_config) and \
            self.random_seed == other.random_seed and \
            self.ode_time_step == other.ode_time_step and \
            self.dfba_time_step == other.dfba_time_step and \
            self.checkpoint_period == other.checkpoint_period and \
            self.submodels_to_skip == other.submodels_to_skip


Change = wc_lang.transform.ChangeValueTransform


class Perturbation(object):
    """ Represents a perturbation to simulate:

    - Change: desired value for a target (state/parameter)
    - Start time: time in seconds when the target should be perturbed
    - End time: optional, time in seconds when the perturbation should end. That is,
      the target will be held to the desired value from the start to the end time.

    Attributes:
        change (:obj:`Change`): desired value for a target
        start_time (:obj:`float`): perturbation start time in seconds
        end_time (:obj:`float`): perturbation end time in seconds
    """
    ATTRIBUTES = ['change', 'start_time', 'end_time']

    def __init__(self, change, start_time, end_time=float('nan')):
        """
        Args:
            change (:obj:`wc.sim.conf.Change`): desired value for a target
            start_time (:obj:`float`): perturbation start time in seconds
            end_time (:obj:`float`, optional): perturbation end time in seconds
        """
        self.change = change
        self.start_time = start_time
        self.end_time = end_time

    def __eq__(self, other):
        """ Compare two `Perturbation` objects

        Args:
            other (:obj:`Object`): other object

        Returns:
            :obj:`bool`: true if `Perturbation` objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        for attr in ['change', 'start_time']:
            if getattr(other, attr) != getattr(self, attr):
                return False

        if math.isnan(other.end_time) and not math.isnan(self.end_time):
            return False

        if not math.isnan(other.end_time) and math.isnan(self.end_time):
            return False

        if not math.isnan(other.end_time) and not math.isnan(self.end_time):
            return other.end_time == self.end_time

        return True


class SedMl(object):
    """ Reads and writes simulation configurations to/from SED-ML."""

    @staticmethod
    def read(file_name):
        """ Imports simulation configuration from SED-ML file

        Args:
            file_name (:obj:`str`): path to SED-ML file from which to import simulation configuration

        Returns:
            :obj:`WCSimulationConfig`: simulation configuration
        """

        # initialize optional configuration
        opt_config = {}

        """ read SED-ML """
        # read XML file
        cfg_ml = libsedml.readSedML(file_name)

        """ validate SED-ML """
        # one model with
        # - zero or more attribute changes
        if cfg_ml.getNumModels() != 1:
            raise SedMlError('SED-ML configuration must have one model')

        mdl = cfg_ml.getModel(0)
        if mdl.getId() or mdl.getName() or mdl.getLanguage() or mdl.getSource():
            warnings.warn('SED-ML import ignoring all model metadata (id, name, language, source)',
                SedMlWarning)

        for change_ml in mdl.getListOfChanges():
            if not isinstance(change_ml, libsedml.SedChangeAttribute):
                raise SedMlError('Cannot import model changes except attribute changes')

        # 1 simulation with
        #- Initial time = 0
        #- Output start time = 0
        #- Algorithm = KISAO_0000352 (hybrid)
        #- 1 algorithm parameters
        #  - random_seed (KISAO_0000488): int
        if cfg_ml.getNumSimulations() != 1:
            raise SedMlError('SED-ML configuration must have 1 simulation')

        sim_ml = cfg_ml.getSimulation(0)
        if sim_ml.getInitialTime() != sim_ml.getOutputStartTime():
            raise SedMlError('Simulation initial time and output start time must be equal')

        alg_ml = sim_ml.getAlgorithm()
        if alg_ml.getKisaoID() != 'KISAO_0000352':
            raise SedMlError('Unsupported algorithm. The only supported algorithm is KISAO_0000352 (hybrid)')

        param_kisao_ids = set(param_ml.getKisaoID() for param_ml in alg_ml.getListOfAlgorithmParameters())
        if len(param_kisao_ids) < alg_ml.getNumAlgorithmParameters():
            raise SedMlError('Algorithm parameter KISAO ids must be unique')
        unsupported_param_kisao_ids = param_kisao_ids - set(['KISAO_0000488'])
        if unsupported_param_kisao_ids:
            raise SedMlError('Algorithm parameters ({}) are not supported'.format(', '.join(unsupported_param_kisao_ids)))

        # zero or more repeated tasks each with
        # - 1 set value task change
        #   - math equal to constant floating point value
        # - 1 uniform or vector range of 1 value
        # - no subtasks
        for task_ml in cfg_ml.getListOfTasks():
            if isinstance(task_ml, libsedml.SedRepeatedTask):
                if task_ml.getNumTaskChanges() != 1:
                    raise SedMlError('Each repeated task must have one task change')
                change_ml = task_ml.getTaskChange(0)
                try:
                    float(libsedml.formulaToString(change_ml.getMath()))
                except ValueError:
                    raise SedMlError('Set value maths must be floating point values')

                if task_ml.getNumRanges() != 1:
                    raise SedMlError('Each repeated task must have one range')
                range_ml = task_ml.getRange(0)
                if not isinstance(range_ml, (libsedml.SedUniformRange, libsedml.SedVectorRange)):
                    raise SedMlError('Task ranges must be uniform or vector ranges')
                if isinstance(range_ml, libsedml.SedUniformRange) and range_ml.getNumberOfPoints() != 2**31 - 1:
                    raise SedMlError('Cannot import number of points')
                if isinstance(range_ml, libsedml.SedUniformRange) and range_ml.getType() != 'linear':
                    raise SedMlError('Only linear ranges are supported')
                if isinstance(range_ml, libsedml.SedVectorRange) and len(list(range_ml.getValues())) != 1:
                    raise SedMlError('Task vector ranges must have length 1')

                if task_ml.getNumSubTasks() > 0:
                    raise SedMlError('Cannot import subtasks')
            else:
                raise SedMlError('Cannot import tasks except repeated tasks')

        # no data generators
        if cfg_ml.getNumDataGenerators() > 0:
            raise SedMlError('Cannot import data generator information')

        # no data descriptions
        if cfg_ml.getNumDataDescriptions() > 0:
            raise SedMlError('Cannot import data description information')

        # no outputs
        if cfg_ml.getNumOutputs() > 0:
            raise SedMlError('Cannot import output information')

        """ Read simulation configuration information from SED-ML document """

        # changes
        changes = []
        mdl = cfg_ml.getModel(0)
        for change_ml in mdl.getListOfChanges():
            target = change_ml.getTarget()
            change = Change()
            change.attr_path = Change.attr_path_from_str(change_ml.getTarget())
            change.value = float(change_ml.getNewValue())
            changes.append(change)

        # perturbations
        perturbations = []
        for task_ml in cfg_ml.getListOfTasks():
            change_ml = task_ml.getTaskChange(0)

            change = Change()
            change.attr_path = Change.attr_path_from_str(change_ml.getTarget())
            change.value = float(libsedml.formulaToString(change_ml.getMath()))

            range_ml = task_ml.getRange(0)
            if isinstance(range_ml, libsedml.SedUniformRange):
                start_time = range_ml.getStart()
                end_time = range_ml.getEnd()
            else:
                start_time = range_ml.getValues()[0]
                end_time = float('nan')

            perturbations.append(Perturbation(change, start_time, end_time))

        # time
        sim = cfg_ml.getSimulation(0)
        time_init = sim.getOutputStartTime()
        time_max = sim.getOutputEndTime()
        ode_time_step = (time_max - time_init) / sim.getNumberOfPoints()

        # algorithm parameters
        alg = sim.getAlgorithm()
        for param in alg.getListOfAlgorithmParameters():
            if param.getKisaoID() == 'KISAO_0000488':
                try:
                    opt_config['random_seed'] = float(param.getValue())
                except ValueError:
                    raise SedMlError('random_seed must be an integer')

                if opt_config['random_seed'] != math.ceil(opt_config['random_seed']):
                    raise SedMlError('random_seed must be an integer')

                opt_config['random_seed'] = int(opt_config['random_seed'])

        """ build simulation configuration object """
        de_simulation_config = SimulationConfig(time_max=time_max, time_init=time_init)
        cfg = WCSimulationConfig(de_simulation_config=de_simulation_config,
                                 ode_time_step=ode_time_step,
                                 changes=changes, perturbations=perturbations,
                                 **opt_config)

        return cfg

    @staticmethod
    def write(cfg, file_name):
        """ Exports simulation configuration to SED-ML file

        Args:
            cfg (:obj:`WCSimulationConfig`): simulation configuration
            file_name (:obj:`str`): path to write SED-ML file
        """

        # initialize SED-ML document
        cfg_ml = libsedml.SedDocument()
        cfg_ml.setLevel(1)
        cfg_ml.setVersion(2)

        # write changes
        mdl = cfg_ml.createModel()
        for change in cfg.changes:
            change_ml = mdl.createChangeAttribute()
            change_ml.setTarget(change.attr_path_to_str())
            change_ml.setNewValue(str(change.value))

        # write perturbations
        for perturbation in cfg.perturbations:
            task_ml = cfg_ml.createRepeatedTask()

            change_ml = task_ml.createTaskChange()
            change_ml.setTarget(perturbation.change.attr_path_to_str())
            change_ml.setMath(libsedml.parseFormula(str(perturbation.change.value)))

            if perturbation.end_time and not numpy.isnan(perturbation.end_time):
                range_ml = task_ml.createUniformRange()
                range_ml.setStart(perturbation.start_time)
                range_ml.setEnd(perturbation.end_time)
                range_ml.setType('linear')
            else:
                range_ml = task_ml.createVectorRange()
                range_ml.setValues([perturbation.start_time])

        # simulation (maximum time, step size)
        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(cfg.de_simulation_config.time_init)
        sim.setOutputStartTime(cfg.de_simulation_config.time_init)
        sim.setOutputEndTime(cfg.de_simulation_config.time_max)
        duration = cfg.de_simulation_config.time_max - cfg.de_simulation_config.time_init
        sim.setNumberOfPoints(int(duration / cfg.ode_time_step))

        # simulation algorithm
        alg = sim.createAlgorithm()
        # .. todo:: add KISAO term for multi-algorithm method
        alg.setKisaoID("KISAO_0000352")  # hybrid method

        # random number generator seed, state
        if cfg.random_seed is not None:
            param = alg.createAlgorithmParameter()
            param.setKisaoID("KISAO_0000488")
            param.setValue(str(cfg.random_seed))

        # write to XML file
        libsedml.writeSedML(cfg_ml, file_name)


class SedMlError(Exception):
    """ SED-ML import/export error """
    pass


class SedMlWarning(UserWarning):
    """ SED-ML import/export warning """
    pass
