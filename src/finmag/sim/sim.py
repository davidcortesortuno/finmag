import os
import inspect
import logging
import tempfile
import dolfin as df
import numpy as np
from finmag.sim.llg import LLG
from finmag.util.timings import mtimed
from finmag.util.consts import exchange_length, bloch_parameter
from finmag.util.meshes import mesh_info, mesh_volume
from finmag.util.fileio import Tablewriter, FieldSaver
from finmag.util import helpers
from finmag.util.vtk_saver import VTKSaver
from finmag.sim.hysteresis import hysteresis as hyst, hysteresis_loop as hyst_loop
from finmag.sim import sim_helpers
from finmag.energies import Exchange, Zeeman, TimeZeeman, Demag, UniaxialAnisotropy, DMI
from finmag.integrators.llg_integrator import llg_integrator
from finmag.integrators import scheduler, events
from finmag.integrators.common import run_with_schedule
from finmag.util.pbc2d import PeriodicBoundary1D, PeriodicBoundary2D
from finmag.llb.sllg import SLLG

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

log = logging.getLogger(name="finmag")


class Simulation(object):
    """
    Unified interface to finmag's micromagnetic simulations capabilities.

    Attributes:
        t           the current simulation time

    """
    @mtimed
    def __init__(self, mesh, Ms, unit_length=1, name='unnamed', kernel='llg' ,integrator_backend="sundials", pbc=None, average=False):
        """Simulation object.

        *Arguments*

          mesh : a dolfin mesh

          Ms   : Magnetisation saturation (in A/m) of the material.

          unit_length: the distance (in metres) associated with the
                       distance 1.0 in the mesh object.

          name : the Simulation name (used for writing data files, for examples)

          pbc : Periodic boundary type: None or '2d'
          
          kernel : 'llg' or 'sllg'
          
          average : take the cell averaged effective field, only for test, will delete it if doesn't work.

        """
        # Store the simulation name and a 'sanitized' version of it which
        # contains only alphanumeric characters and underscores. The latter
        # will be used as a prefix for .log/.ndt files etc.
        self.name = name
        self.sanitized_name = helpers.clean_filename(name)

        self.logfilename = self.sanitized_name + '.log'
        self.ndtfilename = self.sanitized_name + '.ndt'

        self.logging_handler = helpers.start_logging_to_file(self.logfilename, mode='w', level=logging.DEBUG)

        # Create a Tablewriter object for ourselves which will be used
        # by various methods to save the average magnetisation at given
        # timesteps.
        self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

        self.tablewriter.entities['E_total'] = {
            'unit': '<J>',
            'get': lambda sim: sim.total_energy(),
            'header': 'E_total'}
        self.tablewriter.entities['H_total'] = {
            'unit': '<A/m>',
            'get': lambda sim: helpers.average_field(sim.effective_field()),
            'header': ('H_total_x', 'H_total_y', 'H_total_z')}
        self.tablewriter.update_entity_order()

        log.info("Creating Sim object '{}' (rank={}/{}).".format(
            self.name, df.MPI.process_number(), df.MPI.num_processes()))
        log.info(mesh)

        self.pbc = pbc
        if pbc == '2d':
            self.pbc = PeriodicBoundary2D(mesh)
        elif pbc == '1d':
            self.pbc = PeriodicBoundary1D(mesh)


        self.mesh = mesh
        self.Ms = Ms
        self.unit_length = unit_length
        self.integrator_backend = integrator_backend
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=self.pbc)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3, constrained_domain=self.pbc)
        
        if kernel=='llg':
            self.llg = LLG(self.S1, self.S3, average = average)
        elif kernel=='sllg':
            self.llg = SLLG(self.S1, self.S3, unit_length=unit_length)
        else:
            raise ValueError("kernel must be either llg or sllg.")
        
        self.kernel = kernel
        
        self.llg.Ms = Ms
        self.Volume = mesh_volume(mesh)

        self.scheduler = scheduler.Scheduler()
        self.callbacks_at_scheduler_events = []

        self.domains = df.CellFunction("uint", self.mesh)
        self.domains.set_all(0)
        self.region_id = 0

        # XXX TODO: this separation between vtk_savers and
        # field_savers is artificial and should/will be removed once
        # we have a robust, unified field saving mechanism.
        self.vtk_savers = {}
        self.field_savers = {}

        self.scheduler_shortcuts = {
            'save_restart_data': sim_helpers.save_restart_data,
            'save_ndt': sim_helpers.save_ndt,
            'save_averages': sim_helpers.save_ndt,
            'save_vtk': self.save_vtk,
            'save_field': Simulation._save_field_incremental,
            'switch_off_H_ext': Simulation.switch_off_H_ext,
        }

        # At the moment, we can only have cvode as the driver, and thus do
        # time development of a system. We may have energy minimisation at some
        # point (the driver would be an optimiser), or something else.
        self.driver = 'cvode'


    def __str__(self):
        """String briefly describing simulation object"""
        return "finmag.Simulation(name='%s') with %s" % (self.name, self.mesh)

    def __get_m(self):
        """The unit magnetisation"""
        return self.llg.m

    def set_m(self, value, **kwargs):
        """
        Set the magnetisation (it is automatically normalised to unit length).

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        self.llg.set_m(value, **kwargs)

    m = property(__get_m, set_m)

    @property
    def m_average(self):
        """
        Compute and return the average magnetisation over the entire
        mesh, according to the formula :math:`\\langle m \\rangle =
        \\frac{1}{V} \int m \: \mathrm{d}V`
        """
        return self.llg.m_average

    def save_m_in_region(self,region,name='unnamed'):

        self.region_id += 1
        helpers.mark_subdomain_by_function(region, self.mesh, self.region_id, self.domains)
        self.dx = df.Measure("dx")[self.domains]

        if name=='unnamed':
            name='region_'+str(self.region_id)

        region_id=self.region_id
        self.tablewriter.entities[name]={
                        'unit': '<>',
                        'get': lambda sim: sim.llg.m_average_fun(dx=self.dx(region_id)),
                        'header': (name+'_m_x', name+'_m_y', name+'_m_z')}

        self.tablewriter.update_entity_order()

    @property
    def t(self):
        """
        Returns the current simulation time.

        """
        if hasattr(self, "integrator"):
            return self.integrator.cur_t  # the real thing
        return 0.0

    def add(self, interaction, with_time_update=None):
        """
        Add an interaction (such as Exchange, Anisotropy, Demag).

        *Arguments:*

        interaction

             The interaction to be added.

         with_time_update (optional)

             A function of the form f(t), which accepts a time step
             `t` as its only single parameter and updates the internal
             state of the interaction accordingly.
        """
        # Make sure that interaction names are unique
        if interaction.name in [i.name for i in self.llg.effective_field.interactions]:
            raise ValueError("Interaction names must be unique, but an "
                             "interaction with the same name already "
                             "exists: {}".format(interaction.name))

        log.debug("Adding interaction %s to simulation '%s'" % (str(interaction),self.name))
        interaction.setup(self.S3, self.llg._m, self.llg._Ms_dg, self.unit_length)
        # TODO: The following feels somewhat hack-ish because we
        #       explicitly check for TimeZeeman and it's likely that
        #       there will be other classes in the future that also
        #       come with time updates which would then also have to
        #       be added here by hand. Is there a more elegant and
        #       automatic solution?
        if isinstance(interaction, TimeZeeman):
            # The following line ensures that the time integrator is notified
            # about the correct field values at the time steps it chooses.
            with_time_update = interaction.update
            # The following line ensures that the field value is updated
            # correctly whenever the time integration reaches a scheduler
            # "checkpoint" (i.e. whenever integrator.advance_time(t) finishes
            # successfully).
            self.callbacks_at_scheduler_events.append(interaction.update)
        self.llg.effective_field.add(interaction, with_time_update)

        energy_name = 'E_{}'.format(interaction.name)
        field_name = 'H_{}'.format(interaction.name)
        self.tablewriter.entities[energy_name] = {
            'unit': '<J>',
            'get': lambda sim: sim.get_interaction(interaction.name).compute_energy(),
            'header': energy_name}
        self.tablewriter.entities[field_name] = {
            'unit': '<A/m>',
            'get': lambda sim: sim.get_interaction(interaction.name).average_field(),
            'header': (field_name + '_x', field_name + '_y', field_name + '_z')}
        self.tablewriter.update_entity_order()

    def effective_field(self):
        """
        Compute and return the effective field.

        """
        return self.llg.effective_field.compute(self.t)

    def total_energy(self):
        """
        Compute and return the total energy of all fields present in
        the simulation.

        """
        return self.llg.effective_field.total_energy()

    def get_interaction(self, interaction_name):
        """
        Returns the interaction with the given name

        *Arguments*

        interaction_name: string

            Name of the interaction.

        *Returns*

        The matching interaction object. If no or more than one matching
        interaction is found, a ValueError is raised.

        """
        return self.llg.effective_field.get_interaction(interaction_name)

    def remove_interaction(self, interaction_type):
        """
        Remove the interaction of the given type.

        *Arguments*

        interaction_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman'.
        """
        log.debug("Removing interaction '{}' from simulation '{}'".format(
                interaction_type, self.name))
        return self.llg.effective_field.remove_interaction(interaction_type)

    def set_H_ext(self, H_ext):
        """
        Convenience function to set the external field.
        """
        H = self.get_interaction("Zeeman")
        H.set_value(H_ext)

    def switch_off_H_ext(self, remove_interaction=True):
        """
        Convenience function to switch off the external field.

        If `remove_interaction` is True (the default), the Zeeman
        interaction will be completely removed from the Simulation
        class, which should make the time integration run faster.
        Otherwise its value is just set to zero.
        """
        if remove_interaction:
            dbg_str = "(removing Zeeman interaction)"
            self.remove_interaction("Zeeman")
        else:
            dbg_str = "(setting value to zero)"
            H = self.get_interaction("Zeeman")
            H.set_value([0, 0, 0])

        log.debug("Switching off external field {}".format(dbg_str))

    def get_field_as_dolfin_function(self, field_type):
        """
        Returns the field of the interaction of the given type or of the
        magnetisation as a dolfin function.

        *Arguments*

        field_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman',
            as well as 'm' which stands for the normalised magnetisation.

        *Returns*

        A dolfin.Function representing the given field. If no or more than one
        matching field is found, a ValueError is raised.

        """
        if field_type == 'm':
            return self.llg._m
        return self.llg.effective_field.get_dolfin_function(field_type)

    def probe_field(self, field_type, pts):
        """
        Probe the field of type `field_type` at point(s) `pts`, where
        the point coordinates must be specified in metres (not in
        multiples of unit_length!).

        See the documentation of the method get_field_as_dolfin_function
        to know which ``field_type`` is allowed, and helpers.probe for the
        shape of ``pts``.

        """
        pts = np.array(pts) / self.unit_length
        return helpers.probe(self.get_field_as_dolfin_function(field_type), pts)

    def create_integrator(self, backend=None, **kwargs):

        if not hasattr(self, "integrator"):
            if backend == None:
                backend = self.integrator_backend
            log.info("Create integrator {} with kwargs={}".format(backend, kwargs))
            if self.kernel == 'sllg':
                self.integrator = self.llg
            else:
                self.integrator = llg_integrator(self.llg, self.llg.m, backend=backend, **kwargs)
        else:
            log.warning("Cannot create integrator - exists already: {}".format(self.integrator))
        return self.integrator

    def advance_time(self, t):
        """
        The lower-level counterpart to run_until, this runs without a schedule.

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()

        log.debug("Advancing time to t = {} s.".format(t))
        self.integrator.advance_time(t)
        # The following line is necessary because the time integrator may
        # slightly overshoot the requested end time, so here we make sure
        # that the field values represent that requested time exactly.
        self.llg.effective_field.update(t)

    def run_until(self, t):
        """
        Run the simulation until the given time `t` is reached.

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()

        log.info("Simulation will run until t = {:.2g} s.".format(t))
        exit_at = events.StopIntegrationEvent(t)
        self.scheduler._add(exit_at)

        run_with_schedule(self.integrator, self.scheduler, self.callbacks_at_scheduler_events)
        # The following line is necessary because the time integrator may
        # slightly overshoot the requested end time, so here we make sure
        # that the field values represent that requested time exactly.
        self.llg.effective_field.update(t)
        log.info("Simulation has reached time t = {:.2g} s.".format(self.t))

        self.scheduler._remove(exit_at)

    def relax(self, stopping_dmdt=1.0, dt_limit=1e-10,
              dmdt_increased_counter_limit=10):
        """
        Run the simulation until the magnetisation has relaxed.

        This means the magnetisation reaches a state where its change over time
        at each node is smaller than the threshold `stopping_dm_dt` (which
        should be given in multiples of degree/nanosecond).

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()
        log.info("Simulation will run until relaxation of the magnetisation.")
        log.debug("Relaxation parameters: stopping_dmdt={} (degrees per nanosecond), "
                  "dt_limit={}, dmdt_increased_counter_limit={}".format(
                              stopping_dmdt, dt_limit, dmdt_increased_counter_limit))

        if hasattr(self, "relaxation"):
            del(self.relaxation)

        self.relaxation = events.RelaxationEvent(self, stopping_dmdt*ONE_DEGREE_PER_NS, dmdt_increased_counter_limit, dt_limit)
        self.scheduler._add(self.relaxation)

        run_with_schedule(self.integrator, self.scheduler, self.callbacks_at_scheduler_events)
        self.integrator.reinit()
        self.set_m(self.m)
        log.info("Relaxation finished at time t = {:.2g}.".format(self.t))

        self.scheduler._remove(self.relaxation)
        del(self.relaxation.sim) # help the garbage collection by avoiding circular reference

    save_restart_data = sim_helpers.save_restart_data

    def restart(self, filename=None, t0=None):
        """If called, we look for a filename of type sim.name + '-restart.npz',
        and load it. The magnetisation in the restart file will be assigned to
        sim.llg._m. If this is from a cvode time integration run, it will also
        initialise (create) the integrator with that m, and the time at which the
        restart data was saved.

        The time can be overriden with the optional parameter t0 here.

        The method will fail if no restart file exists.
        """

        if filename == None:
            filename = sim_helpers.canonical_restart_filename(self)
        log.debug("Loading restart data from {}. ".format(filename))

        data = sim_helpers.load_restart_data(filename)

        if not data['driver'] in ['cvode']:
            log.error("Requested unknown driver `{}` for restarting. Known: {}.".format(data["driver"], "cvode"))
            raise NotImplementedError("Unknown driver `{}` for restarting.".format(data["driver"]))

        self.llg._m.vector()[:] = data['m']

        self.reset_time(data["simtime"] if (t0 == None) else t0)

        log.info("Reloaded and set m (<m>=%s) and time=%s from %s." % \
            (self.llg.m_average, self.t, filename))

    def reset_time(self, t0):
        """
        Reset the internal clock time of the simulation to `t0`.

        This also adjusts the internal time of the scheduler and time integrator.
        """
        self.integrator = llg_integrator(self.llg, self.llg.m,
                                         backend=self.integrator_backend, t0=t0)
        self.scheduler.reset(t0)
        assert self.t == t0  # self.t is read from integrator

    save_averages = sim_helpers.save_ndt
    hysteresis = hyst
    hysteresis_loop = hyst_loop

    def __get_pins(self):
        return self.llg.pins

    def __set_pins(self, nodes):
        pinlist=[]
        if hasattr(nodes, '__call__'):
            coords = self.mesh.coordinates()
            for i,c in enumerate(coords):
                if nodes(c):
                    pinlist.append(i)
            pinlist=np.array(pinlist)
            self.llg.pins=pinlist
        else:
            self.llg.pins = nodes

    pins = property(__get_pins, __set_pins)

    def __get_alpha(self):
        return self.llg.alpha

    def __set_alpha(self, value):
        self.llg.alpha = value

    alpha = property(__get_alpha, __set_alpha)

    def spatial_alpha(self, alpha, multiplicator):
        self.llg.spatially_varying_alpha(alpha, multiplicator)

    def __get_gamma(self):
        return self.llg.gamma

    def __set_gamma(self, value):
        self.llg.gamma = value

    gamma = property(__get_gamma, __set_gamma)
    
    def __get_dt(self):
        return self.llg.dt
    
    def __set_dt(self, value):
        self.llg.dt = value
    
    dt = property(__get_dt, __set_dt)
    
    def __get_T(self):
        return self.llg.T
    
    def __set_T(self, value):
        self.llg.T = value
    
    T = property(__get_T, __set_T)

    run_normal_modes_computation = sim_helpers.run_normal_modes_computation

    def reinit_integrator(self):
        """
        If an integrator is already present in the simulation, call
        its reinit() method. Otherwise do nothing.
        """
        if hasattr(self, "integrator"):
            self.integrator.reinit()
        else:
            log.warning("Integrator reinit was requested, but no integrator "
                        "is present in the simulation!")

    def set_stt(self, current_density, polarisation, thickness, direction, with_time_update=None):
        """
        Activate the computation of the Slonczewski spin-torque term
        in the LLG.

        *Arguments*

        - Current density in A/m^2 as a number, dolfin function or expression.

        - Polarisation is between 0 and 1. It is defined as P = (x-y)/(x+y),
          where x and y are the fractions of spin up/down electrons).

        - Thickness of the free layer in m.

        - Direction of the polarisation as a triple (is automatically
          normalised to unit length).

        - with_time_update:

             A function of the form J(t), which accepts a time step `t`
             as its only argument and returns the new current density.

             N.B.: For efficiency reasons, the return value is currently
                   assumed to be a number, i.e. J is assumed to be spatially
                   constant (and only varying with time).

        """
        self.llg.use_slonczewski(current_density, polarisation, thickness,
                                 direction, with_time_update=with_time_update)

    def toggle_stt(self, new_state=None):
        """
        Toggle the computation of the Slonczewski spin-torque term.

        You can optionally pass in a new state.
        """
        if new_state:
            self.llg.do_slonczewski = new_state
        else:
            self.llg.do_slonczewski = not self.llg.do_slonczewski

    def clear_schedule(self):
        self.scheduler.clear()
        self.scheduler.reset(self.t)

    def schedule(self, func, *args, **kwargs):
        """
        Register a function that should be called during the simulation.
        To un-schedule this function later, store the return value `item`
        of this function and call sim.unschedule(item) later. Alternatively,
        you can call `sim.clear_schedule`, which unregisters *all* scheduled
        functions.

        By default, the schedule operates on simulation time expressed in
        seconds. Use either the `at` keyword argument to define a single
        point in time at which your function is called, or use the `every`
        keyword to specify an interval between subsequent calls to your
        function. When specifying the interval, you can optionally use the
        `after` keyword to delay the first execution of your function.
        Additionally, you can set the `at_end` option to `True` (default
        is `False`) to have your function called at the end of a simulation
        stage (e.g. when the run_until() command has reached its end time,
        or when the relax() command has reached relaxation). This can be
        combined with `at` and `every`.

        Note that if the internal simulation time is not zero (i.e.. if the
        simulation has already run for some time) then using the 'every'
        keyword will implicitly set 'after' to the current simulation time,
        so that the event repeats in regular intervals from the current time
        onwards). If this is undesired, you should explicitly provide 'after'
        (which is interpreted as an 'absolute' time, i.e. not as an offset to
        the current simulation time).

        You can also schedule actions using real time instead of simulation
        time by setting the `realtime` option to True. In this case you can
        use the `after` keyword on its own.

        The function func(sim) you provide should expect the simulation object
        as its first argument. All arguments to the 'schedule' function (except
        the special ones 'at', 'every', 'at_end' and 'realtime' mentioned
        above) will be passed on to this function.

        If func is a string, it will be looked up in self.scheduler_shortcuts,
        which includes 'save_restart_data', 'save_ndt', 'save_vtk' and
        'save_field'. For example, to save a vtk snapshot of the magnetisation
        every nanosecond, use:

            sim.schedule('save_vtk', every=1e-9, filename='m.pvd')

        and to save the magnetisation to a .npy file every 2 nanoseconds, use:

            sim.schedule('save_field', 'm', every=2e-9, filename='m.npy')

        In both of these cases, a suffix representing the number of
        the current snapshot will be added automatically, e.g.:
        'm_00000.npy', 'm_000001.npy', etc.

        """
        if isinstance(func, str):
            if func in self.scheduler_shortcuts:
                if func == 'save_vtk':
                    # This is a special case which needs some pre-processing
                    # as we need to open a .pvd file first.
                    filename = kwargs.pop('filename', None)
                    overwrite = kwargs.pop('overwrite', False)
                    try:
                        # Check whether a vtk_saver for this filename already exists; this is
                        # necessary to if 'save_vtk' is scheduled multiple times with the same
                        # filename.
                        vtk_saver = self._get_vtk_saver(filename=filename, overwrite=False)
                    except IOError:
                        # If none exists, create a new one.
                        vtk_saver = self._get_vtk_saver(filename=filename, overwrite=overwrite)

                    def aux_save(sim):
                        sim._save_m_to_vtk(vtk_saver)

                    func = aux_save
                    func = lambda sim: sim._save_m_to_vtk(vtk_saver)
                else:
                    func = self.scheduler_shortcuts[func]
            else:
                msg = "Scheduling keyword '%s' unknown. Known values are %s" \
                    % (func, self.scheduler_shortcuts.keys())
                log.error(msg)
                raise KeyError(msg)

        try:
            func_args = inspect.getargspec(func).args
        except TypeError:
            # This can happen when running the binary distribution, since compiled
            # functions cannot be inspected. Not a great problem, though, because
            # this will result in an error once the scheduled function is called,
            # even though it would be preferable to catch this early.
            func_args = None

        if func_args != None:
            illegal_argnames = ['at', 'after', 'every', 'at_end', 'realtime']
            for kw in illegal_argnames:
                if kw in func_args:
                    raise ValueError(
                        "The scheduled function must not use any of the following "
                        "argument names: {}".format(illegal_argnames))

        at = kwargs.pop('at', None)
        every = kwargs.pop('every', None)
        after = kwargs.pop('after', self.t if (every != None) else None)
        at_end = kwargs.pop('at_end', False)
        realtime = kwargs.pop('realtime', False)

        scheduled_item = self.scheduler.add(func, [self] + list(args), kwargs,
                                            at=at, at_end=at_end, every=every,
                                            after=after, realtime=realtime)
        return scheduled_item

    def unschedule(self, item):
        """
        Unschedule a previously scheduled callback function. The
        argument `item` should be the object returned by the call to
        sim.schedule(...) which scheduled that function.
        """
        self.scheduler._remove(item)

    def snapshot(self, filename="", directory="", force_overwrite=False):
        """
        Deprecated. Use 'save_vtk' instead.

        """
        log.warning("Method 'snapshot' is deprecated. Use 'save_vtk' instead.")
        self.vtk(self, filename, directory, force_overwrite)

    def _get_vtk_saver(self, filename=None, overwrite=False):
        if filename == None:
            filename = self.sanitized_name + '.pvd'

        if self.vtk_savers.has_key(filename) and (overwrite == False):
            # Retrieve an existing VTKSaver for appending data
            s = self.vtk_savers[filename]
        else:
            # Create a  new VTKSaver and store it for later re-use
            s = VTKSaver(filename, overwrite=overwrite)
            self.vtk_savers[filename] = s

        return s

    def _save_m_to_vtk(self, vtk_saver):
        vtk_saver.save_field(self.llg._m, self.t)

    def save_vtk(self, filename=None, overwrite=False):
        """
        Save the magnetisation to a VTK file.
        """
        vtk_saver = self._get_vtk_saver(filename, overwrite)
        self._save_m_to_vtk(vtk_saver)

    def _get_field_saver(self, field_name, filename=None, overwrite=False, incremental=False):
        if filename is None:
            filename = '{}_{}.npy'.format(self.sanitized_name, field_name.lower())
        if not filename.endswith('.npy'):
            filename += '.npy'

        s = None
        if self.field_savers.has_key(filename) and self.field_savers[filename].incremental == incremental:
            s = self.field_savers[filename]

        if s is None:
            s = FieldSaver(filename, overwrite=overwrite, incremental=incremental)
            self.field_savers[filename] = s

        return s

    def save_field(self, field_name, filename=None, incremental=False, overwrite=False):
        """
        Save the given field data to a .npy file.

        *Arguments*

        field_name : string

            The name of the field to be saved. This should be either 'm'
            or the name of one of the interactions present in the
            simulation (e.g. Demag, Zeeman, Exchange, UniaxialAnisotropy).

        filename : string

            Output filename. If not specified, a default name will be
            generated automatically based on the simulation name and the
            name of the field to be saved. If a file with the same name
            already exists, an exception of type IOError will be raised.

        incremental : bool

        """
        field_data = self.get_field_as_dolfin_function(field_name)
        field_saver = self._get_field_saver(field_name, filename, incremental=incremental, overwrite=overwrite)
        field_saver.save(field_data.vector().array())

    def _save_field_incremental(self, field_name, filename=None, overwrite=False):
        self.save_field(field_name, filename, incremental=True, overwrite=overwrite)

    def mesh_info(self):
        """
        Return a string containing some basic information about the
        mesh (such as the number of cells, interior/surface triangles,
        vertices, etc.).

        Also print a distribution of edge lengths present in the mesh
        and how they compare to the exchange length and the Bloch
        parameter (if these can be computed, which requires an
        exchange interaction (plus anisotropy for the Bloch
        parameter); note that for spatially varying exchange strength
        and anisotropy constant the average values are used). This
        information is relevant to estimate whether the mesh
        discretisation is too coarse and might result in numerical
        artefacts (see W. Rave, K. Fabian, A. Hubert, "Magnetic states
        ofsmall cubic particles with uniaxial anisotropy", J. Magn.
        Magn. Mater. 190 (1998), 332-348).

        """
        info_string = "{}\n".format(mesh_info(self.mesh))

        edgelengths = [e.length() * self.unit_length for e in df.edges(self.mesh)]

        def added_info(L, name, abbrev):
            (a, b), _ = np.histogram(edgelengths, bins=[0, L, np.infty])
            if b == 0.0:
                msg = "All edges are shorter"
                msg2 = ""
            else:
                msg = "Warning: {:.2f}% of edges are longer".format(100.0 * b / (a + b))
                msg2 = " (this may lead to discretisation artefacts)"
            info = "{} than the {} {} = {:.2f} nm{}.\n".format(msg, name, abbrev, L * 1e9, msg2)
            return info

        if hasattr(self.llg.effective_field, "exchange"):
            A = self.llg.effective_field.exchange.A.vector().array().mean()
            Ms = self.llg.Ms.vector().array().mean()
            l_ex = exchange_length(A, Ms)
            info_string += added_info(l_ex, 'exchange length', 'l_ex')
            if hasattr(self.llg.effective_field, "anisotropy"):
                K1 = float(self.llg.effective_field.anisotropy.K1)
                l_bloch = bloch_parameter(A, K1)
                info_string += added_info(l_bloch, 'Bloch parameter', 'l_bloch')

        return info_string

    def render_scene(self, **kwargs):
        """
        This is a convenience wrapper around the helper function
        `finmag.util.visualization.render_paraview_scene`. It saves
        the current magnetisation to a temporary file and uses
        `render_paraview_scene` to plot it. All keyword arguments
        are passed on to `render_paraview_scene`; see its docstring
        for details (one useful option is `outfile`, which can be used
        to save the resulting image to a png file).

        Returns the IPython.core.display.Image produced by
        `render_paraview_scene`.

        """
        from finmag.util.visualization import render_paraview_scene
        tmpdir = tempfile.mkdtemp()
        basename = os.path.join(tmpdir, 'paraview_scene_{}'.format(self.name))
        self.save_vtk(filename=basename + '.pvd')
        return render_paraview_scene(basename + '000000.vtu', **kwargs)

    def close_logfile(self):
        """
        Stop logging to the logfile associated with this simulation object.

        This closes the file and removed the associated logging
        handler from the 'finmag' logger. Note that logging to other
        files (in particular the global logfile) is not affected.

        """
        self.logging_handler.stream.close()
        log.removeHandler(self.logging_handler)
        self.logging_handler = None


def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend="sundials",
             A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK',
             demag_solver_params={}, D=None, name="unnamed"):
    """
    Create a Simulation instance based on the given parameters.

    This is a convenience function which allows quick creation of a
    common simulation type where at most one exchange/anisotropy/demag
    interaction is present and the initial magnetisation is known.

    If a value for any of the optional arguments A, K1 (and K1_axis),
    or demag_solver are provided then the corresponding exchange /
    anisotropy / demag interaction is created automatically and added
    to the simulation. For example, providing the value A=13.0e-12 in
    the function call is equivalent to:

       exchange = Exchange(A)
       sim.add(exchange)

    The argument `demag_solver_params` can be used to configure the demag solver
    (if the chosen solver class supports this). Example with the 'FK' solver:

       demag_solver_params = {'phi_1_solver': 'cg', phi_1_preconditioner: 'ilu'}

    See the docstring of the individual solver classes (e.g. finmag.energies.demag.FKDemag)
    for more information on possible parameters.
    """
    sim = Simulation(mesh, Ms, unit_length=unit_length,
                     integrator_backend=integrator_backend,
                     name=name)

    sim.set_m(m_init)
    sim.alpha = alpha

    # If any of the optional arguments are provided, initialise
    # the corresponding interactions here:
    if A is not None:
        sim.add(Exchange(A))
    if (K1 != None and K1_axis is None) or (K1 is None and K1_axis != None):
        log.warning(
            "Not initialising uniaxial anisotropy because only one of K1, "
            "K1_axis was specified (values given: K1={}, K1_axis={}).".format(
                K1, K1_axis))
    if K1 != None and K1_axis != None:
        sim.add(UniaxialAnisotropy(K1, K1_axis))
    if H_ext != None:
        sim.add(Zeeman(H_ext))
    if D != None:
        sim.add(DMI(D))
    if demag_solver != None:
        demag = Demag(solver=demag_solver)
        if demag_solver_params != {}:
            demag.parameters.update(demag_solver_params)
            for (k, v) in demag_solver_params.items():
                log.debug("Setting demag solver parameter {}='{}' for simulation '{}'".format(k, v, sim.name))
        sim.add(demag)

    return sim
