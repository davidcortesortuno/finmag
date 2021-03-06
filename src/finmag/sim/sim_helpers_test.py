import os
import pytest
import numpy as np
import dolfin as df
import sim_helpers
from datetime import datetime, timedelta
from distutils.version import LooseVersion
from finmag import Simulation
from finmag.example import barmini
from finmag.util.meshes import nanodisk
from finmag.drivers.llg_integrator import llg_integrator
from finmag.util.helpers import assert_number_of_files
from finmag.util.fileio import Tablereader


def test_can_read_restart_file(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.run_until(1e-21)  # create integrator
    sim_helpers.save_restart_data(sim)
    data = sim_helpers.load_restart_data(sim)

    assert data['simname'] == sim.name
    assert data['simtime'] == sim.t
    assert data['stats'] == sim.integrator.stats()
    assert np.all(data['m'] == sim.integrator.llg.m_numpy)
    # writing and reading the data should take less than 10 seconds
    assert datetime.now() - data['datetime'] < timedelta(0, 10)


def test_try_to_restart_a_simulation(tmpdir):
    os.chdir(str(tmpdir))
    t0 = 10e-12
    t1 = 20e-12

    # Create simulation, integrate until t0, save restart data,
    # and integrate until t1.
    sim1 = barmini()
    sim1.run_until(t0)
    print("Stats for sim1 at t = {} s:\n{}.".format(
        sim1.t, sim1.integrator.stats()))
    sim_helpers.save_restart_data(sim1)
    sim1.run_until(t1)
    print("Stats for sim1 at t = {} s:\n{}.".format(
        sim1.t, sim1.integrator.stats()))

    # Bring new simulation object into previously saved
    # state (which is at t0)...
    data = sim_helpers.load_restart_data(sim1)
    sim2 = barmini()
    sim2.set_m(data['m'])
    sim2.integrator = llg_integrator(sim2.llg, sim2.llg.m_field,
                                     backend=sim2.integrator_backend, t0=data['simtime'])
    # ... and integrate until t1.
    sim2.run_until(t1)
    print("Stats for sim2 at t = {} s:\n{}.".format(
        sim2.t, sim2.integrator.stats()))

    # Check that we have the same data in both simulation objects.
    print "Time for sim1: {} s, time for sim2: {} s.".format(sim1.t, sim2.t)
    assert abs(sim1.t - sim2.t) < 1e-16
    print "Average magnetisation for sim1:\n\t{}\nfor sim2:\n\t{}.".format(
        sim1.m_average, sim2.m_average)
    assert np.allclose(sim1.m, sim2.m, atol=5e-6, rtol=1e-8)

    # Check that sim2 had less work to do, since it got the
    # results up to t0 for free.
    stats1 = sim1.integrator.stats()
    stats2 = sim2.integrator.stats()
    assert stats2['nsteps'] < stats1['nsteps']


def test_create_backup_if_file_exists():
    # remove file
    testfilename = 'tmp-testfile.txt'
    if os.path.exists(testfilename):
        os.remove(testfilename)

    backupfilename = 'tmp-testfile.txt.backup'
    if os.path.exists(backupfilename):
        os.remove(backupfilename)

    assert os.path.exists(testfilename) == False
    assert os.path.exists(backupfilename) == False

    # create file
    os.system('echo "Hello World" > ' + testfilename)

    assert os.path.exists(testfilename) == True
    assert open(testfilename).readline()[:-1] == "Hello World"
    assert os.path.exists(backupfilename) == False

    sim_helpers.create_backup_file_if_file_exists(testfilename)
    assert os.path.exists(backupfilename) == True

    assert open(backupfilename).readline()[:-1] == "Hello World"

    assert open(backupfilename).read() == open(testfilename).read()


def test_run_normal_modes_computation(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini(name='barmini')
    # XXX TODO: Do we need stopping keywords?
    #sim.run_normal_modes_computation(alpha_relax=1.0, alpha_precess=0.01, t_end=10e-9, save_ndt_every=1e-11, save_vtk_every=None, )
    params_relax = {
        'alpha': 1.0,
        'H_ext': [1e3, 0, ],
        'save_ndt_every': None,
        'save_vtk_every': None,
        'save_npy_every': None,
        'save_relaxed_state': True,
        'filename': None
    }
    params_precess = {
        'alpha': 0.0,
        'H_ext': [1e3, 0, 0],
        't_end': 1e-10,
        'save_ndt_every': 1e-11,
        'save_vtk_every': 3e-11,
        'save_npy_every': 2e-11,
        'filename': None,
    }
    sim.run_normal_modes_computation(params_relax, params_precess)

    # XXX TODO: We should change the filename of the .ndt file to 'barmini_precess.ndt'
    #           once different filenames for .ndt files are supported.
    f = Tablereader('barmini.ndt')
    ts, mx, my, mz = f['time', 'm_x', 'm_y', 'm_z']
    assert(len(ts) == 11)
    assert_number_of_files('*_relaxed.npy', 1)
    assert_number_of_files('*_relaxed.pvd', 1)
    assert_number_of_files('*_relaxed*.vtu', 1)
    assert_number_of_files('*_precess.pvd', 1)
    assert_number_of_files('*_precess*.vtu', 4)
    assert_number_of_files('*_precess*.npy', 6)


def test_create_non_existing_parent_directories(tmpdir):
    os.chdir(str(tmpdir))

    dirs = ['foo1/bar',
            os.path.join(str(tmpdir), 'foo2', 'bar', 'baz'),
            '',
            os.curdir]

    for d in dirs:
        sim_helpers.create_non_existing_parent_directories(
            os.path.join(d, 'filename.txt'))
        assert(os.path.exists(os.path.abspath(d)))


def test_save_restart_data_creates_non_existing_directories(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.run_until(0.0)
    sim.save_restart_data('foo/restart_data.npz')
    assert(os.path.exists('foo'))


@pytest.mark.xfail(reason='dolfin 1.5')
def test_get_submesh(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini(mark_regions=True)
    mesh_full = sim.get_submesh(None)
    mesh_top = sim.get_submesh('top')
    mesh_bottom = sim.get_submesh('bottom')
    with pytest.raises(ValueError):
        mesh_full = sim.get_submesh('foo')

    id_top = sim.region_ids['top']
    id_bottom = sim.region_ids['bottom']
    submesh_top = df.SubMesh(sim.mesh, sim.region_markers, id_top)
    submesh_bottom = df.SubMesh(sim.mesh, sim.region_markers, id_bottom)

    assert(mesh_full.num_vertices() == sim.mesh.num_vertices())
    assert(mesh_top.num_vertices() == submesh_top.num_vertices())
    assert(mesh_bottom.num_vertices() == submesh_bottom.num_vertices())


def test_skyrmion_number():
    """
    Test to see if the skyrmion number density function helper integrated over
    a mesh yields the same result as the skyrmion number function itself. Also
    test that they both give sane values by testing on a system with a finite
    number of skyrmions.
    """
    # Create simulation object with four skyrmions in it.
    mesh = df.RectangleMesh(df.Point(-100, -100), df.Point(100, 100), 50, 50)
    sim = Simulation(mesh, 1e5, unit_length=1e-9)
    skCentres = np.array([[0, 0], [-50, 70], [40, -80], [70, 70]])
    sim.initialise_skyrmions(skyrmionRadius=30, centres=skCentres)

    # Obtain skyrmion number and skyrmion number density, and test them.
    skX = sim.skyrmion_number()
    skX_density_function = sim.skyrmion_number_density_function()
    skX_integral = df.assemble(skX_density_function * df.dx)

    assert(abs(skX - skX_integral) < 1e-10)
    assert(abs(skX - 4) < 3e-1)  # There are four skyrmions here...

    # Repeat for a 3D mesh

    mesh3D = df.BoxMesh(df.Point(-100, -100, -5), df.Point(100, 100, 5), 50, 50, 5)
    sim3D = Simulation(mesh3D, 1e5, unit_length=1e-9)
    skCentres3D = np.array(
        [[0, 0, 0], [-50, 70, 0], [40, -80, 0], [70, 70, 0]])
    sim3D.initialise_skyrmions(skyrmionRadius=30, centres=skCentres3D)

    skX_3D = sim3D.skyrmion_number()
    assert(abs(skX_3D - 4) < 4e-1)  # There are four skyrmions here...

    # Test using another geometry (a nanodisk)
    meshNanodisk = nanodisk(d=100, h=10, maxh=3.0, save_result=False)
    simNanodisk = Simulation(meshNanodisk, Ms=1e5, unit_length=1e-9)

    # Initialise with a theoretical skyrmion, which should have skyrmion number as 1.
    simNanodisk.initialise_skyrmions(skyrmionRadius=50)

    skX_nanodisk = simNanodisk.skyrmion_number()

    assert(abs(skX_nanodisk - 1) < 5e-2)
