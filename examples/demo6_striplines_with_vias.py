import emerge as em
from emerge.pyvista import PVDisplay
import matplotlib.pyplot as plt
import numpy as np

"""PCB Vias

This demonstration shows how to add vias with the PCB router. Make sure to go through the other
PCB related demos (demo1 and demo3) to get more information on the PCBLayouter.

"""
em.superlu_info()

mm = 0.001
th = 1
with em.Simulation3D('Stripline_test', PVDisplay, loglevel='DEBUG') as m:
    # As usual we start by creating our layouter
    ly = em.PCBLayouter(th, mm, em.cs.GCS, em.material.ROGERS_4350B)

    # Here we define a simple stripline path that makes a knick turn and a via jump to a new layer.
    # None of the transmission lines are conciously matched in any way, this is just about the routing

    # The .via(...) method allows one to add a via geometry to the PCBLayouter that can be extracted later
    # A user may decide whether to proceed or not beyond the via. More information can be found in the 
    # docstring of the method. The vias can be created later.
    ly.new(0,0,2,(1,0), -th/2).name('p1').straight(10).turn(90).straight(10).turn(-90)\
        .straight(2).via(0, 0.5, True).straight(8).via(-th/2, 0.5).straight(2)\
            .turn(-90).straight(10).turn(90).straight(10).name('p2')
    
    # As usual we compile the traces as a merger of polygons
    trace = ly.compile_paths(True)

    # Now that we have via's defined, we can do the same with vias. I set Merge to True so that I get back
    # One GeoObject.
    vias = ly.generate_vias(True)

    # Here I use lumped ports instead of wave ports. I use the references made earlier to generate the port.
    # By default, all lumped port sheets will be shorted to z=-thickness. You can change this as an optional
    # argument.
    lp1 = ly.lumped_port(ly.ref('p1'))
    lp2 = ly.lumped_port(ly.ref('p2'))
    
    # Because lumped ports don't stop at the edge of our domain, we make sure to add some margins everywhere.
    ly.determine_bounds(5,5,5,5)

    # Finally we can generate the PCB volumes. Because the trace start halfway through the PCB we turn
    # on the split-z function which cuts the PCB in multiple layers. This improves meshing around the striplines.
    diel = ly.gen_pcb(True, merge=True)
    # We also define the air-box
    air = ly.gen_air(3)

    # The rest is as usual
    m.define_geometry(diel, lp1, lp2, trace, air, vias)

    m.view()

    m.physics.set_frequency_range(2e9, 3e9, 21)
    m.mesher.set_boundary_size(trace, 0.001)

    m.generate_mesh()
    
    # We display the geometry with extra attention to the vias. With the vias.outside() method we can
    # specifically show the outside faces of the via.
    m.view(selections=[vias.outside()])

    # We setup the lumped port boundary conditions. Because of an added functionality in the PCBLayouter 
    # class, you don't have to specify the width, height and direction of the lumped port, this information
    # is contained in the lumped port sheet. You can see this information as its stored in the lp1._aux_data
    # dictionary.

    p1 = em.bc.LumpedPort(lp1, 1)
    p2 = em.bc.LumpedPort(lp2, 2)

    pec = em.bc.PEC(trace)
    
    #We also add a PEC for the outsides of our via.
    pecvia = em.bc.PEC(vias.outside())

    m.physics.assign(p1, p2, pec, pecvia)

    # Finally we run the simulation!
    data = m.physics.frequency_domain_par(njobs=4)

    freq, S11 = data.ax('freq').S(1,1)
    freq, S21 = data.ax('freq').S(2,1)

    plt.plot(freq/1e9, 20*np.log10(np.abs(S11)))
    plt.plot(freq/1e9, 20*np.log10(np.abs(S21)))
    plt.legend(['S11','S21'])
    plt.grid(True)
    plt.show()


    m.display.add_object(diel, opacity=0.2)
    m.display.add_object(trace)
    m.display.add_object(vias)

    # In the latest version, you can use the cutplane method of the dataset class
    # which is equivalent to the interpolate method except it automatically generates
    # the point cloud based on a plane x,y or z coordinate.
    m.display.add_quiver(*data.item(11).cutplane(ds=0.001, z=-0.00025).vector('E'))
    m.display.add_surf(*data.item(11).cutplane(ds=0.001, z=-0.00075).scalar('Ez','real'))
    m.display.show()