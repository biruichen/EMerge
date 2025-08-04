from ..mesh3d import Mesh3D
from ..geometry import GeoObject
from ..selection import Selection
from ..physics.microwave.microwave_bc import PortBC
from typing import Iterable, Literal
import numpy as np

cmap_names = Literal['bgy','bgyw','kbc','blues','bmw','bmy','kgy','gray','dimgray','fire','kb','kg','kr',
                     'bkr','bky','coolwarm','gwv','bjy','bwy','cwr','colorwheel','isolum','rainbow','fire',
                     'cet_fire','gouldian','kbgyw','cwr','CET_CBL1','CET_CBL3','CET_D1A']


class BaseDisplay:

    def __init__(self, mesh: Mesh3D):
        self._mesh: Mesh3D = mesh

    def show(self):
        raise NotImplementedError('This method is not implemented')
        
    def add_object(self, obj: GeoObject | Selection,*args, **kwargs):
        """ Adds an object to the display

        Keyword arguments
        ----------
        color : ColorLike, optional
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

            Defaults to :attr:`pyvista.global_theme.color
            <pyvista.plotting.themes.Theme.color>`.

        style : str, optional
            Visualization style of the mesh.  One of the following:
            ``style='surface'``, ``style='wireframe'``, ``style='points'``,
            ``style='points_gaussian'``. Defaults to ``'surface'``. Note that
            ``'wireframe'`` only shows a wireframe of the outer geometry.
            ``'points_gaussian'`` can be modified with the ``emissive``,
            ``render_points_as_spheres`` options.

        scalars : str | numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name
            of an array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active
            scalars are used.

        clim : sequence[float], optional
            Two item color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng`` is
            also an accepted alias for this.

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        edge_color : ColorLike, optional
            The solid color to give the edges when ``show_edges=True``.
            Either a string, RGB list, or hex color string.

            Defaults to :attr:`pyvista.global_theme.edge_color
            <pyvista.plotting.themes.Theme.edge_color>`.

        point_size : float, optional
            Point size of any nodes in the dataset plotted. Also
            applicable when style='points'. Default ``5.0``.

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default ``None``.

        opacity : float | str| array_like
            Opacity of the mesh. If a single float value is given, it
            will be the global opacity of the mesh and uniformly
            applied everywhere - should be between 0 and 1. A string
            can also be specified to map the scalars range to a
            predefined opacity transfer function (options include:
            ``'linear'``, ``'linear_r'``, ``'geom'``, ``'geom_r'``).
            A string could also be used to map a scalars array from
            the mesh to the opacity (must have same number of elements
            as the ``scalars`` argument). Or you can pass a custom
            made transfer function that is an array either
            ``n_colors`` in length or shorter.

        flip_scalars : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r``
            suffix to do this as well.

        lighting : bool, optional
            Enable or disable view direction lighting. Default ``False``.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        interpolate_before_map : bool, optional
            Enabling makes for a smoother scalars display.  Default is
            ``True``.  When ``False``, OpenGL will interpolate the
            mapped colors which can result is showing colors that are
            not present in the color map.

        cmap : str | list | pyvista.LookupTable, default: :attr:`pyvista.plotting.themes.Theme.cmap`
            If a string, this is the name of the ``matplotlib`` colormap to use
            when mapping the ``scalars``.  See available Matplotlib colormaps.
            Only applicable for when displaying ``scalars``.
            ``colormap`` is also an accepted alias
            for this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.Plotter.add_legend`.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene. The default
            setting is ``None``, where the camera is only reset if this plotter
            has already been shown. If ``False``, the camera is not reset
            regardless of the state of the ``Plotter``. When ``True``, the
            camera is always reset.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.Plotter.add_scalar_bar`.

        show_scalar_bar : bool, optional
            If ``False``, a scalar bar will not be added to the
            scene.

        multi_colors : bool | str | cycler.Cycler | sequence[ColorLike], default: False
            If a :class:`pyvista.MultiBlock` dataset is given this will color
            each block by a solid color using a custom cycler.

            If ``True``, the default 'matplotlib' color cycler is used.

            See :func:`set_color_cycler<Plotter.set_color_cycler>` for usage of
            custom color cycles.

        name : str, optional
            The name for the added mesh/actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        texture : pyvista.Texture or np.ndarray, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets.

        render_points_as_spheres : bool, optional
            Render points as spheres rather than dots.

        render_lines_as_tubes : bool, optional
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        smooth_shading : bool, optional
            Enable smooth shading when ``True`` using the Phong
            shading algorithm.  When ``False``, use flat shading.
            Automatically enabled when ``pbr=True``.  See
            :ref:`shading_example`.

        split_sharp_edges : bool, optional
            Split sharp edges exceeding 30 degrees when plotting with smooth
            shading.  Control the angle with the optional keyword argument
            ``feature_angle``.  By default this is ``False`` unless overridden
            by the global or plotter theme.  Note that enabling this will
            create a copy of the input mesh within the plotter.  See
            :ref:`shading_example`.

        ambient : float, optional
            When lighting is enabled, this is the amount of light in
            the range of 0 to 1 (default 0.0) that reaches the actor
            when not directed at the light source emitted from the
            viewer.

        diffuse : float, optional
            The diffuse lighting coefficient. Default 1.0.

        specular : float, optional
            The specular lighting coefficient. Default 0.0.

        specular_power : float, optional
            The specular power. Between 0.0 and 128.0.

        nan_color : ColorLike, optional
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        nan_opacity : float, optional
            Opacity of ``NaN`` values.  Should be between 0 and 1.
            Default 1.0.

        culling : str, optional
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. This can be helpful for dense
            surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed.  Defaults to
            ``False``.

        rgb : bool, optional
            If an 2 dimensional array is passed as the scalars, plot
            those values as RGB(A) colors. ``rgba`` is also an
            accepted alias for this.  Opacity (the A) is optional.  If
            a scalars array ending with ``"_rgba"`` is passed, the default
            becomes ``True``.  This can be overridden by setting this
            parameter to ``False``.

        categories : bool, optional
            If set to ``True``, then the number of unique values in
            the scalar array will be used as the ``n_colors``
            argument.

        silhouette : dict, bool, optional
            If set to ``True``, plot a silhouette highlight for the
            mesh. This feature is only available for a triangulated
            ``PolyData``.  As a ``dict``, it contains the properties
            of the silhouette to display:

                * ``color``: ``ColorLike``, color of the silhouette
                * ``line_width``: ``float``, edge width
                * ``opacity``: ``float`` between 0 and 1, edge transparency
                * ``feature_angle``: If a ``float``, display sharp edges
                  exceeding that angle in degrees.
                * ``decimate``: ``float`` between 0 and 1, level of decimation

        use_transparency : bool, optional
            Invert the opacity mappings and make the values correspond
            to transparency.

        below_color : ColorLike, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'below'``.

        above_color : ColorLike, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'above'``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the string annotations.

        pickable : bool, optional
            Set whether this actor is pickable.

        preference : str, default: "point"
            When ``mesh.n_points == mesh.n_cells`` and setting
            scalars, this parameter sets how the scalars will be
            mapped to the mesh.  Default ``'point'``, causes the
            scalars will be associated with the mesh points.  Can be
            either ``'point'`` or ``'cell'``.

        log_scale : bool, default: False
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float.

        pbr : bool, optional
            Enable physics based rendering (PBR) if the mesh is
            ``PolyData``.  Use the ``color`` argument to set the base
            color.

        metallic : float, optional
            Usually this value is either 0 or 1 for a real material
            but any value in between is valid. This parameter is only
            used by PBR interpolation.

        roughness : float, optional
            This value has to be between 0 (glossy) and 1 (rough). A
            glossy material has reflections and a high specular
            part. This parameter is only used by PBR
            interpolation.

        render : bool, default: True
            Force a render when ``True``.

        user_matrix : np.ndarray | vtk.vtkMatrix4x4, default: np.eye(4)
            Matrix passed to the Actor class before rendering. This affects the
            actor/rendering only, not the input volume itself. The user matrix is the
            last transformation applied to the actor before rendering. Defaults to the
            identity matrix.

        component : int, optional
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        emissive : bool, optional
            Treat the points/splats as emissive light sources. Only valid for
            ``style='points_gaussian'`` representation.

        copy_mesh : bool, default: False
            If ``True``, a copy of the mesh will be made before adding it to
            the plotter.  This is useful if you would like to add the same
            mesh to a plotter multiple times and display different
            scalars. Setting ``copy_mesh`` to ``False`` is necessary if you
            would like to update the mesh after adding it to the plotter and
            have these updates rendered, e.g. by changing the active scalars or
            through an interactive widget. This should only be set to ``True``
            with caution. Defaults to ``False``. This is ignored if the input
            is a ``vtkAlgorithm`` subclass.

        backface_params : dict | pyvista.Property, optional
            A :class:`pyvista.Property` or a dict of parameters to use for
            backface rendering. This is useful for instance when the inside of
            oriented surfaces has a different color than the outside. When a
            :class:`pyvista.Property`, this is directly used for backface
            rendering. When a dict, valid keys are :class:`pyvista.Property`
            attributes, and values are corresponding values to use for the
            given property. Omitted keys (or the default of
            ``backface_params=None``) default to the corresponding frontface
            properties.

        show_vertices : bool, optional
            When ``style`` is not ``'points'``, render the external surface
            vertices. The following optional keyword arguments may be used to
            control the style of the vertices:

            * ``vertex_color`` - The color of the vertices
            * ``vertex_style`` - Change style to ``'points_gaussian'``
            * ``vertex_opacity`` - Control the opacity of the vertices

        edge_opacity : float, optional
            Edge opacity of the mesh. A single float value that will be applied globally
            edge opacity of the mesh and uniformly applied everywhere - should be
            between 0 and 1.

            .. note::
                `edge_opacity` uses ``SetEdgeOpacity`` as the underlying method which
                requires VTK version 9.3 or higher. If ``SetEdgeOpacity`` is not
                available, `edge_opacity` is set to 1.

        **kwargs : dict, optional
            Optional keyword arguments.
        """
        raise NotImplementedError('This method is not implemented')

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        raise NotImplementedError('This method is not implemented')

    def add_portmode(self, port: PortBC, 
                     Npoints: int = 10, 
                     dv=(0,0,0), 
                     XYZ=None,
                     field: Literal['E','H'] = 'E', 
                     k0: float | None = None,
                     mode_number: int | None = None):
        raise NotImplementedError('This method is not implemented')

    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              color: tuple[float, float, float] | None = None,
              scalemode: Literal['lin','log'] = 'lin') -> None:
        
        raise NotImplementedError('This method is not implemented')
    
    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 scale: Literal['lin','log','symlog'] = 'lin',
                 cmap: cmap_names = 'coolwarm',
                 clim: tuple[float, float] | None = None,
                 opacity: float = 1.0,
                 symmetrize: bool = True,
                 _fieldname: str | None = None,
                 **kwargs,):
        """Add a surface plot to the display
        The X,Y,Z coordinates must be a 2D grid of data points. The field must be a real field with the same size.

        Args:
            x (np.ndarray): The X-grid array
            y (np.ndarray): The Y-grid array
            z (np.ndarray): The Z-grid array
            field (np.ndarray): The scalar field to display
            scale (Literal["lin","log","symlog"], optional): The colormap scaling¹. Defaults to 'lin'.
            cmap (cmap_names, optional): The colormap. Defaults to 'coolwarm'.
            clim (tuple[float, float], optional): Specific color limits (min, max). Defaults to None.
            opacity (float, optional): The opacity of the surface. Defaults to 1.0.
            symmetrize (bool, optional): Wether to force a symmetrical color limit (-A,A). Defaults to True.
        
        (¹): lin: f(x)=x, log: f(x)=log₁₀(|x|), symlog: f(x)=sgn(x)·log₁₀(1+|x·ln(10)|)
        """
        raise NotImplementedError('This method is not implemented')