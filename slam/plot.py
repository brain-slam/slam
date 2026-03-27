"""
plots.py

Centralized functions for plotting and figures.

Authors: Zoë LAFFITTE, Guillaume Auzias
Date: 2026
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_graph_on_mesh(graph, mesh, vertex_index_attribute="pit_index",
                       out_shift=0.1, show_edge_ridge=False, text=None,
                       fig=None, **kwargs):
    """


    Parameters
    ----------
    points : array-like
        Coordinates of the points (N, 3).

    text : list
        Hover information for each point.

    **kwargs : dict
        Additional arguments to customize the trace (e.g., marker, hoverlabel).

    Returns
    -------
    plotly.graph_objects.Scatter3d
        The 3D trace with hover functionality.
    """
    # plot the nodes
    nodes_coords = coords_outward_shift(
        list(nx.get_node_attributes(graph, vertex_index_attribute).values()),
        mesh, out_shift=out_shift)
    trace_nodes = go.Scatter3d(
        x=nodes_coords[:, 0],
        y=nodes_coords[:, 1],
        z=nodes_coords[:, 2],
        mode="markers",
        marker=dict(
            color='red',
            size=6,
            line=dict(
                color='black',
                width=2
            )
        ),
        text=text,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    )
    if fig is None:
        fig = go.Figure()
    fig.add_trace(trace_nodes)
    # plot the edges
    line_marker = dict(color='black', width=4)
    if show_edge_ridge:
        for edj in graph.edges.items():
            ridge_coords = coords_outward_shift(edj[1]['ridge_index'],
                                                mesh, out_shift=out_shift)
            edj1_x = [nodes_coords[edj[0][0], 0], ridge_coords[0]]
            edj1_y = [nodes_coords[edj[0][0], 1], ridge_coords[1]]
            edj1_z = [nodes_coords[edj[0][0], 2], ridge_coords[2]]
            fig.add_trace(go.Scatter3d(x=edj1_x, y=edj1_y, z=edj1_z,
                                       mode='lines', line=line_marker))
            edj2_x = [ridge_coords[0], nodes_coords[edj[0][1], 0]]
            edj2_y = [ridge_coords[1], nodes_coords[edj[0][1], 1]]
            edj2_z = [ridge_coords[2], nodes_coords[edj[0][1], 2]]
            fig.add_trace(go.Scatter3d(x=edj2_x, y=edj2_y, z=edj2_z,
                                       mode='lines', line=line_marker))
    else:
        for edj in graph.edges.items():
            # print(edj)
            edj_x = [nodes_coords[edj[0][0], 0], nodes_coords[edj[0][1], 0]]
            edj_y = [nodes_coords[edj[0][0], 1], nodes_coords[edj[0][1], 1]]
            edj_z = [nodes_coords[edj[0][0], 2], nodes_coords[edj[0][1], 2]]
            fig.add_trace(go.Scatter3d(x=edj_x, y=edj_y, z=edj_z,
                                       mode='lines', line=line_marker))
    return fig


def plot_graph(graph, coords_attribute="3dcoords",
               text=None, fig=None, **kwargs):
    """


    Parameters
    ----------
    points : array-like
        Coordinates of the points (N, 3).

    text : list
        Hover information for each point.

    **kwargs : dict
        Additional arguments to customize the trace (e.g., marker, hoverlabel).

    Returns
    -------
    plotly.graph_objects.Scatter3d
        The 3D trace with hover functionality.
    """
    # plot the nodes
    nodes_coords = np.array(
        list(nx.get_node_attributes(graph, coords_attribute).values()))
    trace_nodes = go.Scatter3d(
        x=nodes_coords[:, 0],
        y=nodes_coords[:, 1],
        z=nodes_coords[:, 2],
        mode="markers",
        text=text,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    )
    if fig is None:
        fig = go.Figure()
    fig.add_trace(trace_nodes)
    # plot the edges
    line_marker = dict(color='black', width=4)
    for edj in graph.edges:
        # print(edj)
        edj_x = [nodes_coords[edj[0], 0], nodes_coords[edj[1], 0]]
        edj_y = [nodes_coords[edj[0], 1], nodes_coords[edj[1], 1]]
        edj_z = [nodes_coords[edj[0], 2], nodes_coords[edj[1], 2]]
        fig.add_trace(go.Scatter3d(x=edj_x, y=edj_y, z=edj_z,
                                   mode='lines', line=line_marker))
    return fig


def coords_outward_shift(vertex_indices, mesh, out_shift=0.1):
    """

    Parameters
    ----------
    vertex_indices: list or array of int
        list of vertex indices in the mesh.
    mesh: Trimesh mesh
        mesh object.
    out_shift: float
        shit the coords of the points outside of the mesh along vertex normals*
        of a magnitude computed as out_shift * vertex_normal

    Returns
    -------

    """
    vert_coords = mesh.vertices[vertex_indices, :]
    vert_normals = mesh.vertex_normals[vertex_indices, :]
    vert_coords = vert_coords + out_shift * vert_normals
    return vert_coords


def plot_points(points, text=None, **kwargs):
    """
    Creates a 3D trace for a Plotly figure with customizable options.

    Parameters
    ----------
    points : array-like
        Coordinates of the points (N, 3).

    text : list
        Hover information for each point.

    **kwargs : dict
        Additional arguments to customize the trace (e.g., marker, hoverlabel).

    Returns
    -------
    plotly.graph_objects.Scatter3d
        The 3D trace with hover functionality.
    """

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        text=text,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    )


def plot_mesh(
    mesh_data, intensity_data=None, display_settings=None, caption=None
):
    """
    Creates a 3D projection of a mesh, with or without intensity data.

    Parameters
    ----------
    mesh_data : dict
        Contains the vertex coordinates and face indices of the mesh.

    intensity_data : dict, optional
        Contains intensity values and display mode.
        If None, the mesh is plotted without a texture.

    display_settings : dict, optional
        Display parameters (e.g., colorscale, labels).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting Plotly figure object.
    """

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    if "opacity" in mesh_data.keys():
        opacity = mesh_data["opacity"]
    else:
        opacity = 1
    title = mesh_data.get("title", "")

    if display_settings is None:
        display_settings = {}
    template = display_settings.get("template", None)

    lighting = {
        "ambient": 0.7,
        "diffuse": 0.8,
        "specular": 0.5,
        "roughness": 0.2,
        "fresnel": 0.1,
    }

    mesh_kwargs = {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
        "color": "ghostwhite",
        "opacity": opacity,
        "flatshading": False,
        "lighting": lighting,
    }

    camera = {
        # Camera position from lateral side
        "eye": {"x": 2.5, "y": 0, "z": 0.0},
        # Looking at center
        "center": {"x": 0, "y": 0, "z": 0},
        # Up vector points in positive z direction
        "up": {"x": 0, "y": 0, "z": 1},
    }

    aff_dict = {
        "title": title,
        "title_x": 0.2,
        "height": 900,
        "width": 1200,
        "template": template,
        "legend": {
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
        "scene": {"camera": camera},
    }

    if intensity_data is not None:
        mesh_kwargs.update(
            {
                "intensity": intensity_data["values"],
                "intensitymode": intensity_data.get("mode", "cell"),
                "colorscale": display_settings.get("colorscale", "Turbo"),
                "cmin": intensity_data.get("cmin", None),
                "cmax": intensity_data.get("cmax", None),
                "colorbar": {
                    "title": display_settings.get("colorbar_label", ""),
                    "len": 0.85,
                    "thickness": 25,
                    "tickfont": {"size": 16},
                },
                "flatshading": True,
                "lighting": {
                    "ambient": 1,
                    "diffuse": 0,
                    "specular": 0,
                    "roughness": 1,
                    "fresnel": 0,
                },
                "colorbar_tickvals": display_settings.get("tickvals", None),
                "colorbar_ticktext": display_settings.get("ticktext", None),
            }
        )

    if caption:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            horizontal_spacing=0.03,
        )
        for col in [1, 2]:
            fig.add_trace(go.Mesh3d(**mesh_kwargs), row=1, col=col)

        aff_dict["scene2"] = {
            "camera": {
                # Camera position from lateral side
                "eye": {"x": -2.5, "y": 0, "z": 0.0},
                # Looking at center
                "center": {"x": 0, "y": 0, "z": 0},
                # Up vector points in positive z direction
                "up": {"x": 0, "y": 0, "z": 1},
            }
        }

    else:
        fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])

    if template is None:
        aff_dict["template"] = "simple_white"
        aff_dict["scene"].update(
            {
                "aspectmode": "data",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    fig.update_layout(**aff_dict)
    return fig
