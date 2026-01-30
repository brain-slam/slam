"""
plots.py

Centralized functions for plotting and figures.

Author: Zoë LAFFITTE
Date: 2026
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_hover_trace(points, text, mode="markers", **kwargs):
    """
    Creates a 3D trace for a Plotly figure with customizable options.

    Parameters
    ----------
    points : array-like
        Coordinates of the points (N, 3).

    text : list
        Hover information for each point.

    mode : str, optional
        Display mode for the points (e.g., "markers", "lines"),
        default is "markers".

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
        mode=mode,
        text=text,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    )


def mesh_projection(
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
    title = mesh_data.get("title", "")
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
