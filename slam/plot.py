"""
plots.py.

Centralisation des fonctions pour plots et figures.

Auteur : Zoë LAFFITTE
Date : 2026
"""

from slam import io as sio
import plotly.graph_objects as go

def create_hover_trace(points, text, mode, **kwargs):
    """
    Crée une trace 3D pour un graphique Plotly.

    Parameters
    ----------
    points : array-like
        Coordonnées des points.

    labels : array-like
        Labels associés aux points.

    text : list
        Les informations de survol.

    mode : str
        Mode d'affichage des points (e.g., "markers", "lines").

    **kwargs : dict
        Arguments supplémentaires pour personnaliser la trace.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        La trace 3D avec survol.
    """

    # Création d'une trace 3D avec survol pour chaque point
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode=mode,
        marker={"size": 1, "color": "rgba(0,0,0,0)"},
        text=text,
        hoverinfo="text",
        showlegend=False,
    )


def mes3d_projection(mesh_data, intensity_data=None, display_settings=None):
    """
    Crée une projection 3D d'un maillage, avec ou sans intensités.

    Parameters
    ----------
    mesh_data : dict
        Contient les coordonnées des sommets et les indices des faces.
    intensity_data : dict, optional
        Contient les valeurs d'intensité et le mode d'affichage.
        Si None, le maillage est tracé sans texture.
    display_settings : dict, optional
        Paramètres d'affichage (e.g., colorscale, labels).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        L'objet figure Plotly.
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    title = mesh_data.get("title", "")

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

    if intensity_data is not None:
        mesh_kwargs.update({
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
            "lighting":{
                "ambient": 1,
                "diffuse": 0,
                "specular": 0,
                "roughness": 1,
                "fresnel": 0,
            },
            "colorbar_tickvals": display_settings.get("tickvals", None),
            "colorbar_ticktext": display_settings.get("ticktext", None),
        })
    camera = dict(
        eye=dict(x=2, y=0, z=0),  # Camera position from lateral side
        center=dict(x=0, y=0, z=0),  # Looking at center
        up=dict(x=0, y=0, z=1)  # Up vector points in positive z direction
    )
    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])

    fig.update_layout(
        title=title,
        title_x=0.2,
        template="seaborn",
        scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=camera
            ),
        legend={
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig

