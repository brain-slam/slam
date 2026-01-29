"""
plots.py.

Centralisation des fonctions pour plots et figures.

Auteur : Zoë LAFFITTE
Date : 2026
"""

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


def mes3d_projection(mesh_data, intensity_data, display_settings):
    """
    Crée une projection 3D d'un maillage avec intensités.

    Parameters
    ----------
    mesh_data : dict
        Contient les coordonnées des sommets et
        les indices des faces du maillage.

    intensity_data : dict
        Contient les valeurs d'intensité et le
        mode d'affichage ("vertex" ou "cell").

    display_settings : dict
        Paramètres d'affichage (e.g., colorscale, labels).

    Returns
    -------
    fig
        L'objet plotly.
    """
    # Extraction des données du maillage et des intensités
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    title = mesh_data["title"]

    # Extraction pour la proj de la feature
    intensities = intensity_data["values"]
    intensitymode = intensity_data.get(
        "mode", "cell"
    )  # choix du mode de projection, par défaut par face

    # Colorbar
    colorscale = display_settings.get(
        "colorscale", "Turbo"
    )  # choix de la colorbar, par défaut Turbo (continue)
    cmin = intensity_data.get(
        "cmin", None
    )  # si besoin, on modifie la plage de la colorbar
    cmax = intensity_data.get("cmax", None)
    title_colorbar = display_settings.get("colorbar_label", "")
    colorbar_tickvals = display_settings.get(
        "tickvals", None
    )  # si besoin, on remplace les vals de la texture sur la colorbar
    colorbar_ticktext = display_settings.get(
        "ticktext", None
    )  # si besoin, on remplace les graduation de la colorbar

    print("Plotting ...")

    # Création de la figure Plotly avec le maillage 3D
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale=colorscale,
                intensity=intensities,
                intensitymode=intensitymode,
                flatshading=True,
                cmin=cmin,
                cmax=cmax,
                colorbar={
                    "title": title_colorbar,
                    "len": 0.85,
                    "thickness": 25,
                    "tickfont": {"size": 16},
                },
                colorbar_tickvals=colorbar_tickvals,
                colorbar_ticktext=colorbar_ticktext,
                # suppression lumière caméra / scene
                lighting={
                    "ambient": 1,
                    "diffuse": 0,
                    "specular": 0,
                    "roughness": 1,
                    "fresnel": 0,
                },
            )
            # On peut également utiliser create_hover_trace()
            # On peut jouer sur le nombre de scène
        ]
    )
    camera = dict(
        eye=dict(x=2, y=0, z=0),  # Camera position from lateral side
        center=dict(x=0, y=0, z=0),  # Looking at center
        up=dict(x=0, y=0, z=1)  # Up vector points in positive z direction
    )
    # Configuration de la mise en page de la figure
    fig.update_layout(
        title=title,
        title_x=0.2,
        template="seaborn",  # theme
        scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=camera
            ),
        legend={
            "x": 0,  # position horizontale (0 = gauche)
            "y": 1,  # position verticale (1 = haut)
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig
