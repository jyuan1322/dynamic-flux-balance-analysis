def plot_full_spectra_from_excel(
    excel_path,
    header_rows=2,
    plot_title=None,
    nidxs=20,
    downsample_ppm=None
):
    """
    Load stacked NMR spectra from Excel and plot full spectra.

    Parameters
    ----------
    excel_path : str
        Path to Excel file.
    header_rows : int
        Number of header rows before numeric data starts.
        (Your script uses 2.)
    plot_title : str or None
        Title of plot.
    nidxs : int
        Number of evenly spaced traces to show.
    downsample_ppm : int or None
        Downsample ppm axis for performance.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os

    # Load Excel
    df = pd.read_excel(excel_path, header=None)

    # Extract numeric data (skip header rows)
    data = df.iloc[header_rows:].reset_index(drop=True)
    data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
    data = data.astype(float)

    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    # Optional downsampling
    if downsample_ppm is not None:
        ppm = ppm[::downsample_ppm]
        traces = traces[::downsample_ppm, :]

    # Evenly spaced traces
    indices = np.linspace(0, n_traces - 1, min(nidxs, n_traces), dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6))

    colormap = cm.viridis
    norm = mcolors.Normalize(vmin=indices.min(), vmax=indices.max())

    for t in indices:
        ax.plot(ppm, traces[:, t], color=colormap(norm(t)), linewidth=1)

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Trace index")

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")

    if plot_title is None:
        plot_title = os.path.basename(excel_path)

    ax.set_title(plot_title)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


plot_full_spectra_from_excel(
    "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Jan302026_UGA_HRMAS_13C_Cells/processed_xlsx/raw_1H_mixture.xlsx",
    header_rows=2,      # matches your current format
    nidxs=30,
    downsample_ppm=2    # optional
)