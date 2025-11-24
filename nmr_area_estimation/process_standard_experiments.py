import nmrglue as ng
import pandas as pd
import numpy as np
import glob
import os

def load_trace(pdata_path):
    """Load processed 1D Bruker spectrum and return ppm axis and intensity"""
    dic, data = ng.bruker.read_pdata(pdata_path)
    size = data.shape[-1]
    sw = float(dic['procs']['SW_p'])
    sf = float(dic['procs']['SF'])
    offset = float(dic['procs']['OFFSET'])
    ppm_scale = np.linspace(offset, offset - sw/sf, size)
    return ppm_scale, data

def detect_nucleus(exp_dir):
    """Return the NUC1 nucleus (<1H> or <13C>)"""
    dic, _ = ng.bruker.read(exp_dir)
    return dic['acqus']['NUC1']


def main():
    # base_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to directory containing Bruker experiment folders"
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    # Only include numeric directories (skip xlsx, txt, etc.)
    exp_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*"))
                    if os.path.isdir(d) and os.path.basename(d).isdigit()])

    traces_1H = None
    traces_13C = None

    for exp_dir in exp_dirs:
        pdata_path = os.path.join(exp_dir, "pdata", "1")
        
        # Skip if no pdata/1 folder
        if not os.path.isdir(pdata_path):
            print(f"Skipping {exp_dir}: pdata/1 folder not found")
            continue
        # Skip if no 1D spectrum file (1r)
        if not any(fname.startswith("1r") for fname in os.listdir(pdata_path)):
            print(f"Skipping {exp_dir}: no 1D spectrum found in {pdata_path}")
            continue

        try:
            nucleus = detect_nucleus(exp_dir)
        except Exception as e:
            print(f"Skipping {exp_dir}: cannot read NUC1 ({e})")
            continue

        exp_name = f"{os.path.basename(exp_dir)}_{nucleus.strip('<>')}"

        ppm, intensity = load_trace(pdata_path)

        print(f"nucleus: {nucleus}")

        # Initialize DataFrame with ppm axis
        if traces_1H is None and nucleus == "1H":
            traces_1H = pd.DataFrame({"ppm": ppm})
        if traces_13C is None and nucleus == "13C":
            traces_13C = pd.DataFrame({"ppm": ppm})

        # Add intensity column
        if nucleus == "1H":
            traces_1H[exp_name] = intensity
        elif nucleus == "13C":
            traces_13C[exp_name] = intensity

    # --- Export ---
    if traces_1H is not None:
        traces_1H.to_excel(os.path.join(base_dir, "traces_1H.xlsx"), index=False)
    if traces_13C is not None:
        traces_13C.to_excel(os.path.join(base_dir, "traces_13C.xlsx"), index=False)

    print("Done! 1H and 13C traces exported separately.")

if __name__ == "__main__":
    main()