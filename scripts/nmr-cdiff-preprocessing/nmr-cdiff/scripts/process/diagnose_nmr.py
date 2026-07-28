"""
COMPREHENSIVE NMR PROCESSING DIAGNOSTIC TOOL

This will help identify why your spectra look corrupted (huge spike in middle,
both positive and negative peaks).

Common causes:
1. Wrong GRPDLY (group delay) - causes carrier artifact in center
2. Wrong DECIM - causes aliasing/folding artifacts  
3. Wrong carrier frequency (CAR) - shifts spectrum incorrectly
4. Digital filter not properly removed - leaves spike at carrier position
"""

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os


def diagnose_nmr_processing(fid_path, expt='13C'):
    """
    Comprehensive diagnostic to identify NMRPipe parameter issues.
    
    This will:
    1. Show what parameters were extracted from acqus
    2. Try multiple GRPDLY values (most common issue)
    3. Try with/without digital filter correction
    4. Compare to what nmrglue alone produces
    """
    
    print("\n" + "="*70)
    print("NMR PROCESSING DIAGNOSTIC")
    print("="*70)
    print(f"FID: {fid_path}")
    print(f"Experiment: {expt}")
    print("="*70 + "\n")
    
    # Read the raw data
    print("Reading Bruker data...")
    dic, data = ng.bruker.read(fid_path)
    acqus = dic['acqus']
    
    # Display all relevant acquisition parameters
    print("\nACQUISITION PARAMETERS:")
    print("-" * 70)
    important_params = ['TD', 'SW', 'SW_h', 'SFO1', 'O1', 'BF1', 'NUC1', 
                       'PULPROG', 'NS', 'DS', 'DECIM', 'DSPFVS', 'GRPDLY',
                       'DIGMOD', 'DIGTYP', 'AQ_mod']
    
    for param in important_params:
        if param in acqus:
            print(f"  {param:15s} = {acqus[param]}")
        else:
            print(f"  {param:15s} = NOT FOUND")
    
    # Calculate derived parameters
    td = acqus['TD']
    sw_ppm = acqus.get('SW', 0)
    sw_h = acqus.get('SW_h', 0)
    sfo1 = acqus['SFO1']
    
    if sw_h == 0 and sw_ppm != 0:
        sw_h = sw_ppm * sfo1
    
    print(f"\n  Calculated SW_h: {sw_h:.3f} Hz")
    print(f"  Acquisition time: {td/sw_h:.3f} seconds")
    
    # Get digital filter info
    decim = acqus.get('DECIM', 0)
    dspfvs = acqus.get('DSPFVS', 0)
    grpdly = acqus.get('GRPDLY', 0)
    digmod = acqus.get('DIGMOD', 0)
    
    print("\nDIGITAL FILTER INFORMATION:")
    print("-" * 70)
    print(f"  DECIM: {decim}")
    print(f"  DSPFVS: {dspfvs}")
    print(f"  GRPDLY: {grpdly}")
    print(f"  DIGMOD: {digmod}")
    
    # Check if digital filtering was used
    if decim > 1:
        print("\n  ⚠️  Digital filtering WAS used (DECIM > 1)")
        print("  This means group delay correction is CRITICAL")
        if decim > 256:
            print(f"\n  ⚠️⚠️  WARNING: DECIM={decim} is VERY HIGH!")
            print("  This is unusual and may indicate:")
            print("    - Special acquisition mode")
            print("    - Potential issue with acqus file")
            print("    - May need special processing")
    else:
        print("\n  ✓ No digital filtering (DECIM = 1)")
    
    # Now try different processing approaches
    print("\n" + "="*70)
    print("TESTING DIFFERENT PROCESSING APPROACHES")
    print("="*70)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Processing Diagnostics: {os.path.basename(fid_path)}', fontsize=14)
    axes = axes.flatten()
    
    # Test 1: nmrglue processing only (no NMRPipe)
    print("\n1. Processing with nmrglue only (no NMRPipe)...")
    try:
        dic_ng, data_ng = ng.bruker.read_pdata(fid_path + '/pdata/1')
        udic = ng.bruker.guess_udic(dic_ng, data_ng)
        uc = ng.fileiobase.uc_from_udic(udic)
        ppm = uc.ppm_scale()
        axes[0].plot(ppm, data_ng.real)
        axes[0].set_title('Bruker processed (pdata/1)')
        axes[0].set_xlim(200, -20)
        axes[0].grid(True, alpha=0.3)
    except Exception as e:
        # If no processed data, do basic processing
        print(f"    Note: {e}")
        dic_ng = dic
        data_ng = ng.bruker.remove_digital_filter(dic, data)
        data_ng = ng.proc_base.em(data_ng, lb=1)
        data_ng = ng.proc_base.zf(data_ng, 2)
        data_ng = ng.proc_base.fft(data_ng)
        udic = ng.bruker.guess_udic(dic_ng, data_ng)
        uc = ng.fileiobase.uc_from_udic(udic)
        ppm = uc.ppm_scale()
        axes[0].plot(ppm, data_ng.real)
        axes[0].set_title('nmrglue basic processing')
        axes[0].set_xlim(200, -20)
        axes[0].grid(True, alpha=0.3)
    
    # Test 2-4: Try different GRPDLY values
    print("\n2-4. Testing different GRPDLY values...")
    grpdly_tests = [grpdly, 67.9840, 0]
    
    for i, test_grpdly in enumerate(grpdly_tests, start=1):
        print(f"  Testing GRPDLY = {test_grpdly}")
        try:
            script = generate_test_script(expt, fid_path, dic, override_grpdly=test_grpdly)
            dic_test, data_test = run_pipe_script(script)
            uc_test = ng.pipe.make_uc(dic_test, data_test)
            ppm_test = uc_test.ppm_scale()
            axes[i].plot(ppm_test, data_test.real)
            axes[i].set_title(f'GRPDLY = {test_grpdly}')
            axes[i].set_xlim(200, -20)
            axes[i].grid(True, alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Failed:\n{str(e)[:50]}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # Test 5-6: With and without digital filter flags
    print("\n5-6. Testing digital filter settings...")
    
    # Without any digital filter correction
    print("  Testing without digital filter correction...")
    try:
        script = generate_test_script(expt, fid_path, dic, 
                                     override_grpdly=0, 
                                     override_decim=1)
        dic_test, data_test = run_pipe_script(script)
        uc_test = ng.pipe.make_uc(dic_test, data_test)
        ppm_test = uc_test.ppm_scale()
        axes[4].plot(ppm_test, data_test.real)
        axes[4].set_title('No digital filter correction')
        axes[4].set_xlim(200, -20)
        axes[4].grid(True, alpha=0.3)
    except Exception as e:
        axes[4].text(0.5, 0.5, f'Failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[4].transAxes)
    
    # With automatic group delay calculation
    print("  Testing with auto GRPDLY...")
    try:
        # Calculate expected group delay based on DSPFVS and DECIM
        auto_grpdly = calculate_grpdly(dspfvs, decim)
        print(f"    Calculated GRPDLY: {auto_grpdly}")
        script = generate_test_script(expt, fid_path, dic, override_grpdly=auto_grpdly)
        dic_test, data_test = run_pipe_script(script)
        uc_test = ng.pipe.make_uc(dic_test, data_test)
        ppm_test = uc_test.ppm_scale()
        axes[5].plot(ppm_test, data_test.real)
        axes[5].set_title(f'Auto GRPDLY = {auto_grpdly:.2f}')
        axes[5].set_xlim(200, -20)
        axes[5].grid(True, alpha=0.3)
    except Exception as e:
        axes[5].text(0.5, 0.5, f'Failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[5].transAxes)
    
    # Test 7: Try nmrglue's digital filter removal
    print("\n7. Testing nmrglue digital filter removal...")
    try:
        data_filtered = ng.bruker.remove_digital_filter(dic, data)
        data_processed = ng.proc_base.em(data_filtered, lb=1)
        data_processed = ng.proc_base.zf(data_processed, 2)
        data_processed = ng.proc_base.fft(data_processed)
        udic_proc = ng.bruker.guess_udic(dic, data_processed)
        uc_proc = ng.fileiobase.uc_from_udic(udic_proc)
        ppm_proc = uc_proc.ppm_scale()
        axes[6].plot(ppm_proc, data_processed.real)
        axes[6].set_title('nmrglue digital filter removal')
        axes[6].set_xlim(200, -20)
        axes[6].grid(True, alpha=0.3)
    except Exception as e:
        axes[6].text(0.5, 0.5, f'Failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[6].transAxes)
    
    # Test 8: Your current parameters
    print("\n8. Testing your current parameters...")
    try:
        script = generate_test_script(expt, fid_path, dic)
        dic_test, data_test = run_pipe_script(script)
        uc_test = ng.pipe.make_uc(dic_test, data_test)
        ppm_test = uc_test.ppm_scale()
        axes[7].plot(ppm_test, data_test.real)
        axes[7].set_title('Your current parameters')
        axes[7].set_xlim(200, -20)
        axes[7].grid(True, alpha=0.3)
    except Exception as e:
        axes[7].text(0.5, 0.5, f'Failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[7].transAxes)
    
    # Hide the last subplot
    axes[8].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("1. Look for the plot that shows NORMAL NMR peaks:")
    print("   - Peaks should all be positive (pointing up)")
    print("   - No huge spike in the middle")
    print("   - Baseline should be flat and near zero")
    print("")
    print("2. Common issues and fixes:")
    print("   - Huge spike in center → Wrong GRPDLY")
    print("   - Peaks both positive/negative → Phase correction needed")
    print("   - Spectrum flipped → Wrong carrier frequency")
    print("   - Severe distortion → Wrong DECIM or DSPFVS")
    print("")
    print("3. If nmrglue processing (plot 1 or 7) looks good:")
    print("   → Don't use NMRPipe, use nmrglue instead")
    print("")
    print("4. If one of the GRPDLY tests looks good:")
    print("   → Use that GRPDLY value in your processing")
    print("="*70 + "\n")


def calculate_grpdly(dspfvs, decim):
    """Calculate expected group delay based on firmware version and decimation."""
    # This is based on Bruker's digital filter lookup table
    # DSPFVS = digital signal processing firmware version
    grpdly_table = {
        # DSPFVS: {DECIM: GRPDLY}
        10: {2: 44.75, 3: 33.5, 4: 66.625, 6: 59.083, 8: 68.563, 12: 60.375, 16: 69.531, 24: 61.021, 32: 70.015, 48: 61.345, 64: 70.265, 96: 61.506, 128: 70.390, 192: 61.600, 256: 70.465, 384: 61.652, 512: 70.515, 768: 61.684, 1024: 70.548, 1536: 61.704, 2048: 70.571},
        11: {2: 46.0, 3: 36.5, 4: 48.0, 6: 50.167, 8: 53.25, 12: 69.5, 16: 72.25, 24: 70.167, 32: 72.75, 48: 70.5, 64: 73.0, 96: 70.667, 128: 72.5, 192: 71.333, 256: 72.25, 384: 71.667, 512: 72.125, 768: 71.833, 1024: 72.063, 1536: 71.917, 2048: 72.031},
        12: {2: 46.311, 3: 36.53, 4: 47.87, 6: 50.229, 8: 53.289, 12: 69.551, 16: 71.6, 24: 70.184, 32: 72.138, 48: 70.528, 64: 72.348, 96: 70.7, 128: 72.524, 192: 71.34, 256: 72.611, 384: 71.68, 512: 72.676, 768: 71.84, 1024: 72.72, 1536: 71.92, 2048: 72.75},
        13: {2: 2.75, 3: 2.8333, 4: 2.875, 6: 2.9167, 8: 2.9375, 12: 2.9583, 16: 2.9688, 24: 2.9792, 32: 2.9844, 48: 2.9896, 64: 2.9922, 96: 2.9948, 128: 2.9961, 192: 2.9974, 256: 2.998, 384: 2.9987, 512: 2.999, 768: 2.9993, 1024: 2.9995, 1536: 2.9997, 2048: 2.9998},
        20: {2: 2.75, 3: 2.8333, 4: 2.875, 6: 2.9167, 8: 2.9375, 12: 2.9583, 16: 2.9688, 24: 2.9792, 32: 2.9844, 48: 2.9896, 64: 2.9922, 96: 2.9948, 128: 2.9961, 192: 2.9974, 256: 2.998, 384: 2.9987, 512: 2.999, 768: 2.9993, 1024: 2.9995, 1536: 2.9997, 2048: 2.9998},
        21: {2: 10.375, 3: 10.5, 4: 10.5625, 6: 10.625, 8: 10.6563, 12: 10.6875, 16: 10.7031, 24: 10.7188, 32: 10.7266, 48: 10.7344, 64: 10.7383, 96: 10.7422, 128: 10.7441, 192: 10.7461, 256: 10.7471, 384: 10.748, 512: 10.7485, 768: 10.749, 1024: 10.7493, 1536: 10.7495, 2048: 10.7496},
    }
    
    if dspfvs in grpdly_table and decim in grpdly_table[dspfvs]:
        return grpdly_table[dspfvs][decim]
    else:
        print(f"    Warning: No lookup table entry for DSPFVS={dspfvs}, DECIM={decim}")
        # For DSPFVS 20/13, high DECIM values approach 3.0
        if dspfvs in [13, 20] and decim > 2048:
            print(f"    Using approximation: GRPDLY ≈ 3.0 for high DECIM")
            return 3.0
        return 0


def generate_test_script(expt, fid, dic, override_grpdly=None, override_decim=None):
    """Generate test script with optional parameter overrides."""
    acqus = dic['acqus']
    
    td = int(acqus['TD'])
    sfo1 = float(acqus['SFO1'])
    sw_h = float(acqus['SW_h'])
    o1 = float(acqus['O1'])
    car_ppm = o1 / sfo1
    nuc = str(acqus.get('NUC1', '1H'))
    
    xN = td
    xT = td // 2
    
    decim = int(acqus.get('DECIM', 1)) if override_decim is None else int(override_decim)
    dspfvs = int(acqus.get('DSPFVS', 0))
    grpdly = float(acqus.get('GRPDLY', 0)) if override_grpdly is None else float(override_grpdly)
    
    if expt == '13C':
        lb = 1.0
        zf_factor = 3
    else:
        lb = 0.3
        zf_factor = 2
    
    script_lines = [
        "#!/bin/csh",
        "",
        "bruk2pipe -in {}/fid \\".format(fid),
        "  -bad 0.0 -ext -aswap -AMX -decim {} -dspfvs {} -grpdly {} \\".format(decim, dspfvs, grpdly),
        "  -xN             {} \\".format(xN),
        "  -xT             {} \\".format(xT),
        "  -xMODE            DQD \\",
        "  -xSW        {:.3f} \\".format(sw_h),
        "  -xOBS         {:.3f} \\".format(sfo1),
        "  -xCAR          {:.3f} \\".format(car_ppm),
        "  -xLAB             {} \\".format(nuc),
        "  -ndim               1 \\",
        "| nmrPipe -fn EM -lb {} -c 1.0 \\".format(lb),
        "| nmrPipe -fn ZF -zf {} \\".format(zf_factor),
        "| nmrPipe -fn FT",
        ""
    ]
    
    return "\n".join(script_lines)


def run_pipe_script(script):
    """Execute NMRPipe script and return processed data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.com', delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name
    
    try:
        os.chmod(tmp_path, 0o755)
        pipe_output = subprocess.run(
            ["csh", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30)
        
        if pipe_output.returncode != 0:
            raise RuntimeError(f"NMRPipe failed: {pipe_output.stderr.decode()}")
        
        return ng.fileio.pipe.read(pipe_output.stdout)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <path_to_fid> [experiment_type]")
        print("Example: python diagnose.py /data/nmr/23 13C")
        sys.exit(1)
    
    fid_path = sys.argv[1]
    expt = sys.argv[2] if len(sys.argv) > 2 else '13C'
    
    diagnose_nmr_processing(fid_path, expt)
