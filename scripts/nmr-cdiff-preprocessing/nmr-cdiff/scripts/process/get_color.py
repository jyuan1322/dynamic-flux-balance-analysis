#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  11 15:59:50 2022

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Get colors for matplotlib visualization
 - Use python 3.8+ for best results
 - See /nmr-cdiff/venv/requirements.txt for dependencies

Copyright 2022 Massachusetts Host-Microbiome Center

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
cmap = {
    "Glucose": [28, 18, 195],   # 1. glc
    "Pyruvate": [251, 129, 255], # 2. pyr
    "Formate": [148, 103, 250], # 3. for
    "Bicarbonate": [255, 129, 23],  # 4. hco3
    "Lactate": [162, 62, 0],    # 5. lac
    "Acetyl-CoA": [53, 236, 191],  # 6. acoa
    "Acetate": [253, 0, 80],    # 7. ac
    "Ethanol": [24, 173, 0],    # 8. etoh
    "Butyrate": [245, 202, 61],  # 9. buty
    "Alanine": [74, 153, 255],  # 10. alaL
    "Leucine": [0, 128, 128],   # 11. leuL
    "Isovalerate": [75, 223, 46],   # 12. ival
    "Isocaproate": [255, 118, 116], # 13. icap
    "Proline": [0, 0, 128],     # 14. proL
    "5-aminovalerate": [142, 0, 65],    # 15. 5av
    "Butanol": [188, 189, 34],  # 16. butnl
    "gray": [127, 127, 127], # 17. gray
    "black": [0, 0, 0],       # 18. black
    "white": [255, 255, 255], # 19. white
    "Acetate13C": [0, 0, 0],
    "AcetateNA": [100, 100, 100],
    "Threonine": [28, 18, 195],
    "Propionate": [253, 0, 80],
    "n-Propanol": [24, 173, 0],
    "2-hydroxybutyrate": [162, 62, 0],
    "2-aminobutyrate": [74, 153, 255],
    "Valine": [0, 0, 0],
    "Isoleucine": [0, 0, 0],
    "Isobutyrate": [0, 0, 0],
    "2-methylbutyrate": [0, 0, 0]
}

def get_cmap():
    outmap = {k: tuple(ele/255. for ele in v) for k, v in cmap.items()}
    return outmap
