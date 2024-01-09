import os
import sys
from typing import List
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem

from twinbooster.scripts.model import TwinBooster

from eval_zero_shot_lgbm_mf import get_metrics, generate_data, init_text_emb


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process and analyse assay data.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save results.")
    parser.add_argument("--aid", type=int, default=2732, help="Assay ID for analysis.")
    parser.add_argument("--description", type=str, default="2732", help="Assay description.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Assay data processing...
    if args.description == "2732":
        description = """
        HTS for small molecule inhibitors of CHOP to regulate the unfolded protein response to ER stress.
        Many genetic and environmental diseases result from defective protein folding within the secretory pathway so that aberrantly folded proteins are recognized by the cellular surveillance system and retained within the endoplasmic reticulum (ER). Under conditions of malfolded protein accumulation, the cell activates the Unfolded Protein Response (UPR) to clear the malfolded proteins, and if unsuccessful, initiates a cell death response. Preliminary studies have shown that CHOP is a crucial factor in the apoptotic arm of the UPR; XBP1 activates genes encoding ER protein chaperones and thereby mediates the adaptive UPR response to increase clearance of malfolded proteins. Inhibition of CHOP is hypothesized to enhance survival by preventing UPR programmed cell death. There are currently no known small molecule CHOP inhibitors either for laboratory or clinical use.
        To identify small molecule inhibitors of the UPR pathway, mediated by CHOP, a cell-based luciferase reporter assay using stably transfected CHO-K1 cells with luciferase driven by the CHOP promoter has been developed. The assay have been optimized and validated in 384-well format and used to screen for inhibitors of tunicamycin-induced CHOP in HTS. These identified compounds will have potential therapeutic application to diverse disease states ranging from diabetes, Alzheimer's disease, and Parkinson's disease, to hemophilia, lysosomal storage diseases, and alpha-1 antitrypsin deficiency. 
        Reagents:
        1. Cell line: CHO-CHOP cells with a luciferase reporter driven by the CHOP promoter (provided by assay PI)
        2. Cell growth media (Ham's F12 + Glutamax, 10% FBS, 1X non-essential amino acids, and penicillin:streptomycin) (Invitrogen)
        3. Tunicamycin (Calbiochem)
        4. SteadyGlo reagent (Promega)
        Protocol:
        1. 40 uL of medium containing CHO-CHOP cells (3000-4000) were dispensed to 384 well white opaque plates (Corning #3570) using a Multidrop combi (Thermo-Fisher Scientific). Plates were then incubated for 24 hrs at 37 degrees C, 5% CO2.
        2. 0.5 uL of library compounds (1 mM in DMSO) was added to wells using Sciclone (Caliper LifeSciences). The final concentration of compound is 10 uM.
        3. 10 uL of fresh medium containing tunicamycin (Tm) (2.0 ug/ml, final concentration,) was then added and the plates were incubated for 15-18 hrs.
        4. Medium was aspirated with an Elx405 plate washer (BioTek), leaving 10 uL of medium in the well. 10 uL of Steady-Glo was added to each well using a multildrop combi.
        5. Luminescence signal was measured on an Envision Multilable plate reader (PerkinElmer). 
        """
        
    else:
        description = args.description

    print("word count:", len(description.split(" ")))

    aid = args.aid
    url = f"https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&aid={aid}&version=1.1&response_type=save"
    assay = pd.read_csv(url, low_memory=False)
    assay_data = assay[["PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME"]]
    assay_data.columns = ["smiles", "activity"]
    assay_data["activity"] = assay_data["activity"].replace(
        {"Inactive": 0, "Active": 1, "Inconclusive": np.nan}
    )
    assay_data.dropna(inplace=True)

    tb = TwinBooster()
    pred, confid = tb.predict(assay_data["smiles"], description, get_confidence=True)

    assay_data["pred"] = pred
    assay_data["confidence"] = confid

    # Save results
    assay_data.to_csv(os.path.join(args.results_path, f"aid{aid}_results.csv"), index=False)

    roc, pr= get_metrics(assay_data["activity"], assay_data[f"pred"])
    baseline = np.sum(assay_data["activity"]) / len(assay_data["activity"])
    dpr  = pr_iter - baseline

    pd.DataFrame({"roc": roc, "pr": pr, f"dpr_{baseline}": dpr}).to_csv(
        os.path.join(args.results_path, f"aid{aid}_metrics.csv"), index=False
    )


if __name__ == "__main__":
    main()
