
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print(">>> 开始 import", flush=True)
from rdkit import Chem
print(">>> rdkit OK", flush=True)
from ligand_generator import find_amino_nitrogen, find_carboxyl_carbon
print(">>> ligand_generator OK", flush=True)

test_cases = {
    "Ala":  "N[C@@H](C)C(=O)O",
    "Gly":  "NCC(=O)O",
    "Val":  "N[C@@H](C(C)C)C(=O)O",
    "Leu":  "N[C@@H](CC(C)C)C(=O)O",
    "Phe":  "N[C@@H](Cc1ccccc1)C(=O)O",
    "Trp":  "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "His":  "N[C@@H](Cc1c[nH]cn1)C(=O)O",
    "Lys":  "NCCCC[C@@H](N)C(=O)O",
    "Arg":  "NC(=N)NCCC[C@@H](N)C(=O)O",
    "Asp":  "N[C@@H](CC(=O)O)C(=O)O",
    "Glu":  "N[C@@H](CCC(=O)O)C(=O)O",
    "Ser":  "N[C@@H](CO)C(=O)O",
    "Cys":  "N[C@@H](CS)C(=O)O",
    "Pro":  "N1CCCC1C(=O)O",
    "Aib":  "CC(C)(N)C(=O)O",
    "Nle":  "CCCC[C@@H](C(=O)O)N",
}

print("=" * 70, flush=True)
for name, smi in test_cases.items():
    print(f"--- {name} ---", flush=True)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  ❌ SMILES解析失败: {smi}", flush=True)
            continue
        n_idx = find_amino_nitrogen(mol)
        c_idx, o_idx = find_carboxyl_carbon(mol)
        print(f"  N={n_idx}, C={c_idx}, O={o_idx}", flush=True)
    except Exception as e:
        import traceback
        print(f"  ❌ 异常: {e}", flush=True)
        traceback.print_exc()
print("=" * 70, flush=True)
print("DONE", flush=True)
