import json
from functools import partial
from pathlib import Path

from torch_geometric.datasets import QM9, MoleculeNet

from utils.aq_sol_db import AqSolDB
from utils.chargeset import Chargeset
from utils.chember import Chember, ChemberWithNegative
from utils.graphchem import GraphChem, GraphChemTxt
from utils.rbace import RBACE
from utils.transforms import (
    AllPreFilters,
    AtomicNumberDeterminationFormat,
    PreFilterCheckValidSMILES,
    PreFilterOnlyTheseAtoms,
    PreFilterRemoveLongSMILES,
    pre_filter_carbon_containing,
    transform_from_odb_to_gnn,
    transform_from_qm9_to_gnn,
)

ATOMS_ALLOWED = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 34, 35, 50, 51, 52, 53, 83}

DATASET_MAPPING = {
    "QM9": QM9,
    "chargeset": Chargeset,
    "AqSolDB": AqSolDB,
    "RBACE": RBACE,
    "ESOL": partial(MoleculeNet, name="ESOL"),
    "FreeSolv": partial(MoleculeNet, name="Freesolv"),
    "Lipophilicity": partial(MoleculeNet, name="lipo"),
    "PCBA": partial(MoleculeNet, name="PCBA"),
    "MUV": partial(MoleculeNet, name="muv"),
    "HIV": partial(MoleculeNet, name="hiv"),
    "BACE": partial(MoleculeNet, name="bace"),
    "BBBP": partial(MoleculeNet, name="bbbp"),
    "Tox21": partial(MoleculeNet, name="tox21"),
    "ToxCast": partial(MoleculeNet, name="toxcast"),
    "SIDER": partial(MoleculeNet, name="sider"),
    "ClinTox": partial(MoleculeNet, name="clintox"),
    "chember": Chember,
    "chember_with_negative": ChemberWithNegative,
    "GraphChem77M": GraphChem,
    "GraphChem10M": partial(GraphChem, size=10000000),
    "GraphChem5M": partial(GraphChem, size=5000000),
    "GraphChem1M": partial(GraphChem, size=1000000),
    "GraphChem1K": partial(GraphChem, size=1000),
    "GraphChem77M_txt": partial(
        GraphChemTxt,
        tokenizer_name="basic",
        external_name="GraphChem77M_txt",
    ),
    "GraphChem10M_txt": partial(
        GraphChemTxt,
        size=10000000,
        tokenizer_name="basic",
        external_name="GraphChem10M_txt",
    ),
    "GraphChem5M_txt": partial(
        GraphChemTxt,
        size=5000000,
        tokenizer_name="basic",
        external_name="GraphChem5M_txt",
    ),
    "GraphChem1M_txt": partial(
        GraphChemTxt,
        size=1000000,
        tokenizer_name="basic",
        external_name="GraphChem1M_txt",
    ),
    "GraphChem1K_txt": partial(
        GraphChemTxt,
        size=1000,
        tokenizer_name="basic",
        external_name="GraphChem1K_txt",
    ),
    "GraphChemH5M": partial(
        GraphChem,
        masked_fractions={"nodes": 0.25, "edge": 0},
        size=5000000,
    ),
    "GraphChemH5M_txt": partial(
        GraphChemTxt,
        masked_fractions={"nodes": 0.25, "edge": 0},
        size=5000000,
        tokenizer_name="basic",
        external_name="GraphChemH5M_txt",
    ),
    "GraphChemH1M": partial(
        GraphChem,
        masked_fractions={"nodes": 0.30, "edge": 0},
        size=1000000,
    ),
    "GraphChemR1M": partial(
        GraphChem,
        masked_fractions={"nodes": 0.2, "edge": 0},
        masking_method="recursive",
        size=1000000,
    ),
    "GraphChemR1K": partial(
        GraphChem,
        masked_fractions={"nodes": 0.2, "edge": 0},
        masking_method="recursive",
        size=1000,
        buffer_size=1000,
    ),
    "GraphChem2M": partial(GraphChem, size=2000000),
    "GraphChem2M_txt": partial(
        GraphChemTxt,
        size=2000000,
        tokenizer_name="basic",
        external_name="GraphChem2M_txt",
    ),
}

TRANSFORM_MAPPING = {
    "QM9": transform_from_qm9_to_gnn,
    "chargeset": None,
    "AqSolDB": None,
    "ESOL": transform_from_odb_to_gnn,
    "FreeSolv": transform_from_odb_to_gnn,
    "Lipophilicity": transform_from_odb_to_gnn,
    "PCBA": transform_from_odb_to_gnn,
    "MUV": transform_from_odb_to_gnn,
    "HIV": transform_from_odb_to_gnn,
    "BACE": transform_from_odb_to_gnn,
    "BBBP": transform_from_odb_to_gnn,
    "Tox21": transform_from_odb_to_gnn,
    "ToxCast": transform_from_odb_to_gnn,
    "SIDER": transform_from_odb_to_gnn,
    "ClinTox": transform_from_odb_to_gnn,
    "RBACE": transform_from_odb_to_gnn,
}

PRE_FILTER_MAPPING = {
    "QM9": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
                AtomicNumberDeterminationFormat.Z,
            ),
            PreFilterCheckValidSMILES(),
            pre_filter_carbon_containing,
        ],
    ),
    "RBACE": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "chargeset": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
            pre_filter_carbon_containing,
        ],
    ),
    "AqSolDB": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
            pre_filter_carbon_containing,
        ],
    ),
    "ESOL": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "FreeSolv": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "Lipophilicity": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "PCBA": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "MUV": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "HIV": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "BACE": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "BBBP": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "Tox21": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "ToxCast": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "SIDER": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "ClinTox": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterOnlyTheseAtoms(
                ATOMS_ALLOWED,
            ),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem77M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem10M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem5M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem1M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChemH1M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChemR1M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChemR1K": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem1K": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChemH5M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
    "GraphChem2M": AllPreFilters(
        [
            PreFilterRemoveLongSMILES(300),
            PreFilterCheckValidSMILES(),
        ],
    ),
}

with (Path(__file__).parent / "supporting" / "pcba_targets_descriptions.json").open(
    encoding="utf-8",
) as file:
    pcba_targets_descriptions = json.load(file)

with (Path(__file__).parent / "supporting" / "toxcast_targets_descriptions.json").open(
    encoding="utf-8",
) as file:
    toxcast_targets_descriptions = json.load(file)

TARGETS_DESCRIPTIONS = {
    "QM9": [
        "Dipole moment",
        "Isotropic polarizability",
        "Highest occupied molecular orbital energy",
        "Lowest unoccupied molecular orbital energy",
        "Gap between ε_HOMO and ε_LUMO",
        "Electronic spatial extent",
        "Zero point vibrational energy",
        "Internal energy at 0K",
        "Internal energy at 298.15K",
        "Enthalpy at 298.15K",
        "Free energy at 298.15K",
        "Heat capacity at 298.15K",
        "Atomization energy at 0K",
        "Atomization energy at 298.15K",
        "Atomization enthalpy at 298.15K",
        "Atomization free energy at 298.15K",
        # "Rotational constant",
        # "Rotational constant",
        # "Rotational constant",
    ],
    "chargeset": [
        "Gasteiner charge",
    ],
    "AqSolDB": [
        "Experimental aqueous solubility value (LogS)",
        "Octanol-water partition coefficient",
        "Molar refractivity",
        "Topological polar surface area",
        "Labute's Approximate Surface Area",
        "A topological complexity index of compound",
    ],
    "RBACE": [
        "pIC50",
    ],
    "ESOL": [
        "Measured log solubility in mols per litre",
    ],
    "FreeSolv": [
        "Hydratation free energy",
    ],
    "Lipophilicity": ["Octanol-water partition coefficient"],
    "PCBA": pcba_targets_descriptions,
    "MUV": [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ],
    "HIV": ["HIV active"],
    "BACE": ["Human beta-secretase inhibition activity"],
    "BBBP": ["Blood-brain barrier permeability"],
    "Tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
    "ToxCast": toxcast_targets_descriptions,
    "SIDER": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "ClinTox": ["Clinical trial toxicity", "FDA approval"],
}

TARGETS_NAMES = {
    "QM9": [
        "μ",
        "α",  # noqa:RUF001
        "ε_homo",
        "ε_lumo",
        "ε_homo - ε_lumo",
        "⟨R²⟩",
        "ZPVE",
        "U_0",
        "U_298",
        "H_298",
        "G_298",
        "C_v_298",
        "U_0_atom",
        "U_298_atom",
        "H_298_atom",
        "G_298_atom",
        # "A",
        # "B",
        # "C",
    ],
    "chargeset": [
        "Charge_G",
    ],
    "AqSolDB": [
        "Solubility",
        "MolLogP",
        "MolMR",
        "TPSA",
        "LabuteASA",
        "BertzCT",
    ],
    "ESOL": ["logS"],
    "FreeSolv": ["ΔG_hydr"],
    "Lipophilicity": ["logP"],
    "RBACE": ["pIC50"],
}

CLASSES_NAMES = {}

DATASET_OBJECTIVES = {
    "QM9": "regression",
    "chargeset": "regression",
    "AqSolDB": "regression",
    "ESOL": "regression",
    "FreeSolv": "regression",
    "Lipophilicity": "regression",
    "PCBA": "classification",
    "MUV": "classification",
    "HIV": "classification",
    "BACE": "classification",
    "BBBP": "classification",
    "Tox21": "classification",
    "ToxCast": "classification",
    "SIDER": "classification",
    "ClinTox": "classification",
    "RBACE": "regression",
}


class NoDescriptionForDatasetError(Exception):
    """Dataset provided, but description is missing."""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(f"No description for dataset {dataset_name}")


DATASET_PURPOSES = {
    "QM9": "validation",
    "chargeset": "validation",
    "AqSolDB": "validation",
    "ESOL": "validation",
    "FreeSolv": "validation",
    "Lipophilicity": "validation",
    "PCBA": "validation",
    "MUV": "validation",
    "HIV": "validation",
    "BACE": "validation",
    "BBBP": "validation",
    "Tox21": "validation",
    "ToxCast": "validation",
    "SIDER": "validation",
    "ClinTox": "validation",
    "RBACE": "validation",
    "Chember": "training",
    "ChemberWithNegative": "training",
    "GraphChem77M": "training",
    "GraphChem10M": "training",
    "GraphChem5M": "training",
    "GraphChem1M": "training",
    "GraphChem1K": "training",
    "GraphChem77M_txt": "training",
    "GraphChem10M_txt": "training",
    "GraphChem5M_txt": "training",
    "GraphChem1M_txt": "training",
    "GraphChem1K_txt": "training",
    "GraphChemH5M": "training",
    "GraphChemH5M_txt": "training",
    "GraphChem2M": "training",
    "GraphChem2M_txt": "training",
    "GraphChemH1M": "training",
    "GraphChemR1M": "training",
    "GraphChemR1K": "training",
}

IS_TORCH_GEOMETRIC_DATASET = {
    "QM9": True,
    "chargeset": True,
    "AqSolDB": True,
    "ESOL": True,
    "FreeSolv": True,
    "Lipophilicity": True,
    "PCBA": True,
    "MUV": True,
    "HIV": True,
    "BACE": True,
    "BBBP": True,
    "RBACE": True,
    "Tox21": True,
    "ToxCast": True,
    "SIDER": True,
    "ClinTox": True,
    "Chember": True,
    "ChemberWithNegative": True,
    "GraphChem77M": True,
    "GraphChem10M": True,
    "GraphChem5M": True,
    "GraphChem1M": True,
    "GraphChem1K": True,
    "GraphChem77M_txt": False,
    "GraphChem10M_txt": False,
    "GraphChem5M_txt": False,
    "GraphChem1M_txt": False,
    "GraphChem1K_txt": False,
    "GraphChemH5M": True,
    "GraphChemH5M_txt": False,
    "GraphChem2M": True,
    "GraphChem2M_txt": False,
    "GraphChemH1M": True,
    "GraphChemR1M": True,
    "GraphChemR1K": True,
}

VALIDATION_DATASETS = {
    dsname for dsname, purpose in DATASET_PURPOSES.items() if purpose == "validation"
}

for key in DATASET_MAPPING:
    if key not in TRANSFORM_MAPPING:
        TRANSFORM_MAPPING[key] = lambda x: x
    if key not in PRE_FILTER_MAPPING:
        PRE_FILTER_MAPPING[key] = lambda _: True
for key in VALIDATION_DATASETS:
    if key not in TARGETS_DESCRIPTIONS:
        raise NoDescriptionForDatasetError(key)
    if key not in TARGETS_NAMES:
        TARGETS_NAMES[key] = TARGETS_DESCRIPTIONS[key]
    if DATASET_OBJECTIVES[key] == "classification" and key not in CLASSES_NAMES:
        CLASSES_NAMES[key] = [("Negative", "Positive") for _ in TARGETS_DESCRIPTIONS[key]]


def get_filtered_targets_names(dataset_name: str, target_indices: list[int] | None) -> list[str]:
    """Return TARGETS_NAMES[dataset_name] filtered by target_indices (or unmodified if None)."""
    full = TARGETS_NAMES[dataset_name]
    if target_indices is None:
        return list(full)
    return [full[i] for i in target_indices]


def get_filtered_targets_descriptions(
    dataset_name: str,
    target_indices: list[int] | None,
) -> list[str]:
    """Return TARGETS_DESCRIPTIONS[dataset_name] filtered by target_indices (or unmodified if None)."""
    full = TARGETS_DESCRIPTIONS[dataset_name]
    if target_indices is None:
        return list(full)
    return [full[i] for i in target_indices]
