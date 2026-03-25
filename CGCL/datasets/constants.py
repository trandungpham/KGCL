"""
================================================================================
CGCL Dataset Constants
================================================================================
Centralized label definitions for skin cancer classification tasks.
These constants are used across the codebase for:
- Text generation from structured annotations
- Classification labels for chaos and clues tasks
- Human-readable display names
"""

# =============================================================================
# DIAGNOSIS MAPPINGS
# =============================================================================

DIAGNOSIS_NAMES = {
    'MEL': 'melanoma',
    'NV': 'melanocytic nevus',
    'BCC': 'basal cell carcinoma',
    'AK': 'actinic keratosis',
    'BKL': 'benign keratosis',
    'DF': 'dermatofibroma',
    'VASC': 'vascular lesion',
    'SCC': 'squamous cell carcinoma',
    'UNK': 'unknown lesion type'
}

# =============================================================================
# CHAOS LABELS (2 binary classification targets)
# =============================================================================

CHAOS_LABELS = ['structure_is_chaotic', 'colour_is_chaotic']

# =============================================================================
# CLUES LABELS (multiclass classification targets)
# =============================================================================

CLUES_NAMES = [
    'Eccentric Structureless Area',
    'Thick Lines',
    'Grey Blue Structures',
    'Black Dots & Clods',
    'Radial Lines / Pseudopods',
    'White Lines',
    'Polymorphous Vessels',
    'Parallel Ridge Lines',
    'Angulated Lines',
]

# =============================================================================
# DIAGNOSIS CLASSIFICATION LABELS (binary: NV vs MEL)
# =============================================================================

DIAGNOSIS_LABELS = ['NV', 'MEL']  # 0 = NV (benign), 1 = MEL (malignant)
NUM_DIAGNOSIS_CLASSES = 2

DIAGNOSIS_TO_IDX = {'NV': 0, 'MEL': 1}
IDX_TO_DIAGNOSIS = {0: 'NV', 1: 'MEL'}
