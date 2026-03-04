"""
================================================================================
MGCA-ISIC Constants
================================================================================
Centralized label definitions for ISIC-2019 skin cancer classification tasks.
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
# ANATOMICAL SITE MAPPINGS
# =============================================================================

SITE_NAMES = {
    'anterior torso': 'the anterior torso',
    'posterior torso': 'the posterior torso',
    'lateral torso': 'the lateral torso',
    'upper extremity': 'the upper extremity',
    'lower extremity': 'the lower extremity',
    'head/neck': 'the head and neck area',
    'palms/soles': 'the palms or soles',
}

# =============================================================================
# CHAOS LABELS (2 binary classification targets)
# =============================================================================

CHAOS_LABELS = ['structure_is_chaotic', 'colour_is_chaotic']

# =============================================================================
# CLUE LABELS (10 multi-label classification targets)
# =============================================================================

CLUE_LABELS = [
    'clue_1_eccentric_structureless_area',
    'clue_2_thick_lines',
    'clue_3_grey_blue_structures',
    'clue_4_black_dots_clods',
    'clue_5_lines_radial_pseudopods',
    'clue_6_white_lines',
    'clue_7_polymorphous_vessels',
    'clue_8_parallel_ridge_lines',
    'clue_9_angulated_lines',
    'clue_10_no_clues'
]

# Dermoscopic clue column to natural language description mapping
CLUE_DESCRIPTIONS = {
    'clue_1_eccentric_structureless_area': 'eccentric structureless area',
    'clue_2_thick_lines': 'thick lines',
    'clue_3_grey_blue_structures': 'grey blue structures',
    'clue_4_black_dots_clods': 'black dots and clods',
    'clue_5_lines_radial_pseudopods': 'lines radial pseudopods',
    'clue_6_white_lines': 'white lines',
    'clue_7_polymorphous_vessels': 'polymorphous vessels',
    'clue_8_parallel_ridge_lines': 'parallel ridge lines',
    'clue_9_angulated_lines': 'angulated lines',
    'clue_10_no_clues': 'no specific dermoscopic clues'
}

# Human-readable display names for clues
CLUE_NAMES = {
    'clue_1_eccentric_structureless_area': 'Eccentric Structureless Area',
    'clue_2_thick_lines': 'Thick Lines',
    'clue_3_grey_blue_structures': 'Grey Blue Structures',
    'clue_4_black_dots_clods': 'Black Dots & Clods',
    'clue_5_lines_radial_pseudopods': 'Radial Lines / Pseudopods',
    'clue_6_white_lines': 'White Lines',
    'clue_7_polymorphous_vessels': 'Polymorphous Vessels',
    'clue_8_parallel_ridge_lines': 'Parallel Ridge Lines',
    'clue_9_angulated_lines': 'Angulated Lines',
    'clue_10_no_clues': 'No Clues'
}