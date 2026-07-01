import numpy as np

# Risk weights (lower = safer for surgical path to traverse)
W_MOTOR    = 0.95   # near-absolute barrier — permanent motor deficit if damaged
W_LANGUAGE = 0.95   # near-absolute barrier — permanent speech deficit if damaged
W_VISUAL   = 0.80   # high risk — serious visual field loss
W_EDEMA    = 0.50   # medium-high — inflamed, bleeds easily
W_NORMAL   = 0.10   # small traversal cost
W_TARGET   = 0.00   # tumor = destination, zero cost


def build_risk_map(masks_dict, functional_masks):
    """
    Compose a 3D risk cost map by layering tissue regions in priority order.
    Higher-priority regions overwrite lower ones (tumor target is always last = 0).
    """
    shape = masks_dict['Whole Tumor'].shape
    zeros = np.zeros(shape, dtype=np.float32)

    motor_bin    = (functional_masks.get('motor',    zeros) > 0).astype(np.float32)
    language_bin = (functional_masks.get('language', zeros) > 0).astype(np.float32)
    visual_bin   = (functional_masks.get('visual',   zeros) > 0).astype(np.float32)
    edema_bin    = (masks_dict['Edema']        > 0).astype(np.float32)
    whole_bin    = (masks_dict['Whole Tumor']  > 0).astype(np.float32)

    functional_union = np.clip(motor_bin + language_bin + visual_bin, 0, 1)
    normal_tissue    = ((functional_union == 0) & (whole_bin == 0)).astype(np.float32)

    # Build layered map: low-priority first, high-priority overwrites
    risk_map = np.zeros(shape, dtype=np.float32)
    risk_map = np.where(normal_tissue > 0, W_NORMAL,   risk_map)
    risk_map = np.where(edema_bin     > 0, W_EDEMA,    risk_map)
    risk_map = np.where(visual_bin    > 0, W_VISUAL,   risk_map)
    risk_map = np.where(language_bin  > 0, W_LANGUAGE, risk_map)
    risk_map = np.where(motor_bin     > 0, W_MOTOR,    risk_map)
    risk_map = np.where(whole_bin     > 0, W_TARGET,   risk_map)  # always last

    return risk_map
