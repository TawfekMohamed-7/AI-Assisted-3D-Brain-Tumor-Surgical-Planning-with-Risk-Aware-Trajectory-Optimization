import os
import numpy as np
import nibabel as nib
import ants
from nilearn import datasets

ATLAS_DATA_DIR = "data/atlas"

# ── Functional groups (for risk map / A* cost) ────────────────
REGION_INDICES = {
    'motor':    [7, 17, 26],   # Precentral, Postcentral, Supplementary Motor
    'language': [5, 6, 10],    # Broca x2, Wernicke (MTG)
    'visual':   [22, 23],      # Lateral Occipital superior + inferior
}

# ── Individual cortical regions for 3D viewer ─────────────────
# atlas_value: (hex_color, scatter_opacity, display_name)
CORTICAL_DISPLAY = {
    7:  ('#3b82f6', 0.55, 'Precentral Gyrus (Motor Cortex)'),
    17: ('#93c5fd', 0.45, 'Postcentral Gyrus (Sensory Cortex)'),
    26: ('#1d4ed8', 0.50, 'Supplementary Motor Area'),
    5:  ('#10b981', 0.55, "Broca's Area – IFG triangularis"),
    6:  ('#059669', 0.55, "Broca's Area – IFG opercularis"),
    10: ('#6ee7b7', 0.50, "Wernicke's Area (Middle Temporal Gyrus)"),
    22: ('#a855f7', 0.55, 'Lateral Occipital Cortex sup. (Visual)'),
    23: ('#7c3aed', 0.50, 'Lateral Occipital Cortex inf. (Visual)'),
    3:  ('#f59e0b', 0.28, 'Superior Frontal Gyrus'),
    4:  ('#d97706', 0.28, 'Middle Frontal Gyrus'),
    29: ('#f472b6', 0.28, 'Superior Parietal Lobule'),
    30: ('#db2777', 0.28, 'Angular Gyrus (Inferior Parietal)'),
}

# ── Subcortical structures for 3D viewer ─────────────────────
# tuple-of-atlas-values: (hex_color, scatter_opacity, display_name)
SUBCORTICAL_DISPLAY = {
    (1, 12): ('#cbd5e1', 0.06, 'Cerebral White Matter'),
    (4, 15): ('#fde047', 0.45, 'Thalamus'),
    (9, 19): ('#fb923c', 0.45, 'Hippocampus'),
}

_cache = {}


def _parse_labels(labels_list):
    """Map atlas value (1-based) → region name, regardless of nilearn version."""
    if labels_list and labels_list[0].strip().lower() in ('background', ''):
        return {i: name for i, name in enumerate(labels_list[1:], start=1)}
    return {i: name for i, name in enumerate(labels_list, start=1)}


def ensure_atlas(progress_cb=None):
    if _cache.get('ready'):
        return _cache

    # Ensure ATLAS_DATA_DIR is absolute to avoid relative path issues
    abs_atlas_dir = os.path.abspath(ATLAS_DATA_DIR)
    os.makedirs(abs_atlas_dir, exist_ok=True)

    # Check if pre-processed atlas files already exist locally
    mni_path = os.path.join(abs_atlas_dir, 'mni152_template.nii.gz')
    
    # Check functional masks
    mni_mask_paths = {}
    functional_ok = True
    for region in REGION_INDICES.keys():
        path = os.path.join(abs_atlas_dir, f'{region}_mask_mni.nii.gz')
        if not os.path.exists(path):
            functional_ok = False
        mni_mask_paths[region] = path

    # Check cortical display regions
    cort_display_paths = {}
    cortical_ok = True
    for val, (color, opacity, fallback_name) in CORTICAL_DISPLAY.items():
        path = os.path.join(abs_atlas_dir, f'cort_{val}.nii.gz')
        if not os.path.exists(path):
            cortical_ok = False
        cort_display_paths[val] = {'path': path, 'color': color, 'opacity': opacity, 'name': fallback_name}

    # Check subcortical display regions
    sub_display_paths = {}
    subcortical_ok = True
    for vals, (color, opacity, name) in SUBCORTICAL_DISPLAY.items():
        fname = f'sub_{"_".join(str(v) for v in vals)}.nii.gz'
        path  = os.path.join(abs_atlas_dir, fname)
        if not os.path.exists(path):
            subcortical_ok = False
        sub_display_paths[name] = {'path': path, 'color': color, 'opacity': opacity}

    if os.path.exists(mni_path) and functional_ok and cortical_ok and subcortical_ok:
        if progress_cb:
            progress_cb("Using pre-processed Harvard-Oxford atlas files found locally.")
        _cache.update({
            'mni_path':           mni_path,
            'mni_mask_paths':     mni_mask_paths,
            'cort_display_paths': cort_display_paths,
            'sub_display_paths':  sub_display_paths,
            'ready': True,
        })
        return _cache

    if progress_cb:
        progress_cb("Downloading Harvard-Oxford cortical atlas...")

    ho_cort = datasets.fetch_atlas_harvard_oxford(
        'cort-maxprob-thr25-1mm',
        data_dir=abs_atlas_dir,
        symmetric_split=False
    )
    cort_img = ho_cort['maps']

# compatibility with older nilearn versions
    if isinstance(cort_img, str):
        cort_img = nib.load(cort_img)
    cort_data   = cort_img.get_fdata().astype(np.int32)
    cort_labels = _parse_labels(list(ho_cort['labels']))

    # ── Functional group masks (risk map) ────────────────────
    mni_mask_paths = {}
    for region, indices in REGION_INDICES.items():
        mask = np.isin(cort_data, indices).astype(np.uint8)
        path = os.path.join(abs_atlas_dir, f'{region}_mask_mni.nii.gz')
        nib.save(nib.Nifti1Image(mask, cort_img.affine), path)
        mni_mask_paths[region] = path

    # ── Individual cortical display regions ──────────────────
    cort_display_paths = {}
    for val, (color, opacity, fallback_name) in CORTICAL_DISPLAY.items():
        name = cort_labels.get(val, fallback_name)
        mask = (cort_data == val).astype(np.uint8)
        path = os.path.join(abs_atlas_dir, f'cort_{val}.nii.gz')
        nib.save(nib.Nifti1Image(mask, cort_img.affine), path)
        cort_display_paths[val] = {'path': path, 'color': color, 'opacity': opacity, 'name': name}

    if progress_cb:
        progress_cb("Downloading Harvard-Oxford subcortical atlas...")

    ho_sub   = datasets.fetch_atlas_harvard_oxford(
        'sub-maxprob-thr25-1mm',
        data_dir=abs_atlas_dir,
        symmetric_split=False
    )
    sub_img = ho_sub['maps']

    if isinstance(sub_img, str):
        sub_img = nib.load(sub_img)
    sub_data = sub_img.get_fdata().astype(np.int32)

    # ── Subcortical display regions ───────────────────────────
    sub_display_paths = {}
    for vals, (color, opacity, name) in SUBCORTICAL_DISPLAY.items():
        mask = np.isin(sub_data, list(vals)).astype(np.uint8)
        fname = f'sub_{"_".join(str(v) for v in vals)}.nii.gz'
        path  = os.path.join(abs_atlas_dir, fname)
        nib.save(nib.Nifti1Image(mask, sub_img.affine), path)
        sub_display_paths[name] = {'path': path, 'color': color, 'opacity': opacity}

    if progress_cb:
        progress_cb("Downloading MNI152 template...")

    mni_img  = datasets.load_mni152_template(resolution=1)
    mni_path = os.path.join(abs_atlas_dir, 'mni152_template.nii.gz')
    nib.save(mni_img, mni_path)

    _cache.update({
        'mni_path':           mni_path,
        'mni_mask_paths':     mni_mask_paths,
        'cort_display_paths': cort_display_paths,
        'sub_display_paths':  sub_display_paths,
        'ready': True,
    })
    return _cache


def register_atlas_to_patient(patient_t1_path, progress_cb=None):
    """
    Register MNI atlas to patient T1 space using ANTs rigid registration.

    Returns
    -------
    functional_masks : dict  {motor|language|visual → np.uint8 array in patient voxel space}
    display_regions  : dict  {label_name → {mask, color, opacity}}
    """
    cache = ensure_atlas(progress_cb)

    if progress_cb:
        progress_cb("Registering MNI template → patient space (ANTs rigid, 2–4 min)...")

    fixed_ants  = ants.image_read(patient_t1_path)
    moving_ants = ants.image_read(cache['mni_path'])

    registration = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform='Affine',
        verbose=False,
    )
    fwdtransforms = registration['fwdtransforms']

    # Reference shape from nibabel so we can assert consistency
    ref_shape = nib.load(patient_t1_path).get_fdata().shape

    def _warp(nii_path):
        mask_ants = ants.image_read(nii_path)
        warped    = ants.apply_transforms(
            fixed=fixed_ants,
            moving=mask_ants,
            transformlist=fwdtransforms,
            interpolator='nearestNeighbor',
        )
        arr = warped.numpy()
        # ANTs and nibabel both use (x, y, z) order for NIfTI — shapes must match
        def _warp(nii_path):

            mask_ants = ants.image_read(nii_path)

            warped = ants.apply_transforms(
                fixed=fixed_ants,
                moving=mask_ants,
                transformlist=fwdtransforms,
                interpolator='nearestNeighbor'
            )

            arr = warped.numpy()

            # fix orientation mismatch
            if arr.shape != ref_shape:
                arr = np.transpose(arr,(1,0,2))

            if arr.shape != ref_shape:
                raise RuntimeError(
                    f"Shape mismatch {arr.shape} vs {ref_shape}"
                )

            return (arr>0.5).astype(np.uint8)
        return (arr > 0.5).astype(np.uint8)

    if progress_cb:
        progress_cb("Warping functional masks → patient space...")

    functional_masks = {
        region: _warp(path)
        for region, path in cache['mni_mask_paths'].items()
    }

    if progress_cb:
        progress_cb("Warping 15 labelled atlas regions → patient space...")

    display_regions = {}

    for val, meta in cache['cort_display_paths'].items():
        warped = _warp(meta['path'])
        if warped.sum() > 0:
            display_regions[meta['name']] = {
                'mask':    warped,
                'color':   meta['color'],
                'opacity': meta['opacity'],
            }

    for name, meta in cache['sub_display_paths'].items():
        warped = _warp(meta['path'])
        if warped.sum() > 0:
            display_regions[name] = {
                'mask':    warped,
                'color':   meta['color'],
                'opacity': meta['opacity'],
            }

    return functional_masks, display_regions
