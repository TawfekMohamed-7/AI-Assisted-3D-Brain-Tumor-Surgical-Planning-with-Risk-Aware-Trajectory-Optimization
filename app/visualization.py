import io
import base64
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from skimage.measure import marching_cubes
from scipy.ndimage import binary_dilation, binary_erosion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────

def _vox_to_mm(pts_vox, affine):
    """Convert Nx3 integer voxel indices to Nx3 mm world coordinates."""
    pts_h = np.hstack([pts_vox.astype(float), np.ones((len(pts_vox), 1))])
    return (affine @ pts_h.T).T[:, :3]


def _single_vox_to_mm(vox, affine):
    """Convert a single (x,y,z) voxel tuple to mm world coordinates."""
    v = np.array([[vox[0], vox[1], vox[2], 1.0]])
    return (affine @ v.T).T[0, :3]


# ─────────────────────────────────────────────────────────────
# Mesh helpers
# ─────────────────────────────────────────────────────────────

def make_mesh_data(mask_array, affine):
    mask_dilated = binary_dilation(mask_array, iterations=1)
    if not np.any(mask_dilated):
        return None, None
    verts, faces, _, _ = marching_cubes(mask_dilated, level=0.5)
    verts_h  = np.hstack([verts, np.ones((len(verts), 1))])
    verts_mm = (affine @ verts_h.T).T[:, :3]
    return verts_mm, faces


def save_as_vtk(mask_array, affine, filename):
    verts, faces = make_mesh_data(mask_array, affine)
    if verts is None:
        return None
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces])
    mesh     = pv.PolyData(verts, faces_pv)
    mesh     = mesh.smooth(n_iter=100)
    mesh.save(filename)
    return filename


def save_path_vtk(path_coords, filename):
    if path_coords is None or len(path_coords) < 2:
        return None
    pts = np.array(path_coords, dtype=float)
    n   = len(pts)
    cells      = np.empty(n + 1, dtype=int)
    cells[0]   = n
    cells[1:]  = np.arange(n)
    mesh        = pv.PolyData()
    mesh.points = pts
    mesh.lines  = cells
    mesh.save(filename)
    return filename


# ─────────────────────────────────────────────────────────────
# Atlas overlap report (optional)
# ─────────────────────────────────────────────────────────────

def get_atlas_overlap(mask, atlas_img, labels_dict):
    atlas_data    = atlas_img.get_fdata()
    unique_labels = np.unique(atlas_data[mask > 0])
    overlaps      = []
    for label in unique_labels:
        if label == 0:
            continue
        name   = labels_dict.get(int(label), f"Unknown_{int(label)}")
        volume = int(np.sum((mask > 0) & (atlas_data == label)))
        overlaps.append({"label": name, "volume_voxels": volume})
    return overlaps


# ─────────────────────────────────────────────────────────────
# Functional region colours  (motor / language / visual)
# ─────────────────────────────────────────────────────────────

_FUNCTIONAL_CFG = {
    #  key        hex colour   opacity  legend label
    'motor':    ('#3b82f6',   0.55,    'Motor Cortex'),
    'language': ('#10b981',   0.60,    'Language Cortex'),
    'visual':   ('#a855f7',   0.55,    'Visual Cortex'),
}


# ─────────────────────────────────────────────────────────────
# 2D view helpers
# ─────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _overlay_panel(ax, bg, overlays, title):
    ax.imshow(bg, cmap='gray', origin='lower', aspect='auto')
    for item in overlays:
        if item is None:
            continue
        data, cmap, alpha = item
        if data is not None and data.max() > 0:
            ax.imshow(data, cmap=cmap, alpha=alpha, origin='lower',
                      aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, color='white', fontsize=9, pad=4)
    ax.axis('off')


def _generate_2d_views_html(masks_dict, functional_masks, t1_data):
    whole_tumor = masks_dict.get('Whole Tumor',     np.zeros_like(t1_data, dtype=np.uint8))
    et_mask     = masks_dict.get('Enhancing Tumor', np.zeros_like(t1_data, dtype=np.uint8))
    ncr_mask    = masks_dict.get('Necrotic Core',   np.zeros_like(t1_data, dtype=np.uint8))

    def _safe(key):
        """Return functional mask only if shape matches t1_data."""
        if not functional_masks:
            return None
        m = functional_masks.get(key)
        if m is None:
            return None
        if m.shape != t1_data.shape:
            print(f"[2D] shape mismatch for {key}: {m.shape} vs {t1_data.shape}")
            return None
        return m.astype(float)

    motor    = _safe('motor')
    language = _safe('language')
    visual   = _safe('visual')

    t1_norm = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min() + 1e-8)

    # Best slice: largest tumor cross-section
    z_idx = int(np.argmax(np.sum(whole_tumor, axis=(0, 1))))  # axial
    y_idx = int(np.argmax(np.sum(whole_tumor, axis=(0, 2))))  # coronal

    bg_dark = '#0d1117'

    def sz(arr):   # axial slice
        return arr[:, :, z_idx].T if arr is not None else None

    def sy(arr):   # coronal slice
        return arr[:, y_idx, :].T if arr is not None else None

    panels_ax = [
        ("Tumor Regions", [
            (sz(whole_tumor), 'Reds',   0.55),
            (sz(et_mask),     'YlOrRd', 0.50),
            (sz(ncr_mask),    'hot',    0.45),
        ]),
        ("Functional Cortex", [
            (sz(motor),    'Blues',   0.65),
            (sz(language), 'Greens',  0.65),
            (sz(visual),   'Purples', 0.65),
        ]),
        ("Full Overlay", [
            (sz(motor),       'Blues',   0.40),
            (sz(language),    'Greens',  0.40),
            (sz(visual),      'Purples', 0.40),
            (sz(whole_tumor), 'Reds',    0.55),
            (sz(et_mask),     'YlOrRd',  0.45),
        ]),
    ]

    fig_ax, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=bg_dark)
    fig_ax.patch.set_facecolor(bg_dark)
    for ax, (title, overlays) in zip(axes, panels_ax):
        ax.set_facecolor(bg_dark)
        _overlay_panel(ax, t1_norm[:, :, z_idx].T, overlays, title)
    fig_ax.suptitle(f'Axial View  (z = {z_idx})', color='white', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    axial_b64 = _fig_to_b64(fig_ax)

    panels_cor = [
        ("Tumor Regions", [
            (sy(whole_tumor), 'Reds',   0.55),
            (sy(et_mask),     'YlOrRd', 0.50),
            (sy(ncr_mask),    'hot',    0.45),
        ]),
        ("Functional Cortex", [
            (sy(motor),    'Blues',   0.65),
            (sy(language), 'Greens',  0.65),
            (sy(visual),   'Purples', 0.65),
        ]),
        ("Full Overlay", [
            (sy(motor),       'Blues',   0.40),
            (sy(language),    'Greens',  0.40),
            (sy(visual),      'Purples', 0.40),
            (sy(whole_tumor), 'Reds',    0.55),
            (sy(et_mask),     'YlOrRd',  0.45),
        ]),
    ]

    fig_cor, axes2 = plt.subplots(1, 3, figsize=(15, 5), facecolor=bg_dark)
    fig_cor.patch.set_facecolor(bg_dark)
    for ax, (title, overlays) in zip(axes2, panels_cor):
        ax.set_facecolor(bg_dark)
        _overlay_panel(ax, t1_norm[:, y_idx, :].T, overlays, title)
    fig_cor.suptitle(f'Coronal View  (y = {y_idx})', color='white', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    coronal_b64 = _fig_to_b64(fig_cor)

    func_note = (
        "Motor cortex = <span style='color:#3b82f6'>blue</span> &nbsp;|&nbsp;"
        "Language = <span style='color:#10b981'>green</span> &nbsp;|&nbsp;"
        "Visual = <span style='color:#a855f7'>purple</span> &nbsp;|&nbsp;"
        "Whole Tumor = <span style='color:#f87171'>red</span> &nbsp;|&nbsp;"
        "Enhancing = <span style='color:#fbbf24'>yellow-orange</span>"
    ) if (motor is not None or language is not None or visual is not None) else (
        "<span style='color:#f87171'>Atlas registration unavailable — "
        "functional regions not shown</span>"
    )

    html = f"""
<div class="views2d-wrap">
  <button class="views2d-toggle" onclick="
    var p=this.nextElementSibling;
    var open=p.style.display==='block';
    p.style.display=open?'none':'block';
    this.textContent=open?'▶  Show 2D Axial / Coronal Views':'▼  Hide 2D Views';
  ">▶  Show 2D Axial / Coronal Views</button>
  <div class="views2d-panel" style="display:none">
    <p class="views2d-hint">{func_note}</p>
    <h3 class="views2d-sub">Axial (z = {z_idx})</h3>
    <img src="data:image/png;base64,{axial_b64}"
         style="width:100%;border-radius:6px;">
    <h3 class="views2d-sub" style="margin-top:20px">Coronal (y = {y_idx})</h3>
    <img src="data:image/png;base64,{coronal_b64}"
         style="width:100%;border-radius:6px;">
  </div>
</div>
"""
    return html


# ─────────────────────────────────────────────────────────────
# Main HTML builder
# ─────────────────────────────────────────────────────────────

_MESH_COLORS = {
    'Whole Tumor':     'red',
    'Edema':           'orange',
    'Enhancing Tumor': 'yellow',
    'Necrotic Core':   'purple',
}

_PATH_COLORS = ['lime', 'orange']


def create_interactive_html(masks_dict, affine, output_path,
                             functional_masks=None, display_regions=None,
                             paths=None, target_vox=None,
                             t1_data=None, t1ce_data=None):
    fig = go.Figure()

    # ── 1. Brain surface ──────────────────────────────────────
    # Computed from t1ce in voxel space, then converted to mm via affine.
    # All other structures use the same affine so they align correctly.
    brain_verts = None
    if t1ce_data is not None:
        brain_mask     = t1ce_data > t1ce_data.mean() * 0.20
        brain_interior = binary_erosion(brain_mask, iterations=2)
        shell          = brain_mask & ~brain_interior
        surface_vox    = np.array(np.where(shell)).T
        if len(surface_vox) > 8000:
            surface_vox = surface_vox[
                np.random.choice(len(surface_vox), 8000, replace=False)
            ]
        surface_mm = _vox_to_mm(surface_vox, affine)
        brain_verts = surface_mm
        fig.add_trace(go.Scatter3d(
            x=surface_mm[:, 0], y=surface_mm[:, 1], z=surface_mm[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='lightgray', opacity=0.18),
            name='Brain Surface',
            legendgroup='brain',
        ))

    # ── 2. Functional regions (motor / language / visual) ─────
    # These masks are in patient voxel space (XYZ nibabel order),
    # exactly the same space as t1ce_data. We convert with the same
    # affine so they overlay correctly on the brain surface.
    # Functional cortex as surfaces instead of point clouds
    if functional_masks:
        for key,(color,opacity,label) in _FUNCTIONAL_CFG.items():

            m = functional_masks.get(key)

            if m is None or m.sum()==0:
                continue

            verts, faces = make_mesh_data(m, affine)

            if verts is None:
                continue

            fig.add_trace(go.Mesh3d(
                x=verts[:,0],
                y=verts[:,1],
                z=verts[:,2],
                i=faces[:,0],
                j=faces[:,1],
                k=faces[:,2],

                color=color,
                opacity=0.30,

                name=label,
                legendgroup='functional',
                showlegend=True,
                visible='legendonly',
            ))
    else:
        print("[3D] functional_masks is None or empty — no cortex regions plotted")

    # ── 3. Fine-grained display regions (per-gyrus) ───────────
    # Hidden by default; togglable from legend.
    if display_regions:
        for region_name, meta in display_regions.items():
            m = meta['mask']
            if m.sum() == 0:
                continue
            pts_vox = np.array(np.where(m > 0)).T
            max_pts = 500 if meta['opacity'] < 0.10 else 4000
            if len(pts_vox) > max_pts:
                pts_vox = pts_vox[
                    np.random.choice(len(pts_vox), max_pts, replace=False)
                ]
            pts_mm = _vox_to_mm(pts_vox, affine)
            fig.add_trace(go.Scatter3d(
                x=pts_mm[:, 0], y=pts_mm[:, 1], z=pts_mm[:, 2],
                mode='markers',
                marker=dict(size=2.0, color=meta['color'], opacity=meta['opacity']),
                name=region_name,
                legendgroup='atlas',
                visible='legendonly',
                showlegend=False,
                hovertemplate=f'<b>{region_name}</b><extra></extra>',
            ))

    # ── 4. Tumor meshes ───────────────────────────────────────
    ncr_mask = masks_dict.get('Necrotic Core')
    edema_mask = masks_dict.get('Edema')
    et_mask = masks_dict.get('Enhancing Tumor')

    ncr_verts, ncr_faces = make_mesh_data(ncr_mask, affine) if ncr_mask is not None else (None, None)
    edema_verts, edema_faces = make_mesh_data(edema_mask, affine) if edema_mask is not None else (None, None)
    et_verts, et_faces = make_mesh_data(et_mask, affine) if et_mask is not None else (None, None)

    # Calculate offset to move tumor outside brain
    if brain_verts is not None:
        brain_x_max = brain_verts[:, 0].max()
        tumor_ref = ncr_verts if ncr_verts is not None else (edema_verts if edema_verts is not None else et_verts)
        if tumor_ref is not None:
            TUMOR_OFFSET_X = (brain_x_max - tumor_ref[:, 0].mean()) + 40
        else:
            TUMOR_OFFSET_X = 80
    else:
        TUMOR_OFFSET_X = 80

    tumor_name_tags = {
        'Necrotic Core': 'Necrotic Core',
        'Edema': 'Edema',
        'Enhancing Tumor': 'Enhancing Tumor',
    }
    for name, mask in masks_dict.items():
        if name == 'Necrotic Core':
            verts, faces = ncr_verts, ncr_faces
        elif name == 'Edema':
            verts, faces = edema_verts, edema_faces
        elif name == 'Enhancing Tumor':
            verts, faces = et_verts, et_faces
        else:
            verts, faces = make_mesh_data(mask, affine)
        if verts is None:
            continue
        trace_name = tumor_name_tags.get(name, name)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            name=trace_name,
            color=_MESH_COLORS.get(name, 'red'),
            opacity=0.6,
        ))

    # ── 5. Surgical paths ─────────────────────────────────────
    if paths:
        for rank, (score, path) in enumerate(paths[:2]):
            arr_vox = np.array(path)
            arr_mm  = _vox_to_mm(arr_vox, affine)
            color   = _PATH_COLORS[rank]
            fig.add_trace(go.Scatter3d(
                x=arr_mm[:, 0], y=arr_mm[:, 1], z=arr_mm[:, 2],
                mode='lines',
                line=dict(color=color, width=10),
                name=f'Path #{rank + 1}  (score {score:.2f})',
                showlegend=True
            ))
            entry_mm = arr_mm[0]
            fig.add_trace(go.Scatter3d(
                x=[entry_mm[0]], y=[entry_mm[1]], z=[entry_mm[2]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='circle'),
                name=f'Entry #{rank + 1}',
                showlegend=True
            ))

    # ── 6. Surgical target ────────────────────────────────────
    if target_vox is not None:
        tgt_mm = _single_vox_to_mm(target_vox, affine)
        fig.add_trace(go.Scatter3d(
            x=[tgt_mm[0]], y=[tgt_mm[1]], z=[tgt_mm[2]],
            mode='markers',
            marker=dict(size=14, color='red', symbol='diamond'),
            name='Surgical Target',
            showlegend=True
        ))

    fig.update_layout(
        title=dict(text='Integrated 3D Surgical Planning', font=dict(color='white')),
        paper_bgcolor='#0d1117',
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='#0d1117', gridcolor='#333'),
            yaxis=dict(title='Y (mm)', backgroundcolor='#0d1117', gridcolor='#333'),
            zaxis=dict(title='Z (mm)', backgroundcolor='#0d1117', gridcolor='#333'),
            aspectmode='data',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(color='white'), 
            bgcolor='rgba(0,0,0,0.4)'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='brainPlot')

    # ── 7. 2D collapsible views ───────────────────────────────
    views_html = ''
    if t1_data is not None:
        try:
            views_html = _generate_2d_views_html(
                masks_dict, functional_masks, t1_data
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            views_html = (
                f'<p style="color:#f87171;padding:12px">'
                f'2D views error: {exc}</p>'
            )

    _write_combined_html(output_path, plotly_html, views_html, TUMOR_OFFSET_X)
    return output_path


def _write_combined_html(path, plotly_html, views_html, tumor_offset_x):
    css = """
body { margin:0; background:#0d1117; color:#e6edf3;
       font-family:'Segoe UI',sans-serif; }
.container { max-width:1400px; margin:0 auto; padding:16px; }
.views2d-wrap { margin-top:24px; }
.views2d-toggle {
  background:#161b22; border:1px solid #30363d; color:#58a6ff;
  padding:10px 20px; border-radius:8px; cursor:pointer; font-size:14px;
  transition:background 0.2s;
}
.views2d-toggle:hover { background:#21262d; }
.views2d-panel { margin-top:16px; padding:16px; background:#161b22;
                 border:1px solid #30363d; border-radius:10px; }
.views2d-hint  { font-size:13px; color:#8b949e; margin-bottom:12px; }
.views2d-sub   { font-size:14px; color:#c9d1d9; margin:0 0 8px 0; }
.viz-controls { display:flex; gap:10px; margin:8px 0 12px 0; }
"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surgical Planning Viewer</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<div class="viz-controls">
        <button id="btnTumor" onclick="toggleTumor()"
            style="background:#161b22; border:1px solid #30363d; color:#58a6ff; padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; transition:background 0.2s;">
            🟠 Hide Tumor
        </button>
</div>
{plotly_html}
{views_html}
</div>
<script>
const TUMOR_OFFSET_X = {tumor_offset_x};
let tumorVisible = true;
const tumorNames = ['Necrotic Core', 'Edema', 'Enhancing Tumor'];
function getTumorIndices() {{
    return document.getElementById('brainPlot').data
        .map((t, i) => tumorNames.includes(t.name) ? i : -1)
        .filter(i => i !== -1);
}}
function toggleTumor() {{
    const btn = document.getElementById('btnTumor');
    const indices = getTumorIndices();
    if (!indices.length) return;

    tumorVisible = !tumorVisible;
    const update = {{ visible: tumorVisible ? true : 'legendonly' }};
    Plotly.restyle('brainPlot', update, indices);
    btn.innerText = tumorVisible ? '🟠 Hide Tumor' : '🟠 Show Tumor';
}}
</script>
</body>
</html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
