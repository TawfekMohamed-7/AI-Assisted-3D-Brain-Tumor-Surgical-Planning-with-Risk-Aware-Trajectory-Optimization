# AI-Assisted 3D Brain Tumor Surgical Planning with Risk-Aware Trajectory Optimization

An AI-assisted neurosurgical planning framework for **3D brain tumor analysis**, **functional brain mapping**, and **risk-aware surgical trajectory optimization** using multi-modal MRI from **BraTS 2020**. The system combines tumor segmentation, atlas-guided functional localization, 3D risk map construction, candidate path generation, and interactive visualization in a unified workflow.

---

## Key Features

- Multi-Modal MRI Processing: Supports T1, T1ce, T2, and FLAIR MRI modalities.
- Tumor Segmentation with SegResNet: Automated brain tumor analysis using a trained deep learning model.
- Atlas-Based Functional Mapping: Localizes critical regions such as motor, language, and visual cortex.
- Risk Map Generation: Builds a voxel-wise 3D surgical risk map from tumor and functional structures.
- Risk-Aware Path Planning: Generates and ranks candidate surgical trajectories using path cost and anatomical safety constraints.
- Interactive 3D Visualization: Displays brain surface, tumor structures, functional regions, entry points, and planned paths.
- FastAPI Backend: Provides API endpoints for MRI upload, processing, visualization, and downloadable outputs.
- Frontend and App Integration Ready: Structured to support web/mobile interaction with the AI pipeline.

---

## Project Architecture

The framework follows a modular medical-imaging pipeline:

1. Input Acquisition.
   - Patient MRI volumes: T1, T1ce, T2, FLAIR.
   - Brain atlas / functional region data.

2. Preprocessing.
   - Volume loading.
   - Registration / alignment.
   - Reorientation and normalization.
   - Multi-modal preparation for inference.

3. Tumor Analysis.
   - SegResNet-based segmentation.
   - Tumor mask generation.
   - Tumor subregion extraction.
   - 3D tumor reconstruction.

4. Functional Mapping.
   - Atlas registration to patient space.
   - Localization of eloquent cortex and critical structures.

5. Risk Modeling.
   - Voxel-wise risk map creation.
   - Combination of tumor, edema, white matter, and functional regions.

6. Path Planning.
   - Candidate skull entry point generation.
   - A* based trajectory planning.
   - Path scoring and ranking.

7. Visualization and Export.
   - Interactive HTML visualization.
   - VTK mesh and path export.
   - Segmentation mask and ZIP packaging.

---

## Repository Structure

> Based on the current project files and folders.

| File/Folder | Description |
| :--- | :--- |
| `app/` | Core AI and backend modules. |
| `app/main.py` | Main FastAPI application and pipeline orchestration. |
| `app/segmentation.py` | Tumor segmentation model loading and inference logic. |
| `app/atlas.py` | Atlas registration and functional cortex alignment. |
| `app/riskmap.py` | Surgical risk map generation using anatomical priorities. |
| `app/pathplanning.py` | Candidate path generation, A* search, scoring, and ranking. |
| `app/visualization.py` | 2D/3D rendering, mesh generation, and interactive HTML output. |
| `app/clip_filter.py` | MRI input validation utilities. |
| `models/` | Trained model weights and notebooks. |
| `models/SegResNet_BraTS_best.pth` | Trained SegResNet model used for segmentation. |
| `data/` | Input MRI files, uploaded volumes, or auxiliary data. |
| `outputs/` | Generated segmentation masks, meshes, visualizations, and ZIP results. |
| `requirements.txt` | Python dependencies. |
| `README.md` | Project documentation. |

If your frontend or mobile project is included in a separate folder, add it here after upload, for example:
- `frontend/` for web interface
- `mobile_app/` for Flutter application

---

## Dataset Information

The project uses the **BraTS 2020** dataset for brain tumor analysis and segmentation.

### Dataset
- Name: Brain Tumor Segmentation (BraTS) 2020.
- Modalities: T1, T1ce, T2, FLAIR.
- Format: `.nii` / `.nii.gz`.
- Task: Multi-modal brain tumor segmentation and analysis.

### MRI Modalities Used
| Modality | Purpose |
| :--- | :--- |
| `T1` | Structural anatomical information |
| `T1ce` | Highlights enhancing tumor regions |
| `T2` | Emphasizes edema and fluid abnormalities |
| `FLAIR` | Improves edema and infiltrative region visibility |

### Notes
- Each case is processed as a 3D volumetric study.
- Multi-modal MRI improves segmentation quality and planning reliability.
- Functional atlas information is integrated after MRI alignment to support surgical safety analysis.

---

## Model and Planning Details

### 1. Tumor Segmentation
The project uses a **SegResNet** model to segment tumor structures from preprocessed MRI volumes. The generated masks are used to identify tumor regions and reconstruct 3D anatomy.

### 2. Functional Brain Mapping
Atlas-based registration is used to identify critical functional regions such as:
- Motor Cortex
- Language Cortex
- Visual Cortex
- White Matter Tracts

### 3. Risk Map Creation
A voxel-wise surgical risk map is built by assigning different traversal costs to tissue classes. Functional regions receive high risk values, while the tumor target receives the lowest cost to support surgical access planning.

### 4. Surgical Path Optimization
Candidate entry points are sampled from the brain surface, then possible trajectories are generated using A* search. Paths are evaluated based on:
- Cumulative risk
- Path length
- Proximity to eloquent regions
- Smoothness

### 5. Interactive Visualization
The final output includes an interactive 3D environment showing:
- Brain surface
- Tumor subregions
- Functional areas
- Surgical target
- Ranked surgical paths

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/AI-Assisted-3D-Brain-Tumor-Surgical-Planning-with-Risk-Aware-Trajectory-Optimization.git
cd AI-Assisted-3D-Brain-Tumor-Surgical-Planning-with-Risk-Aware-Trajectory-Optimization
```

### 2. Create Environment
```bash
python -m venv venv
source venv/bin/activate
```

For Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Start the FastAPI Server
```bash
uvicorn app.main:app --reload
```

### API Workflow
The backend supports:
- Uploading MRI modalities
- Running segmentation and planning
- Monitoring job status
- Viewing visualization output
- Downloading generated files

---

## Pipeline Workflow

The system processes each case through the following steps:

1. Load patient MRI volumes (T1, T1ce, T2, FLAIR).
2. Register and preprocess the volumes.
3. Run SegResNet inference for tumor segmentation.
4. Generate tumor masks and volumetric statistics.
5. Register atlas-derived functional regions to patient space.
6. Build a voxel-wise anatomical risk map.
7. Generate brain-surface candidate entry points.
8. Plan candidate surgical trajectories with A*.
9. Score and rank trajectories based on safety and practicality.
10. Export interactive 3D visualization and downloadable artifacts.

---

## Outputs

The pipeline can generate:

- `segmentation.nii.gz`
- Tumor mesh `.vtk` files
- Surgical path `.vtk` files
- Interactive `visualization.html`
- Packaged `surgicalplanning.zip`

---

## Current System Capabilities

- Automated tumor localization from multi-modal MRI.
- Atlas-guided identification of eloquent regions.
- Risk-aware path generation.
- Interactive neurosurgical planning visualization.
- Backend-ready integration with external frontend and mobile interfaces.

---

## Limitations

- Performance depends on MRI quality and preprocessing consistency.
- Atlas registration quality may vary across cases.
- Path planning is a research-oriented support tool and not a clinical replacement.
- Further clinical validation and larger-scale testing are needed before real-world deployment.

---

## Future Work

- Improve segmentation boundary refinement.
- Add more advanced trajectory optimization methods.
- Integrate richer white matter tract constraints.
- Extend frontend and mobile interfaces.
- Support real-time or near-real-time planning.
- Validate on larger and more diverse datasets.

---

## Keywords

`Brain Tumor Segmentation` `BraTS 2020` `SegResNet` `Medical Imaging` `3D Visualization` `Surgical Planning` `Risk Map` `Trajectory Optimization` `FastAPI` `MRI Analysis`

---

## Authors

- Tawfek Mohamed Tawfek
- Adham Osama Alrifaie
- Basmala Hashim Abdelrahman
- Arwa Hisham Abdelaziz
- Mariam Salah Aldin AlSayed
- Malak Arfa Hussien

Supervisor: Prof. Dr. Hanaa Salem Marie

---

## Academic Context

This project was prepared at Delta University for Sciences and Technology, Faculty of Artificial Intelligence, as part of:

**AI317 - Work-based Professional Project in Artificial Intelligence (II)** 
