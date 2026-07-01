# Brain Tumor Segmentation Backend

This is a FastAPI backend adapted from your brain tumor segmentation notebook. It is designed to be integrated with a React frontend.

## Features
- **Automatic Registration**: Aligns T1, T2, and FLAIR images to the T1ce image using ANTs.
- **Segmentation**: Uses your 3D SegResNet model for tumor segmentation.
- **3D Visualization**: Generates an interactive Plotly HTML view of the tumor regions.
- **File Export**: Provides `.nii.gz` and `.vtk` files for download, compatible with 3D Slicer.
- **Local Optimization**: Designed to run on a local laptop (single patient at a time).

## Setup Instructions (Windows)

1. **Install Python**: Ensure you have Python 3.9+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For ANTs on Windows, you might need to install `antspyx` via pre-built wheels if the standard pip install fails.*
3. **Model File**:
   Place your model file `SegResNet_BraTS_best.pth` into the `models/` directory.
4. **Run the Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints
- `POST /segment`: Upload 4 MRI files (t1, t1ce, t2, flair) to start processing.
- `GET /view/{job_id}`: Returns the interactive 3D HTML visualization.
- `GET /download/{job_id}/{filename}`: Download the generated `.nii.gz` or `.vtk` files.

## Integration with React
In your React frontend, you can use an `<iframe>` to display the visualization:
```jsx
<iframe src={`http://localhost:8000/view/${jobId}`} width="100%" height="600px" />
```
And provide download buttons using the links returned by the `/segment` endpoint.
