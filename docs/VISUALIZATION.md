# Local Neuroglancer viewer

In a VS Code PowerShell terminal:

```powershell
cd <project-root>
. .\scripts\activate.ps1
vessel2graph view --config config.yaml
```

This opens the default browser and prints a localhost link. Leave the terminal running;
Ctrl+C stops the server. The server listens only on 127.0.0.1. No data are uploaded and no
pipeline stages run. A new invocation creates a new session link from the saved YAML.
The terminal command supervises a dedicated viewer process. Ctrl+C terminates that read-only
worker rather than waiting for native meshing to finish; the OS releases its file handles and
memory. A two-second termination timeout is followed by a bounded kill fallback.
Browser camera/layer changes are session-local, not automatically written back to YAML.
For a new environment install `python -m pip install -r requirements-view.txt` in addition
to the core environment/package. Neuroglancer remains an optional dependency.

## Select layers

The macaque YAML already names both dense outputs and all nine supported layer types:

```yaml
visualization:
  layers: [input_mask, branch_segmentation, caliber_segmentation,
           skeleton_by_branch, skeleton_by_caliber, skeleton_by_corrected_caliber,
           skeleton_by_radius, junctions, corrected_nodes]
  hidden_layers: [input_mask, caliber_segmentation, skeleton_by_caliber,
                  skeleton_by_corrected_caliber, skeleton_by_radius, junctions, corrected_nodes]
  branch_volume: ../runs/macaque/segmentation.h5
  caliber_volume: ../runs/macaque-caliber/segmentation.h5
  open_browser: true
  port: 0
```

Only listed layers are created. `hidden_layers` are loaded but initially invisible.
Click a layer name to toggle it; control-click to open its controls. The question-mark
button lists navigation shortcuts. The initial layout provides three slice views and a 3D
view. Meshes load on demand and can take time/memory, particularly on larger datasets.
The Python mesh generator may materialize a whole volume; lazy HDF5 slice access does not
make meshing out-of-core. This initial viewer is validated on macaque, not production volumes.
Native mesh requests are serialized and mesh simplification is disabled to avoid the problematic
OpenMesh processing path observed during interactive use. Meshes preserve the extracted surface
but have more triangles than simplified meshes. Eleven tests and a full 16-object macaque mesh
stress test pass. This does not establish production-scale performance or explain every possible
native crash. The terminal enables Python faulthandler for diagnostic output if one occurs.
The adapter implements the mesher's transpose interface by reading slabs into an in-memory
array, after checking the configured memory budget. First-time 3D mesh generation can take
tens of seconds or longer; slice views and skeletons may appear first. Struck-through layer
names mean hidden layers: click once to show. To inspect a skeleton color mode, hide the other
skeleton layers so they do not obscure it. Restart the terminal viewer after code updates;
refreshing the browser alone does not reload Python code.

Start by inspecting branch_segmentation with skeleton_by_branch. Hide them and show
caliber_segmentation with skeleton_by_corrected_caliber to inspect the caliber result.
Show corrected_nodes (magenta markers) to locate corrected branch ends, or junctions
(white markers). Annotation descriptions identify node IDs and original/assigned classes.
Avoid showing multiple opaque skeleton color modes simultaneously.

Branch and caliber colors are deterministic and shared between corresponding volumes and
skeletons. Radius colors run from blue (small) to red (large), normalized to the maximum
saved skeleton radius; this is a continuous display, not another caliber classification.
Original skeleton caliber is per branch path; corrected caliber uses per-node segmentation
labels. Skeleton paths include the shared junction endpoint to display connections. This
does not give junctions branch-ID seeds in dense segmentation.

Paths resolve relative to the dataset YAML. Null volume paths fall back to the run directory's
segmentation.h5; its attribute must match the requested branch or caliber layer. The viewer
checks volume shape, axis metadata and spacing. It cannot prove an arbitrary same-shaped
external volume has the same lineage: use matching outputs from the intended experiment.
Missing or incompatible artifacts fail without generating replacements. Graph-dependent
layers read graph.npz, branches.json and metrics.json from output.directory.

The input layer masks out unselected labels on each requested slice. HDF5 files stay open
read-only until the viewer exits. Do not recompute/replace these artifacts during a session;
stop the viewer, rerun the pipeline separately, then launch it again.

This command does not generate plots, videos, uncorrected comparison volumes, or precomputed
exports. Statistical plots remain separate future work. Rendering works locally; the accuracy
of the segmentation boundaries is for scientific inspection, not established by a viewer test.

## Dense radius heatmap

`radius_volume` is an additional supported layer. Set visualization.radius_volume to the
radius-valued HDF5 path. The macaque config points to runs/macaque-radius/segmentation.h5,
generated with configs/macaque-radius.yaml. This float32 image stores propagated skeleton radii
in nanometers, not new local EDT measurements. Background is zero and renders transparent.
The ImageLayer shader maps raw values continuously to blue/small through red/large; maxRadius
in the rendering controls adjusts the color range. Its initial maximum matches skeleton_by_radius.
This is a slice heatmap (including slice planes in the 3D panel), not a segmented surface mesh.
Radius downsampling is disabled: coarse image averaging with zero background otherwise invents
smaller radii at boundaries. Wide zoom uses full-resolution chunks and may cost more bandwidth
and GPU memory. Display pixel sampling can still hide subpixel structures; the server does not
average their radius values. No change to the stored HDF5 values is needed.
It uses no per-radius segment IDs or OpenMesh processing. The new layer starts hidden in the
main config to preserve the existing branch inspection view.

Skeleton gradients use a vec3 per-vertex attribute, node_color, and the skeleton shader
`void main() { emitRGB(node_color); }`. Dense image gradients instead read each voxel with
`toRaw(getDataValue())` and calculate a color in the fragment shader. Neither is limited to
Neuroglancer's default one-color-per-segment behavior.
# Phase-2 branch comparison

`visualization.skeleton_overlays` accepts saved physical-coordinate graphs. For example:

```yaml
visualization:
  layers: [input_mask]
  hidden_layers: [input_mask]
  skeleton_overlays:
    - name: Old by branch
      path: ../runs/old/graph.npz
      mode: branch
      visible: true
    - name: New by branch
      path: ../runs/new/graph.npz
      mode: branch
      visible: true
```

Each graph must have its own adjacent `branches.json`. Branch IDs/colors are preserved
within each run; IDs do not imply correspondence between runs. Both graphs must already
use the same physical ZYX-nanometer frame; the viewer does not register them. `mode: solid`
and `color: '#00e5ff'` are available for whole-graph overlays. Viewing never recomputes stages.
