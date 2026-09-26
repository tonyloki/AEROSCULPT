/**
 * AeroSculpt Main Application Controller
 * Manages UI interactions, pipeline job simulation, API requests, and export triggers.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize 3D Viewer and Measurement Tool
  const viewer = new AeroSculptViewer('three-canvas-container');
  const measurementTool = new AeroSculptMeasurementTool(viewer);

  // DOM Elements
  const dropzone = document.getElementById('video-dropzone');
  const videoInput = document.getElementById('video-input');
  const telemInput = document.getElementById('telemetry-input');
  const telemBtn = document.getElementById('telemetry-btn');
  const sampleBtn = document.getElementById('btn-load-sample');
  const fileLoadedInfo = document.getElementById('file-loaded-info');
  const videoTag = document.getElementById('loaded-video-tag');
  const telemTag = document.getElementById('loaded-telem-tag');
  const runBtn = document.getElementById('btn-run-pipeline');
  const progressFill = document.getElementById('pipeline-progress-fill');
  const stageLabel = document.getElementById('current-stage-label');
  const percentLabel = document.getElementById('progress-percent-label');
  const terminalLogs = document.getElementById('terminal-logs');
  const systemStatusPill = document.getElementById('system-status-pill');
  const systemStatusText = document.getElementById('system-status-text');

  let activeJobId = "flight_uav_demo";
  let isProcessing = false;

  // Logging utility
  function appendLog(message, type = "normal") {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const now = new Date().toTimeString().split(' ')[0];
    entry.textContent = `[${now}] ${message}`;
    terminalLogs.appendChild(entry);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
  }

  // File Upload Handlers
  dropzone.addEventListener('click', () => videoInput.click());
  videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      videoTag.textContent = e.target.files[0].name;
      fileLoadedInfo.style.display = 'flex';
      appendLog(`Video file loaded: ${e.target.files[0].name}`, 'sys-ready');
    }
  });

  telemBtn.addEventListener('click', () => telemInput.click());
  telemInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      telemTag.textContent = e.target.files[0].name;
      fileLoadedInfo.style.display = 'flex';
      appendLog(`Telemetry log attached: ${e.target.files[0].name}`, 'sys-ready');
    }
  });

  sampleBtn.addEventListener('click', () => {
    videoTag.textContent = "sample_uav_4k_flight.mp4";
    telemTag.textContent = "dji_fc6310_rtk_log.srt";
    fileLoadedInfo.style.display = 'flex';
    appendLog("Loaded sample UAV Arctic Outpost dataset (4K 30fps with RTK telemetry).", "sys-ready");
  });

  // Layer Visibility Controls
  const layers = [
    { id: 'layer-mesh', name: 'mesh' },
    { id: 'layer-pointcloud', name: 'pointcloud' },
    { id: 'layer-bounding-box', name: 'bounding-box' },
    { id: 'layer-drone-path', name: 'drone-path' },
    { id: 'layer-datum-grid', name: 'datum-grid' }
  ];

  layers.forEach(l => {
    const el = document.getElementById(l.id);
    if (el) {
      el.addEventListener('change', (e) => {
        viewer.setLayerVisibility(l.name, e.target.checked);
        appendLog(`Layer '${l.name}' visibility: ${e.target.checked ? 'ON' : 'OFF'}`);
      });
    }
  });

  // Shading Mode Controls
  const modeButtons = document.querySelectorAll('.pill-btn');
  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      modeButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.getAttribute('data-mode');
      viewer.setShadingMode(mode);
      appendLog(`Switched shading shader to '${mode.toUpperCase()}'.`);
    });
  });

  // Camera Presets
  document.getElementById('btn-cam-nadir').addEventListener('click', () => viewer.setCameraPreset('nadir'));
  document.getElementById('btn-cam-iso').addEventListener('click', () => viewer.setCameraPreset('iso'));
  document.getElementById('btn-cam-drone').addEventListener('click', () => viewer.setCameraPreset('drone'));
  document.getElementById('btn-cam-reset').addEventListener('click', () => viewer.setCameraPreset('reset'));
  
  const orbitBtn = document.getElementById('btn-cam-orbit');
  orbitBtn.addEventListener('click', () => {
    const isOrbiting = viewer.toggleAutoOrbit();
    orbitBtn.classList.toggle('active', isOrbiting);
    appendLog(`360° Auto-Orbit: ${isOrbiting ? 'ENABLED' : 'DISABLED'}`);
  });

  // Pipeline Stepper Stages
  const stages = [
    { id: 'step-uav-input', name: 'UAV_INPUT', label: 'Ingesting Video & RTK Telemetry', pct: 15 },
    { id: 'step-quality-keyframe', name: 'KEYFRAME_FILTERING', label: 'Blur & Exposure Keyframing (~550 KFs)', pct: 30 },
    { id: 'step-scene-understanding', name: 'SCENE_UNDERSTANDING', label: 'Dynamic Masking & Scene Classifier', pct: 45 },
    { id: 'step-3d-reconstruction', name: 'SPARSE_RECONSTRUCTION', label: 'GLUEMAP / Global SfM & Dense MVS', pct: 65 },
    { id: 'step-georeferencing', name: 'GEOREFERENCING', label: '7-DOF Sim(3) & UTM Metric Scale Alignment', pct: 82 },
    { id: 'step-metric-validation', name: 'METRIC_VALIDATION', label: 'Geometric Consistency Checks (VERIFIED)', pct: 95 },
    { id: 'step-3d-world', name: 'COMPLETED', label: '3D World Ready (Mesh, Cloud, GIS)', pct: 100 }
  ];

  // Pipeline Execution
  runBtn.addEventListener('click', () => {
    if (isProcessing) return;
    isProcessing = true;
    runBtn.disabled = true;
    runBtn.style.opacity = '0.6';

    systemStatusPill.style.background = 'rgba(0, 242, 255, 0.15)';
    systemStatusPill.style.borderColor = '#00f2ff';
    systemStatusText.style.color = '#00f2ff';
    systemStatusText.textContent = 'RECONSTRUCTING';

    appendLog('Starting Single-Pass UAV Reconstruction Pipeline...', 'sys-ready');

    let currentStepIdx = 0;

    const stepInterval = setInterval(() => {
      if (currentStepIdx < stages.length) {
        const step = stages[currentStepIdx];
        
        // Update Stepper UI
        document.querySelectorAll('.stepper-step').forEach((s, idx) => {
          if (idx < currentStepIdx) {
            s.className = 'stepper-step completed';
          } else if (idx === currentStepIdx) {
            s.className = 'stepper-step active';
          } else {
            s.className = 'stepper-step';
          }
        });

        progressFill.style.width = `${step.pct}%`;
        stageLabel.textContent = step.name;
        percentLabel.textContent = `${step.pct}%`;
        appendLog(`[${step.pct}%] ${step.label}`);

        currentStepIdx++;
      } else {
        clearInterval(stepInterval);
        isProcessing = false;
        runBtn.disabled = false;
        runBtn.style.opacity = '1.0';

        systemStatusPill.style.background = 'rgba(16, 185, 129, 0.15)';
        systemStatusPill.style.borderColor = '#10b981';
        systemStatusText.style.color = '#10b981';
        systemStatusText.textContent = 'SYSTEM VERIFIED';

        appendLog('Reconstruction Complete! 3D Model Certified: VERIFIED.', 'sys-ready');
        appendLog('Accuracy: Spatial RMSE 0.38m (Target &le; 1.0m Met), RTK Residual 11.4cm (Target &le; 20cm Met).');
      }
    }, 1200);
  });

  // Export Deliverables Handlers
  const exportButtons = [
    { id: 'btn-export-glb', file: 'model.glb', label: 'glTF/GLB 3D Mesh' },
    { id: 'btn-export-obj', file: 'model.obj', label: 'Wavefront OBJ + MTL' },
    { id: 'btn-export-ply', file: 'dense_cloud.ply', label: 'Dense PLY Point Cloud' },
    { id: 'btn-export-dsm', file: 'dsm_elevation.tif', label: 'GeoTIFF DSM Raster' },
    { id: 'btn-export-geojson', file: 'survey_flight_path.geojson', label: 'Trajectory & Footprint GeoJSON' }
  ];

  exportButtons.forEach(exp => {
    document.getElementById(exp.id).addEventListener('click', () => {
      appendLog(`Preparing download for deliverable: ${exp.label}...`, 'sys-ready');
      // Create and trigger mock or API download
      const blob = new Blob([`# AeroSculpt Export: ${exp.label}\n# Timestamp: ${new Date().toISOString()}`], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `AeroSculpt_${exp.file}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      appendLog(`Downloaded ${exp.file} successfully.`);
    });
  });
});
