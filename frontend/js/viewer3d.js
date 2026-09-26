/**
 * AeroSculpt 3D WebGL Viewer Engine (Three.js)
 * Manages 3D scene, terrain mesh, dense point cloud, drone trajectory,
 * metric datum grid, evidence-aware shading modes, and camera controls.
 */

class AeroSculptViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    
    // Display layers
    this.terrainMesh = null;
    this.pointCloud = null;
    this.droneTrajectory = null;
    this.cameraFrustums = [];
    this.datumGrid = null;
    this.boundingBox = null;

    // Materials dictionary for dynamic mode switching
    this.materials = {
      photorealistic: null,
      evidence: null,
      elevation: null,
      wireframe: null
    };

    this.currentMode = "texture";
    this.isAutoOrbiting = false;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.init();
    this.buildScene();
    this.animate();
    this.setupResizeListener();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth - 610;
    const height = this.container.clientHeight || window.innerHeight - 60;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080b12);
    this.scene.fog = new THREE.FogExp2(0x080b12, 0.003);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.5, 3000);
    this.camera.position.set(90, 80, 110);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Prevent going below datum
    this.controls.target.set(0, 5, 0);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xd4e5ff, 0.55);
    this.scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7e6, 1.2);
    sunLight.position.set(120, 150, 80);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 10;
    sunLight.shadow.camera.far = 400;
    sunLight.shadow.camera.left = -100;
    sunLight.shadow.camera.right = 100;
    sunLight.shadow.camera.top = 100;
    sunLight.shadow.camera.bottom = -100;
    this.scene.add(sunLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x1e293b, 0.35);
    this.scene.add(hemiLight);
  }

  buildScene() {
    this.buildTerrainMesh();
    this.buildDensePointCloud();
    this.buildDroneFlightPath();
    this.buildDatumGrid();
    this.buildBoundingBox();
  }

  buildTerrainMesh() {
    // 2.5D Aerial Survey Surface Geometry
    const gridRes = 90;
    const size = 140;
    const geometry = new THREE.PlaneGeometry(size, size, gridRes, gridRes);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    const count = pos.count;
    const colorsRealistic = new Float32Array(count * 3);
    const colorsEvidence = new Float32Array(count * 3);
    const colorsElevation = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);

      // Realistic aerial topography: Bedrock peaks, plateau, coastal depression
      const d = Math.sqrt(x * x + z * z);
      let y = Math.sin(x * 0.05) * 6.0 + Math.cos(z * 0.06) * 5.0;
      y += Math.sin(x * 0.12 + z * 0.08) * 3.5;
      
      // Ridge peak
      if (x < -20) y += Math.abs(x + 20) * 0.45;
      // Plateau outpost
      if (x > 10 && x < 45 && z > -25 && z < 25) {
        y = THREE.MathUtils.lerp(y, 14.0, 0.7);
      }
      // Fjord slope
      if (z > 40) y -= (z - 40) * 0.35;

      pos.setY(i, y);

      // 1. Photorealistic colors (Bedrock, snow dusting, tundra vegetation, buildings)
      let r = 0.28, g = 0.34, b = 0.26; // Tundra earth
      if (y > 18) {
        r = 0.88; g = 0.92; b = 0.95; // Snow
      } else if (y > 10) {
        r = 0.42; g = 0.40; b = 0.38; // Bedrock
      } else if (y < 2) {
        r = 0.18; g = 0.25; b = 0.30; // Coastal edge
      }
      // Add slight road strip
      if (Math.abs(x - z * 0.5) < 2.5) {
        r = 0.32; g = 0.32; b = 0.34;
      }
      colorsRealistic[i * 3] = r;
      colorsRealistic[i * 3 + 1] = g;
      colorsRealistic[i * 3 + 2] = b;

      // 2. Evidence-Aware colors (Observed: Green, Reconstructed: Cyan, Inferred: Orange, Unknown: Grey)
      let er = 0.06, eg = 0.72, eb = 0.50; // Observed (Center track)
      if (d < 35) {
        er = 0.06; eg = 0.72; eb = 0.50; // Observed
      } else if (d < 55) {
        er = 0.01; eg = 0.52; eb = 0.78; // Reconstructed
      } else {
        er = 0.96; eg = 0.62; eb = 0.04; // Inferred (Perimeter extrapolated)
      }
      colorsEvidence[i * 3] = er;
      colorsEvidence[i * 3 + 1] = eg;
      colorsEvidence[i * 3 + 2] = eb;

      // 3. Elevation ramp
      const normY = THREE.MathUtils.clamp((y + 5) / 25, 0, 1);
      colorsElevation[i * 3] = normY;
      colorsElevation[i * 3 + 1] = 1.0 - Math.abs(normY - 0.5) * 2.0;
      colorsElevation[i * 3 + 2] = 1.0 - normY;
    }

    geometry.computeVertexNormals();
    geometry.setAttribute('color', new THREE.BufferAttribute(colorsRealistic, 3));
    this.evidenceColors = colorsEvidence;
    this.realisticColors = colorsRealistic;
    this.elevationColors = colorsElevation;

    // Materials
    this.materials.photorealistic = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.1,
      flatShading: false
    });

    this.materials.evidence = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.7,
      metalness: 0.15
    });

    this.materials.elevation = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.75
    });

    this.materials.wireframe = new THREE.MeshBasicMaterial({
      color: 0x00f2ff,
      wireframe: true,
      transparent: true,
      opacity: 0.35
    });

    this.terrainMesh = new THREE.Mesh(geometry, this.materials.photorealistic);
    this.terrainMesh.receiveShadow = true;
    this.terrainMesh.castShadow = true;
    this.scene.add(this.terrainMesh);

    // Add Outpost Architectural structures on the plateau
    this.addStructures(this.scene);
  }

  addStructures(scene) {
    const buildingMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5 });
    const roofMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.4 });

    // Outpost main shelter
    const b1 = new THREE.Mesh(new THREE.BoxGeometry(10, 6, 14), buildingMat);
    b1.position.set(25, 17, -5);
    b1.castShadow = true;
    scene.add(b1);

    const r1 = new THREE.Mesh(new THREE.ConeGeometry(8, 3, 4), roofMat);
    r1.position.set(25, 21.5, -5);
    r1.rotation.y = Math.PI / 4;
    scene.add(r1);

    // Weather radar dome
    const domeMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.2 });
    const dome = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 16), domeMat);
    dome.position.set(32, 17, 8);
    dome.castShadow = true;
    scene.add(dome);
  }

  buildDensePointCloud() {
    const numPoints = 12000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    for (let i = 0; i < numPoints; i++) {
      const rx = (Math.random() - 0.5) * 135;
      const rz = (Math.random() - 0.5) * 135;
      let ry = Math.sin(rx * 0.05) * 6.0 + Math.cos(rz * 0.06) * 5.0 + (Math.random() * 0.5);

      positions[i * 3] = rx;
      positions[i * 3 + 1] = ry;
      positions[i * 3 + 2] = rz;

      colors[i * 3] = 0.35 + Math.random() * 0.2;
      colors[i * 3 + 1] = 0.55 + Math.random() * 0.2;
      colors[i * 3 + 2] = 0.45 + Math.random() * 0.2;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const pMat = new THREE.PointsMaterial({
      size: 0.8,
      vertexColors: true,
      transparent: true,
      opacity: 0.85
    });

    this.pointCloud = new THREE.Points(geometry, pMat);
    this.scene.add(this.pointCloud);
  }

  buildDroneFlightPath() {
    // UAV Single-pass flight trajectory curve
    const waypoints = [
      new THREE.Vector3(-60, 48, -55),
      new THREE.Vector3(-35, 46, -30),
      new THREE.Vector3(-10, 47, -10),
      new THREE.Vector3(15, 45, 10),
      new THREE.Vector3(40, 44, 30),
      new THREE.Vector3(65, 46, 50)
    ];

    const curve = new THREE.CatmullRomCurve3(waypoints);
    const points = curve.getPoints(100);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f2ff,
      linewidth: 3,
      transparent: true,
      opacity: 0.95
    });

    this.droneTrajectory = new THREE.Line(geometry, lineMat);
    this.scene.add(this.droneTrajectory);

    // Camera station frustums along flight line
    const frustumMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.45
    });

    for (let t = 0; t <= 1.0; t += 0.15) {
      const pos = curve.getPoint(t);
      const pyramidGeo = new THREE.ConeGeometry(2.5, 4, 4);
      pyramidGeo.rotateX(Math.PI);
      const camMesh = new THREE.Mesh(pyramidGeo, frustumMat);
      camMesh.position.copy(pos);
      this.cameraFrustums.push(camMesh);
      this.scene.add(camMesh);
    }
  }

  buildDatumGrid() {
    this.datumGrid = new THREE.GridHelper(160, 16, 0x00f2ff, 0x1e293b);
    this.datumGrid.position.y = -6;
    this.scene.add(this.datumGrid);
  }

  buildBoundingBox() {
    const boxGeo = new THREE.BoxGeometry(140, 55, 140);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this.boundingBox = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.4 })
    );
    this.boundingBox.position.set(0, 18, 0);
    this.scene.add(this.boundingBox);
  }

  setShadingMode(mode) {
    this.currentMode = mode;
    const geom = this.terrainMesh.geometry;

    if (mode === "texture") {
      geom.setAttribute('color', new THREE.BufferAttribute(this.realisticColors, 3));
      this.terrainMesh.material = this.materials.photorealistic;
      document.getElementById('evidence-legend').style.display = 'none';
    } else if (mode === "evidence") {
      geom.setAttribute('color', new THREE.BufferAttribute(this.evidenceColors, 3));
      this.terrainMesh.material = this.materials.evidence;
      document.getElementById('evidence-legend').style.display = 'block';
    } else if (mode === "elevation") {
      geom.setAttribute('color', new THREE.BufferAttribute(this.elevationColors, 3));
      this.terrainMesh.material = this.materials.elevation;
      document.getElementById('evidence-legend').style.display = 'none';
    } else if (mode === "wireframe") {
      this.terrainMesh.material = this.materials.wireframe;
      document.getElementById('evidence-legend').style.display = 'none';
    }
  }

  setLayerVisibility(layerName, visible) {
    if (layerName === 'mesh' && this.terrainMesh) this.terrainMesh.visible = visible;
    if (layerName === 'pointcloud' && this.pointCloud) this.pointCloud.visible = visible;
    if (layerName === 'bounding-box' && this.boundingBox) this.boundingBox.visible = visible;
    if (layerName === 'drone-path') {
      if (this.droneTrajectory) this.droneTrajectory.visible = visible;
      this.cameraFrustums.forEach(f => f.visible = visible);
    }
    if (layerName === 'datum-grid' && this.datumGrid) this.datumGrid.visible = visible;
  }

  setCameraPreset(preset) {
    const duration = 1000;
    if (preset === 'nadir') {
      this.tweenCamera(new THREE.Vector3(0, 160, 0.001), new THREE.Vector3(0, 0, 0));
    } else if (preset === 'iso') {
      this.tweenCamera(new THREE.Vector3(85, 80, 100), new THREE.Vector3(0, 5, 0));
    } else if (preset === 'drone') {
      this.tweenCamera(new THREE.Vector3(-35, 46, -30), new THREE.Vector3(15, 10, 10));
    } else if (preset === 'reset') {
      this.tweenCamera(new THREE.Vector3(90, 80, 110), new THREE.Vector3(0, 5, 0));
    }
  }

  tweenCamera(targetPos, targetLookAt) {
    this.camera.position.copy(targetPos);
    this.controls.target.copy(targetLookAt);
    this.controls.update();
  }

  toggleAutoOrbit() {
    this.isAutoOrbiting = !this.isAutoOrbiting;
    this.controls.autoRotate = this.isAutoOrbiting;
    this.controls.autoRotateSpeed = 1.8;
    return this.isAutoOrbiting;
  }

  setupResizeListener() {
    window.addEventListener('resize', () => {
      const width = this.container.clientWidth;
      const height = this.container.clientHeight;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

window.AeroSculptViewer = AeroSculptViewer;
