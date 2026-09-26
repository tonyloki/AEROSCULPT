/**
 * AeroSculpt 3D Spatial Measurement Tool
 * Allows interactive raycasted 2-point distance, elevation relief, and slope angle inspection.
 */

class AeroSculptMeasurementTool {
  constructor(viewer) {
    this.viewer = viewer;
    this.active = false;
    this.points = [];
    this.markers = [];
    this.measureLine = null;

    this.banner = document.getElementById('measure-banner');
    this.distVal = document.getElementById('m-dist-val');
    this.horizVal = document.getElementById('m-horiz-val');
    this.elevVal = document.getElementById('m-elev-val');
    this.slopeVal = document.getElementById('m-slope-val');
    this.toggleBtn = document.getElementById('btn-toggle-measure');
    this.clearBtn = document.getElementById('btn-clear-measure');
    this.closeBtn = document.getElementById('btn-close-measure');

    this.bindEvents();
  }

  bindEvents() {
    this.toggleBtn.addEventListener('click', () => this.toggleActivation());
    this.clearBtn.addEventListener('click', () => this.clearMeasurement());
    this.closeBtn.addEventListener('click', () => {
      this.banner.style.display = 'none';
      this.clearMeasurement();
    });

    const dom = this.viewer.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => this.onPointerDown(e));
  }

  toggleActivation() {
    this.active = !this.active;
    if (this.active) {
      this.toggleBtn.style.background = 'rgba(16, 185, 129, 0.25)';
      this.toggleBtn.style.borderColor = '#10b981';
      this.toggleBtn.innerHTML = '&#10003; Click 2 Points on Terrain';
      this.viewer.renderer.domElement.style.cursor = 'crosshair';
    } else {
      this.toggleBtn.style.background = 'rgba(0, 242, 255, 0.1)';
      this.toggleBtn.style.borderColor = '#00f2ff';
      this.toggleBtn.innerHTML = '<span class="tool-icon">&#128207;</span> Activate 2-Point Measure';
      this.viewer.renderer.domElement.style.cursor = 'default';
    }
  }

  onPointerDown(event) {
    if (!this.active) return;
    // Only primary left button
    if (event.button !== 0) return;

    const rect = this.viewer.renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.viewer.raycaster.setFromCamera(new THREE.Vector2(x, y), this.viewer.camera);
    const intersects = this.viewer.raycaster.intersectObject(this.viewer.terrainMesh);

    if (intersects.length > 0) {
      const hitPoint = intersects[0].point;
      this.addPoint(hitPoint);
    }
  }

  addPoint(vec3) {
    if (this.points.length >= 2) {
      this.clearMeasurement();
    }

    this.points.push(vec3);
    const marker = this.createMarker(vec3, this.points.length === 1 ? 0x00f2ff : 0x10b981);
    this.markers.push(marker);
    this.viewer.scene.add(marker);

    if (this.points.length === 2) {
      this.renderMeasurementLine();
      this.computeAndDisplayMetrics();
      this.toggleActivation(); // Deactivate tool once pair selected
    }
  }

  createMarker(position, color) {
    const group = new THREE.Group();
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 16, 16),
      new THREE.MeshBasicMaterial({ color: color })
    );
    sphere.position.copy(position);

    const pin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 3.0),
      new THREE.MeshBasicMaterial({ color: color })
    );
    pin.position.set(position.x, position.y + 1.5, position.z);

    group.add(sphere);
    group.add(pin);
    return group;
  }

  renderMeasurementLine() {
    const geometry = new THREE.BufferGeometry().setFromPoints(this.points);
    const material = new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 1.5,
      gapSize: 0.8,
      linewidth: 2
    });
    this.measureLine = new THREE.Line(geometry, material);
    this.measureLine.computeLineDistances();
    this.viewer.scene.add(this.measureLine);
  }

  computeAndDisplayMetrics() {
    const p1 = this.points[0];
    const p2 = this.points[1];

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dz = p2.z - p1.z;

    const euclidean = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const horizontal = Math.sqrt(dx * dx + dz * dz);
    const vertical = Math.abs(dy);
    const slopeDeg = (Math.atan2(vertical, horizontal + 1e-6) * 180) / Math.PI;

    this.distVal.textContent = `${euclidean.toFixed(2)} m`;
    this.horizVal.textContent = `${horizontal.toFixed(2)} m`;
    this.elevVal.textContent = `${dy >= 0 ? '+' : ''}${dy.toFixed(2)} m`;
    this.slopeVal.textContent = `${slopeDeg.toFixed(1)}°`;

    this.banner.style.display = 'flex';
  }

  clearMeasurement() {
    this.points = [];
    this.markers.forEach(m => this.viewer.scene.remove(m));
    this.markers = [];
    if (this.measureLine) {
      this.viewer.scene.remove(this.measureLine);
      this.measureLine = null;
    }
    this.banner.style.display = 'none';
  }
}

window.AeroSculptMeasurementTool = AeroSculptMeasurementTool;
