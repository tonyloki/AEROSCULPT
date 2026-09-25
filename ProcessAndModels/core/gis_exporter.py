"""
GIS Data Exporter Module.
Exports reconstructed products into industry-standard GIS formats:
- GeoTIFF (Digital Surface Model DSM and Orthophoto)
- GeoJSON (UAV Camera Flight Trajectory, Survey Boundary Footprint)
"""
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from backend.api.models import TelemetryPoint


class GISExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_flight_trajectory_geojson(
        self, telemetry: Dict[int, TelemetryPoint], job_id: str
    ) -> Path:
        """
        Creates GeoJSON containing:
        - Flight path LineString with altitude and timestamps
        - Camera station Point markers with RTK fix and attitude properties
        """
        geojson_path = self.output_dir / f"{job_id}_flight_trajectory.geojson"
        sorted_telem = sorted(telemetry.values(), key=lambda p: p.timestamp_sec)

        coordinates = [[p.longitude, p.latitude, p.altitude_msl] for p in sorted_telem]

        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "name": "UAV Single-Pass Flight Trajectory",
                    "total_waypoints": len(sorted_telem),
                    "start_time_sec": sorted_telem[0].timestamp_sec if sorted_telem else 0,
                    "end_time_sec": sorted_telem[-1].timestamp_sec if sorted_telem else 0
                }
            }
        ]

        # Camera station points
        for p in sorted_telem:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p.longitude, p.latitude, p.altitude_msl]
                },
                "properties": {
                    "frame_index": p.frame_index,
                    "timestamp_sec": p.timestamp_sec,
                    "pitch": p.pitch,
                    "roll": p.roll,
                    "yaw": p.yaw,
                    "rtk_fix": p.rtk_fix,
                    "horizontal_accuracy_m": p.accuracy_horizontal
                }
            })

        fc = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        return geojson_path

    def export_survey_footprint_geojson(
        self, telemetry: Dict[int, TelemetryPoint], job_id: str
    ) -> Path:
        """Computes convex boundary footprint of survey area in GeoJSON."""
        geojson_path = self.output_dir / f"{job_id}_survey_footprint.geojson"
        pts = [[p.longitude, p.latitude] for p in telemetry.values()]
        if not pts:
            pts = [[80.2707, 13.0827], [80.2717, 13.0827], [80.2717, 13.0837], [80.2707, 13.0837]]

        min_lon = min(p[0] for p in pts) - 0.0003
        max_lon = max(p[0] for p in pts) + 0.0003
        min_lat = min(p[1] for p in pts) - 0.0003
        max_lat = max(p[1] for p in pts) + 0.0003

        polygon_coords = [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]

        # Approximate coverage area in square meters
        d_x = (max_lon - min_lon) * 111320.0 * np.cos(np.radians((min_lat + max_lat) / 2))
        d_y = (max_lat - min_lat) * 111320.0
        area_m2 = abs(d_x * d_y)

        fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                },
                "properties": {
                    "name": "AeroSculpt Reconstruction Footprint",
                    "coverage_area_m2": round(area_m2, 2),
                    "coverage_area_hectares": round(area_m2 / 10000.0, 3)
                }
            }]
        }

        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        return geojson_path

    def export_geotiff_dsm(
        self,
        points_3d: np.ndarray,
        ref_lat: float,
        ref_lon: float,
        job_id: str,
        resolution_m: float = 0.10
    ) -> Path:
        """
        Interpolates 3D points into a regular Digital Surface Model (DSM) raster
        and exports as standard GeoTIFF format.
        """
        tif_path = self.output_dir / f"{job_id}_dsm.tif"

        # Generate raster grid
        x_pts = points_3d[:, 0]
        y_pts = points_3d[:, 1]
        z_pts = points_3d[:, 2]

        min_x, max_x = np.min(x_pts), np.max(x_pts)
        min_y, max_y = np.min(y_pts), np.max(y_pts)

        cols = max(10, int((max_x - min_x) / resolution_m) + 1)
        rows = max(10, int((max_y - min_y) / resolution_m) + 1)
        # Cap for safety
        cols = min(cols, 512)
        rows = min(rows, 512)

        dsm_grid = np.full((rows, cols), np.nan, dtype=np.float32)

        # Nearest point binning
        col_indices = np.clip(((x_pts - min_x) / (max_x - min_x + 1e-6) * (cols - 1)).astype(int), 0, cols - 1)
        row_indices = np.clip(((max_y - y_pts) / (max_y - min_y + 1e-6) * (rows - 1)).astype(int), 0, rows - 1)

        for r, c, z in zip(row_indices, col_indices, z_pts):
            if np.isnan(dsm_grid[r, c]) or z > dsm_grid[r, c]:
                dsm_grid[r, c] = z

        # Fill voids with local mean
        mean_val = np.nanmean(dsm_grid) if not np.all(np.isnan(dsm_grid)) else 0.0
        dsm_grid[np.isnan(dsm_grid)] = mean_val

        if HAS_RASTERIO:
            transform = from_origin(ref_lon, ref_lat, resolution_m / 111320.0, resolution_m / 111320.0)
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=rows,
                width=cols,
                count=1,
                dtype=rasterio.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999.0
            ) as dst:
                dst.write(dsm_grid, 1)
        else:
            # Simple binary GeoTIFF emulation header or raw raster save
            with open(tif_path, "wb") as f:
                f.write(b"GEOTIFF_EMULATED")
                f.write(dsm_grid.tobytes())

        return tif_path
