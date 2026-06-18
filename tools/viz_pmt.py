"""Standalone visualizer for extracted PMT component meshes.

4 views (2x2 grid):
  [Barrel segment] [Barrel components]
  [LUX bottom]     [ETEL top]

Usage:
    python tools/viz_pmt.py
    # Open http://localhost:8002
"""

import xml.etree.ElementTree as ET
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse
import json

MESH_DIR = Path("pmt_meshes")

# Panel definitions: (label, [(gdml_filename, component_label, color, is_togglable, is_wireframe), ...])
# Togglable components get checkboxes; non-togglable ones are always shown.
PANELS = [
    (
        '10" barrel PMT',
        [
            ("pmt_10inch_body.gdml",
             "PMT body", "#ff8844", True, False),
            ("pmt_10inch_hardware.gdml",
             "Hardware", "#44aaff", True, False),
        ],
    ),
    (
        '8" barrel PMT',
        [
            ("pmt_8inch_body.gdml",
             "PMT glass", "#44aaff", True, False),
            ("pmt_8inch_hardware.gdml",
             "Hardware", "#66cc66", True, False),
        ],
    ),
    (
        "LUX bottom",
        [
            ("pmt_lux_bottom.gdml", "LUX bottom (12 inst)", "#ff66aa", False, False),
        ],
    ),
    (
        "ETEL top",
        [
            ("pmt_etel_top.gdml", "ETEL top (12 inst)", "#aa66ff", False, False),
        ],
    ),
]

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  body { margin: 0; overflow: hidden; font-family: sans-serif; background: #1a1a2e; }
  #container { display: grid; grid-template-columns: 1fr 1fr;
               grid-template-rows: 1fr 1fr; width: 100vw; height: 100vh; }
  .panel { position: relative; overflow: hidden; border: 1px solid #333; }
  .panel canvas { display: block; width: 100% !important; height: 100% !important; }
  .label { position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
           color: #ddd; font-size: 13px; background: rgba(0,0,0,0.7);
           padding: 3px 12px; border-radius: 5px; pointer-events: none; z-index: 10;
           white-space: nowrap; }
  .controls { position: absolute; bottom: 8px; left: 8px;
              color: #aaa; font-size: 11px; z-index: 10; }
  .controls label { display: block; margin: 2px 0; cursor: pointer; user-select: none; }
  .controls input { vertical-align: middle; margin-right: 4px; }
  .stats { position: absolute; bottom: 8px; right: 8px;
           color: #888; font-size: 10px; background: rgba(0,0,0,0.5);
           padding: 2px 8px; border-radius: 3px; pointer-events: none; z-index: 10; }
</style>
</head><body>
<div id="container"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const PANEL_SPECS = %SPECS%;

async function fetchBuf(url) {
  const resp = await fetch(url);
  return new Float32Array(await resp.arrayBuffer());
}

function initPanel(panelIdx) {
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.id = 'panel' + panelIdx;
  document.getElementById('container').appendChild(panel);

  const spec = PANEL_SPECS[panelIdx];
  const labelDiv = document.createElement('div');
  labelDiv.className = 'label';
  labelDiv.textContent = spec.label;
  panel.appendChild(labelDiv);

  const statsDiv = document.createElement('div');
  statsDiv.className = 'stats';
  statsDiv.id = 'stats' + panelIdx;
  panel.appendChild(statsDiv);

  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'controls';
  controlsDiv.id = 'controls' + panelIdx;
  panel.appendChild(controlsDiv);

  const rect = panel.getBoundingClientRect();
  const W = rect.width || window.innerWidth / 2;
  const H = rect.height || window.innerHeight / 2;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(35, W / H, 1, 5000);
  camera.up.set(0, 0, 1);
  camera.position.set(600, -400, 500);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  panel.insertBefore(renderer.domElement, panel.firstChild);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  const dir1 = new THREE.DirectionalLight(0xffffff, 1.2);
  dir1.position.set(1, 2, 1);
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dir2.position.set(-1, -1, 0.5);
  scene.add(dir2);

  const axes = new THREE.AxesHelper(400);
  scene.add(axes);

  const grid = new THREE.GridHelper(800, 16, 0x666666, 0x444444);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  return { scene, camera, renderer, controls, panel, controlsDiv };
}

async function main() {
  // Fetch all mesh data
  const allUrls = [];
  for (const spec of PANEL_SPECS) {
    for (const comp of spec.components) {
      allUrls.push('/api/mesh/' + comp.gdml);
    }
  }
  const allBufs = await Promise.all(allUrls.map(fetchBuf));

  // Build lookup: gdml filename -> flat float32 array
  const meshCache = {};
  let idx = 0;
  for (const spec of PANEL_SPECS) {
    for (const comp of spec.components) {
      meshCache[comp.gdml] = allBufs[idx++];
    }
  }

  const panels = [];

  // Init each panel
  for (let i = 0; i < PANEL_SPECS.length; i++) {
    const spec = PANEL_SPECS[i];
    const p = initPanel(i);
    panels.push(p);

    // Build component meshes
    const comps = [];
    for (const comp of spec.components) {
      const flat = meshCache[comp.gdml];
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      geo.computeVertexNormals();
      const mat = new THREE.MeshPhysicalMaterial({
        color: comp.color,
        transparent: !comp.wireframe,
        opacity: comp.wireframe ? 1.0 : 0.8,
        roughness: comp.wireframe ? 1.0 : 0.3,
        metalness: 0.0,
        side: THREE.DoubleSide,
        wireframe: !!comp.wireframe,
      });
      const mesh = new THREE.Mesh(geo, mat);
      p.scene.add(mesh);
      comps.push({ mesh, label: comp.label, togglable: comp.togglable });
    }

    // Build controls (checkboxes for togglable components only)
    for (const comp of comps) {
      if (!comp.togglable) continue;
      const labelEl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => { comp.mesh.visible = cb.checked; });
      labelEl.appendChild(cb);
      labelEl.appendChild(document.createTextNode(' ' + comp.label));
      p.controlsDiv.appendChild(labelEl);
    }

    // Stats
    document.getElementById('stats' + i).textContent =
      comps.map(c => {
        const n = c.mesh.geometry.attributes.position.count;
        return `${(n / 3).toLocaleString()} tris`;
      }).join(' | ');
  }

  function resize() {
    for (let i = 0; i < PANEL_SPECS.length; i++) {
      const p = panels[i];
      if (!p) continue;
      const rect = p.panel.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      if (w === 0 || h === 0) continue;
      p.camera.aspect = w / h;
      p.camera.updateProjectionMatrix();
      p.renderer.setSize(w, h);
    }
  }
  window.addEventListener('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    for (const p of panels) {
      p.controls.update();
      p.renderer.render(p.scene, p.camera);
    }
  }
  animate();
}

main();
</script>
</body></html>"""


def parse_gdml_flattened(path: Path, recenter=True):
    """Parse GDML and return flattened float32 array (9 floats per triangle)."""
    tree = ET.parse(path)
    root = tree.getroot()
    positions = root.findall(".//position")
    verts = {
        p.attrib["name"]: (float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
        for p in positions
    }
    triangles = root.findall(".//triangular")
    out = []
    for tri in triangles:
        for key in ("vertex1", "vertex2", "vertex3"):
            out.extend(verts[tri.attrib[key]])
    arr = np.array(out, dtype=np.float32).reshape(-1, 3)
    if recenter:
        centroid = arr.mean(axis=0)
        arr -= centroid
    return arr.tobytes()


# Verify all files exist and pre-parse
# body/hardware files are pre-centered at origin; LUX/ETEL need recentering
RECENTER_FILES = {"pmt_lux_bottom.gdml", "pmt_etel_top.gdml"}
MESH_CACHE = {}
for label, components in PANELS:
    for gdml_name, comp_label, color, togglable, wireframe in components:
        path = MESH_DIR / gdml_name
        if not path.exists():
            print(f"WARNING: {path} not found!")
            continue
        if gdml_name not in MESH_CACHE:
            data = parse_gdml_flattened(path, recenter=(gdml_name in RECENTER_FILES))
            MESH_CACHE[gdml_name] = data
            n_tri = len(data) // 36
            print(f"  {gdml_name}: {n_tri} triangles")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            # Build spec JSON
            spec_list = []
            for label, components in PANELS:
                comp_list = []
                for gdml_name, comp_label, color, togglable, wireframe in components:
                    comp_list.append({
                        "gdml": gdml_name,
                        "label": comp_label,
                        "color": color,
                        "togglable": togglable,
                        "wireframe": wireframe,
                    })
                spec_list.append({"label": label, "components": comp_list})

            html = HTML.replace("%SPECS%", json.dumps(spec_list))
            self.wfile.write(html.encode())

        elif parsed.path.startswith("/api/mesh/"):
            gdml_name = parsed.path[len("/api/mesh/"):]
            if gdml_name in MESH_CACHE:
                data = MESH_CACHE[gdml_name]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[PMT_VIZ] {args[0]} {args[1]} {args[2]}")


def main():
    print(f"Loaded {len(MESH_CACHE)} meshes:")
    for name, data in MESH_CACHE.items():
        print(f"  {name}: {len(data) // 36} triangles")

    port = 8002
    server = HTTPServer(("", port), Handler)
    print(f"\nPMT visualizer at http://localhost:{port}")
    print('  4 views: 10" barrel | 8" barrel | LUX bottom | ETEL top')
    print("  Meshes are centered at origin for easier viewing.")
    print("  Use OrbitControls to pan/zoom/rotate.")
    print("\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
