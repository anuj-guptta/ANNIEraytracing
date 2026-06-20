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

# Panel definitions: (label, rotatable, [(gdml_filename, component_label, color,
#                                        is_togglable, is_wireframe), ...])
# Togglable components get checkboxes; non-togglable ones are always shown.
PANELS = [
    (
        '10" barrel PMT', True,
        [
            ("pmt_10inch_body.gdml",
             "PMT body", "#ff8844", True, False),
            ("pmt_10inch_hardware.gdml",
             "Hardware", "#44aaff", True, False),
        ],
    ),
    (
        '8" barrel PMT', False,
        [
            ("pmt_8inch_body.gdml",
             "PMT glass", "#44aaff", True, False),
            ("pmt_8inch_hardware.gdml",
             "Hardware", "#66cc66", True, False),
        ],
    ),
    (
        "LUX bottom", False,
        [
            ("pmt_lux_bottom.gdml", "LUX bottom", "#ff66aa", False, False),
        ],
    ),
    (
        "ETEL top", False,
        [
            ("pmt_etel_top.gdml", "ETEL top", "#aa66ff", False, False),
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

  // Axes
  const axes = new THREE.AxesHelper(400);
  scene.add(axes);

  // Origin marker
  const dotGeo = new THREE.SphereGeometry(6, 12, 8);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.position.set(0, 0, 0);
  scene.add(dot);

  // Axis labels (only on first panel)
  if (panelIdx === 0) {
    const axes = [
      { dir: [1,0,0], label: '+X', color: 0xff4444, pos: [260,0,0] },
      { dir: [0,1,0], label: '+Y', color: 0x44ff44, pos: [0,260,0] },
      { dir: [0,0,1], label: '+Z', color: 0x4444ff, pos: [0,0,260] },
      { dir: [-1,0,0], label: '-X', color: 0x884444, pos: [-260,0,0] },
      { dir: [0,-1,0], label: '-Y', color: 0x448844, pos: [0,-260,0] },
      { dir: [0,0,-1], label: '-Z', color: 0x444488, pos: [0,0,-260] },
    ];
    for (const a of axes) {
      const v = new THREE.Vector3(a.dir[0], a.dir[1], a.dir[2]);
      const arrow = new THREE.ArrowHelper(v, new THREE.Vector3(0,0,0), 200, a.color, 30, 15);
      scene.add(arrow);
      const c = document.createElement('canvas');
      c.width = 80; c.height = 40;
      const ctx = c.getContext('2d');
      ctx.font = 'Bold 24px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = '#' + a.color.toString(16).padStart(6,'0');
      ctx.fillText(a.label, 40, 20);
      const tex = new THREE.CanvasTexture(c);
      tex.needsUpdate = true;
      const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(a.pos[0], a.pos[1], a.pos[2]);
      sprite.scale.set(80, 40, 1);
      scene.add(sprite);
    }
  }

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

    // Build component meshes — wrapped in a group for rotation
    const comps = [];
    const group = new THREE.Group();
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
      group.add(mesh);
      comps.push({ mesh, label: comp.label, togglable: comp.togglable });
    }
    p.scene.add(group);

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

    // Rotation sliders (for rotatable panels): Y (roll) and Z (tilt)
    if (spec.rotatable) {
      for (const axis of ['Y', 'Z']) {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:4px;margin:2px 0;';

        const label = document.createElement('span');
        label.textContent = axis + ':';
        label.style.cssText = 'width:16px;font-size:11px;color:#aaa;';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = -180;
        slider.max = 180;
        slider.step = 0.5;
        slider.value = 0;
        slider.style.cssText = 'flex:1;min-width:100px;height:16px;';

        const numInput = document.createElement('input');
        numInput.type = 'number';
        numInput.min = -180;
        numInput.max = 180;
        numInput.step = 0.5;
        numInput.value = '0';
        numInput.style.cssText = 'width:52px;font-size:11px;text-align:center;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;';

        const update = (deg) => {
          deg = Math.max(-180, Math.min(180, deg));
          group.rotation[axis.toLowerCase()] = deg * Math.PI / 180;
          slider.value = deg;
          numInput.value = deg;
        };

        slider.addEventListener('input', () => update(parseFloat(slider.value)));
        numInput.addEventListener('change', () => update(parseFloat(numInput.value)));
        numInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') update(parseFloat(numInput.value)); });

        row.appendChild(label);
        row.appendChild(slider);
        row.appendChild(numInput);
        p.controlsDiv.appendChild(row);
      }
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


def parse_gdml_flattened(path: Path, centroid=None):
    """Parse GDML and return flattened float32 array (9 floats per triangle).
    
    If centroid is given, shift so that centroid becomes the origin.
    Otherwise compute the mesh's own centroid and center there.
    """
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
    if centroid is None:
        centroid = arr.mean(axis=0)
    arr -= centroid
    return arr.tobytes()


# GDML files were extracted using extract_barrel_assemblies.py which
# tessellates ALL components from the STEP file at their STEP positions,
# then shifts by the combined vertex centroid. Body and hardware files
# use the SAME shift, so their step-relative positions are preserved.
# Do NOT recenter — pass centroid=(0,0,0) to keep the extracted positions.
ZERO = np.array([0.0, 0.0, 0.0])
ASSEMBLY_CENTROIDS: dict[str, np.ndarray] = {}
for label, _, components in PANELS:
    for gdml_name, *_ in components:
        ASSEMBLY_CENTROIDS[gdml_name] = ZERO

# Pre-parse all meshes. Single-file panels are centered on their own centroid.
# Multi-file panels use the combined assembly centroid so components stay aligned.
MESH_CACHE = {}
for label, _, components in PANELS:
    for gdml_name, comp_label, color, togglable, wireframe in components:
        path = MESH_DIR / gdml_name
        if not path.exists():
            print(f"WARNING: {path} not found!")
            continue
        if gdml_name not in MESH_CACHE:
            centroid = ASSEMBLY_CENTROIDS.get(gdml_name)
            data = parse_gdml_flattened(path, centroid=centroid)
            MESH_CACHE[gdml_name] = data
            n_tri = len(data) // 36
            tag = "(assembly)" if gdml_name in ASSEMBLY_CENTROIDS else "(self)"
            print(f"  {gdml_name}: {n_tri} triangles {tag}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            # Build spec JSON
            spec_list = []
            for label, rotatable, components in PANELS:
                comp_list = []
                for gdml_name, comp_label, color, togglable, wireframe in components:
                    comp_list.append({
                        "gdml": gdml_name,
                        "label": comp_label,
                        "color": color,
                        "togglable": togglable,
                        "wireframe": wireframe,
                    })
                spec_list.append({"label": label, "rotatable": rotatable,
                                  "components": comp_list})

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
    print('  10" panel has Z-rotation slider to align PMT axis.')
    print("  Use OrbitControls to pan/zoom/rotate.")
    print("\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
