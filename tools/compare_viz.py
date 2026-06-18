"""Side-by-side comparison viz: original GDML vs closed reconstruction.

Left:  original (blue wireframe)
Right: closed (blue solid, semi-transparent)

Usage:
    python tools/compare_viz.py
    # Open http://localhost:8001
"""

import xml.etree.ElementTree as ET
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse

ORIG_PATH = Path("PHASE2_INNER_STRUCTURE.gdml")
CLOSED_PATH = Path("PHASE2_INNER_STRUCTURE_closed.gdml")

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  body { margin: 0; overflow: hidden; font-family: sans-serif; background: #1a1a2e; }
  #container { display: flex; width: 100vw; height: 100vh; }
  .panel { flex: 1; position: relative; overflow: hidden; }
  .panel canvas { display: block; width: 100% !important; height: 100% !important; }
  .label { position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
           color: #ccc; font-size: 14px; background: rgba(0,0,0,0.6);
           padding: 4px 14px; border-radius: 6px; pointer-events: none; z-index: 10; }
  .stats { position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
           color: #888; font-size: 12px; background: rgba(0,0,0,0.5);
           padding: 3px 10px; border-radius: 4px; pointer-events: none; z-index: 10; }
  .divider { width: 2px; background: #333; flex-shrink: 0; }
</style>
</head><body>
<div id="container">
  <div class="panel" id="panel0">
    <div class="label">Original (wireframe)</div>
    <div class="stats" id="stats0">loading...</div>
  </div>
  <div class="divider"></div>
  <div class="panel" id="panel1">
    <div class="label">Closed (solid)</div>
    <div class="stats" id="stats1">loading...</div>
  </div>
</div>

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

async function fetchBuf(url) {
  const resp = await fetch(url);
  return new Float32Array(await resp.arrayBuffer());
}

function initPanel(panelId, statsId) {
  const panel = document.getElementById(panelId);
  const rect = panel.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(40, W / H, 1, 20000);
  camera.up.set(0, 0, 1);
  camera.position.set(3500, 2500, 3000);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.NoToneMapping;
  panel.insertBefore(renderer.domElement, panel.firstChild);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 2000);
  controls.update();

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  const dir1 = new THREE.DirectionalLight(0xffffff, 1.0);
  dir1.position.set(1, 2, 1);
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dir2.position.set(-1, -1, 0.5);
  scene.add(dir2);

  const axes = new THREE.AxesHelper(500);
  scene.add(axes);

  return { scene, camera, renderer, controls, panel };
}

async function main() {
  const [origFlat, closedFlat] = await Promise.all([
    fetchBuf('/api/orig'),
    fetchBuf('/api/closed'),
  ]);

  document.getElementById('stats0').textContent =
    `${(origFlat.length / 9).toLocaleString()} triangles`;
  document.getElementById('stats1').textContent =
    `${(closedFlat.length / 9).toLocaleString()} triangles`;

  // Panel 0: original wireframe
  const p0 = initPanel('panel0', 'stats0');
  const origGeo = new THREE.BufferGeometry();
  origGeo.setAttribute('position', new THREE.BufferAttribute(origFlat, 3));
  origGeo.computeVertexNormals();
  const wireMesh = new THREE.Mesh(origGeo, new THREE.MeshBasicMaterial({
    color: 0x88bbff,
    wireframe: true,
    transparent: true,
    opacity: 0.35,
  }));
  p0.scene.add(wireMesh);

  // Panel 1: closed solid (semi-transparent blue)
  const p1 = initPanel('panel1', 'stats1');
  const closedGeo = new THREE.BufferGeometry();
  closedGeo.setAttribute('position', new THREE.BufferAttribute(closedFlat, 3));
  closedGeo.computeVertexNormals();
  const solidMesh = new THREE.Mesh(closedGeo, new THREE.MeshPhysicalMaterial({
    color: 0x4488ff,
    transparent: true,
    opacity: 0.55,
    roughness: 0.3,
    metalness: 0.0,
    side: THREE.DoubleSide,
  }));
  p1.scene.add(solidMesh);

  function resize() {
    for (const p of [p0, p1]) {
      const rect = p.panel.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      p.camera.aspect = w / h;
      p.camera.updateProjectionMatrix();
      p.renderer.setSize(w, h);
    }
  }
  window.addEventListener('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    p0.controls.update();
    p0.renderer.render(p0.scene, p0.camera);
    p1.controls.update();
    p1.renderer.render(p1.scene, p1.camera);
  }
  animate();
}

main();
</script>
</body></html>"""


def parse_gdml_flattened(path: Path):
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
    return np.array(out, dtype=np.float32).tobytes()


orig_data = parse_gdml_flattened(ORIG_PATH)
closed_data = parse_gdml_flattened(CLOSED_PATH)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif parsed.path == "/api/orig":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(orig_data)))
            self.end_headers()
            self.wfile.write(orig_data)
        elif parsed.path == "/api/closed":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(closed_data)))
            self.end_headers()
            self.wfile.write(closed_data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[COMPARE] {args[0]} {args[1]} {args[2]}")


def main():
    print(f"Original:  {len(orig_data) // 36} triangles")
    print(f"Closed:    {len(closed_data) // 36} triangles")

    port = 8001
    server = HTTPServer(("", port), Handler)
    print(f"\nComparison viz at http://localhost:{port}")
    print(f"  Left:  Original (wireframe)")
    print(f"  Right: Closed (solid, semi-transparent)")
    print("\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
