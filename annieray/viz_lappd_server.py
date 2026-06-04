"""Standalone LAPPD module visualizer for ANNIE.

Serves a minimal Three.js page showing only the ANNIE LAPPD housing
box and photocathode, with OrbitControls for inspection.

Usage:
    python -m annieray viz-lappd
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

from annieray.lappd_model import build_housing, housing_to_arrays

housing_json = "{}"


def _build_housing_json() -> dict:
    housing = build_housing((0, 0, 0), (0, 0, 1))
    hd, ad = housing_to_arrays(housing)
    h = hd[0]
    a = ad[0]
    return {
        "center": [float(h[0]), float(h[1]), float(h[2])],
        "axis_x": [float(h[3]), float(h[4]), float(h[5])],
        "axis_y": [float(h[6]), float(h[7]), float(h[8])],
        "axis_z": [float(h[9]), float(h[10]), float(h[11])],
        "half": [float(h[12]), float(h[13]), float(h[14])],
        "pc_center": [float(a[0]), float(a[1]), float(a[2])],
        "pc_normal": [float(a[3]), float(a[4]), float(a[5])],
        "pc_half": [float(a[6])],
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin:0; overflow:hidden; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#1a1a2e; }
  #status {
    position:absolute; bottom:12px; left:12px; z-index:100;
    color:rgba(255,255,255,0.5); font-size:12px;
  }
  #axis-label {
    position:absolute; bottom:12px; right:12px; z-index:100;
    color:rgba(255,255,255,0.4); font-size:11px;
    text-align:right; line-height:1.6;
  }
  .c { color:#ff4444; } .y { color:#44ff44; } .z { color:#4488ff; }
</style>
</head>
<body>
<div id="status">LAPPD housing · drag to orbit · scroll to zoom</div>
<div id="axis-label"><span class="c">X</span> tangential · <span class="y">Y</span> vertical · <span class="z">Z</span> radial</div>

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

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 10000);
camera.position.set(600, 400, 600);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

// ---- Lights ----
const ambient = new THREE.AmbientLight(0x404060, 0.8);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(500, 800, 600);
dirLight.castShadow = true;
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x6688cc, 0.5);
fillLight.position.set(-400, -200, -300);
scene.add(fillLight);

// ---- Grid & Axes ----
const grid = new THREE.GridHelper(1000, 20, 0x444466, 0x333355);
scene.add(grid);

const axes = new THREE.AxesHelper(300);
scene.add(axes);

// ---- Build housing from API ----
const housingResp = await fetch('/api/geometry');
const h = await housingResp.json();

// Housing box
const boxGeo = new THREE.BoxGeometry(h.half[0]*2, h.half[1]*2, h.half[2]*2);
const boxMat = new THREE.MeshStandardMaterial({
    color: 0x4488aa,
    transparent: true,
    opacity: 0.25,
    roughness: 0.5,
    metalness: 0.05,
    side: THREE.DoubleSide,
});
const boxMesh = new THREE.Mesh(boxGeo, boxMat);
boxMesh.position.set(h.center[0], h.center[1], h.center[2]);
const m4 = new THREE.Matrix4();
m4.set(
    h.axis_x[0], h.axis_y[0], h.axis_z[0], 0,
    h.axis_x[1], h.axis_y[1], h.axis_z[1], 0,
    h.axis_x[2], h.axis_y[2], h.axis_z[2], 0,
    0, 0, 0, 1,
);
boxMesh.quaternion.setFromRotationMatrix(m4);
boxMesh.castShadow = true;
boxMesh.receiveShadow = true;
scene.add(boxMesh);

// Box edges (wireframe overlay)
const edgeGeo = new THREE.EdgesGeometry(boxGeo);
const edgeMat = new THREE.LineBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.6 });
const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
edgeMesh.position.copy(boxMesh.position);
edgeMesh.quaternion.copy(boxMesh.quaternion);
scene.add(edgeMesh);

// Photocathode rectangle
const pcGeo = new THREE.PlaneGeometry(h.pc_half[0]*2, h.pc_half[0]*2);
const pcMat = new THREE.MeshStandardMaterial({
    color: 0x66aadd,
    roughness: 0.3,
    metalness: 0.1,
    side: THREE.DoubleSide,
});
const pcMesh = new THREE.Mesh(pcGeo, pcMat);
pcMesh.position.set(h.pc_center[0], h.pc_center[1], h.pc_center[2]);
pcMesh.quaternion.copy(boxMesh.quaternion);
pcMesh.castShadow = true;
pcMesh.receiveShadow = true;
scene.add(pcMesh);

// Photocathode edges
const pcEdgeGeo = new THREE.EdgesGeometry(pcGeo);
const pcEdgeMat = new THREE.LineBasicMaterial({ color: 0x88ddff, transparent: true, opacity: 0.4 });
const pcEdgeMesh = new THREE.LineSegments(pcEdgeGeo, pcEdgeMat);
pcEdgeMesh.position.copy(pcMesh.position);
pcEdgeMesh.quaternion.copy(pcMesh.quaternion);
scene.add(pcEdgeMesh);

// ---- Window (front face of housing) ----
const winMat = new THREE.MeshStandardMaterial({
    color: 0x88bbdd,
    transparent: true,
    opacity: 0.15,
    roughness: 0.1,
    metalness: 0.0,
    side: THREE.DoubleSide,
});
const winMesh = new THREE.Mesh(boxGeo.clone(), winMat);
winMesh.position.copy(boxMesh.position);
winMesh.quaternion.copy(boxMesh.quaternion);
scene.add(winMesh);

// ---- Animation ----
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

document.title = 'LAPPD Module Viewer';
</script>
</body>
</html>"""


class LAPPDServer(BaseHTTPRequestHandler):
    _housing: dict = {}

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html()
        elif path == "/api/geometry":
            self._send_json(self._housing)
        else:
            self.send_error(404)


def run_server(host: str = "localhost", port: int = 8081) -> None:
    LAPPDServer._housing = _build_housing_json()
    server = HTTPServer((host, port), LAPPDServer)
    print(f"LAPPD module viewer at http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
