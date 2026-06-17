import { app } from "../../scripts/app.js";

const BASE = new URL(".", import.meta.url).href;

let THREE = null, OrbitControls = null;
async function loadThree() {
    if (THREE) return;
    try {
        THREE = await import(BASE + "three.module.js");
        ({ OrbitControls } = await import(BASE + "OrbitControls.js"));
    } catch(e) {
        console.error("[PerspectiveFields3D] Three.js load error:", e);
        throw e;
    }
}

// ── scène Three.js ────────────────────────────────────────────────────────────
function createViewer(canvas, W, H) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H, false);
    renderer.setClearColor(0x111111, 1);

    // empêcher LiteGraph de voler les events
    for (const t of ["pointerdown","pointermove","pointerup","wheel","mousedown","mousemove","mouseup","contextmenu","click"]) {
        canvas.addEventListener(t, e => e.stopPropagation(), true);
    }

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(3, 5, 3);
    scene.add(dl);
    scene.add(new THREE.GridHelper(4, 8, 0x333333, 0x222222));

    const ax = (d, c) => scene.add(new THREE.ArrowHelper(
        new THREE.Vector3(...d).normalize(), new THREE.Vector3(), 0.45, c, 0.07, 0.04));
    ax([1,0,0], 0xff4444);
    ax([0,1,0], 0x44ff44);
    ax([0,0,1], 0x4444ff);

    const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100);
    const DEF = new THREE.Vector3(2.5, 2.0, 3.5);
    camera.position.copy(DEF);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;

    let dyn = new THREE.Group();
    scene.add(dyn);

    function updateVectors(up3d, road2d) {
        scene.remove(dyn);
        dyn = new THREE.Group();
        scene.add(dyn);

        const arr = (dir, color) => dyn.add(new THREE.ArrowHelper(
            new THREE.Vector3(...dir).normalize(),
            new THREE.Vector3(), 1.4, color, 0.2, 0.11));

        // up_3D : caméra(x=droite,y=bas,z=profond) → Three.js(x=droite,y=haut,z=viewer)
        arr([up3d[0], -up3d[1], -up3d[2]], 0x00ff88);

        // road_dir 2D → plan sol
        const rn = Math.hypot(road2d[0], road2d[1]) || 1;
        arr([ road2d[0]/rn, 0,  road2d[1]/rn], 0xffaa00);
        arr([-road2d[0]/rn, 0, -road2d[1]/rn], 0xffaa00);

        // voiture wireframe
        const car = new THREE.Group();
        car.rotation.y = -Math.atan2(road2d[1], road2d[0]);
        const wm = (g, c) => new THREE.Mesh(g, new THREE.MeshBasicMaterial(
            { color: c, wireframe: true, transparent: true, opacity: 0.45 }));
        const body = wm(new THREE.BoxGeometry(0.85, 0.28, 1.9), 0xffffff);
        body.position.y = 0.22; car.add(body);
        const roof = wm(new THREE.BoxGeometry(0.58, 0.22, 0.88), 0xaaddff);
        roof.position.set(0, 0.49, -0.1); car.add(roof);
        for (const [wx, wz] of [[-0.46,-0.72],[0.46,-0.72],[-0.46,0.72],[0.46,0.72]]) {
            const w = wm(new THREE.CylinderGeometry(0.13,0.13,0.12,12), 0x888888);
            w.rotation.z = Math.PI/2; w.position.set(wx, 0.12, wz); car.add(w);
        }
        dyn.add(car);
    }

    let raf = requestAnimationFrame(function loop() {
        raf = requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
    });

    return {
        updateVectors,
        reset: () => { camera.position.copy(DEF); controls.reset(); },
        dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); },
    };
}

// ── extension ComfyUI ─────────────────────────────────────────────────────────
app.registerExtension({
    name: "PerspectiveFields.3DViewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PerspectiveFields3DViewer") return;

        const W = 340, H = 320;

        // override onNodeCreated sur le prototype du type
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function() {
            origCreated?.call(this);
            const node = this;

            // conteneur
            const wrap = document.createElement("div");
            wrap.style.cssText = `width:${W}px;height:${H}px;position:relative;background:#111;border-radius:4px;overflow:hidden;box-sizing:border-box;`;

            // canvas Three.js
            const canvas = document.createElement("canvas");
            canvas.width = W * Math.min(window.devicePixelRatio, 2);
            canvas.height = H * Math.min(window.devicePixelRatio, 2);
            canvas.style.cssText = `width:${W}px;height:${H}px;display:block;`;
            wrap.appendChild(canvas);

            // bouton reset
            const btn = document.createElement("button");
            btn.textContent = "↺ reset";
            btn.style.cssText = "position:absolute;top:6px;right:6px;padding:3px 10px;background:#2a2a2a;color:#ccc;border:1px solid #555;border-radius:3px;cursor:pointer;font:11px monospace;z-index:10;";
            btn.addEventListener("pointerdown", e => e.stopPropagation(), true);
            wrap.appendChild(btn);

            // légende
            const leg = document.createElement("div");
            leg.style.cssText = "position:absolute;bottom:6px;left:8px;font:11px monospace;pointer-events:none;z-index:10;";
            leg.innerHTML = '<span style="color:#00ff88">▶ up_3D</span> &nbsp; <span style="color:#ffaa00">▶ road</span>';
            wrap.appendChild(leg);

            const latLbl = document.createElement("div");
            latLbl.style.cssText = "position:absolute;top:6px;left:8px;font:11px monospace;color:#888;pointer-events:none;z-index:10;";
            latLbl.textContent = "lat: —";
            wrap.appendChild(latLbl);

            // ajout du widget DOM avec computeSize dans les options
            node.addDOMWidget("viewer3d", "preview", wrap, {
                serialize: false,
                computeSize: () => [W, H],
            });
            node.setSize([Math.max(node.size[0], W + 20), node.size[1]]);

            // charger Three.js et init
            try {
                await loadThree();
                const viewer = createViewer(canvas, W, H);
                btn.addEventListener("click", () => viewer.reset());

                node.onExecuted = function(msg) {
                    if (!msg?.up3d || !msg?.road2d) return;
                    const lat = msg.lat?.[0] ?? 0;
                    latLbl.textContent = `lat: ${lat.toFixed(1)}°`;
                    viewer.updateVectors(msg.up3d[0], msg.road2d[0]);
                };

                const origRemove = node.onRemoved?.bind(node);
                node.onRemoved = () => { viewer.dispose(); origRemove?.(); };
            } catch(e) {
                const err = document.createElement("div");
                err.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#f66;font:12px monospace;padding:10px;text-align:center;";
                err.textContent = "Erreur Three.js : " + e.message;
                wrap.appendChild(err);
            }
        };
    },
});
