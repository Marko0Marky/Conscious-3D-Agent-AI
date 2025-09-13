// --- START OF FILE viz-concepts.js ---
// --- UPDATED FILE: viz-concepts.js ---
// Integrated with Sheaf AI: 3D Three.js graph for sheaf vertices; links to game state (e.g., agent pos → node anim).
// Shared clock/scene with ThreeDeeGame; edges weighted by sheaf correlations.

import { logger, clamp, norm2 } from './utils.js';

// Dependencies: THREE, OrbitControls, CSS2DRenderer (global via CDN)

// Globals
let scene, camera, renderer, labelRenderer, controls;
let nodes = {}; // { vertexId: { mesh: THREE.Mesh, label: CSS2DObject, data: { qualia: vec } } }
let edges = []; // Line meshes for correlations
let container;
let clock; // Shared from main.js
let initialized = false;
let sheafInstance = null; // Module-level variable to hold sheaf reference

// Dynamic elements: Link to game/sheaf
let agentStateMesh, agentStateLabel, emergenceCoreMesh, emergenceCoreLabel;

// Sheaf integration: Map vertices to nodes (from qualia-sheaf.js)
const VERTEX_MAP = {
    'agent_x': { name: 'Agent-X', gameLink: 'agentX' },
    'agent_z': { name: 'Agent-Z', gameLink: 'agentZ' },
    'agent_rot': { name: 'Agent-Rot', gameLink: 'agentRot' },
    'target_x': { name: 'Target-X', gameLink: 'targetX' },
    'target_z': { name: 'Target-Z', gameLink: 'targetZ' },
    'vec_dx': { name: 'Vec-DX', gameLink: null },
    'vec_dz': { name: 'Vec-DZ', gameLink: null },
    'dist_target': { name: 'Dist-Target', gameLink: 'dist' }
};

/**
 * Initializes the 3D concept graph, sharing clock with main game.
 * @param {THREE.Clock} mainClock - Shared clock.
 * @param {Object} sheaf - Reference to EnhancedQualiaSheaf for correlations.
 * @returns {boolean} Success.
 */
export function initConceptVisualization(mainClock, sheaf) {
    if (initialized) return true;
    if (typeof THREE === 'undefined') {
        logger.error('Three.js not found for Concept Viz.');
        return false;
    }

    container = document.getElementById('concept-panel');
    if (!container) {
        logger.error('Concept panel not found.');
        return false;
    }
    clock = mainClock;
    sheafInstance = sheaf; // Assign to module-level variable

    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a15);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 0, 50);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    labelRenderer = new THREE.CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    container.appendChild(labelRenderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    buildGraphFromSheaf();

    window.addEventListener('resize', onWindowResize);
    initialized = true;
    logger.info('3D Concept Viz initialized with Sheaf integration.');
    return true;
}

/**
 * Builds nodes/edges from sheaf vertices and correlations.
 */
function buildGraphFromSheaf() {
    if (!sheafInstance) return;

    Object.values(nodes).forEach(n => {
        scene.remove(n.mesh);
        if (n.label) scene.remove(n.label);
    });
    edges.forEach(e => scene.remove(e));
    nodes = {};
    edges = [];

    sheafInstance.graph.vertices.forEach((vertexName, idx) => {
        const data = VERTEX_MAP[vertexName] || { name: vertexName };
        const geometry = new THREE.SphereGeometry(1, 16, 16);
        const material = new THREE.MeshPhongMaterial({ color: 0x4af, emissive: 0x001122 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(new THREE.Vector3(Math.random() * 40 - 20, Math.random() * 20 - 10, Math.random() * 40 - 20));
        mesh.userData = { idx, ...data };
        scene.add(mesh);

        const labelDiv = document.createElement('div');
        labelDiv.className = 'concept-label';
        labelDiv.textContent = data.name;
        labelDiv.style.color = '#4af';
        const label = new THREE.CSS2DObject(labelDiv);
        label.position.copy(mesh.position);
        scene.add(label);

        nodes[vertexName] = { mesh, label, data: { qualia: sheafInstance.stalks.get(vertexName) || [] } };
    });

    sheafInstance.graph.edges.forEach(([u, v]) => {
        const uNode = nodes[u], vNode = nodes[v];
        if (!uNode || !vNode) return;
        const positions = [uNode.mesh.position, vNode.mesh.position];
        const geometry = new THREE.BufferGeometry().setFromPoints(positions);
        const material = new THREE.LineBasicMaterial({ color: 0x888888, opacity: 0.5, transparent: true });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        edges.push(line);
    });

    createAgentStateNode();
    createEmergenceCoreNode(sheafInstance);
}

function createAgentStateNode() {
    const geometry = new THREE.SphereGeometry(1.5, 16, 16);
    const material = new THREE.MeshPhongMaterial({ color: 0x44aaff, emissive: 0x000033 });
    agentStateMesh = new THREE.Mesh(geometry, material);
    agentStateMesh.position.set(20, 0, 0);
    scene.add(agentStateMesh);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'concept-label';
    labelDiv.textContent = 'Agent State';
    labelDiv.style.color = '#44aaff';
    agentStateLabel = new THREE.CSS2DObject(labelDiv);
    agentStateLabel.position.copy(agentStateMesh.position);
    scene.add(agentStateLabel);
}

function createEmergenceCoreNode(sheaf) {
    const geometry = new THREE.SphereGeometry(2, 16, 16);
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff99, emissive: 0x003322 });
    emergenceCoreMesh = new THREE.Mesh(geometry, material);
    emergenceCoreMesh.position.set(-20, 0, 0);
    emergenceCoreMesh.scale.setScalar(sheaf?.phi || 1);
    scene.add(emergenceCoreMesh);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'concept-label';
    labelDiv.textContent = 'Emergence Core (Φ)';
    labelDiv.style.color = '#00ff99';
    emergenceCoreLabel = new THREE.CSS2DObject(labelDiv);
    emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
    scene.add(emergenceCoreLabel);
}

export function updateAgentSimulationVisuals(qualiaTensor, gameState, rihScore) {
    if (!initialized || !sheafInstance) return;

    sheafInstance.graph.vertices.forEach(vertexName => {
        const node = nodes[vertexName];
        if (!node) return;
        const qualia = node.data.qualia;
        const intensity = clamp(norm2(qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        node.mesh.material.emissive.setHex(0x000000).lerp(new THREE.Color(0x4af), intensity);
        node.mesh.scale.lerp(new THREE.Vector3(1 + intensity, 1 + intensity, 1 + intensity), 0.1);
        node.label.element.textContent = `${node.data.name}\n(${intensity.toFixed(2)})`;
    });

    if (agentStateMesh && gameState?.ai) {
        agentStateMesh.position.lerp(new THREE.Vector3(gameState.ai.x / 10, 0, gameState.ai.z / 10), 0.1);
        agentStateLabel.position.copy(agentStateMesh.position);
    }

    if (emergenceCoreMesh && gameState) {
        const dist = gameState.dist || 1;
        const pulse = clamp(1 / (dist + 1e-6), 0.5, 2);
        emergenceCoreMesh.scale.lerp(new THREE.Vector3(pulse, pulse, pulse), 0.1);
        emergenceCoreMesh.material.emissive.setHex(0x222233).lerp(new THREE.Color(0xffffff), rihScore);
        emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
    }

    edges.forEach((edge, idx) => {
        const [u, v] = sheafInstance.graph.edges[idx];
        const uIdx = sheafInstance.graph.vertices.indexOf(u);
        const vIdx = sheafInstance.graph.vertices.indexOf(v);
        if (uIdx === -1 || vIdx === -1 || !sheafInstance.adjacencyMatrix) return;
        
        const weight = sheafInstance.adjacencyMatrix[uIdx]?.[vIdx] || 0.5;
        edge.material.opacity = clamp(weight, 0.2, 0.8);
        edge.material.color.lerp(new THREE.Color(0x4af), weight);
    });
}

export function animateConceptNodes(deltaTime) {
    if (!initialized) return;
    const time = clock.getElapsedTime();

    Object.values(nodes).forEach(node => {
        const qualiaNorm = clamp(norm2(node.data.qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        node.mesh.position.y += Math.sin(time * 2 + node.userData.idx) * qualiaNorm * deltaTime * 0.5;
        node.label.position.copy(node.mesh.position);
    });

    if (emergenceCoreMesh) {
        emergenceCoreMesh.rotation.y += deltaTime * 0.5;
    }
}

export function renderConceptVisualization() {
    if (!initialized) return;
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}

function onWindowResize() {
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width <= 0 || height <= 0) return;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labelRenderer.setSize(width, height);
}

export function cleanupConceptVisualization() {
    if (!initialized) return;

    Object.values(nodes).forEach(n => {
        scene.remove(n.mesh);
        if (n.label) scene.remove(n.label);
        n.mesh.geometry.dispose();
        n.mesh.material.dispose();
    });
    edges.forEach(e => {
        scene.remove(e);
        e.geometry.dispose();
        e.material.dispose();
    });

    if (agentStateMesh) {
        scene.remove(agentStateMesh);
        agentStateMesh.geometry.dispose();
        agentStateMesh.material.dispose();
    }
    if (emergenceCoreMesh) {
        scene.remove(emergenceCoreMesh);
        emergenceCoreMesh.geometry.dispose();
        emergenceCoreMesh.material.dispose();
    }
    if (agentStateLabel) scene.remove(agentStateLabel);
    if (emergenceCoreLabel) scene.remove(emergenceCoreLabel);

    renderer.dispose();
    if (container) container.innerHTML = '';
    controls.dispose();
    
    initialized = false;
    sheafInstance = null;
    logger.info('3D Concept Viz cleaned up.');
}

export { initialized as conceptInitialized };
export function isConceptVisualizationReady() { return initialized; }
// --- END OF FILE viz-concepts.js ---