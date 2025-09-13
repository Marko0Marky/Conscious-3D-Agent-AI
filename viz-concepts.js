// --- START OF FILE viz-concepts.js ---
// --- UPDATED VERSION: Uses InstancedMesh with legacy node styles (fixed color 0x4af, emissive 0x001122) ---
// Integrates high-performance InstancedMesh with game state updates, RIH score, and edge weighting.
import { logger, clamp, norm2, vecZeros } from './utils.js';

// Explicit ES module imports for Three.js and its examples
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/renderers/CSS2DRenderer.js';

// Globals
let scene, camera, renderer, labelRenderer, controls;
let nodes = {}; // { vertexId: { label: CSS2DObject, name: string, qualia: vec, instanceId: int, basePosition: Vector3, gameLink?: string } }
let edges = []; // { line: Line, u: string, v: string }
let instancedNodesMesh = null; // InstancedMesh for all nodes
let container;
let clock;
let initialized = false;
let sheafInstance = null;

// Dynamic elements (not part of the instanced graph)
let agentStateMesh, agentStateLabel, emergenceCoreMesh, emergenceCoreLabel;

// A dummy object to help with matrix transformations for the instanced mesh
const dummy = new THREE.Object3D();

const VERTEX_MAP = {
    'agent_x': { name: 'Agent-X', gameLink: 'agentX' }, 'agent_z': { name: 'Agent-Z', gameLink: 'agentZ' },
    'agent_rot': { name: 'Agent-Rot', gameLink: 'agentRot' }, 'target_x': { name: 'Target-X', gameLink: 'targetX' },
    'target_z': { name: 'Target-Z', gameLink: 'targetZ' }, 'vec_dx': { name: 'Vec-DX', gameLink: null },
    'vec_dz': { name: 'Vec-DZ', gameLink: null }, 'dist_target': { name: 'Dist-Target', gameLink: 'dist' }
};

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
    sheafInstance = sheaf;

    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a15);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 0, 50);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    container.appendChild(labelRenderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
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
    logger.info('3D Concept Viz initialized with InstancedMesh and legacy node styles.');
    return true;
}

function buildGraphFromSheaf() {
    if (!sheafInstance) return;

    // --- Cleanup previous elements ---
    if (instancedNodesMesh) scene.remove(instancedNodesMesh);
    edges.forEach(e => scene.remove(e.line));
    Object.values(nodes).forEach(n => {
        if (n.label) scene.remove(n.label);
    });
    nodes = {};
    edges = [];
    instancedNodesMesh = null;
    
    // --- Build Nodes using InstancedMesh with legacy styles ---
    const nodeCount = sheafInstance.graph.vertices.length;
    const geometry = new THREE.SphereGeometry(1, 16, 16);
    // Match legacy node style: fixed color and emissive
    const material = new THREE.MeshPhongMaterial({ color: 0x4af, emissive: 0x001122 });

    instancedNodesMesh = new THREE.InstancedMesh(geometry, material, nodeCount);
    scene.add(instancedNodesMesh);

    sheafInstance.graph.vertices.forEach((vertexName, idx) => {
        const vertexInfo = VERTEX_MAP[vertexName] || { name: vertexName, gameLink: null };
        const position = new THREE.Vector3(
            Math.random() * 40 - 20, 
            Math.random() * 20 - 10, 
            Math.random() * 40 - 20
        );

        // Set initial transform for this instance
        dummy.position.copy(position);
        dummy.updateMatrix();
        instancedNodesMesh.setMatrixAt(idx, dummy.matrix);

        // Create label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'concept-label';
        labelDiv.textContent = vertexInfo.name;
        labelDiv.style.color = '#4af';
        const label = new CSS2DObject(labelDiv);
        label.position.copy(position);
        scene.add(label);

        nodes[vertexName] = { 
            label, 
            name: vertexInfo.name, 
            gameLink: vertexInfo.gameLink,
            qualia: sheafInstance.stalks.get(vertexName) || vecZeros(sheafInstance.qDim),
            instanceId: idx,
            basePosition: position.clone() // Store base position for animation
        };
    });
    
    // Mark instance data for update
    instancedNodesMesh.instanceMatrix.needsUpdate = true;

    // --- Build Edges as individual lines (performance is acceptable) ---
    sheafInstance.graph.edges.forEach(([u, v]) => {
        const uNode = nodes[u], vNode = nodes[v];
        if (!uNode || !vNode) return;
        
        const p1 = uNode.basePosition;
        const p2 = vNode.basePosition;
        const geometry = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const material = new THREE.LineBasicMaterial({ color: 0x4af, opacity: 0.5, transparent: true });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        edges.push({ line, u, v }); // Store line and its vertices
    });

    createAgentStateNode();
    createEmergenceCoreNode(sheafInstance);
}

function createAgentStateNode() {
    const geometry = new THREE.BoxGeometry(1.5, 1.5, 1.5);
    const material = new THREE.MeshPhongMaterial({ color: 0xffaa00, emissive: 0x112200 });
    agentStateMesh = new THREE.Mesh(geometry, material);
    agentStateMesh.position.set(20, 0, 0);
    scene.add(agentStateMesh);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'concept-label';
    labelDiv.textContent = 'Agent State';
    labelDiv.style.color = '#ffaa00';
    agentStateLabel = new CSS2DObject(labelDiv);
    agentStateLabel.position.copy(agentStateMesh.position);
    scene.add(agentStateLabel);
}

function createEmergenceCoreNode(sheaf) {
    const geometry = new THREE.SphereGeometry(2.5, 16, 16);
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff99, emissive: 0x003322 });
    emergenceCoreMesh = new THREE.Mesh(geometry, material);
    emergenceCoreMesh.position.set(-20, 0, 0);
    emergenceCoreMesh.scale.setScalar(sheaf?.phi || 1);
    scene.add(emergenceCoreMesh);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'concept-label';
    labelDiv.textContent = 'Emergence Core (Φ)';
    labelDiv.style.color = '#00ff99';
    emergenceCoreLabel = new CSS2DObject(labelDiv);
    emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
    scene.add(emergenceCoreLabel);
}

export function updateAgentSimulationVisuals(qualiaTensor, gameState, rihScore) {
    if (!initialized || !sheafInstance) return;

    sheafInstance.graph.vertices.forEach(vertexName => {
        const node = nodes[vertexName];
        if (!node) return;
        const qualia = sheafInstance.stalks.get(vertexName) || node.qualia;
        node.qualia = qualia; // Update internal qualia state

        const intensity = clamp(norm2(qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        
        // Match legacy style: update emissive and scale instead of vertex color
        instancedNodesMesh.material.emissive.setHex(0x000000).lerp(new THREE.Color(0x4af), intensity);
        const scale = 1 + intensity;
        dummy.scale.set(scale, scale, scale);
        dummy.position.copy(node.basePosition);
        dummy.updateMatrix();
        instancedNodesMesh.setMatrixAt(node.instanceId, dummy.matrix);

        // Update label text
        node.label.element.textContent = `${node.name}\n(${intensity.toFixed(2)})`;
    });
    
    // Mark instance matrix buffer for update
    if (instancedNodesMesh) instancedNodesMesh.instanceMatrix.needsUpdate = true;

    // Update agent state mesh position based on gameState
    if (agentStateMesh && gameState?.ai) {
        agentStateMesh.position.lerp(new THREE.Vector3(gameState.ai.x / 10, 0, gameState.ai.z / 10), 0.1);
        agentStateLabel.position.copy(agentStateMesh.position);
    }

    // Update emergence core based on dist and rihScore
    if (emergenceCoreMesh && gameState) {
        const dist = gameState.dist || 1;
        const pulse = clamp(1 / (dist + 1e-6), 0.5, 2);
        emergenceCoreMesh.scale.lerp(new THREE.Vector3(pulse, pulse, pulse), 0.1);
        emergenceCoreMesh.material.emissive.setHex(0x222233).lerp(new THREE.Color(0xffffff), rihScore || 0);
        emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
    }

    // Update edges
    edges.forEach(({ line, u, v }, idx) => {
        const uIdx = sheafInstance.graph.vertices.indexOf(u);
        const vIdx = sheafInstance.graph.vertices.indexOf(v);
        if (uIdx === -1 || vIdx === -1 || !sheafInstance.adjacencyMatrix) return;
        
        const weight = sheafInstance.adjacencyMatrix[uIdx]?.[vIdx] || 0.5;
        line.material.opacity = clamp(weight, 0.2, 0.8);
        line.material.color.lerp(new THREE.Color(0x4af), weight);
    });
}

export function animateConceptNodes(deltaTime) {
    if (!initialized || !instancedNodesMesh) return;
    const time = clock.getElapsedTime();

    sheafInstance.graph.vertices.forEach(vertexName => {
        const node = nodes[vertexName];
        if (!node) return;
        const { instanceId, basePosition } = node;
        const qualiaNorm = clamp(norm2(node.qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);

        // Calculate new position and scale
        const yOffset = Math.sin(time * 2 + instanceId) * qualiaNorm * 2.0; // Animate y-position
        dummy.position.set(basePosition.x, basePosition.y + yOffset, basePosition.z);
        
        const scale = 1 + qualiaNorm * 0.5; // Scale based on intensity
        dummy.scale.set(scale, scale, scale);

        // Update the matrix for this instance
        dummy.updateMatrix();
        instancedNodesMesh.setMatrixAt(instanceId, dummy.matrix);

        // Update the label position to follow the animated node
        node.label.position.copy(dummy.position);
    });

    // Mark instance matrix buffer for update
    instancedNodesMesh.instanceMatrix.needsUpdate = true;

    if (emergenceCoreMesh) {
        emergenceCoreMesh.rotation.y += deltaTime * 0.5;
        const phiPulse = 1 + Math.sin(time * 3) * 0.1 * clamp(sheafInstance.phi / 5.0, 0, 1);
        emergenceCoreMesh.scale.setScalar(phiPulse);
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

    if (instancedNodesMesh) {
        scene.remove(instancedNodesMesh);
        instancedNodesMesh.geometry.dispose();
        instancedNodesMesh.material.dispose();
    }
    edges.forEach(e => {
        scene.remove(e.line);
        e.line.geometry.dispose();
        e.line.material.dispose();
    });

    Object.values(nodes).forEach(n => {
        if (n.label) scene.remove(n.label);
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
    if (controls) controls.dispose();
    
    initialized = false;
    sheafInstance = null;
    instancedNodesMesh = null;
    nodes = {};
    edges = [];
    logger.info('3D Concept Viz cleaned up.');
}

export { initialized as conceptInitialized };
export function isConceptVisualizationReady() { return initialized; }
// --- END OF FILE viz-concepts.js ---
