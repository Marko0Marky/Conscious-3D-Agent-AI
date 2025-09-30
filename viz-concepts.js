// --- START OF FILE viz-concepts.js ---
// FLOQUET-ENHANCED VERSION: InstancedMesh with coherence-modulated animations, Floquet PD spheres, and free-energy cores
// Integrates Th. 1–3 & 14–17: Harmonic flows scale nodes, PD bars birth on rhythmic awareness, emergence pulses with Φ_SA cascade.

import { logger, clamp, norm2, vecZeros } from './utils.js';
import { FloquetPersistentSheaf } from './qualia-sheaf.js';

// Explicit ES module imports for Three.js and its examples
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/renderers/CSS2DRenderer.js';

// Global variables
let scene, camera, renderer, labelRenderer, controls;
let nodes = {}; // { vertexId: { label: CSS2DObject, name: string, qualia: vec, instanceId: int, basePosition: Vector3, gameLink?: string } }
let edges = []; // { line: Line, u: string, v: string }
let instancedNodesMesh = null; // InstancedMesh for all nodes
let container;
let clock;
let initialized = false;
let sheafInstance = null;

// Dynamic elements (not part of the instanced graph)
let agentStateMesh, agentStateLabel, emergenceCoreMesh, emergenceCoreLabel, floquetPDGroup;

// A dummy object to help with matrix transformations for the instanced mesh
const dummy = new THREE.Object3D();

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

export async function initConceptVisualization(mainClock, sheaf) {
    if (initialized) {
        logger.warn('initConceptVisualization: Already initialized.');
        return true;
    }

    if (!THREE) {
        logger.error('initConceptVisualization: Three.js not found.');
        return false;
    }

    container = document.getElementById('concept-panel');
    if (!container) {
        logger.error('initConceptVisualization: Concept panel not found.');
        return false;
    }

    clock = mainClock;
    sheafInstance = sheaf;

    // Validate or initialize sheaf
    if (!sheafInstance || !sheafInstance.complex || !Array.isArray(sheafInstance.complex.vertices)) {
        logger.warn('initConceptVisualization: Invalid sheaf or complex. Initializing default sheaf.');
        sheafInstance = new FloquetPersistentSheaf({}, { qDim: 7, stateDim: 13 });
        try {
            await sheafInstance.initialize();
        } catch (e) {
            logger.error('initConceptVisualization: Failed to initialize default sheaf.', e);
            return false;
        }
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Initialize scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a15);

    // Initialize camera
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 0, 50);

    // Initialize renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Initialize label renderer
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    container.appendChild(labelRenderer.domElement);

    // Initialize controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Build graph
    buildGraphFromSheaf();

    // Initialize Floquet PD group (Th. 17)
    floquetPDGroup = new THREE.Group();
    floquetPDGroup.name = 'floquetPDGroup';
    scene.add(floquetPDGroup);

    // Verify scene initialization
    if (!scene || typeof scene.getObjectByName !== 'function') {
        logger.error('initConceptVisualization: Scene initialization failed.');
        return false;
    }

    // Add resize listener
    window.addEventListener('resize', onWindowResize);
    initialized = true;
    logger.info('3D Concept Viz initialized with Floquet-enhanced InstancedMesh and PD visualization.');
    return true;
}

function buildGraphFromSheaf() {
    if (!sheafInstance || !sheafInstance.complex || !Array.isArray(sheafInstance.complex.vertices)) {
        logger.error('buildGraphFromSheaf: Invalid sheaf or complex. Returning empty graph.');
        return;
    }

    // Clear existing graph elements
    edges.forEach(e => {
        scene.remove(e.line);
        e.line.geometry.dispose();
        e.line.material.dispose();
    });
    edges = [];
    Object.values(nodes).forEach(n => {
        if (n.label) scene.remove(n.label);
    });
    nodes = {};

    if (instancedNodesMesh) {
        scene.remove(instancedNodesMesh);
        instancedNodesMesh.geometry.dispose();
        instancedNodesMesh.material.dispose();
        instancedNodesMesh = null;
    }

    // Create instanced mesh for nodes
    const numVertices = sheafInstance.complex.vertices.length;
    const geometry = new THREE.SphereGeometry(0.5, 16, 16);
    const material = new THREE.MeshPhongMaterial({
        color: 0x4af, // Legacy blue
        emissive: 0x001122
    });
    instancedNodesMesh = new THREE.InstancedMesh(geometry, material, numVertices);
    scene.add(instancedNodesMesh);

    // Position nodes in a circle
    const basePositions = [];
    const radius = 5;
    sheafInstance.complex.vertices.forEach((vertexName, i) => {
        const angle = (i / numVertices) * 2 * Math.PI;
        const basePosition = new THREE.Vector3(
            Math.cos(angle) * radius,
            0,
            Math.sin(angle) * radius
        );
        basePositions.push(basePosition);

        // Create label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'concept-label';
        labelDiv.textContent = VERTEX_MAP[vertexName]?.name || vertexName;
        labelDiv.style.background = 'rgba(0, 0, 0, 0.7)';
        labelDiv.style.color = 'white';
        labelDiv.style.padding = '2px 4px';
        const label = new CSS2DObject(labelDiv);
        label.position.copy(basePosition);
        scene.add(label);

        nodes[vertexName] = {
            label,
            name: vertexName,
            qualia: sheafInstance.stalks?.get(vertexName)?.state || vecZeros(sheafInstance.qDim || 7),
            instanceId: i,
            basePosition,
            gameLink: VERTEX_MAP[vertexName]?.gameLink || null
        };
    });

    // Build edges
    sheafInstance.complex.edges.forEach(([u, v, weight]) => {
        const uIdx = sheafInstance.complex.vertices.indexOf(u);
        const vIdx = sheafInstance.complex.vertices.indexOf(v);
        if (uIdx === -1 || vIdx === -1) {
            logger.warn(`buildGraphFromSheaf: Invalid edge vertices (${u}, ${v}). Skipping.`);
            return;
        }

        const points = [basePositions[uIdx], basePositions[vIdx]];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x4af,
            transparent: true,
            opacity: clamp(weight || 0.5, 0.2, 0.8)
        });
        const line = new THREE.Line(geometry, material);
        scene.add(line);

        edges.push({ line, u, v });
    });

    // Emergence core (Th. 3: Free-energy visualization)
    const coreGeometry = new THREE.SphereGeometry(1, 16, 16);
    const coreMaterial = new THREE.MeshPhongMaterial({ color: 0xff44aa, emissive: 0x220011 });
    emergenceCoreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    emergenceCoreMesh.position.set(0, 0, -10);
    scene.add(emergenceCoreMesh);

    const coreLabelDiv = document.createElement('div');
    coreLabelDiv.className = 'concept-label';
    coreLabelDiv.textContent = 'Emergence Core (Φ)';
    coreLabelDiv.style.background = 'rgba(255, 68, 170, 0.8)';
    coreLabelDiv.style.color = 'white';
    emergenceCoreLabel = new CSS2DObject(coreLabelDiv);
    emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
    scene.add(emergenceCoreLabel);

    // Agent state mesh (game linkage)
    const agentGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const agentMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    agentStateMesh = new THREE.Mesh(agentGeometry, agentMaterial);
    agentStateMesh.position.set(-5, 0, 0);
    scene.add(agentStateMesh);

    const agentLabelDiv = document.createElement('div');
    agentLabelDiv.className = 'concept-label';
    agentLabelDiv.textContent = 'Agent State';
    agentStateLabel = new CSS2DObject(agentLabelDiv);
    agentStateLabel.position.copy(agentStateMesh.position);
    scene.add(agentStateLabel);

    logger.info(`buildGraphFromSheaf: Graph built: ${numVertices} vertices, ${sheafInstance.complex.edges.length} edges.`);
}

export function updateAgentSimulationVisuals(agent, canvas, ctx) {
    if (!(canvas instanceof HTMLCanvasElement)) {
        logger.error('updateAgentSimulationVisuals: Invalid canvas element.', { canvas });
        return;
    }

    if (!(ctx instanceof CanvasRenderingContext2D)) {
        try {
            ctx = canvas.getContext('2d');
            if (!ctx) {
                logger.error('updateAgentSimulationVisuals: Failed to get 2D context from canvas.');
                return;
            }
        } catch (e) {
            logger.error('updateAgentSimulationVisuals: Error getting canvas context:', e);
            return;
        }
    }

    if (!agent || !agent.ai || !agent.player) {
        logger.warn('updateAgentSimulationVisuals: Invalid agent state, using default.', { agent });
        agent = {
            ai: { x: 0, z: 0, rot: 0 },
            player: { x: 0, z: 0, rot: 0 }
        };
    }

    // Apply DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
    }

    try {
        ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    } catch (e) {
        logger.error('updateAgentSimulationVisuals: Error clearing canvas:', e);
        return;
    }

    const scaleX = (canvas.width / dpr) / 100;
    const scaleY = (canvas.height / dpr) / 100;
    const offsetX = (canvas.width / dpr) / 2;
    const offsetY = (canvas.height / dpr) / 2;

    try {
        ctx.save();
        ctx.translate(agent.ai.x * scaleX + offsetX, agent.ai.z * scaleY + offsetY);
        ctx.rotate(agent.ai.rot);
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
        ctx.closePath();
        ctx.restore();
    } catch (e) {
        logger.error('updateAgentSimulationVisuals: Error rendering AI:', e);
    }

    try {
        ctx.save();
        ctx.translate(agent.player.x * scaleX + offsetX, agent.player.z * scaleY + offsetY);
        ctx.rotate(agent.player.rot);
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
        ctx.closePath();
        ctx.restore();
    } catch (e) {
        logger.error('updateAgentSimulationVisuals: Error rendering player:', e);
    }

    if (agent.dist) {
        ctx.font = '12px Arial';
        ctx.fillStyle = 'black';
        ctx.fillText(`Dist: ${agent.dist.toFixed(2)}`, 10, 20);
    }
}

export function updateEdgeWeights() {
    if (!edges.length || !sheafInstance.adjacency) {
        logger.warn('updateEdgeWeights: No edges or adjacency data available.');
        return;
    }

    edges.forEach(({ line, u, v }) => {
        const uIdx = sheafInstance.complex.vertices.indexOf(u);
        const vIdx = sheafInstance.complex.vertices.indexOf(v);
        if (uIdx === -1 || vIdx === -1) {
            logger.warn(`updateEdgeWeights: Invalid edge vertices (${u}, ${v}). Skipping.`);
            return;
        }

        const weight = sheafInstance.adjacency.get(u)?.has(v) ? 0.8 : 0.5;
        line.material.opacity = clamp(weight, 0.2, 0.8);
        line.material.color.lerp(new THREE.Color(0x4af), weight);
    });
}

export function animateConceptNodes(deltaTime) {
    if (!initialized || !instancedNodesMesh) {
        logger.warn('animateConceptNodes: Visualization not initialized or instancedNodesMesh missing.');
        return;
    }

    const time = clock.getElapsedTime();

    sheafInstance.complex.vertices.forEach(vertexName => {
        const node = nodes[vertexName];
        if (!node) return;
        const { instanceId, basePosition } = node;
        const qualiaNorm = clamp(norm2(node.qualia) / Math.sqrt(sheafInstance.qDim || 7), 0, 1);
        const coherence = sheafInstance.coherence || 0; // Th. 1: Coherence modulation

        // Th. 2 & 17: Animate with Floquet phase
        const phase = sheafInstance.floquetPD?.phases[0] || 0;
        const yOffset = Math.sin(time * 2 + instanceId) * qualiaNorm * 2.0 * coherence;
        dummy.position.set(basePosition.x, basePosition.y + yOffset, basePosition.z);

        const scale = 1 + qualiaNorm * 0.5 * Math.cos(phase * time); // Rhythmic scaling
        dummy.scale.set(scale, scale, scale);

        dummy.updateMatrix();
        instancedNodesMesh.setMatrixAt(instanceId, dummy.matrix);

        node.label.position.copy(dummy.position);
    });

    instancedNodesMesh.instanceMatrix.needsUpdate = true;

    if (emergenceCoreMesh) {
        emergenceCoreMesh.rotation.y += deltaTime * 0.5;
        const phiPulse = 1 + Math.sin(time * 3) * 0.1 * clamp(sheafInstance.phi / 5.0, 0, 1);
        emergenceCoreMesh.scale.setScalar(phiPulse * sheafInstance.coherence); // Th. 1: Coherence pulse
    }
}

export function renderConceptVisualization() {
    if (!initialized) {
        logger.warn('renderConceptVisualization: Visualization not initialized.');
        return;
    }

    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);

    // Th. 17: Render Floquet PD every 30 frames
    if (Math.floor(clock.getElapsedTime() * 60) % 30 === 0) {
        // FIX: The function call is corrected to pass the 'sheafInstance' object,
        // which holds the actual cognitive data, instead of the 'scene' object.
        visualizeFloquetPD(sheafInstance);
    }
}

// Th. 17: Floquet PD Visualization
// Th. 17: Floquet PD Visualization
export function visualizeFloquetPD(qualiaSheaf) {
    if (!initialized || !scene || typeof scene.getObjectByName !== 'function') {
        logger.warn('visualizeFloquetPD: Visualization not initialized or scene invalid.', { initialized });
        return null;
    }

    // Validate qualiaSheaf structure
    if (!qualiaSheaf || !qualiaSheaf.complex || !Array.isArray(qualiaSheaf.complex.vertices)) {
        logger.warn('visualizeFloquetPD: Invalid qualiaSheaf structure.', { qualiaSheaf });
        return null;
    }

    // Clear existing PD group
    const existingPD = scene.getObjectByName('floquetPDGroup');
    if (existingPD) {
        existingPD.children.forEach(child => {
            child.geometry?.dispose();
            child.material?.dispose();
        });
        scene.remove(existingPD);
    }

    const pdGroup = new THREE.Group();
    pdGroup.name = 'floquetPDGroup';
    const geometry = new THREE.SphereGeometry(0.1, 8, 8);

    // Check for valid floquetPD data
    if (!qualiaSheaf.floquetPD || !Array.isArray(qualiaSheaf.floquetPD.births)) {
        logger.warn('visualizeFloquetPD: Missing or invalid floquetPD data. Rendering empty PD group.');
        scene.add(pdGroup);
        renderer.render(scene, camera);
        floquetPDGroup = pdGroup;
        return pdGroup;
    }

    // Process PD births and deaths
    const delta = qualiaSheaf.delta || 0.1;
    qualiaSheaf.floquetPD.births.forEach((birth, i) => {
        const death = qualiaSheaf.floquetPD.deaths?.[i] || { time: birth.time + delta, phase: birth.phase };
        const lifetime = death.time - birth.time;
        if (lifetime < delta) return;

        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color(
                0.5 + 0.5 * Math.cos(birth.phase),
                0.5 + 0.5 * Math.sin(birth.phase),
                0.5
            ),
            emissive: new THREE.Color(0.1, 0.1, 0.2)
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(
            birth.time * 0.1,
            lifetime * 0.1,
            birth.phase / (2 * Math.PI)
        );
        pdGroup.add(sphere);
    });

    scene.add(pdGroup);
    renderer.render(scene, camera);
    floquetPDGroup = pdGroup;
    logger.info('visualizeFloquetPD: Successfully rendered Floquet PD with ' + pdGroup.children.length + ' spheres.');
    return pdGroup;
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
    if (floquetPDGroup) {
        scene.remove(floquetPDGroup);
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
    logger.info('3D Concept Viz cleaned up with Floquet PD disposal.');
}

export { initialized as conceptInitialized };
export function isConceptVisualizationReady() {
    return initialized;
}

// --- END OF FILE viz-concepts.js ---
