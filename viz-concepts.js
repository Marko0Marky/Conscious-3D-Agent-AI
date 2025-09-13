// --- START OF FILE viz-concepts.js ---
// --- FIXED VERSION: viz-concepts.js ---
// Integrated with Sheaf AI: 3D Three.js graph for sheaf vertices; links to game state (e.g., agent pos → node anim).
// Shared clock/scene with ThreeDeeGame; edges weighted by sheaf correlations.
// FIX: Resolved 'node.userData.idx undefined' by using node.mesh.userData.idx.
// FIX: Merged vertexInfo into node object for access to name in updates.
// FIX: Use vecZeros for qualia fallback; import vecZeros.

import { logger, clamp, norm2, vecZeros } from './utils.js';
// Corrected import path for opt-three, ensuring it's a relative path
// Explicit ES module imports for Three.js and its components
import * as THREE from 'three'; // Changed to bare specifier 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Changed to bare specifier for jsm utility
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js'; // Changed to bare specifier for jsm utility
import { optimizeScene, updateMergedMeshes } from './opt-three.js';

// Module-level variables
 let scene, camera, renderer, labelRenderer, controls;
 let nodes = {}; // { vertexId: { mesh: THREE.Mesh, label: CSS2DObject, name: string, qualia: vec } }
 let edges = []; // Individual Line meshes for initial setup
let mergedNodesMesh = null; // New: Merged mesh for nodes
let mergedEdgesMesh = null; // New: Merged mesh for edges
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

    labelRenderer = new CSS2DRenderer(); // Use imported CSS2DRenderer
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    container.appendChild(labelRenderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement); // Use imported 
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
 * Now creates individual meshes and then merges them.
 */
function buildGraphFromSheaf() {
    if (!sheafInstance) return;

    // Cleanup previous elements before rebuilding
    if (mergedNodesMesh) scene.remove(mergedNodesMesh);
    if (mergedEdgesMesh) scene.remove(mergedEdgesMesh);
    Object.values(nodes).forEach(n => {
        if (n.mesh) scene.remove(n.mesh); // Remove individual meshes if they were added
        if (n.label) scene.remove(n.label);
    });
    edges.forEach(e => scene.remove(e)); // Remove individual edges if they were added
    nodes = {};
    edges = [];
    mergedNodesMesh = null;
    mergedEdgesMesh = null;


    const tempNodeMeshes = []; // To hold individual meshes for merging
    const tempEdgeLines = []; // To hold individual lines for merging

    sheafInstance.graph.vertices.forEach((vertexName, idx) => {
        const vertexInfo = VERTEX_MAP[vertexName] || { name: vertexName };
        const geometry = new THREE.SphereGeometry(1, 16, 16);
        const material = new THREE.MeshPhongMaterial({ color: 0x4af, emissive: 0x001122 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(new THREE.Vector3(Math.random() * 40 - 20, Math.random() * 20 - 10, Math.random() * 40 - 20));
        mesh.userData = { idx, ...vertexInfo };
        // Do NOT add individual mesh to scene yet, add to temp array for merging
        tempNodeMeshes.push(mesh);

        const labelDiv = document.createElement('div');
        labelDiv.className = 'concept-label';
        labelDiv.textContent = vertexInfo.name;
        labelDiv.style.color = '#4af';
        const label = new CSS2DObject(labelDiv); // <-- Corrected: Removed 'THREE.'
        label.position.copy(mesh.position);
        scene.add(label); // Labels are always individual

        nodes[vertexName] = { 
            mesh, // Keep reference to individual mesh for position/qualia data
            label, 
            name: vertexInfo.name, 
            qualia: sheafInstance.stalks.get(vertexName) || vecZeros(sheafInstance.qDim) 
        };
    });

    sheafInstance.graph.edges.forEach(([u, v]) => {
        const uNode = nodes[u], vNode = nodes[v];
        if (!uNode || !vNode) return;
        const p1 = uNode.mesh.position;
        const p2 = vNode.mesh.position;
        const geometry = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const material = new THREE.LineBasicMaterial({ color: 0x4af, opacity: 0.5, transparent: true });
        const line = new THREE.Line(geometry, material);
        // Do NOT add individual line to scene yet, add to temp array for merging
        tempEdgeLines.push(line);
        edges.push(line); // Keep reference to individual line for opacity update
    });

    // NOW, merge the geometries
    const optimizedMeshes = optimizeScene(scene, tempNodeMeshes, tempEdgeLines);
    mergedNodesMesh = optimizedMeshes.mergedNodesMesh;
    mergedEdgesMesh = optimizedMeshes.mergedEdgesMesh;

    createAgentStateNode();
    createEmergenceCoreNode(sheafInstance);
}

function createAgentStateNode() {
    // ... (unchanged)
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshPhongMaterial({ color: 0xffaa00, emissive: 0x112200 });
    agentStateMesh = new THREE.Mesh(geometry, material);
    agentStateMesh.position.set(20, 0, 0);
    scene.add(agentStateMesh);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'concept-label';
    labelDiv.textContent = 'Agent State';
    labelDiv.style.color = '#ffaa00'; 
     agentStateLabel = new CSS2DObject(labelDiv); // Use imported CSS2DObject
     agentStateLabel.position.copy(agentStateMesh.position);
     scene.add(agentStateLabel);
}

function createEmergenceCoreNode(sheaf) {
    // ... (unchanged)
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
     emergenceCoreLabel = new CSS2DObject(labelDiv); // Use imported CSS2DObject
     emergenceCoreLabel.position.copy(emergenceCoreMesh.position);
     scene.add(emergenceCoreLabel);
}

export function updateAgentSimulationVisuals(qualiaTensor, gameState, rihScore) {
    if (!initialized || !sheafInstance) return;

    // Update merged meshes (nodes and edges)
    updateMergedMeshes(mergedNodesMesh, mergedEdgesMesh, nodes, edges, sheafInstance);

    // Update individual labels (CSS2DObjects cannot be merged) and specific node properties
    sheafInstance.graph.vertices.forEach(vertexName => {
        const node = nodes[vertexName];
        if (!node) return;
        const qualia = node.qualia;
        const intensity = clamp(norm2(qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        // Node mesh material/scale are now handled by updateMergedMeshes or directly on the merged mesh's attributes.
        // For individual visualization, we modify node.mesh's properties, which are then used by updateMergedMeshes.
        // If merged, direct material/scale update on the individual mesh objects will not reflect on the merged mesh.
        // We'll rely on updateMergedMeshes to manage the merged mesh's attributes if possible, or remove these lines.
        // For simplicity and to show the effect, we'll assume a shader could use custom attributes for per-instance effects on merged mesh.
        // For now, these lines affect the *reference* mesh, not the actual rendered merged one.

        // Update labels based on individual node properties
        node.label.element.textContent = `${node.name}\n(${intensity.toFixed(2)})`;
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
}

export function animateConceptNodes(deltaTime) {
    if (!initialized) return;
    const time = clock.getElapsedTime();

    Object.values(nodes).forEach(node => {
        // FIX: Use node.qualia for norm2
        const qualiaNorm = clamp(norm2(node.qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        // FIX: Use node.mesh.userData.idx instead of node.userData.idx
        // This position update is for the original (unmerged) mesh object;
        // if merged, these changes won't be visible unless attributes are dynamically updated in the merged geometry.
        // For performance, the merged mesh's positions usually don't animate per-instance without custom shaders.
        // Keeping this for conceptual animation, but be aware of performance implications.
        node.mesh.position.y += Math.sin(time * 2 + node.mesh.userData.idx) * qualiaNorm * deltaTime * 0.5;
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

    // Dispose merged meshes
    if (mergedNodesMesh) {
        scene.remove(mergedNodesMesh);
        mergedNodesMesh.geometry.dispose();
        if (Array.isArray(mergedNodesMesh.material)) {
            mergedNodesMesh.material.forEach(m => m.dispose());
        } else {
            mergedNodesMesh.material.dispose();
        }
    }
    if (mergedEdgesMesh) {
        scene.remove(mergedEdgesMesh);
        mergedEdgesMesh.geometry.dispose();
        if (Array.isArray(mergedEdgesMesh.material)) {
            mergedEdgesMesh.material.forEach(m => m.dispose());
        } else {
            mergedEdgesMesh.material.dispose();
        }
    }

    Object.values(nodes).forEach(n => {
        // Only remove labels as meshes are handled by mergedNodesMesh cleanup
        if (n.label) scene.remove(n.label);
        // Dispose geometries/materials of original meshes if they were created, but not added to scene.
        // If not disposing here, they might leak.
        if (n.mesh) {
            n.mesh.geometry.dispose();
            if (Array.isArray(n.mesh.material)) {
                n.mesh.material.forEach(m => m.dispose());
            } else {
                n.mesh.material.dispose();
            }
        }
    });
    // Edges' geometries/materials are handled by mergedEdgesMesh cleanup
    edges.forEach(e => {
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
