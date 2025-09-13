// --- START OF FILE opt-three.js ---
// --- NEW FILE: Three.js optimizations for viz-concepts.js ---
import * as THREE from 'three';
import { BufferGeometryUtils } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { logger } from './utils.js';

/**
 * Optimizes Three.js scene for reduced draw calls and better performance.
 * @param {THREE.Scene} scene - The scene to optimize.
 * @param {Object} nodes - Node dictionary from viz-concepts.js.
 * @param {Array} edges - Edge array from viz-concepts.js.
 * @returns {Object} Optimized { mergedNodesMesh, mergedEdgesMesh }
 */
export function optimizeScene(scene, nodes, edges) {
    // Merge node geometries
    const nodeGeometries = [];
    const nodePositions = [];
    Object.values(nodes).forEach(node => {
        const geom = node.mesh.geometry.clone();
        geom.applyMatrix4(node.mesh.matrixWorld);
        nodeGeometries.push(geom);
        nodePositions.push(node.mesh.position.clone());
        scene.remove(node.mesh); // Remove individual meshes
    });

    if (nodeGeometries.length === 0) {
        logger.warn('No node geometries to merge.');
        return { mergedNodesMesh: null, mergedEdgesMesh: null };
    }

    const mergedNodeGeometry = BufferGeometryUtils.mergeBufferGeometries(nodeGeometries);
    const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x44aaff, emissive: 0x001122 });
    const mergedNodesMesh = new THREE.Mesh(mergedNodeGeometry, nodeMaterial);
    scene.add(mergedNodesMesh);

    // Merge edge geometries
    const edgeGeometries = [];
    edges.forEach(edge => {
        const geom = edge.geometry.clone();
        edgeGeometries.push(geom);
        scene.remove(edge);
    });

    const mergedEdgeGeometry = edgeGeometries.length > 0 ? BufferGeometryUtils.mergeBufferGeometries(edgeGeometries) : null;
    let mergedEdgesMesh = null;
    if (mergedEdgeGeometry) {
        const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x44aaff, opacity: 0.5, transparent: true });
        mergedEdgesMesh = new THREE.LineSegments(mergedEdgeGeometry, edgeMaterial);
        scene.add(mergedEdgesMesh);
    }

    // Reattach labels to track merged node positions
    Object.values(nodes).forEach((node, i) => {
        if (node.label) {
            node.label.position.copy(nodePositions[i]);
            scene.add(node.label);
        }
    });

    logger.info('Scene optimized: Merged nodes and edges for reduced draw calls.');
    return { mergedNodesMesh, mergedEdgesMesh };
}

/**
 * Updates merged meshes with dynamic properties (e.g., qualia-driven scaling).
 * @param {THREE.Mesh} mergedNodesMesh - Merged nodes mesh.
 * @param {THREE.LineSegments} mergedEdgesMesh - Merged edges mesh.
 * @param {Object} nodes - Node dictionary.
 * @param {Array} edges - Original edge array.
 * @param {Object} sheafInstance - Sheaf instance for weights.
 */
export function updateMergedMeshes(mergedNodesMesh, mergedEdgesMesh, nodes, edges, sheafInstance) {
    if (!mergedNodesMesh || !sheafInstance) return;

    // Update node scales based on qualia
    const scales = new Float32Array(mergedNodesMesh.geometry.attributes.position.count / 3);
    Object.values(nodes).forEach((node, i) => {
        const qualiaNorm = clamp(norm2(node.qualia) / Math.sqrt(sheafInstance.qDim), 0, 1);
        scales[i] = 1 + qualiaNorm;
    });
    mergedNodesMesh.geometry.setAttribute('scale', new THREE.BufferAttribute(scales, 1));
    mergedNodesMesh.geometry.attributes.scale.needsUpdate = true;

    // Update edge opacities
    if (mergedEdgesMesh && sheafInstance.adjacencyMatrix) {
        const opacities = new Float32Array(edges.length);
        edges.forEach((_, idx) => {
            const [u, v] = sheafInstance.graph.edges[idx];
            const uIdx = sheafInstance.graph.vertices.indexOf(u);
            const vIdx = sheafInstance.graph.vertices.indexOf(v);
            opacities[idx] = uIdx !== -1 && vIdx !== -1 ? clamp(sheafInstance.adjacencyMatrix[uIdx][vIdx] || 0.5, 0.2, 0.8) : 0.5;
        });
        mergedEdgesMesh.geometry.setAttribute('opacity', new THREE.BufferAttribute(opacities, 1));
        mergedEdgesMesh.geometry.attributes.opacity.needsUpdate = true;
    }
}
// --- END OF FILE opt-three.js ---