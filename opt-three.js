 // --- START OF FILE opt-three.js ---
 // --- NEW FILE: Three.js optimizations for viz-concepts.js ---
import * as THREE from 'three'; // Changed to bare specifier 'three'
import { BufferGeometryUtils } from 'three/examples/jsm/utils/BufferGeometryUtils.js'; // Changed to bare specifier for jsm utility
 import { logger, clamp, norm2 } from './utils.js'; // Also import clamp and norm2 for dynamic updates
/**
 * Optimizes Three.js scene elements by merging node and edge geometries.
 * This function is intended to be called during graph initialization in viz-concepts.js.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.Mesh[]} individualNodeMeshes - Array of individual node meshes.
 * @param {THREE.Line[]} individualEdgeLines - Array of individual edge lines.
 * @returns {{mergedNodesMesh: THREE.Mesh|null, mergedEdgesMesh: THREE.LineSegments|null}} Optimized meshes.
 */
export function optimizeScene(scene, individualNodeMeshes, individualEdgeLines) {
    let mergedNodesMesh = null;
    let mergedEdgesMesh = null;

    // Merge node geometries
    if (individualNodeMeshes.length > 0) {
        const nodeGeometries = [];
        const nodePositions = []; // Store original positions to restore labels
        
        individualNodeMeshes.forEach(mesh => {
            if (mesh && mesh.geometry) {
                mesh.updateMatrixWorld(true); // Ensure world matrix is updated for correct merging
                const geom = mesh.geometry.clone();
                geom.applyMatrix4(mesh.matrixWorld);
                nodeGeometries.push(geom);
                nodePositions.push(mesh.position.clone());
                // No need to remove from scene here, viz-concepts.js will manage
            }
        });

        if (nodeGeometries.length > 0) {
            const mergedGeometry = BufferGeometryUtils.mergeBufferGeometries(nodeGeometries);
            const nodeMaterial = new THREE.MeshPhongMaterial({ color: 0x44aaff, emissive: 0x001122 });
            mergedNodesMesh = new THREE.Mesh(mergedGeometry, nodeMaterial);
            mergedNodesMesh.name = 'mergedConceptNodes';
            scene.add(mergedNodesMesh);
            logger.info(`Merged ${nodeGeometries.length} node geometries into one mesh.`);
        } else {
            logger.warn('No valid node geometries to merge.');
        }
    }

    // Merge edge geometries
    if (individualEdgeLines.length > 0) {
        const edgeGeometries = [];
        individualEdgeLines.forEach(line => {
            if (line && line.geometry) {
                line.updateMatrixWorld(true); // Ensure world matrix is updated
                const geom = line.geometry.clone();
                geom.applyMatrix4(line.matrixWorld);
                edgeGeometries.push(geom);
                // No need to remove from scene here
            }
        });

        if (edgeGeometries.length > 0) {
            const mergedGeometry = BufferGeometryUtils.mergeBufferGeometries(edgeGeometries);
            // Note: LineBasicMaterial's opacity can be tricky with merged geometries.
            // For now, we use a single material. If per-edge opacity is needed,
            // a custom shader or breaking up merging would be required.
            const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x44aaff, opacity: 0.5, transparent: true });
            mergedEdgesMesh = new THREE.LineSegments(mergedGeometry, edgeMaterial);
            mergedEdgesMesh.name = 'mergedConceptEdges';
            scene.add(mergedEdgesMesh);
            logger.info(`Merged ${edgeGeometries.length} edge geometries into one line segment mesh.`);
        } else {
            logger.warn('No valid edge geometries to merge.');
        }
    }

    // Dispose of original geometries to free up memory after merging
    individualNodeMeshes.forEach(mesh => mesh.geometry.dispose());
    individualEdgeLines.forEach(line => line.geometry.dispose());

    return { mergedNodesMesh, mergedEdgesMesh };
}

/**
 * Updates dynamic properties of merged meshes (e.g., node colors/scales, edge opacities).
 * This function should be called in the animation loop.
 * NOTE: For per-instance dynamic properties on a merged mesh without re-merging,
 * custom shaders with attributes for color, scale, etc., would be necessary.
 * For simplicity, this current implementation primarily focuses on the merged mesh's
 * overall material properties. To truly animate individual nodes of a merged mesh,
 * a custom shader setup is required. The current `viz-concepts.js` animates individual
 * node.mesh positions (which won't affect the merged mesh's vertices directly without more work).
 * @param {THREE.Mesh} mergedNodesMesh - The merged nodes mesh.
 * @param {THREE.LineSegments} mergedEdgesMesh - The merged edges mesh.
 * @param {Object} nodes - The node dictionary from viz-concepts.js (contains original meshes & qualia).
 * @param {Array} edges - The original edge array from viz-concepts.js (contains original lines).
 * @param {Object} sheafInstance - The EnhancedQualiaSheaf instance for weights and qualia.
 */
export function updateMergedMeshes(mergedNodesMesh, mergedEdgesMesh, nodes, edges, sheafInstance) {
    if (!sheafInstance) return;

    // --- Dynamic Node Properties (Conceptual): Requires custom shader for per-instance effects ---
    // If you want individual nodes within the *mergedNodesMesh* to pulse/change color based on qualia,
    // you would need to implement a custom shader that uses attributes (e.g., `aQualiaIntensity`, `aQualiaHue`)
    // that are updated here and passed to the shader.
    // For now, we'll keep the conceptual update in viz-concepts.js for individual `node.mesh` which isn't directly rendered.
    // A simple, visible effect on the *merged* mesh would be an overall emissive color change based on average qualia activity.

    let totalQualiaNorm = 0;
    let nodeCount = 0;
    Object.values(nodes).forEach(node => {
        const qualia = node.qualia;
        if (qualia && qualia.length > 0) {
            totalQualiaNorm += norm2(qualia);
            nodeCount++;
        }
    });
    const avgQualiaIntensity = nodeCount > 0 ? clamp(totalQualiaNorm / (nodeCount * Math.sqrt(sheafInstance.qDim)), 0, 1) : 0;

    if (mergedNodesMesh && mergedNodesMesh.material) {
        // Adjust overall emissive color based on average qualia activity
        const baseColor = new THREE.Color(0x001122);
        const activeColor = new THREE.Color(0x4af); // Primary blue for active
        mergedNodesMesh.material.emissive.copy(baseColor).lerp(activeColor, avgQualiaIntensity * 0.5); // Subtle glow
    }


    // --- Dynamic Edge Properties (Opacity/Color): Also best with custom shader for per-instance effects ---
    // Similar to nodes, for per-edge opacity/color, a custom shader reading attributes is ideal.
    // Otherwise, we can only set a single opacity/color for the entire mergedEdgesMesh.
    if (mergedEdgesMesh && mergedEdgesMesh.material && sheafInstance.adjacencyMatrix) {
        let totalWeight = 0;
        let edgeCount = 0;
        sheafInstance.graph.edges.forEach(([u, v]) => {
            const uIdx = sheafInstance.graph.vertices.indexOf(u);
            const vIdx = sheafInstance.graph.vertices.indexOf(v);
            if (uIdx !== -1 && vIdx !== -1 && sheafInstance.adjacencyMatrix[uIdx] && Number.isFinite(sheafInstance.adjacencyMatrix[uIdx][vIdx])) {
                totalWeight += sheafInstance.adjacencyMatrix[uIdx][vIdx];
                edgeCount++;
            }
        });
        const avgWeight = edgeCount > 0 ? clamp(totalWeight / edgeCount, 0.1, 1.0) : 0.1;
        mergedEdgesMesh.material.opacity = clamp(avgWeight, 0.2, 0.8);
        mergedEdgesMesh.material.needsUpdate = true; // Essential for transparent materials
    }
}
