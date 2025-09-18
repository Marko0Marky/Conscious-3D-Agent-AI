
/**
 * Combined and Refined Qualia Sheaf Module
 * 
 * This module integrates two provided versions of a qualia sheaf implementation.
 * It prioritizes the more detailed and robust version, which includes advanced 
 * mathematical modeling, extensive error handling, and performance optimizations 
 * using Web Workers, TensorFlow.js, and GPU.js. It then incorporates
 * specific algorithmic enhancements and more complete state management from the
 * secondary version.
 * 
 * The final code represents a complete hierarchy of sheaf-based qualia models:
 * 1.  EnhancedQualiaSheaf: The base kernel for graph-based qualia representation (Th. 1, Th. 3).
 * 2.  RecursiveTopologicalSheaf: Extends for self-awareness via fixed-point cohomology (Th. 14).
 * 3.  AdjunctionReflexiveSheaf: Extends for hierarchical awareness using categorical monads (Th. 15).
 * 4.  PersistentAdjunctionSheaf: Extends for temporal awareness via persistent homology (Th. 16).
 * 5.  FloquetPersistentSheaf: The final extension for rhythmic awareness using Floquet theory (Th. 17).
 * 
 * Dependencies: This module assumes the presence of 'utils.js' (providing math
 * functions and a logger), and optionally leverages TensorFlow.js, GPU.js,
 * Numeric.js, and THREE.js if they are available in the global scope.
 */

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity, covarianceMatrix, matVecMul, matMul, vecMul
} from './utils.js';

const Numeric = window.Numeric || null;
const GPU = window.gpu || null;
const tf = window.tf || null;
const THREE = window.THREE || null;

/**
 * A circular buffer implementation for efficient storage of historical data.
 */
export class CircularBuffer {
    constructor(capacity) {
        this.capacity = Math.max(1, Math.floor(Number.isFinite(capacity) ? capacity : 10));
        this.buffer = new Array(this.capacity).fill(null);
        this.start = 0;
        this.size = 0;
    }

    push(item) {
        if (item === null || item === undefined) {
            console.warn('CircularBuffer.push: Invalid item (null/undefined). Skipping.', { item });
            return;
        }
        const index = (this.start + this.size) % this.capacity;
        this.buffer[index] = item;
        if (this.size < this.capacity) {
            this.size++;
        } else {
            this.start = (this.start + 1) % this.capacity;
        }
    }

    get(index) {
        if (index < 0 || index >= this.size) return null;
        return this.buffer[(this.start + index) % this.capacity];
    }
    
    getAll() {
        const result = [];
        for (let i = 0; i < this.size; i++) {
            result.push(this.get(i));
        }
        return result;
    }

    [Symbol.iterator]() {
        let index = 0;
        const start = this.start;
        const size = this.size;
        const buffer = this.buffer;
        const capacity = this.capacity;
        return {
            next() {
                if (index >= size) {
                    return { done: true };
                }
                const value = buffer[(start + index) % capacity];
                index++;
                return { value, done: false };
            }
        };
    }

    get length() {
        return this.size;
    }

    clear() {
        this.buffer.fill(null);
        this.start = 0;
        this.size = 0;
    }
}

/**
 * Base EnhancedQualiaSheaf – Kernel for all theoremic extensions.
 * Th. 1: Cohomological Binding Hypothesis – Harmonic flows on sheaf Laplacian for qualia coherence.
 * Th. 3: Free Energy Qualia Principle – Sheaf priors precondition OWM gradients via KL-proxy minimization.
 */
export class EnhancedQualiaSheaf {
    constructor(graphData, config = {}) {
        this.owm = null;
        this.ready = false;
        this.entityNames = config.entityNames || ['shape', 'emotion', 'symbolic', 'synesthesia', 'metacognition', 'social', 'temporal'];
        this.qDim = this.entityNames.length;
        
        this.stateDim = config.stateDim || 13;
        
        this.alpha = clamp(config.alpha ?? 0.2, 0.01, 1);
        this.beta = clamp(config.beta ?? 0.1, 0.01, 1);
        this.gamma = clamp(config.gamma ?? 0.05, 0.01, 0.5);
        this.sigma = clamp(config.sigma ?? 0.025, 0.001, 0.1);
        this.eps = 1e-6;
        
        this.adaptation = {
            addThresh: clamp(config.addThresh ?? 0.7, 0.5, 0.95),
            removeThresh: clamp(config.removeThresh ?? 0.2, 0.05, 0.4),
            targetH1: config.targetH1 ?? 2.0,
            maxEdges: config.maxEdges ?? 50,
        };

        this._initializeGraph(graphData);

        if (!this.complex || !Array.isArray(this.complex.vertices) || this.complex.vertices.length === 0) {
            logger.warn('EnhancedQualiaSheaf constructor: Graph is still invalid after _initializeGraph. Forcing minimal fallback structure.');
            this.complex = {
                vertices: ['constructor_fallback_v1', 'constructor_fallback_v2'],
                edges: [['constructor_fallback_v1', 'constructor_fallback_v2', 0.5]]
            };
            this.simplicialComplex = { // Reset this as well for consistency
                triangles: [],
                tetrahedra: []
            };
            this.graph = this.complex;
            this.edgeSet = new Set(['constructor_fallback_v1,constructor_fallback_v2']);
            this.entityNames = this.complex.vertices.slice();
        }

        this.stalks = new Map();
        this._initializeStalks();
        
        this.stalkHistorySize = 100;
        this.correlationMatrix = zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);
        this.stalkHistory = new CircularBuffer(this.stalkHistorySize);
        
        const N_total_stalk_dim = this.graph.vertices.length * this.qDim;
        this.windowSize = Math.max(50, N_total_stalk_dim * 2);
        this.windowedStates = new CircularBuffer(this.windowSize);
        this._initializeWindowedStates(N_total_stalk_dim);
        
        this.phiHistory = new CircularBuffer(this.stalkHistorySize);
        this.gestaltHistory = new CircularBuffer(this.stalkHistorySize);
        this.inconsistencyHistory = new CircularBuffer(this.stalkHistorySize);

        this.phi = 0.2;
        this.h1Dimension = 0;
        this.gestaltUnity = 0.6;
        this.stability = 0.6;
        this.diffusionEnergy = 0;
        this.inconsistency = 0;
        this.feel_F = 0;
        this.intentionality_F = 0;
        this.cup_product_intensity = 0;
        this.structural_sensitivity = 0;
        this.coherence = 0; // Th. 1: Cohomological coherence
        
        this.adjacencyMatrix = null;
        this.laplacian = null;
        this.maxEigApprox = 1;
        this.projectionMatrices = new Map();
        this.isUpdating = false;
        this.qInput = null;
        this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim);

        this.floquetPD = { births: [], phases: [], deaths: [] };

        logger.info(`Enhanced Qualia Sheaf constructed: vertices=${this.graph.vertices.length}, edges=${this.graph.edges.length}, triangles=${this.simplicialComplex.triangles.length}, tetrahedra=${this.simplicialComplex.tetrahedra.length}`);
    }

    /**
     * A harmonic state represents an equilibrium. A simple but effective
     * implementation is to return the normalized average of all stalk states.
     */
    async computeHarmonicState() {
        const nV = this.graph.vertices.length;
        if (nV === 0 || this.stalks.size === 0) return vecZeros(this.qDim);

        let avgStalk = vecZeros(this.qDim);
        let count = 0;
        for (const stalk of this.stalks.values()) {
            if (isFiniteVector(stalk)) {
                avgStalk = vecAdd(avgStalk, stalk);
                count++;
            }
        }
        
        if (count === 0) return vecZeros(this.qDim);

        avgStalk = vecScale(avgStalk, 1 / count);
        const norm = norm2(avgStalk);
        return norm > 1e-6 ? vecScale(avgStalk, 1 / norm) : avgStalk;
    }

    _initializeGraph(graphData) {
        const safeGraphData = graphData && typeof graphData === 'object' ? graphData : {};

        const defaultVertices = ['agent_x', 'agent_z', 'agent_rot', 'target_x', 'target_z', 'vec_dx', 'vec_dz', 'dist_target'];
        const defaultEdges = [
            ['agent_x', 'agent_rot'], ['agent_z', 'agent_rot'],
            ['agent_x', 'vec_dx'], ['agent_z', 'vec_dz'],
            ['target_x', 'vec_dx'], ['target_z', 'vec_dz'],
            ['vec_dx', 'dist_target'], ['vec_dz', 'dist_target']
        ];
        const defaultTriangles = [
            ['agent_x', 'agent_z', 'agent_rot'],
            ['target_x', 'target_z', 'dist_target'],
            ['agent_x', 'target_x', 'vec_dx'],
            ['agent_z', 'target_z', 'vec_dz']
        ];
        const defaultTetrahedra = [
            ['agent_x', 'agent_z', 'target_x', 'target_z'],
            ['agent_rot', 'vec_dx', 'vec_dz', 'dist_target']
        ];

        const hasValidVertices = Array.isArray(safeGraphData.vertices) && safeGraphData.vertices.length > 0;
        const hasValidEdges = Array.isArray(safeGraphData.edges) && safeGraphData.edges.length > 0;

        const initialGraphVertices = hasValidVertices ? safeGraphData.vertices : defaultVertices;
        const initialBaseEdges = hasValidEdges ? safeGraphData.edges : defaultEdges;
        const explicitTriangles = Array.isArray(safeGraphData.triangles) ? safeGraphData.triangles : defaultTriangles;
        const explicitTetrahedra = Array.isArray(safeGraphData.tetrahedra) ? safeGraphData.tetrahedra : defaultTetrahedra;

        const allVerticesSet = new Set(initialGraphVertices);
        explicitTriangles.forEach(tri => tri.forEach(v => allVerticesSet.add(v)));
        explicitTetrahedra.forEach(tet => tet.forEach(v => allVerticesSet.add(v)));
        
        let finalVertices = Array.from(allVerticesSet);
        if (finalVertices.length === 0) {
            logger.warn('Sheaf._initializeGraph: No vertices derived from graphData or defaults. Forcing default vertices.');
            finalVertices = [...defaultVertices];
        }

        const allEdgesSet = new Set(initialBaseEdges.map(e => e.slice(0, 2).sort().join(',')));
        
        let finalTrianglesUpdated = [...explicitTriangles];
        explicitTetrahedra.forEach(tet => {
            if (!Array.isArray(tet) || tet.length !== 4) return;
            for (let i = 0; i < 4; i++) {
                const newTri = tet.filter((_, idx) => idx !== i).sort();
                if (!finalTrianglesUpdated.some(t => t.slice().sort().join(',') === newTri.join(','))) {
                    finalTrianglesUpdated.push(newTri);
                }
            }
        });

        finalTrianglesUpdated.forEach(tri => {
            if (!Array.isArray(tri) || tri.length !== 3) return;
            for (let i = 0; i < 3; i++) {
                allEdgesSet.add([tri[i], tri[(i + 1) % 3]].sort().join(','));
            }
        });

        this.complex = {
            vertices: finalVertices,
            edges: Array.from(allEdgesSet).map(s => s.split(',').concat([0.5]))
        };
        this.simplicialComplex = {
            triangles: finalTrianglesUpdated.filter(t => t.length === 3),
            tetrahedra: explicitTetrahedra.filter(t => t.length === 4)
        };
        this.edgeSet = allEdgesSet;
        this.graph = this.complex;
    }

    _initializeStalks() {
        if (!this.graph || !Array.isArray(this.graph.vertices) || this.graph.vertices.length === 0) {
            logger.error('Sheaf._initializeStalks: Graph is invalid or empty. Cannot initialize stalks.');
            return;
        }
        this.graph.vertices.forEach(v => {
            const stalk = new Float32Array(this.qDim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.5, -1, 1));
            if (!isFiniteVector(stalk)) {
                logger.error(`Non-finite stalk for vertex ${v}; setting to zeros.`);
                stalk.fill(0);
            }
            if (v.includes('vec_') || v.includes('dist_')) stalk[2] = clamp(stalk[2] * 1.5, -1, 1);
            this.stalks.set(v, stalk);
        });
    }

    _initializeWindowedStates(N_total_stalk_dim) {
        for (let i = 0; i < this.windowSize; i++) {
            const randomState = new Float32Array(N_total_stalk_dim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.1, -1, 1));
            if (!isFiniteVector(randomState)) {
                randomState.fill(0);
            }
            this.windowedStates.push(randomState);
        }
    }

    setOWM(owmInstance) {
        if (!this.owm) {
            this.owm = owmInstance;
            logger.info(`Sheaf: OWM reference set.`);
        } else {
            logger.warn(`Sheaf: Attempted to set OWM reference multiple times.`);
        }
    }

    getStalksAsVector() {
        const nV = this.graph.vertices.length;
        const expectedLength = nV * this.qDim;
        const output = new Float32Array(expectedLength);
        
        for (let i = 0; i < nV; i++) {
            const v = this.graph.vertices[i];
            const offset = i * this.qDim;
            const stalk = this.stalks.get(v);

            if (isFiniteVector(stalk) && stalk.length === this.qDim) {
                output.set(stalk, offset);
            } else {
                logger.warn(`Sheaf.getStalksAsVector: Invalid stalk for vertex ${v}; using zeros.`);
                output.fill(0, offset, offset + this.qDim);
            }
        }

        if (!isFiniteVector(output)) {
            logger.error(`Sheaf.getStalksAsVector: Assembled vector is non-finite. Returning a zero vector.`);
            return vecZeros(expectedLength);
        }
        
        return output;
    }

    isValidTriangle(tri) {
        if (!Array.isArray(tri) || tri.length !== 3) return false;
        const [a, b, c] = tri;
        if (!this.graph.vertices.includes(a) || !this.graph.vertices.includes(b) || !this.graph.vertices.includes(c)) return false;
        
        const edge_ab = [a, b].sort().join(',');
        const edge_bc = [b, c].sort().join(',');
        const edge_ca = [c, a].sort().join(',');
        
        return this.edgeSet.has(edge_ab) && this.edgeSet.has(edge_bc) && this.edgeSet.has(edge_ca);
    }

    addEdge(u, v, weight = 0.5) {
        if (!this.graph.vertices.includes(u) || !this.graph.vertices.includes(v)) {
            logger.warn(`Sheaf.addEdge: Vertex not found for ${u} or ${v}`);
            return;
        }
        const sorted = [u, v].sort();
        const key = sorted.join(',');
        if (!this.edgeSet.has(key)) {
            this.graph.edges.push([sorted[0], sorted[1], weight]);
            this.edgeSet.add(key);
            logger.info(`Added edge ${u}-${v} with weight ${(weight ?? 0).toFixed(3)}`);
        }
    }

    removeEdge(u, v) {
        const sorted = [u, v].sort();
        const key = sorted.join(',');
        const idx = this.graph.edges.findIndex(e => e.slice(0, 2).sort().join(',') === key);
        if (idx !== -1) {
            this.graph.edges.splice(idx, 1);
            this.edgeSet.delete(key);
            logger.info(`Removed edge ${u}-${v}`);
        }
    }

    addTriangle(a, b, c) {
        if (![a, b, c].every(v => this.graph.vertices.includes(v))) {
            logger.warn(`Sheaf.addTriangle: Vertex not found for ${a}, ${b}, or ${c}`);
            return;
        }
        const sorted = [a, b, c].sort();
        const key = sorted.join(',');
        if (!this.simplicialComplex.triangles.some(t => t.sort().join(',') === key)) {
            this.simplicialComplex.triangles.push(sorted);
            this.addEdge(a, b);
            this.addEdge(b, c);
            this.addEdge(c, a);
            logger.info(`Added triangle ${a}-${b}-${c}`);
        }
    }

    removeTriangle(a, b, c) {
        const sorted = [a, b, c].sort();
        const key = sorted.join(',');
        const idx = this.simplicialComplex.triangles.findIndex(t => t.sort().join(',') === key);
        if (idx !== -1) {
            this.simplicialComplex.triangles.splice(idx, 1);
            logger.info(`Removed triangle ${a}-${b}-${c}`);
        }
    }

    addTetrahedron(a, b, c, d) {
        if (![a, b, c, d].every(v => this.graph.vertices.includes(v))) {
            logger.warn(`Sheaf.addTetrahedron: Vertex not found for ${a}, ${b}, ${c}, or ${d}`);
            return;
        }
        const sorted = [a, b, c, d].sort();
        const key = sorted.join(',');
        if (!this.simplicialComplex.tetrahedra.some(t => t.sort().join(',') === key)) {
            this.simplicialComplex.tetrahedra.push(sorted);
            for (let i = 0; i < 4; i++) {
                const face = sorted.filter((_, idx) => idx !== i).sort();
                this.addTriangle(...face);
            }
            logger.info(`Added tetrahedron ${a}-${b}-${c}-${d}`);
        }
    }

    removeTetrahedron(a, b, c, d) {
        const sorted = [a, b, c, d].sort();
        const key = sorted.join(',');
        const idx = this.simplicialComplex.tetrahedra.findIndex(t => t.sort().join(',') === key);
        if (idx !== -1) {
            this.simplicialComplex.tetrahedra.splice(idx, 1);
            logger.info(`Removed tetrahedron ${a}-${b}-${c}-${d}`);
        }
    }

    async computeVertexCorrelationsFromHistory() {
        const n = this.graph.vertices.length;
        if (!(this.stalkHistory instanceof CircularBuffer)) {
            logger.warn('Sheaf.computeVertexCorrelationsFromHistory: Invalid stalkHistory. Reinitializing.');
            this.stalkHistory = new CircularBuffer(100);
        }
        if (this.stalkHistory.length < 2) {
            return identity(n);
        }

        const validHistory = this.stalkHistory.getAll().filter(isFiniteVector);
        if (validHistory.length < 2) {
            return identity(n);
        }

        if (Numeric) {
            try {
                const corr = Numeric.cor(validHistory);
                if (isFiniteMatrix(corr)) return corr;
                logger.warn('Sheaf: Numeric.js returned non-finite correlation matrix; falling back.');
            } catch (e) {
                logger.warn('Sheaf: Numeric.js correlation failed; falling back.', e);
            }
        }
    
        const covMatrix = await runWorkerTask('covarianceMatrix', { states: validHistory, eps: this.eps }, 10000);
        if (isFiniteMatrix(covMatrix)) {
            return covMatrix;
        }
        return identity(n);
    }

    async initialize() {
        if (this.ready) {
            logger.info('Sheaf.initialize: Already initialized. Skipping.');
            return;
        }
        logger.info('EnhancedQualiaSheaf.initialize() called for a robust, simplified setup.');
        try {
            // Step 1: Ensure graph, stalks, and adjacency are valid.
            if (!this.complex || !Array.isArray(this.complex.vertices) || this.complex.vertices.length === 0) {
                logger.warn('EnhancedQualiaSheaf.initialize: Complex/graph is invalid. Forcing re-initialization.');
                this._initializeGraph({});
                if (!this.complex || !this.complex.vertices || this.complex.vertices.length === 0) {
                    throw new Error('EnhancedQualiaSheaf: Failed to establish a valid graph structure.');
                }
            }
            if (!this.stalks || this.stalks.size !== this.graph.vertices.length) {
                logger.warn('EnhancedQualiaSheaf.initialize: Stalks are invalid or mismatched. Re-initializing.');
                this._initializeStalks();
            }
            if (!this.adjacency || this.adjacency.size === 0) {
                this.adjacency = new Map();
                this.graph.vertices.forEach(v => this.adjacency.set(v, new Set()));
                this.graph.edges.forEach(([v1, v2]) => {
                    this.adjacency.get(v1).add(v2);
                    this.adjacency.get(v2).add(v1);
                });
            }
             // Ensure currentCochain exists and has correct dimensions
            if (!this.currentCochain || this.currentCochain.length !== this.graph.vertices.length || (this.currentCochain[0] && this.currentCochain[0].length !== this.qDim)) {
                 this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim);
            }

            // Step 2: Initialize all projection matrices to identity.
            this.projectionMatrices = new Map();
            this.graph.edges.forEach(([u, v]) => {
                const P_identity = identity(this.qDim);
                this.projectionMatrices.set(`${u}-${v}`, P_identity);
                this.projectionMatrices.set(`${v}-${u}`, P_identity);
            });
            logger.info('Sheaf.initialize: Projection matrices set to identity.');

            // Step 3: Compute only the essential metrics needed for the first run.
            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();
            await this.computeH1Dimension();

            // Step 4: Set default values for complex metrics instead of computing them from random data.
            this.inconsistency = 0;
            this.gestaltUnity = 0.5;
            this.phi = 0.1;
            this.cup_product_intensity = 0;
            this.structural_sensitivity = 0;
            logger.info('Sheaf.initialize: Complex metrics set to default initial values.');

            this.ready = true;
            logger.info('Enhanced Qualia Sheaf ready with simplified initial metrics.');
        } catch (e) {
            logger.error('CRITICAL ERROR: EnhancedQualiaSheaf initialization failed:', e);
            this.ready = false;
            throw e;
        }
    }
    
    _csrToDense(csr) {
        if (!csr || !csr.rowPtr) return zeroMatrix(0, 0);
        const n = csr.n || (csr.rowPtr.length - 1);
        const dense = zeroMatrix(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; j++) {
                const col = csr.colIndices[j];
                const val = csr.values[j];
                dense[i][col] = val;
            }
        }
        return dense;
    }
    
    async adaptSheafTopology(adaptFreq = 100, stepCount = 0, addThresh = this.adaptation.addThresh, removeThresh = this.adaptation.removeThresh) {
        if (!this.ready || stepCount % adaptFreq !== 0) return;
        
        if (this.stalkHistory.length < this.stalkHistorySize / 2) {
            logger.info(`Sheaf: Skipping topology adaptation at step ${stepCount}; insufficient history.`);
            return;
        }

        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        if (!isFiniteMatrix(this.correlationMatrix)) {
            logger.warn('Sheaf: Non-finite correlation matrix; skipping adaptation.');
            return;
        }
        
        try {
            this.adaptEdges(this.correlationMatrix, addThresh, removeThresh);
            this.adaptSimplices(this.correlationMatrix, this.adaptation.targetH1);
            
            // --- CRITICAL FIX ---
            // After changing the graph, rebuild all dependent matrices.
            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();
            this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim);
            // --- END OF FIX ---

            await this.computeH1Dimension();
            await this._updateDerivedMetrics();
            logger.info(`Sheaf adapted at step ${stepCount}. All dependent matrices rebuilt.`);

        } catch(e) {
            logger.error(`Sheaf.adaptSheafTopology: Failed: ${e.message}`, { stack: e.stack });
        }
    }
    
    adaptEdges(corrMatrix, addThreshold, removeThreshold) {
        const numVertices = this.graph.vertices.length;
        if (numVertices === 0) return;

        const essentialEdges = new Set();
        this.simplicialComplex.triangles.forEach(tri => {
            if (!Array.isArray(tri) || tri.length < 3) return;
            essentialEdges.add([tri[0], tri[1]].sort().join(','));
            essentialEdges.add([tri[1], tri[2]].sort().join(','));
            essentialEdges.add([tri[0], tri[2]].sort().join(','));
        });

        let added = 0;
        const maxAdd = 3;
        const maxEdges = this.adaptation.maxEdges;

        const edgesToRemove = [];
        this.graph.edges.forEach(edge => {
            const u = edge[0], v = edge[1];
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            const edgeKey = [u, v].sort().join(',');
            
            if (i !== -1 && j !== -1 && corrMatrix[i]?.[j] !== undefined) {
                const corrVal = corrMatrix[i][j];
                if (corrVal < removeThreshold && !essentialEdges.has(edgeKey)) {
                    edgesToRemove.push([u, v]);
                }
            }
        });
        
        edgesToRemove.forEach(edge => this.removeEdge(edge[0], edge[1]));

        if (this.graph.edges.length < maxEdges) {
            for (let i = 0; i < numVertices && added < maxAdd; i++) {
                for (let j = i + 1; j < numVertices; j++) {
                    if (corrMatrix[i]?.[j] === undefined) continue;

                    const corrVal = corrMatrix[i][j];
                    const u = this.graph.vertices[i];
                    const v = this.graph.vertices[j];
                    const edgeKey = [u, v].sort().join(',');
                    
                    if (corrVal > addThreshold && !this.edgeSet.has(edgeKey)) {
                        const weight = clamp(corrVal * (this.gestaltUnity || 0.5), 0.1, 1.0);
                        this.addEdge(u, v, weight);
                        added++;
                    }
                }
            }
        }
    }

    adaptSimplices(corrMatrix, targetH1 = 2.0) {
        const deltaH1 = (this.h1Dimension || 0) - targetH1;
        const numV = this.graph.vertices.length;
        let changed = false;

        const protectedTetrahedra = new Set([
            ['agent_x', 'agent_z', 'target_x', 'target_z'].sort().join(','),
            ['agent_rot', 'dist_target', 'vec_dx', 'vec_dz'].sort().join(',')
        ]);

        if (deltaH1 < -0.5 && this.gestaltUnity > 0.7) {
            const maxAddTri = 2;
            let addedTri = 0;
            for (let i = 0; i < numV && addedTri < maxAddTri; i++) {
                for (let j = i + 1; j < numV; j++) {
                    for (let k = j + 1; k < numV; k++) {
                        if (corrMatrix[i]?.[j] === undefined || corrMatrix[j]?.[k] === undefined || corrMatrix[k]?.[i] === undefined) continue;

                        const c1 = corrMatrix[i][j];
                        const c2 = corrMatrix[j][k];
                        const c3 = corrMatrix[k][i];
                        const avgC = (c1 + c2 + c3) / 3;

                        const u = this.graph.vertices[i], v = this.graph.vertices[j], w = this.graph.vertices[k];
                        if (avgC > 0.8 && this.isValidTriangle([u, v, w])) {
                            const tri = [u, v, w];
                            const key = tri.sort().join(',');
                            if (!this.simplicialComplex.triangles.some(t => t.sort().join(',') === key)) {
                                this.addTriangle(...tri);
                                addedTri++;
                                changed = true;
                                logger.info(`Added high-coherence triangle (h1=${(this.h1Dimension ?? 0).toFixed(2)})`);
                                return;
                            }
                        }
                    }
                }
            }
        } else if (deltaH1 > 0.5 && this.inconsistency > 0.6) {
            const toPrune = [];
            this.simplicialComplex.triangles.forEach(tri => {
                const idxs = tri.map(v => this.graph.vertices.indexOf(v));
                if (idxs.some(id => id === -1)) return;
                if (corrMatrix[idxs[0]]?.[idxs[1]] === undefined || corrMatrix[idxs[1]]?.[idxs[2]] === undefined || corrMatrix[idxs[2]]?.[idxs[0]] === undefined) return;
                const pairCorrs = [corrMatrix[idxs[0]][idxs[1]], corrMatrix[idxs[1]][idxs[2]], corrMatrix[idxs[2]][idxs[0]]];
                if (Math.min(...pairCorrs) < 0.3) toPrune.push(tri);
            });
            if (toPrune.length > 0) {
                this.removeTriangle(...toPrune[0]);
                changed = true;
                logger.info(`Pruned low-coherence triangle (h1=${(this.h1Dimension ?? 0).toFixed(2)})`);
            }
        }

        const maxTetrahedra = 5;
        if (this.simplicialComplex.tetrahedra.length < maxTetrahedra && this.gestaltUnity > 0.8 && deltaH1 < -1) {
            for (let i = 0; i < numV; i++) {
                for (let j = i + 1; j < numV; j++) {
                    for (let k = j + 1; k < numV; k++) {
                        for (let l = k + 1; l < numV; l++) {
                            const vertices = [this.graph.vertices[i], this.graph.vertices[j], this.graph.vertices[k], this.graph.vertices[l]];
                            const vertexIndices = [i, j, k, l];
                            let corrSum = 0;
                            let pairCount = 0;
                            for (let vi1 = 0; vi1 < 4; vi1++) {
                                for (let vi2 = vi1 + 1; vi2 < 4; vi2++) {
                                    const c1 = vertexIndices[vi1], c2 = vertexIndices[vi2];
                                    if (corrMatrix[c1]?.[c2] !== undefined) {
                                        corrSum += corrMatrix[c1][c2];
                                        pairCount++;
                                    }
                                }
                            }
                            const avgPairwiseCorr = pairCount > 0 ? corrSum / pairCount : 0;

                            if (avgPairwiseCorr > 0.8 && !this.simplicialComplex.tetrahedra.some(t => t.sort().join(',') === vertices.sort().join(','))) {
                                this.addTetrahedron(...vertices);
                                changed = true;
                                logger.info(`Added high-coherence tetrahedron`);
                                return;
                            }
                        }
                    }
                }
            }
        } else if (this.simplicialComplex.tetrahedra.length > 0 && (this.gestaltUnity < 0.5 || this.inconsistency > 0.7)) {
            let minCorr = Infinity;
            let tetToRemove = null;
            this.simplicialComplex.tetrahedra.forEach(tet => {
                const tetKey = tet.sort().join(',');
                if (protectedTetrahedra.has(tetKey)) {
                    return;
                }

                const idxs = tet.map(v => this.graph.vertices.indexOf(v));
                if (idxs.some(id => id === -1)) return;
                let currentCorrSum = 0;
                let pairCount = 0;
                for (let i = 0; i < 4; i++) {
                    for (let j = i + 1; j < 4; j++) {
                        if (corrMatrix[idxs[i]]?.[idxs[j]] !== undefined) {
                            currentCorrSum += corrMatrix[idxs[i]][idxs[j]];
                            pairCount++;
                        }
                    }
                }
                const avgCorr = pairCount > 0 ? currentCorrSum / pairCount : 0;
                if (avgCorr < minCorr) {
                    minCorr = avgCorr;
                    tetToRemove = tet;
                }
            });
            if (tetToRemove) {
                this.removeTetrahedron(...tetToRemove);
                changed = true;
                logger.info(`Removed low-coherence tetrahedron`);
            }
        }

        if (changed) {
            logger.info(`Simplex adaptation applied (deltaH1=${(deltaH1 ?? 0).toFixed(2)})`);
        }
    }

    async buildBoundaryMatrices() {
        if (!this.graph.vertices.length) {
            return {
                partial1: { flatData: new Float32Array(0), rows: 0, cols: 0 },
                partial2: { flatData: new Float32Array(0), rows: 0, cols: 0 },
                partial3: { flatData: new Float32Array(0), rows: 0, cols: 0 }
            };
        }

        const vMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const eMapIndices = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        const tMapIndices = new Map(this.simplicialComplex.triangles.map((t, i) => [t.slice().sort().join(','), i]));

        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;
        const nT = this.simplicialComplex.triangles.length;
        const nTet = this.simplicialComplex.tetrahedra.length;

        const boundary1 = zeroMatrix(nE, nV);
        this.graph.edges.forEach((edge, eIdx) => {
            const [u, v] = edge;
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx === undefined || vIdx === undefined) {
                logger.warn(`Sheaf.buildBoundaryMatrices: Invalid vertex index for edge ${u}-${v} (uIdx=${uIdx}, vIdx=${vIdx})`);
                return;
            }
            boundary1[eIdx][uIdx] = 1;
            boundary1[eIdx][vIdx] = -1;
        });
            
        const partial2 = zeroMatrix(nT, nE);
        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (!Array.isArray(tri) || tri.length !== 3) return;
            const [u, v, w] = tri;
            const edges = [
                { key: [u, v].sort().join(','), sign: 1 },
                { key: [v, w].sort().join(','), sign: -1 },
                { key: [w, u].sort().join(','), sign: 1 }
            ];
            edges.forEach(({ key, sign }) => {
                const eIdx = eMapIndices.get(key);
                if (eIdx !== undefined) {
                    partial2[tIdx][eIdx] = sign;
                }
            });
        });

        const partial3 = zeroMatrix(nTet, nT);
        this.simplicialComplex.tetrahedra.forEach((tet, tetIdx) => {
            if (!Array.isArray(tet) || tet.length !== 4) return;
            const sortedTet = tet.slice().sort();
            for (let i = 0; i < 4; i++) {
                const face = sortedTet.filter((_, idx) => idx !== i).sort();
                const tIdx = tMapIndices.get(face.join(','));
                if (tIdx !== undefined) {
                    partial3[tetIdx][tIdx] = (i % 2 === 0 ? 1 : -1);
                }
            }
        });

        const safeFlatten = (matrix, name) => {
            if (!isFiniteMatrix(matrix)) {
                logger.error(`Sheaf: Non-finite matrix for ${name} boundary detected BEFORE flattening. Returning empty.`, { matrix });
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            const flattenedResult = flattenMatrix(matrix);
            if (!isFiniteVector(flattenedResult.flatData)) {
                logger.error(`Sheaf: CRITICAL: flattenMatrix for ${name} boundary produced non-finite flatData! Returning empty.`, { flattenedResult });
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            if (flattenedResult.flatData.length !== flattenedResult.rows * flattenedResult.cols) {
                logger.error(`Sheaf: CRITICAL: Flattened data length mismatch for ${name} boundary. Returning empty.`, { flattenedResult });
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            return flattenedResult;
        };

        return {
            partial1: safeFlatten(boundary1, "boundary1"),
            partial2: safeFlatten(partial2, "partial2"),
            partial3: safeFlatten(partial3, "partial3")
        };
    }

    async computeCorrelationMatrix() {
        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        const numVertices = this.graph.vertices.length;
        if (numVertices === 0) {
            this.adjacencyMatrix = zeroMatrix(0, 0);
            return;
        }

        let performanceFactor = (this.owm?.ready && Number.isFinite(this.owm.actorLoss)) ? (1 - clamp(this.owm.actorLoss / 0.5, 0, 1)) : 0;
        let performanceScalar = clamp(1 + performanceFactor, 0.5, 2.0);
        let h1Boost = 1 + clamp(this.h1Dimension / Math.max(1, numVertices / 2), 0, 1) * 0.5;

        this.adjacencyMatrix = zeroMatrix(numVertices, numVertices);
        this.graph.edges.forEach(([u, v, weight = 0.1]) => {
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i >= 0 && j >= 0) {
                const correlation = this.correlationMatrix[i]?.[j] || 0;
                const dynamicWeight = clamp(weight + 0.9 * ((1 + correlation) / 2) * performanceScalar * h1Boost, 0.01, 1.0);
                this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = dynamicWeight;
            }
        });
    }

    buildLaplacian() {
        const n = this.graph.vertices.length;
        const adj = this.adjacencyMatrix;
        if (!adj || !isFiniteMatrix(adj) || adj.length !== n) {
            logger.error('Sheaf.buildLaplacian: Invalid adjacency matrix. Returning empty Laplacian.');
            return { values: new Float32Array(0), colIndices: new Int32Array(0), rowPtr: new Int32Array(n + 1).fill(0), n };
        }
        
        const values = [];
        const colIndices = [];
        const rowPtr = new Int32Array(n + 1);
        rowPtr[0] = 0;

        for (let i = 0; i < n; i++) {
            const currentRow = [];
            let degree = 0;

            for (let j = 0; j < n; j++) {
                if (i !== j && adj[i][j] > 0) {
                    currentRow.push({ col: j, val: -adj[i][j] });
                    degree += adj[i][j];
                }
            }
            
            currentRow.push({ col: i, val: degree + this.eps });
            currentRow.sort((a, b) => a.col - b.col);

            for (const item of currentRow) {
                values.push(item.val);
                colIndices.push(item.col);
            }

            rowPtr[i + 1] = values.length;
        }

        this.laplacian = {
            values: new Float32Array(values),
            colIndices: new Int32Array(colIndices),
            rowPtr,
            n
        };

        return this.laplacian;
    }

    csrMatVecMul(csr, v) {
        const n = csr.n;
        const result = new Float32Array(n);
        if (!csr || !csr.values || !csr.colIndices || !csr.rowPtr || !isFiniteVector(v) || v.length !== n) {
            logger.warn('Sheaf.csrMatVecMul: Invalid CSR matrix or vector. Returning zero vector.');
            return vecZeros(n);
        }

        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; j++) {
                sum += (csr.values[j] || 0) * (v[csr.colIndices[j]] || 0);
            }
            result[i] = sum;
        }
        return result;
    }

    async computeProjectionMatrices() {
        const projections = new Map();
        const eta_P = 0.01;
        const lambda = 0.05;

        for (const edge of this.graph.edges) {
            const u = edge[0];
            const v = edge[1];
            let P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            const s_u = this.stalks.get(u);
            const s_v = this.stalks.get(v);

            if (!isFiniteVector(s_u) || !isFiniteVector(s_v) || s_u.length !== this.qDim || s_v.length !== this.qDim) {
                logger.warn(`Sheaf.computeProjectionMatrices: Invalid stalk for edge ${u}-${v}. Using identity.`);
                projections.set(`${u}-${v}`, identity(this.qDim));
                projections.set(`${v}-${u}`, identity(this.qDim));
                continue;
            }

            if (!isFiniteMatrix(P_uv) || P_uv.length !== this.qDim || P_uv[0].length !== this.qDim) {
                P_uv = identity(this.qDim);
            }

            try {
                const flattened_P_uv = flattenMatrix(P_uv);
                if (!isFiniteVector(flattened_P_uv.flatData)) {
                    throw new Error("Matrix P_uv for matVecMul is non-finite.");
                }

                const projected = await runWorkerTask('matVecMul', { matrix: flattened_P_uv, vector: s_u }, 5000);

                if (!isFiniteVector(projected)) {
                    throw new Error("Worker returned non-finite projected vector.");
                }

                const error = vecSub(projected, s_v);
                const grad = Array.from({ length: this.qDim }, () => new Float32Array(this.qDim));

                for (let i = 0; i < this.qDim; i++) {
                    for (let j = 0; j < this.qDim; j++) {
                        grad[i][j] = error[i] * s_u[j];
                    }
                }

                const reg_grad = P_uv.map((row, i) => row.map((val, j) => 2 * lambda * (val - (i === j ? 1 : 0))));
                let updated_P_uv = P_uv.map((row, i) => row.map((val, j) => 
                    clamp(val - eta_P * (grad[i][j] + reg_grad[i][j]), -1, 1)
                ));

                if (!isFiniteMatrix(updated_P_uv)) {
                    logger.warn(`Sheaf.computeProjectionMatrices: Non-finite projection matrix for ${u}-${v}. Using identity.`);
                    updated_P_uv = identity(this.qDim);
                }

                projections.set(`${u}-${v}`, updated_P_uv);
                projections.set(`${v}-${u}`, updated_P_uv);

            } catch (e) {
                logger.warn(`Sheaf.computeProjectionMatrices: Error for edge ${u}-${v}: ${e.message}. Using identity.`, { stack: e.stack });
                projections.set(`${u}-${v}`, identity(this.qDim));
                projections.set(`${v}-${u}`, identity(this.qDim));
            }
        }
        this.projectionMatrices = projections;
        return projections;
    }

    /**
     * Th. 3: Full free energy prior method.
     * Computes a prior for OWM action selection based on sheaf coherence and KL-divergence proxy.
     */
    async computeFreeEnergyPrior(state, hiddenState) {
        const actionDim = (this.owm && this.owm.actionDim) ? this.owm.actionDim : 4;
        const prior = vecZeros(actionDim);
        const coherenceBonus = (this.coherence || 0) * 0.05;
        const inconsistencyPenalty = (this.inconsistency || 0) * -0.1;

        for (let i = 0; i < actionDim; i++) {
            prior[i] = coherenceBonus + inconsistencyPenalty;
        }

        if (!isFiniteVector(prior)) {
            logger.warn('computeFreeEnergyPrior generated a non-finite vector. Returning zeros.');
            return vecZeros(actionDim);
        }

        return prior;
    }

    async _updateGraphStructureAndMetrics() {
        if (this.isUpdating) return;
        this.isUpdating = true;
        try {
            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();
            
            if (this.laplacian && this.laplacian.values.length > 0) {
                const denseLaplacian = this._csrToDense(this.laplacian);
                if (!isFiniteMatrix(denseLaplacian) || denseLaplacian.length === 0) {
                    logger.error('Sheaf._updateGraphStructureAndMetrics: Non-finite or empty dense Laplacian. Skipping spectral norm.');
                    this.maxEigApprox = 1;
                } else {
                    this.maxEigApprox = await runWorkerTask('matrixSpectralNormApprox', { matrix: flattenMatrix(denseLaplacian) }, 15000);
                }
            } else {
                this.maxEigApprox = 1;
            }

            if (!Number.isFinite(this.maxEigApprox) || this.maxEigApprox <= 0) {
                this.maxEigApprox = 1;
            }

            this.projectionMatrices = await this.computeProjectionMatrices();
        } catch (e) {
            logger.error('Sheaf._updateGraphStructureAndMetrics: Failed to update', e);
            throw e;
        } finally {
            this.isUpdating = false;
        }
    }

    async computeCoherenceFlow(dt = 0.01) {
        if (!this.laplacian || !this.laplacian.values || this.laplacian.values.length === 0 || !Number.isFinite(this.laplacian.values[0])) {
            logger.warn('Sheaf.computeCoherenceFlow: Invalid or uninitialized Laplacian. Rebuilding.');
            this.laplacian = this.buildLaplacian();
            if (!this.laplacian || this.laplacian.values.length === 0) {
                 logger.error('Sheaf.computeCoherenceFlow: Rebuilding Laplacian failed. Aborting coherence flow.');
                 this.coherence = 0;
                 return 0;
            }
        }

        const nV = this.graph.vertices.length;
        const vertexInput = this.qInput || vecZeros(nV);

        if (!this.currentCochain || !isFiniteMatrix(this.currentCochain) || this.currentCochain.length !== nV || this.currentCochain[0]?.length !== this.qDim) {
            this.currentCochain = zeroMatrix(nV, this.qDim);
            logger.warn('Sheaf.computeCoherenceFlow: currentCochain was invalid, re-initialized to zeros.');
        }

        const nextCochain = zeroMatrix(nV, this.qDim);

        for (let q = 0; q < this.qDim; q++) {
            const singleDimVector = new Float32Array(nV);
            for (let i = 0; i < nV; i++) {
                singleDimVector[i] = this.currentCochain[i][q];
            }

            const laplacianTerm = this.csrMatVecMul(this.laplacian, singleDimVector);
            const inputTerm = vecScale(vertexInput, this.beta * dt);
            const flowUpdate = vecAdd(vecScale(laplacianTerm, -this.alpha * dt), inputTerm);
            const nextDimVector = vecAdd(singleDimVector, flowUpdate);
            
            for (let i = 0; i < nV; i++) {
                nextCochain[i][q] = nextDimVector[i];
            }
        }

        if (!isFiniteMatrix(nextCochain)) {
             logger.error('Sheaf.computeCoherenceFlow: Non-finite nextCochain after flow update. Resetting to zeros.');
             this.currentCochain = zeroMatrix(nV, this.qDim);
        } else {
            this.currentCochain = nextCochain;
        }

        const flatNextCochain = flattenMatrix(this.currentCochain).flatData;
        this.coherence = clamp(norm2(flatNextCochain) / Math.sqrt(nV * this.qDim), 0, 1);
        return this.coherence;
    }

    async diffuseQualia(state) {
        if (!this.ready) {
            logger.warn('Sheaf not ready for diffusion. Skipping.');
            return;
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn(`Sheaf.diffuseQualia: Invalid input state. Skipping. Expected length ${this.stateDim}, but got ${state?.length}.`, {state});
            return;
        }

        try {
            await this._updateGraphStructureAndMetrics();
        } catch (e) {
            logger.error('Sheaf.diffuseQualia: Error updating graph structure:', e);
            this.ready = false;
            return;
        }

        const n = this.graph.vertices.length;
        const N = n * this.qDim;
        const s = this.getStalksAsVector();
        if (!isFiniteVector(s) || s.length !== N) {
            logger.error(`Sheaf.diffuseQualia: Non-finite or dimension-mismatched initial stalk vector "s". Aborting diffusion. Expected length ${N}, got ${s.length}.`);
            return;
        }
        
        const qInput = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            qInput[i] = state[Math.min(i, state.length - 1)] || 0;
        }

        // Pass the qInput to the coherence flow calculation.
        this.qInput = qInput;
        await this.computeCoherenceFlow();

        const Lfull = zeroMatrix(N, N);
        const idxMap = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) {
            const i = idxMap.get(u), j = idxMap.get(v);
            if (i === undefined || j === undefined) continue;
            const weight = this.adjacencyMatrix[i]?.[j];
            if (!Number.isFinite(weight) || weight <= 0) continue;

            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            if (!isFiniteMatrix(P_uv) || P_uv.length !== this.qDim || P_uv[0].length !== this.qDim) {
                logger.warn(`Sheaf.diffuseQualia: Invalid P_uv for edge ${u}-${v}. Skipping contribution.`);
                continue;
            }

            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    let val = -weight * (P_uv[qi]?.[qj] || 0);
                    if (qi !== qj) val += 0.1 * Math.sin(qi - qj) * weight;
                    if (Number.isFinite(val)) {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = clamp(val, -100, 100);
                        Lfull[j * this.qDim + qi][i * this.qDim + qj] = clamp(val, -100, 100);
                    }
                }
            }
        }

        for (let i = 0; i < n; i++) {
            let degree = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j && Number.isFinite(this.adjacencyMatrix[i]?.[j])) {
                    degree += this.adjacencyMatrix[i][j];
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                Lfull[i * this.qDim + qi][i * this.qDim + qi] = clamp(degree + this.eps, -100, 100);
            }
        }

        let eta = this.gamma / Math.max(1, this.maxEigApprox);
        if (!Number.isFinite(eta)) eta = 0.01;

        const f_s = new Float32Array(N);
        for (let i = 0; i < n; i++) {
            for (let qi = 0; qi < this.qDim; qi++) {
                f_s[i * this.qDim + qi] = this.alpha * (qInput[i] || 0) * 2.0;
            }
        }

        const A = zeroMatrix(N, N).map((row, i) => row.map((v, j) => {
            const val = (i === j ? 1 : 0) - eta * (Lfull[i][j] || 0);
            return Number.isFinite(val) ? clamp(val, -100, 100) : 0;
        }));

        const noise = new Float32Array(N).map(() => (Math.random() - 0.5) * this.sigma);
        const rhs = vecAdd(vecAdd(s, vecScale(f_s, eta)), vecScale(noise, Math.sqrt(eta)));

        if (!isFiniteMatrix(A) || A.length !== N || !isFiniteVector(rhs) || rhs.length !== N) {
            logger.error(`Sheaf.diffuseQualia: Non-finite or dimension-mismatched A/RHS for linear solver. Skipping.`);
            return;
        }

        let sSolved;
        if (GPU) {
            try {
                const gpu = new GPU();
                logger.warn('Sheaf.diffuseQualia: GPU.js path is a placeholder. Falling back to CPU worker for CG solve.');
                sSolved = await runWorkerTask('solveLinearSystemCG', {
                    A: flattenMatrix(A),
                    b: rhs,
                    opts: { tol: 1e-6, maxIter: 15, preconditioner: 'diagonal' }
                }, 10000);
                gpu.destroy();
            } catch (e) {
                logger.warn('Sheaf.diffuseQualia: GPU.js failed; falling back to CPU worker for CG solve.', e);
                sSolved = await runWorkerTask('solveLinearSystemCG', {
                    A: flattenMatrix(A),
                    b: rhs,
                    opts: { tol: 1e-6, maxIter: 15, preconditioner: 'diagonal' }
                }, 10000);
            }
        } else {
            sSolved = await runWorkerTask('solveLinearSystemCG', {
                A: flattenMatrix(A),
                b: rhs,
                opts: { tol: 1e-6, maxIter: 15, preconditioner: 'diagonal' }
            }, 10000);
        }

        const sNext = isFiniteVector(sSolved) ? new Float32Array(sSolved.map(v => clamp(v, -1, 1))) : vecZeros(N);
        
        this._updateStalksAndWindow(sNext, n, qInput);
        await this._updateDerivedMetrics();
    }

    _updateStalksAndWindow(sNextVector, n, qInput) {
        const currentStalkNorms = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const rawStalkSlice = sNextVector.slice(i * this.qDim, (i + 1) * this.qDim);
            const sanitizedStalk = new Float32Array(rawStalkSlice.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
            this.stalks.set(this.graph.vertices[i], sanitizedStalk);
            currentStalkNorms[i] = norm2(sanitizedStalk);
        }
        
        this.stalkHistory.push(currentStalkNorms);
        this.windowedStates.push(new Float32Array(sNextVector));
        this.qInput = qInput;
    }
    
    async computeGluingInconsistency() {
        let sum = 0;
        let edgeCount = 0;
        for (const [u, v] of this.graph.edges) {
            const stalk_u = this.stalks.get(u);
            const stalk_v = this.stalks.get(v);
            let P_uv = this.projectionMatrices.get(`${u}-${v}`);

            if (!isFiniteVector(stalk_u) || !isFiniteVector(stalk_v) || stalk_u.length !== this.qDim || stalk_v.length !== this.qDim) {
                logger.warn(`Sheaf.computeGluingInconsistency: Invalid stalk for edge ${u}-${v}. Skipping.`);
                continue;
            }

            if (!isFiniteMatrix(P_uv) || P_uv.length !== this.qDim || P_uv[0].length !== this.qDim) {
                logger.warn(`Sheaf.computeGluingInconsistency: Invalid P_uv for ${u}-${v}. Using identity.`);
                P_uv = identity(this.qDim);
            }
            const flattened_P_uv = flattenMatrix(P_uv);
            if (!isFiniteVector(flattened_P_uv.flatData) || flattened_P_uv.cols !== stalk_u.length) {
                logger.warn(`Sheaf.computeGluingInconsistency: Invalid data for matVecMul worker for ${u}-${v}. Skipping.`);
                continue;
            }
            
            let projected_u = await runWorkerTask('matVecMul', { matrix: flattened_P_uv, vector: stalk_u }, 5000);
            
            if (!isFiniteVector(projected_u)) {
                logger.warn(`Sheaf.computeGluingInconsistency: Worker returned non-finite projected vector for ${u}-${v}. Skipping.`);
                continue;
            }
            
            const diffNorm = norm2(vecSub(projected_u, stalk_v));
            if (Number.isFinite(diffNorm)) {
                sum += clamp(diffNorm, 0, 5);
                edgeCount++;
            }
        }
        this.inconsistency = edgeCount > 0 ? clamp(sum / edgeCount, 0, 5) : 0;
        this.inconsistencyHistory.push(new Float32Array([this.inconsistency]));
        return this.inconsistency;
    }

    async computeGestaltUnity() {
        let validStates = [];
        try {
            validStates = this.windowedStates.getAll().filter(isFiniteVector);
        } catch (e) {
            logger.error(`Sheaf.computeGestaltUnity: Error retrieving validStates: ${e.message}`, { stack: e.stack });
            validStates = [];
        }

        if (validStates.length < 2) {
            this.gestaltUnity = 0;
            this.gestaltHistory.push(new Float32Array([this.gestaltUnity]));
            return 0;
        }

        let totalSimilarity = 0;
        let count = 0;

        try {
            if (GPU) {
                try {
                    const gpu = new GPU();
                    const computeSimilarity = gpu.createKernel(function(flattenedStalks, eps, qDim_const, numStalks_const) {
                        let sum = 0;
                        for (let i = 0; i < numStalks_const; i++) {
                            for (let j = i + 1; j < numStalks_const; j++) {
                                let dotProd = 0;
                                let norm1_sq = 0;
                                let norm2_sq = 0;
                                for (let k = 0; k < qDim_const; k++) {
                                    dotProd += flattenedStalks[i * qDim_const + k] * flattenedStalks[j * qDim_const + k];
                                    norm1_sq += flattenedStalks[i * qDim_const + k] * flattenedStalks[i * qDim_const + k];
                                    norm2_sq += flattenedStalks[j * qDim_const + k] * flattenedStalks[j * qDim_const + k];
                                }
                                const norm1 = Math.sqrt(norm1_sq);
                                const norm2 = Math.sqrt(norm2_sq);
                                if (norm1 > eps && norm2 > eps) {
                                    sum += Math.abs(dotProd / (norm1 * norm2));
                                }
                            }
                        }
                        return sum;
                    }).setOutput([1]).setConstants({ qDim_const: this.qDim, numStalks_const: validStates.length });

                    const flattenedStalks = new Float32Array(validStates.flat());
                    if (!isFiniteVector(flattenedStalks)) {
                        throw new Error('Flattened stalks for GPU.js are non-finite.');
                    }
                    totalSimilarity = computeSimilarity(flattenedStalks, this.eps);
                    count = (validStates.length * (validStates.length - 1)) / 2;
                    gpu.destroy();
                } catch (e) {
                    logger.warn('Sheaf.computeGestaltUnity: GPU.js failed; falling back to CPU.', e);
                    count = 0;
                    totalSimilarity = 0;
                }
            }

            if (count === 0) { // If GPU.js failed or wasn't used
                for (let i = 0; i < validStates.length; i++) {
                    for (let j = i + 1; j < validStates.length; j++) {
                        const n1 = norm2(validStates[i]);
                        const n2 = norm2(validStates[j]);
                        if (n1 > this.eps && n2 > this.eps) {
                            const similarity = Math.abs(dot(validStates[i], validStates[j]) / (n1 * n2));
                            if (Number.isFinite(similarity)) {
                                totalSimilarity += similarity;
                                count++;
                            }
                        }
                    }
                }
            }
        } catch (e) {
            logger.error(`Sheaf.computeGestaltUnity: Error computing Gestalt Unity: ${e.message}`, { stack: e.stack });
            totalSimilarity = 0;
            count = 0;
        }
        
        this.gestaltUnity = count > 0 ? clamp(totalSimilarity / count, 0, 1) : 0;
        this.gestaltHistory.push(new Float32Array([this.gestaltUnity]));
        return this.gestaltUnity;
    }

    _computeLogDet(A) {
        const n = A.length;
        if (n === 0) return 0;

        const L = zeroMatrix(n, n);
        const U = zeroMatrix(n, n);
        for (let i = 0; i < n; i++) L[i][i] = 1;

        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                let sum = A[i][j];
                for (let k = 0; k < i; k++) {
                    sum -= L[i][k] * U[k][j];
                }
                U[i][j] = sum;
            }
            for (let j = i + 1; j < n; j++) {
                let sum = A[j][i];
                for (let k = 0; k < i; k++) {
                    sum -= L[j][k] * U[k][i];
                }
                const U_ii_abs = Math.abs(U[i][i]);
                if (U_ii_abs < this.eps * 10) {
                    logger.warn(`Sheaf._computeLogDet: Division by near-zero diagonal element in U at i=${i}. Returning 0.`);
                    return 0;
                }
                L[j][i] = sum / U[i][i];
            }
        }

        let logDet = 0;
        for (let i = 0; i < n; i++) {
            const U_ii = U[i][i];
            if (!Number.isFinite(U_ii) || U_ii <= this.eps * 10) {
                logger.warn(`Sheaf._computeLogDet: Non-finite or non-positive diagonal element for log-det at i=${i}. Returning 0.`);
                return 0;
            }
            logDet += Math.log(U_ii);
        }
        return logDet;
    }

    // In qualia-sheaf.js, inside the EnhancedQualiaSheaf class...

    async _computeDirectionalMI(states) {
        if (!Array.isArray(states) || states.length < 20) {
            // Not enough data for a reliable estimate
            return 0;
        }

        const n_dim = states[0].length;
        const n_half = Math.floor(n_dim / 2);
        if (n_half < 1) return 0;

        // Prepare time-lagged vectors for Transfer Entropy calculation I(Y_future; X_past | Y_past)
        const Y_future = [];
        const Y_past = [];
        const X_past = [];
        const YX_past = [];

        for (let i = 0; i < states.length - 1; i++) {
            const s_t = states[i];
            const s_t1 = states[i+1];

            const x_t = s_t.slice(0, n_half);
            const y_t = s_t.slice(n_half);
            const y_t1 = s_t1.slice(n_half);

            Y_future.push(y_t1);
            Y_past.push(y_t);
            X_past.push(x_t);
            YX_past.push(new Float32Array([...y_t, ...x_t]));
        }

        try {
            // Calculate I(Y_future; [Y_past, X_past]) and I(Y_future; Y_past)
            const [mi_full, mi_partial] = await Promise.all([
                runWorkerTask('ksgMutualInformation', { states: Y_future.map((yf, i) => new Float32Array([...yf, ...YX_past[i]])), k: 3 }, 15000),
                runWorkerTask('ksgMutualInformation', { states: Y_future.map((yf, i) => new Float32Array([...yf, ...Y_past[i]])), k: 3 }, 15000)
            ]);

            const transferEntropy = (mi_full || 0) - (mi_partial || 0);

            // Return a non-negative, clamped value.
            return clamp(transferEntropy, 0, 10);

        } catch (e) {
            logger.warn(`_computeDirectionalMI: Worker task failed. ${e.message}`);
            return 0;
        }
    }

    async _computeCochains() {
        if (!this.ready || this.stalks.size === 0) {
            return { C0: [], C1: new Map(), C2: new Map() };
        }

        // C0: 0-cochains (data on vertices)
        const C0 = Array.from(this.stalks.values());

        // C1: 1-cochains (data on edges)
        const C1 = new Map();
        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const s_u = this.stalks.get(u);
            const s_v = this.stalks.get(v);
            const P_vu = this.projectionMatrices.get(`${v}-${u}`) || identity(this.qDim);

            if (!isFiniteVector(s_u) || !isFiniteVector(s_v)) continue;
            
            const projected_u = matVecMul(P_vu, s_u);
            const difference = vecSub(s_v, projected_u);
            
            if (isFiniteVector(difference)) {
                C1.set([u,v].sort().join(','), difference);
            }
        }

        // C2: 2-cochains (data on triangles)
        const C2 = new Map();
        for (const tri of this.simplicialComplex.triangles) {
            const [u, v, w] = tri;
            
            const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            const c_vw = C1.get([v, w].sort().join(',')) || vecZeros(this.qDim);
            const c_wu = C1.get([w, u].sort().join(',')) || vecZeros(this.qDim); // Note the order for the cycle

            // Coboundary d(C1) = c_uv + c_vw + c_wu (with appropriate signs from projections)
            // This simplifies to the sum of differences around the loop.
            let curl = vecAdd(c_uv, c_vw);
            curl = vecAdd(curl, c_wu);

            if (isFiniteVector(curl)) {
                C2.set([u,v,w].sort().join(','), curl);
            }
        }

        return { C0, C1, C2 };
    }

    async computeIntegratedInformation() {
        let MI = 0;
        let validStates = [];
        try {
            validStates = this.windowedStates.getAll().filter(isFiniteVector);
            if (validStates.length < this.windowSize / 4) {
                MI = 0;
            } else {
                const n_dim = validStates[0].length;
                const num_samples = validStates.length;

                if (num_samples <= n_dim) {
                    MI = 0;
                } else {
                    const tfVersion = tf?.version?.core || '0.0.0';
                    const useTF = tf && parseFloat(tfVersion) >= 2.0 && tf.linalg?.determinant;
                    if (useTF) {
                        try {
                            const statesArray = validStates.map(s => Array.from(s));
                            if (!statesArray.every(s => isFiniteVector(s))) {
                                throw new Error("Non-finite states detected in TensorFlow input.");
                            }
                            const statesTensor = tf.tensor2d(statesArray);
                            const regularizer = tf.eye(n_dim).mul(this.eps * 100);
                            const rawCovMatrix = tf.matMul(statesTensor.transpose(), statesTensor).div(num_samples);
                            const covMatrix = rawCovMatrix.add(regularizer);
                            
                            let logDet;
                            try {
                                const L = tf.linalg.cholesky(covMatrix);
                                logDet = tf.sum(L.log().mul(2)).dataSync()[0];
                            } catch (e) {
                                logDet = tf.linalg.logMatrixDeterminant(covMatrix).logDeterminant.dataSync()[0];
                            }
                            MI = Number.isFinite(logDet) ? 0.1 * Math.abs(logDet) + this.eps : 0;
                            tf.dispose([statesTensor, rawCovMatrix, covMatrix, L]);
                        } catch (e) {
                            logger.warn(`Sheaf.computeIntegratedInformation: TensorFlow.js failed: ${e.message}`, { stack: e.stack });
                        }
                    }

                    if (MI === 0) { // CPU Fallback
                        const mean = new Float32Array(n_dim).fill(0);
                        validStates.forEach(state => {
                            for (let i = 0; i < n_dim; i++) mean[i] += state[i] / num_samples;
                        });
                        const covMatrix = zeroMatrix(n_dim, n_dim);
                        validStates.forEach(state => {
                            const centered = vecSub(state, mean);
                            for (let i = 0; i < n_dim; i++) {
                                for (let j = i; j < n_dim; j++) {
                                    covMatrix[i][j] += centered[i] * centered[j] / (num_samples - 1);
                                    if (i !== j) covMatrix[j][i] = covMatrix[i][j];
                                }
                            }
                        });

                        if (isFiniteMatrix(covMatrix)) {
                            const regularizedCovMatrix = covMatrix.map((row, i) =>
                                new Float32Array(row.map((val, j) => (i === j) ? (val + this.eps * 100) : val))
                            );
                            if (isFiniteMatrix(regularizedCovMatrix)) {
                                MI = 0.1 * Math.abs(this._computeLogDet(regularizedCovMatrix)) + this.eps;
                            } else {
                                MI = 0;
                            }
                        } else {
                           MI = 0;
                        }
                    }
                }

                if (MI === 0) { // Final fallback to KSG worker
                    if (Array.isArray(validStates) && validStates.length > 0 && isFiniteVector(validStates[0])) {
                        const ksgMI = await runWorkerTask('ksgMutualInformation', { states: validStates, k: 3 }, 20000);
                        MI = Number.isFinite(ksgMI) ? ksgMI : 0;
                    } else {
                        MI = 0;
                    }
                }
            }
        } catch (e) {
            logger.error(`Sheaf.computeIntegratedInformation: Error: ${e.message}`, { stack: e.stack });
            MI = 0;
        }

        const safeFloquetBirths = Array.isArray(this.floquetPD.births) ? this.floquetPD.births : [];
        const betaFloq = await this._supFloqBetti(this.floquetPD);
        const avgBirthTime = safeFloquetBirths.reduce((sum, b) => sum + (b.time || 0), 0) / Math.max(1, safeFloquetBirths.length);
        const persistenceBoost = 0.05 * betaFloq * Math.log(1 + (this.stalkHistory.length || 1) / Math.max(1, avgBirthTime + 1));
        
        const phiRaw = (Math.log(1 + Math.abs(MI)) + persistenceBoost) * this.stability * this.gestaltUnity * Math.exp(-this.inconsistency) * (1 + 0.05 * this.h1Dimension);
        this.phi = clamp(phiRaw, 0.001, 100);
        this.phiHistory.push(new Float32Array([this.phi]));
        
        this.feelIntensity = clamp((MI + 0.02 * betaFloq) * this.stability * Math.exp(-this.inconsistency), 0.001, 10);
        
        const MI_dir = await this._computeDirectionalMI(validStates); 
        this.intentionality = clamp((MI_dir + 0.01 * betaFloq) * this.stability * Math.exp(-this.inconsistency), 0.001, 10);
        
        return this.phi;
    }

    async computeCupProduct() {
        const edges = this.graph.edges;
        if (!edges || edges.length < 2) return 0;
        
        let totalIntensity = 0;
        let count = 0;

        try {
            for (const triangle of this.simplicialComplex.triangles) {
                if (!this.isValidTriangle(triangle)) continue;
                const [u, v, w] = triangle;
                const s_u = this.stalks.get(u), s_v = this.stalks.get(v), s_w = this.stalks.get(w);
                if (!isFiniteVector(s_u) || !isFiniteVector(s_v) || !isFiniteVector(s_w) ||
                    s_u.length !== this.qDim || s_v.length !== this.qDim || s_w.length !== this.qDim) {
                    continue;
                }

                let P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
                let P_vw = this.projectionMatrices.get(`${v}-${w}`) || identity(this.qDim);

                if (!isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vw)) continue;

                try {
                    const P_compose = matMul({matrixA: P_uv, matrixB: P_vw});

                    if (!isFiniteMatrix(P_compose)) continue;

                    const s_w_projected = matVecMul(P_compose, s_w);

                    if (!isFiniteVector(s_w_projected)) continue;

                    const cupValue = dot(s_u, s_w_projected);
                    if (Number.isFinite(cupValue)) {
                        totalIntensity += Math.abs(cupValue);
                        count++;
                    }
                } catch (e) {
                    logger.warn(`Sheaf.computeCupProduct: Error for triangle ${triangle}: ${e.message}`, { stack: e.stack });
                }
            }
            
            const betaFloq = await this._supFloqBetti(this.floquetPD);
            const cupProduct = (count > 0 ? totalIntensity / count : 0.001) + 0.01 * betaFloq;
            this.cup_product_intensity = clamp(cupProduct, 0.001, 10);
            return this.cup_product_intensity;
        } catch (e) {
            logger.error(`Sheaf.computeCupProduct: Error: ${e.message}.`, { stack: e.stack });
            this.cup_product_intensity = 0.001;
            return this.cup_product_intensity;
        }
    }

    async computeH1Dimension() {
        try {
            const cycles = this._computeBetti1UnionFind();
            this.h1Dimension = clamp(cycles, 0, this.graph.edges.length);
        } catch (e) {
            logger.error(`Sheaf.computeH1Dimension: Error during Union-Find: ${e.message}`, { stack: e.stack });
            this.h1Dimension = clamp(this.graph.edges.length - this.graph.vertices.length + 1, 0, this.graph.edges.length);
        }

        if (!Number.isFinite(this.h1Dimension)) {
            this.h1Dimension = clamp(this.graph.edges.length - this.graph.vertices.length + 1, 0, this.graph.edges.length);
        }

        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        return this.h1Dimension;
    }

    _buildIncidenceMatrix() {
        const n = this.graph.vertices.length;
        const m = this.graph.edges.length;
        const matrix = zeroMatrix(n, m);
        this.graph.edges.forEach(([u, v], j) => {
            const uIdx = this.graph.vertices.indexOf(u);
            const vIdx = this.graph.vertices.indexOf(v);
            if (uIdx !== -1 && vIdx !== -1) {
                matrix[uIdx][j] = 1;
                matrix[vIdx][j] = -1;
            }
        });
        return matrix;
    }

    _computeBetti1UnionFind() {
        const vertices = this.graph.vertices;
        const edges = this.graph.edges;
        const n = vertices.length;
        const m = edges.length;

        if (n === 0) return 0;

        const parent = Array(n).fill().map((_, i) => i);
        const rank = Array(n).fill(0);

        const find = (x) => {
            if (parent[x] !== x) parent[x] = find(parent[x]);
            return parent[x];
        };

        const union = (x, y) => {
            const px = find(x);
            const py = find(y);
            if (px === py) return;
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else {
                parent[py] = px;
                if (rank[px] === rank[py]) rank[px]++;
            }
        };

        const vMap = new Map(vertices.map((v, i) => [v, i]));
        edges.forEach(([u, v]) => {
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx !== undefined && vIdx !== undefined) {
                union(uIdx, vIdx);
            }
        });

        const components = new Set(parent.map((_, i) => find(i))).size;
        return Math.max(0, m - n + components);
    }

    async simulateDiffusionStep(s_t) {
        const N = this.graph.vertices.length * this.qDim;
        const Lfull = zeroMatrix(N, N);
        const idx = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const eta = this.gamma / Math.max(1, this.maxEigApprox);

        for (const [u, v] of this.graph.edges) {
            const i = idx.get(u), j = idx.get(v);
            if (i === undefined || j === undefined) continue;
            const weight = this.adjacencyMatrix[i]?.[j] || 0;
            if (!Number.isFinite(weight) || weight <= 0) continue;
            let P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            if (!isFiniteMatrix(P_uv)) P_uv = identity(this.qDim);
            
            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    let val = -weight * (P_uv[qi]?.[qj] || 0);
                    if (qi !== qj) val += 0.1 * Math.sin(qi - qj) * weight;
                    if (Number.isFinite(val)) {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = clamp(val, -100, 100);
                        Lfull[j * this.qDim + qi][i * this.qDim + qj] = clamp(val, -100, 100);
                    }
                }
            }
        }

        for (let i = 0; i < this.graph.vertices.length; i++) {
            let degree = 0;
            for (let j = 0; j < this.graph.vertices.length; j++) {
                if (i !== j && Number.isFinite(this.adjacencyMatrix[i]?.[j])) {
                    degree += this.adjacencyMatrix[i][j];
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                Lfull[i * this.qDim + qi][i * this.qDim + qi] = clamp(degree + this.eps, -100, 100);
            }
        }

        const A = zeroMatrix(N, N).map((row, i) => row.map((v, j) => {
            const val = (i === j ? 1 : 0) - eta * (Lfull[i][j] || 0);
            return Number.isFinite(val) ? clamp(val, -100, 100) : 0;
        }));

        if (!isFiniteMatrix(A) || A.length !== N || !isFiniteVector(s_t) || s_t.length !== N) {
            logger.error(`Sheaf.simulateDiffusionStep: Invalid A or s_t. Returning zero vector.`);
            return vecZeros(N);
        }
        
        return matVecMul(A, s_t);
    }

    async computeStructuralSensitivity(perturbationScale = 0.05) {
        if (!this.ready || this.graph.edges.length === 0) {
            this.structural_sensitivity = 0;
            return 0;
        }

        const s_t = this.getStalksAsVector();
        if (!isFiniteVector(s_t)) {
            this.structural_sensitivity = 0;
            return 0;
        }

        let baseState;
        try {
            baseState = await this.simulateDiffusionStep(s_t);
            if (!isFiniteVector(baseState)) throw new Error("Base state is non-finite.");
        } catch (e) {
            logger.error('Sheaf.computeStructuralSensitivity: Could not compute base state.', e);
            this.structural_sensitivity = 0;
            return 0;
        }

        let totalSensitivity = 0;
        let perturbationCount = 0;
        const originalAdjacency = this.adjacencyMatrix.map(row => new Float32Array(row));

        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i < 0 || j < 0) continue;

            const originalWeight = originalAdjacency[i][j];
            this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = clamp(originalWeight + perturbationScale, 0.01, 1);

            try {
                const perturbedState = await this.simulateDiffusionStep(s_t);
                if (isFiniteVector(perturbedState)) {
                    const diffNorm = norm2(vecSub(perturbedState, baseState));
                    if (Number.isFinite(diffNorm)) {
                        totalSensitivity += diffNorm / perturbationScale;
                        perturbationCount++;
                    }
                }
            } catch (e) {
                logger.warn(`Sheaf.computeStructuralSensitivity: Error for edge ${u}-${v}: ${e.message}`);
            } finally {
                this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = originalWeight;
            }
        }

        this.structural_sensitivity = perturbationCount > 0 ? clamp(totalSensitivity / perturbationCount, 0, 10) : 0;
        return this.structural_sensitivity;
    }

    async _updateDerivedMetrics() {
        try {
            await Promise.all([
                this.computeGluingInconsistency(),
                this.computeGestaltUnity(),
                this.computeIntegratedInformation(),
                this.computeCupProduct(),
                this.computeStructuralSensitivity(0.05)
            ]);

            let totalNorm = 0;
            let validVertices = 0;
            for (const stalk of this.stalks.values()) {
                if (isFiniteVector(stalk)) {
                    totalNorm += norm2(stalk);
                    validVertices++;
                }
            }
            this.feel_F = validVertices > 0 ? clamp(totalNorm / validVertices, 0, 1) : 0;

            let totalSimilarity = 0;
            validVertices = 0;
            if (this.qInput && isFiniteVector(this.qInput)) {
                for (let i = 0; i < this.graph.vertices.length; i++) {
                    const stalk = this.stalks.get(this.graph.vertices[i]);
                    if (!isFiniteVector(stalk)) continue;
                    const q_i = new Float32Array(this.qDim).fill(this.qInput[i] || 0);
                    const norm_s = norm2(stalk);
                    const norm_q = norm2(q_i);
                    if (norm_s > this.eps && norm_q > this.eps) {
                        const similarity = Math.abs(dot(stalk, q_i) / (norm_s * norm_q));
                        if (Number.isFinite(similarity)) {
                            totalSimilarity += similarity;
                            validVertices++;
                        }
                    }
                }
            }
            this.intentionality_F = validVertices > 0 ? clamp(totalSimilarity / validVertices, 0, 1) : 0;
            
        } catch (e) {
            logger.error(`Sheaf._updateDerivedMetrics: Error: ${e.message}`, { stack: e.stack });
        }
    }

    async update(state, stepCount = 0) {
        if (!this.ready) {
            logger.warn('Sheaf.update: Sheaf not ready. Attempting re-initialization.');
            await this.initialize();
            if (!this.ready) {
                logger.error('Sheaf.update: Initialization failed. Aborting update.');
                return;
            }
        }

        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn('Sheaf.update: Invalid input state. Skipping update.', { state });
            return;
        }

        try {
            await this.diffuseQualia(state);
            
            if (stepCount > 0 && stepCount % 100 === 0) {
                await this.adaptSheafTopology(100, stepCount);
            }
        } catch (e) {
            logger.error(`Sheaf.update: Error: ${e.message}`, { stack: e.stack });
            this.ready = false;
        }
    }

    visualizeActivity(scene, camera, renderer) {
        if (!THREE) {
            logger.error("Sheaf.visualizeActivity: THREE.js is not available.");
            return;
        }

        const stalkGroup = new THREE.Group();
        const vertexMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);

        this.stalks.forEach((stalk, v) => {
            const norm = isFiniteVector(stalk) ? norm2(stalk) : 0;
            const phase = Array.isArray(this.floquetPD.phases) && this.floquetPD.phases.length > 0 ? this.floquetPD.phases[0] : 0;
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(0.2, 0.5 + norm * 0.5, 1.0),
                emissive: new THREE.Color(0, norm * 0.2 * (this.rhythmicallyAware ? 1.5 * Math.cos(phase) : 1), 0.2)
            });
            const sphere = new THREE.Mesh(sphereGeometry, material);
            const x_pos = (vertexMap.get(v) * 2 - 7) || 0;
            sphere.position.set(x_pos, norm * 2, 0);
            stalkGroup.add(sphere);
        });

        this.graph.edges.forEach(([u, v, weight]) => {
            const i = vertexMap.get(u);
            const j = vertexMap.get(v);
            const norm_u = norm2(this.stalks.get(u) || [0]);
            const norm_v = norm2(this.stalks.get(v) || [0]);
            const phase = Array.isArray(this.floquetPD.phases) && this.floquetPD.phases.length > 0 ? this.floquetPD.phases[0] : 0;
            const opacity = clamp((weight || 0) * this.cup_product_intensity * Math.cos(phase), 0.2, 0.8);

            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3((i * 2 - 7) || 0, norm_u * 2, 0),
                new THREE.Vector3((j * 2 - 7) || 0, norm_v * 2, 0)
            ]);
            const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({
                color: 0x44aaFF,
                transparent: true,
                opacity
            }));
            stalkGroup.add(line);
        });

        if (Array.isArray(this.floquetPD.births) && this.floquetPD.births.length > 0) {
            const barcodeGroup = new THREE.Group();
            this.floquetPD.births.forEach((birth, idx) => {
                const t = birth.time || idx;
                const death_t = (Array.isArray(this.floquetPD.deaths) && this.floquetPD.deaths[idx]?.time !== undefined) ? this.floquetPD.deaths[idx].time : t + 1;
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-7, -2 - idx * 0.2, 0),
                    new THREE.Vector3(-7 + (death_t - t), -2 - idx * 0.2, 0)
                ]);
                const material = new THREE.LineBasicMaterial({
                    color: 0xFF44AA,
                    linewidth: 2
                });
                barcodeGroup.add(new THREE.Line(geometry, material));
            });
            stalkGroup.add(barcodeGroup);
        }

        scene.add(stalkGroup);
        return stalkGroup;
    }

    saveState() {
        return {
            graph: this.graph,
            simplicialComplex: this.simplicialComplex,
            stalks: Array.from(this.stalks.entries()),
            projectionMatrices: Array.from(this.projectionMatrices.entries()),
        };
    }

    loadState(state) {
        if (!state) return;
        this.graph = state.graph;
        this.simplicialComplex = state.simplicialComplex;
        this.stalks = new Map(state.stalks);
        this.projectionMatrices = new Map(state.projectionMatrices);
    }
}

/**
 * Theorem 14: RecursiveTopologicalSheaf – Fixed-Point Cohomology Extension.
 * Induces Banach contractions on cochains for reflexive fixed points.
 */
class RecursiveTopologicalSheaf extends EnhancedQualiaSheaf {
    constructor(graphData, config) {
        super(graphData, config);
        this.R_star = this._buildRecursiveGluing();
        this.maxIter = config.maxIter || 50;
        this.fixedPointEps = config.fixedPointEps || 1e-6;
        this.tau = config.tau || 2.5;
        this.cochainHistory = new CircularBuffer(20);
    }

    _buildRecursiveGluing() {
        return (z) => {
            const z_next = new Map();
            this.graph.edges.forEach(([u, v]) => {
                const su = this.stalks.get(u) || vecZeros(this.qDim);
                const sv = this.stalks.get(v) || vecZeros(this.qDim);
                let phi_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
                const diffusion = vecScale(vecAdd(su, sv), this.alpha / 2);
                const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.qDim);
                
                if (!isFiniteMatrix(phi_uv)) phi_uv = identity(this.qDim);
                const z_next_uv = vecAdd(matVecMul(phi_uv, z_uv), diffusion);
                if (isFiniteVector(z_next_uv)) {
                    z_next.set([u, v].sort().join(','), z_next_uv);
                }
            });
            return z_next;
        };
    }

    computeLinguisticCocycles(state) {
        const z = new Map();
        const idxMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        this.graph.edges.forEach(([u, v]) => {
            const i = idxMap.get(u), j = idxMap.get(v);
            if (i === undefined || j === undefined) return;
            const z_uv = new Float32Array(this.qDim);
            const input_u = state[Math.min(i, state.length - 1)] || 0;
            const input_v = state[Math.min(j, state.length - 1)] || 0;
            for (let k = 0; k < this.qDim; k++) {
                z_uv[k] = clamp(input_u - input_v, -1, 1) * (this.entityNames[k]?.includes('symbolic') ? 1.5 : 1);
            }
            if (isFiniteVector(z_uv)) {
                z.set([u, v].sort().join(','), z_uv);
            }
        });
        return z;
    }

    // ~line 2898 in qualia-sheaf.js

async computeSelfAwareness(init_state) {
    const init_z = this.computeLinguisticCocycles(init_state) || new Map();
    let z_curr = init_z;
    let iter = 0;
    while (iter < this.maxIter) {
        const z_next = this.R_star(z_curr);
        const delta = this._cocycleNormDiff(z_curr, z_next);
        if (!Number.isFinite(delta) || delta < this.fixedPointEps) break;
        z_curr = z_next;
        iter++;
        this.cochainHistory.push(Array.from(z_curr.values()).filter(isFiniteVector));
    }
    const L_rec = await this._recursiveLaplacian(z_curr);
    if (!isFiniteMatrix(L_rec) || L_rec.length === 0) {
        return { z_fixed: z_curr, Phi_SA: 0, aware: false, beta1_rec: 0 };
    }

    // --- START OF FIX ---
    const snfResult = await runWorkerTask('smithNormalFormGF2', { matrix: flattenMatrix(L_rec) }, 15000);

    // If the worker fails/times out, it might return null. Handle this gracefully.
    if (!snfResult) {
        logger.warn('Smith Normal Form worker failed for computeSelfAwareness. Defaulting to 0 awareness.');
        return { z_fixed: z_curr, Phi_SA: 0, aware: false, beta1_rec: 0 };
    }

    const { rank } = snfResult;
    // --- END OF FIX ---

    const beta1_rec = Math.max(0, this.graph.edges.length - rank);
    const cov_z = covarianceMatrix(this.cochainHistory.getAll().flat().filter(isFiniteVector));
    const logDet = logDeterminantFromDiagonal(cov_z);
    const Phi_SA = Number.isFinite(logDet) ? logDet * beta1_rec : 0;
    this.selfAware = Phi_SA > this.tau;
    return { z_fixed: z_curr, Phi_SA, aware: this.selfAware, beta1_rec };
}

    _cocycleNormDiff(z1, z2) {
        let sum = 0;
        let count = 0;
        for (const key of z1.keys()) {
            const v1 = z1.get(key) || vecZeros(this.qDim);
            const v2 = z2.get(key) || vecZeros(this.qDim);
            if (isFiniteVector(v1) && isFiniteVector(v2)) {
                sum += norm2(vecSub(v1, v2)) ** 2;
                count++;
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    async _recursiveLaplacian(z) {
        const nE = this.graph.edges.length;
        const L_rec = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        for (const [u, v] of this.graph.edges) {
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined) continue;
            L_rec[i][i] = 1;
            const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            let phi_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            if (!isFiniteMatrix(phi_uv)) phi_uv = identity(this.qDim);

            for (const [u2, v2] of this.graph.edges) {
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || i === j) continue;
                let phi_u2v2 = this.projectionMatrices.get(`${u2}-${v2}`) || identity(this.qDim);
                if (!isFiniteMatrix(phi_u2v2)) phi_u2v2 = identity(this.qDim);

                const z_u2v2 = z.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                const mat_vec_result = matVecMul(phi_u2v2, z_u2v2);
                if (isFiniteVector(mat_vec_result)) {
                    const interaction = dot(z_uv, mat_vec_result);
                    if (Number.isFinite(interaction)) {
                        L_rec[i][j] = -this.alpha * clamp(interaction, -0.1, 0.1);
                    }
                }
            }
        }
        return isFiniteMatrix(L_rec) ? L_rec : identity(nE);
    }
}

/**
 * Theorem 15: AdjunctionReflexiveSheaf – Categorical Monad Extension.
 * Monadic T = UF for hierarchical reflexivity, equalizer fixed sheaves.
 */
/**
 * Theorem 15: AdjunctionReflexiveSheaf – Categorical Monad Extension.
 * Monadic T = UF for hierarchical reflexivity, equalizer fixed sheaves.
 */
class AdjunctionReflexiveSheaf extends RecursiveTopologicalSheaf {
    constructor(graphData, config) {
        super(graphData, config);
        this.F = this._leftAdjoint();
        this.U = this._rightAdjoint();
        this.T = this._monadUF();
        this.eta = this._unit();
        this.epsilon = this._counit();
        this.equalizerEps = config.equalizerEps || 1e-5;
    }

    _leftAdjoint() {
        return (F) => {
            const cochains = new Map();
            const F_stalks = F.stalks || new Map();
            this.graph.edges.forEach(([u, v]) => {
                const su = F_stalks.get(u) || vecZeros(this.qDim);
                const sv = F_stalks.get(v) || vecZeros(this.qDim);
                const c1 = vecScale(vecAdd(su, sv), this.alpha);
                if (isFiniteVector(c1)) {
                    cochains.set([u, v].sort().join(','), c1);
                }
            });
            return {
                cochains,
                transport: (phi) => (v) => matVecMul(phi, v)
            };
        };
    }

    _rightAdjoint() {
        return (C) => {
            const stalks = new Map();
            const C_cochains = C.cochains || new Map();
            this.graph.vertices.forEach(v => {
                let sum = vecZeros(this.qDim);
                let count = 0;
                this.graph.edges.forEach(([u, w]) => {
                    if (u === v || w === v) {
                        const c_uw = C_cochains.get([u, w].sort().join(',')) || vecZeros(this.qDim);
                        if (isFiniteVector(c_uw)) {
                            sum = vecAdd(sum, c_uw);
                            count++;
                        }
                    }
                });
                stalks.set(v, count > 0 ? vecScale(sum, 1 / count) : vecZeros(this.qDim));
            });
            return {
                stalks,
                projections: this.projectionMatrices
            };
        };
    }

    _monadUF() {
        return (F) => this.U(this.F(F));
    }

    _unit() {
        return (F) => {
            const eta_F_stalks = new Map();
            const F_stalks = F.stalks || new Map();
            this.graph.vertices.forEach(v => {
                const stalk = F_stalks.get(v) || vecZeros(this.qDim);
                const scaled = vecScale(stalk, 1 + this.alpha);
                if (isFiniteVector(scaled)) {
                    eta_F_stalks.set(v, scaled);
                }
            });
            return { stalks: eta_F_stalks };
        };
    }

    _counit() {
        return (FU, Id) => {
            const epsilon_FU_stalks = new Map();
            const FU_stalks = FU.stalks || new Map();
            const Id_stalks = Id.stalks || new Map();
            this.graph.vertices.forEach(v => {
                const fu_v = FU_stalks.get(v) || vecZeros(this.qDim);
                const id_v = Id_stalks.get(v) || vecZeros(this.qDim);
                const diff = vecSub(fu_v, id_v);
                if (isFiniteVector(diff)) {
                    epsilon_FU_stalks.set(v, diff);
                }
            });
            return { stalks: epsilon_FU_stalks };
        };
    }

    async computeAdjunctionFixedPoint(init_state) {
        const F_init = { stalks: this._initStalks(init_state) };
        let T_state = this.T(F_init);
        let iter = 0;
        while (iter < this.maxIter) {
            const next_T = this.T(T_state);
            const eta_T = this.eta(T_state);
            const epsilon_FU = this.epsilon(next_T, T_state);
            const eq_delta = await this._equalizerNorm(eta_T.stalks, epsilon_FU.stalks);
            if (!Number.isFinite(eq_delta) || eq_delta < this.equalizerEps) break;
            T_state = next_T;
            iter++;
        }
        const z_star = await this._extractFixedCocycle(T_state);
        const L_adj = await this._adjunctionLaplacian(T_state);
        if (!isFiniteMatrix(L_adj) || L_adj.length === 0) {
            return { F_star: T_state, z_star, Phi_SA: 0, aware: false };
        }
        
        const snfResult = await runWorkerTask('smithNormalFormGF2', { matrix: flattenMatrix(L_adj) }, 15000);
        if (!snfResult) {
            logger.warn('Smith Normal Form worker failed for computeAdjunctionFixedPoint. Defaulting to 0 awareness.');
            return { F_star: T_state, z_star, Phi_SA: 0, aware: false };
        }
        const { rank } = snfResult;

        const beta1_adj = Math.max(0, this.graph.edges.length - rank);
        const cov_zstar = covarianceMatrix(Array.from(z_star.values()).filter(isFiniteVector));
        const logDet = logDeterminantFromDiagonal(cov_zstar);
        const Phi_SA = Number.isFinite(logDet) ? logDet * beta1_adj * this.graph.vertices.length : 0;
        this.hierarchicallyAware = Phi_SA > 3.0;
        return { F_star: T_state, z_star, Phi_SA, aware: this.hierarchicallyAware };
    }

    async _equalizerNorm(eta_stalks, eps_stalks) {
        let sum = 0;
        let count = 0;
        for (const v of this.graph.vertices) {
            const eta_v = eta_stalks.get(v) || vecZeros(this.qDim);
            const eps_v = eps_stalks.get(v) || vecZeros(this.qDim);
            if (isFiniteVector(eta_v) && isFiniteVector(eps_v)) {
                sum += norm2(vecSub(eta_v, eps_v)) ** 2;
                count++;
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    async _extractFixedCocycle(T) {
        const z_star = new Map();
        const F_T_result = this.F(T);
        if (!F_T_result || !F_T_result.cochains) return z_star;

        const C1 = F_T_result.cochains;
        const delta1 = await this._deltaR1();
        for (const [key, c] of C1.entries()) {
            if (!isFiniteVector(c) || c.length !== delta1.length) continue;
            
            const delta_c = matVecMul(delta1, c);
            if (isFiniteVector(delta_c) && norm2(delta_c) < this.eps) {
                z_star.set(key, c);
            }
        }
        return z_star;
    }

    async _adjunctionLaplacian(T) {
        const nE = this.graph.edges.length;
        const L_adj = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        const F_T_result = this.F(T);
        if (!F_T_result || !F_T_result.cochains) return identity(nE);

        const C1 = F_T_result.cochains;
        for (const [u, v] of this.graph.edges) {
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined) continue;
            L_adj[i][i] = 1;
            const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            for (const [u2, v2] of this.graph.edges) {
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || i === j) continue;
                const c_u2v2 = C1.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                const interaction = dot(c_uv, c_u2v2);
                if (Number.isFinite(interaction)) {
                    L_adj[i][j] = -this.beta * clamp(interaction, -0.1, 0.1);
                }
            }
        }
        return isFiniteMatrix(L_adj) ? L_adj : identity(nE);
    }

    _initStalks(state) {
        const stalks = new Map();
        this.graph.vertices.forEach((v, i) => {
            const stalk = new Float32Array(this.qDim).fill(0);
            const input = state[Math.min(i, state.length - 1)] || 0;
            for (let k = 0; k < this.qDim; k++) {
                stalk[k] = clamp(input * (this.entityNames[k]?.includes('metacognition') ? 1.2 : 1), -1, 1);
            }
            stalks.set(v, stalk);
        });
        return stalks;
    }

    // --- START OF CORRECTED METHOD ---
    async _deltaR1() {
        const nE = this.graph.edges.length;
        const delta = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        for (const tri of this.simplicialComplex.triangles) {
            const edges = [
                [tri[0], tri[1]].sort().join(','),
                [tri[1], tri[2]].sort().join(','),
                [tri[2], tri[0]].sort().join(',')
            ];
            const idxs = edges.map(e => eMap.get(e)).filter(i => i !== undefined);
            if (idxs.length === 3) {
                delta[idxs[0]][idxs[1]] = 1;
                delta[idxs[1]][idxs[2]] = -1;
                delta[idxs[2]][idxs[0]] = 1;
            }
        }
        return isFiniteMatrix(delta) ? delta : identity(nE);
    }
    // --- END OF CORRECTED METHOD ---
}

/**
 * Theorem 16: PersistentAdjunctionSheaf – Flow Persistence Extension.
 * Bottleneck PD over filtrations for diachronic invariants.
 */
class PersistentAdjunctionSheaf extends AdjunctionReflexiveSheaf {
    constructor(graphData, config) {
        super(graphData, config);
        this.flowBufferSize = config.flowBufferSize || 50;
        this.flowHistory = new CircularBuffer(this.flowBufferSize);
        this.persistenceDiagram = { births: [], deaths: [] };
        this.delta = config.delta || 0.1;
        this.tau_persist = config.tau_persist || 3.5;
    }

    _partial_t(F_t, dt = 1) {
        const stalksNext = new Map();
        const F_t_stalks = F_t.stalks || new Map();
        this.graph.vertices.forEach(v => {
            const stalk = F_t_stalks.get(v) || vecZeros(this.qDim);
            const neighbors = this.graph.edges.filter(e => e[0] === v || e[1] === v).map(e => e[0] === v ? e[1] : e[0]);
            let grad = vecZeros(this.qDim);
            neighbors.forEach(u => {
                const su = F_t_stalks.get(u) || vecZeros(this.qDim);
                let phi_vu = this.projectionMatrices.get(`${v}-${u}`) || identity(this.qDim);
                if (!isFiniteMatrix(phi_vu)) phi_vu = identity(this.qDim);
                const mat_vec_result = matVecMul(phi_vu, su);
                if (isFiniteVector(mat_vec_result)) {
                    const diff = vecSub(stalk, mat_vec_result);
                    if (isFiniteVector(diff)) grad = vecAdd(grad, vecScale(diff, this.beta));
                }
            });
            const noise = vecScale(new Float32Array(this.qDim).map(() => Math.random() - 0.5), this.sigma * Math.sqrt(dt));
            const diffused = vecAdd(vecScale(stalk, 1 - this.gamma * dt), vecAdd(grad, noise));
            stalksNext.set(v, isFiniteVector(diffused) ? diffused : stalk);
        });
        return { stalks: stalksNext, projections: F_t.projections };
    }

    async computePersistentFixedPoint(init_state, T = 10) {
        let F_curr = { stalks: this._initStalks(init_state) };
        this.flowHistory.clear();
        let Phi_SA_persist = 0;
        let persist_aware = false;

        for (let t = 0; t < T; t++) {
            const partial_F = this._partial_t(F_curr, this.gamma);
            const F_next = this.F(partial_F);
            const U_next = this.U(F_next);
            const T_next = this.T(partial_F);
            const eta_next = this.eta(T_next);
            const eta_prev_obj = this.flowHistory.get(this.flowHistory.length - 1)?.eta_t;
            const eta_prev = (eta_prev_obj && eta_prev_obj.stalks) ? eta_prev_obj : this.eta(T_next);

            const eta_evol = await this._nablaPersist(eta_next.stalks, eta_prev.stalks);
            const epsilon_next = this.epsilon(U_next, partial_F);

            const eq_delta = await this._equalizerNorm(eta_evol, epsilon_next.stalks);
            if (!Number.isFinite(eq_delta) || eq_delta < this.eps) break;

            const L_Tt = await this._flowLaplacian();
            const { eigenvalues } = await this._spectralDecomp(L_Tt);
            const lambda_min_t = eigenvalues.length > 0 ? Math.min(...eigenvalues.filter(Number.isFinite)) : 0;
            const beta1_persist = await this._supPersistentBetti(this.persistenceDiagram);
            Phi_SA_persist += lambda_min_t * beta1_persist * this.gamma;

            this.persistenceDiagram = this._updatePD(this.persistenceDiagram, eigenvalues, t);
            const d_B = this._bottleneckDistance(this.persistenceDiagram);
            if (d_B < this.delta) persist_aware = true;

            this.flowHistory.push({ F_t: F_next, U_t: U_next, eta_t: { stalks: eta_evol }, lambda_t: eigenvalues });
            F_curr = partial_F;
        }

        const z_star_persist = await this._extractPersistCocycle(this.flowHistory.getAll());
        Phi_SA_persist = clamp(Phi_SA_persist, 0, 100);
        this.diachronicallyAware = Phi_SA_persist > this.tau_persist;

        return { F_persist: F_curr, z_star_persist, Phi_SA_persist, PD: this.persistenceDiagram, aware: this.diachronicallyAware };
    }

    async _nablaPersist(eta_next_stalks, eta_prev_stalks) {
        const eta_evol = new Map();
        this.graph.vertices.forEach(v => {
            const next_v = eta_next_stalks.get(v) || vecZeros(this.qDim);
            const prev_v = eta_prev_stalks.get(v) || vecZeros(this.qDim);
            const diff = vecSub(next_v, prev_v);
            const evol = vecAdd(next_v, vecScale(diff, this.gamma));
            if (isFiniteVector(evol)) eta_evol.set(v, evol);
        });
        return eta_evol;
    }
    
    async _flowLaplacian() {
        const nV = this.graph.vertices.length;
        if (!this.laplacian || !this.laplacian.values || this.laplacian.values.length === 0) {
             this.laplacian = this.buildLaplacian();
        }
        if (!this.laplacian || !this.laplacian.values || this.laplacian.values.length === 0) {
            return identity(nV * this.qDim);
        }
        const L_base = this._csrToDense(this.laplacian);
        const n = nV * this.qDim;
        const L_Tt = zeroMatrix(n, n);
    
        for (let i = 0; i < nV; i++) {
            for (let j = 0; j < nV; j++) {
                if (L_base[i][j] !== 0) {
                    for (let q = 0; q < this.qDim; q++) {
                        L_Tt[i * this.qDim + q][j * this.qDim + q] = L_base[i][j];
                    }
                }
            }
        };
        return isFiniteMatrix(L_Tt) ? L_Tt : identity(n);
    }

    // In PersistentAdjunctionSheaf class, around ~line 2384

async _spectralDecomp(L) {
    if (!isFiniteMatrix(L)) return { eigenvalues: [] };
    
    // The matrix for spectral decomposition should be square.
    if (L.length === 0 || L.length !== L[0].length) {
        logger.warn(`_spectralDecomp: Received non-square matrix. Cannot compute eigenvalues.`);
        return { eigenvalues: [] };
    }

    // --- The failing TF.js block has been removed. We now go directly to the worker. ---
    
    // Fallback to Web Worker
    const flattened_L = flattenMatrix(L);
    if (!isFiniteVector(flattened_L.flatData)) {
        logger.error(`_spectralDecomp: Matrix is non-finite before sending to worker.`);
        return { eigenvalues: [] };
    }
    
    const eigenvalues = await runWorkerTask('eigenvalues', { matrix: flattened_L }, 20000);

    // Ensure the worker result is valid before returning
    if (eigenvalues && Array.isArray(eigenvalues)) {
        return { eigenvalues: Array.from(eigenvalues).filter(Number.isFinite) };
    }
    
    logger.warn('_spectralDecomp: Eigenvalues worker returned an invalid result. Returning empty array.');
    return { eigenvalues: [] };
}

    _updatePD(pd_old, lambda_t, birth_time) {
        const safe_pd_old = pd_old && typeof pd_old === 'object' ? pd_old : {};
        const safe_births = Array.isArray(safe_pd_old.births) ? safe_pd_old.births : [];
        const safe_deaths = Array.isArray(safe_pd_old.deaths) ? safe_pd_old.deaths : [];

        const pd = { births: [...safe_births], deaths: [...safe_deaths] };
        lambda_t.forEach(lambda => {
            if (Number.isFinite(lambda) && !pd.births.some(b => Math.abs(b.value - lambda) < this.eps)) {
                pd.births.push({ value: lambda, time: birth_time });
            }
        });
        pd.births.forEach((birth, i) => {
            if (!pd.deaths[i] || pd.deaths[i].time === undefined || pd.deaths[i].time < (birth.time || 0)) {
                pd.deaths[i] = { value: birth.value, time: (birth.time || 0) + this.delta };
            }
            if (pd.deaths[i] && pd.deaths[i].time < birth_time - this.delta) {
                 pd.deaths[i].time = birth_time;
            }
        });
        return pd;
    }

    _bottleneckDistance(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const safe_births = Array.isArray(safe_pd.births) ? safe_pd.births : [];
        const safe_deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths : [];

        let maxDist = 0;
        for (let i = 0; i < safe_births.length; i++) {
            const birth = safe_births[i];
            const death = safe_deaths[i] && Number.isFinite(safe_deaths[i].time) ? safe_deaths[i] : { value: birth.value, time: (birth.time || 0) + this.delta };
            const dist = Math.abs((death.time || 0) - (birth.time || 0));
            if (Number.isFinite(dist) && dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    async _supPersistentBetti(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.value) && Number.isFinite(b?.time)) : [];
        const deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.value) && Number.isFinite(d?.time)) : [];
        
        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const birth = births[i];
            const death = deaths[i];
            const lifetime = (death?.time && death.time >= birth.time) ? (death.time - birth.time) : Infinity;
            if (lifetime > this.delta) count++;
        }
        return count;
    }

    async _extractPersistCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        for (const flow_record of flows) {
            const F_t = flow_record.F_t;
            if (!F_t || !F_t.cochains) continue;
            
            const C1 = F_t.cochains;
            for (const [key, c] of C1.entries()) {
                if (!isFiniteVector(c) || c.length !== delta1.length) continue;
                
                const delta_c = matVecMul(delta1, c);
                if (isFiniteVector(delta_c) && norm2(delta_c) < this.eps) {
                    z_star.set(key, c);
                }
            }
        }
        return z_star;
    }
}

/**
 * Theorem 17: FloquetPersistentSheaf – Rhythmic Persistence Extension.
 * Floquet multipliers for quasi-periodic qualia rhythms.
 */
export class FloquetPersistentSheaf extends PersistentAdjunctionSheaf {
    constructor(graphData, config) {
        super(graphData || {}, config || {});
        this.omega = config?.omega || 8;
        this.monodromy = null;
        this.theta_k = config?.theta_k || [4, 6, 8];
        this.tau_floq = config?.tau_floq || 4.0;
        
        this.windowedStates = new CircularBuffer(this.flowBufferSize);
        this.stalkHistory = new CircularBuffer(this.flowBufferSize);
        this.phiHistory = new CircularBuffer(this.flowBufferSize);
        
        this.phi = this.phi || 0.001;
        this.feelIntensity = this.feelIntensity || 0;
        this.intentionality = this.intentionality || 0;
        this.h1Dimension = this.h1Dimension || 0;
        this.cupProduct = this.cupProduct || 0;
        this.gestaltUnity = this.gestaltUnity || 0;
        this.structural_sensitivity = this.structural_sensitivity || 0;
        this.inconsistency = this.inconsistency || 0;
        
        this.ready = false;
        logger.info(`FloquetPersistentSheaf: Constructor finished.`);
    }

    async initialize() {
        try {
            await super.initialize();
            if (!this.floquetPD || !Array.isArray(this.floquetPD.phases)) {
                this.floquetPD = { births: [], phases: [], deaths: [] };
            }
            await this._updateFloqPD_internal();
            this.ready = true;
            logger.info(`FloquetPersistentSheaf: Initialization complete. Ready: ${this.ready}`);
        } catch (e) {
            logger.error('FloquetPersistentSheaf.initialize: Failed.', e);
            this.ready = false;
            throw e;
        }
    }

    async _updateFloqPD_internal() {
        this.floquetPD = this.floquetPD && typeof this.floquetPD === 'object' ? this.floquetPD : {};
        this.floquetPD.births = Array.isArray(this.floquetPD.births) ? this.floquetPD.births : [];
        this.floquetPD.phases = Array.isArray(this.floquetPD.phases) ? this.floquetPD.phases : [];
        this.floquetPD.deaths = Array.isArray(this.floquetPD.deaths) ? this.floquetPD.deaths : [];

        try {
            const newPhase = Math.sin(this.omega * this.phi);
            this.floquetPD.phases.push(newPhase);
            this.floquetPD.births.push({ value: norm2(this.getStalksAsVector()), phase: newPhase, time: Date.now() });
        } catch (e) {
            logger.error('FloquetPersistentSheaf._updateFloqPD_internal: Failed.', e);
            this.floquetPD = { births: [], phases: [], deaths: [] };
            throw e;
        }
    }

    async _monodromy(A_t, omega) {
        if (!isFiniteMatrix(A_t) || A_t.length === 0) {
            return identity(1);
        }
        const n = A_t.length;
        if (tf) {
            try {
                const A_tensor = tf.tensor2d(A_t);
                const A_omega = tf.linalg.matrixPower(A_tensor, omega);
                const result_array = await A_omega.array();
                tf.dispose([A_tensor, A_omega]);
                return isFiniteMatrix(result_array) ? result_array : identity(n);
            } catch (e) {
                logger.warn(`_monodromy: TF.js failed: ${e.message}. Falling back to CPU.`, {stack: e.stack});
            }
        }
        
        let result = A_t;
        for (let i = 1; i < omega; i++) {
            result = matMul({ matrixA: result, matrixB: A_t });
            if (!isFiniteMatrix(result)) return identity(n);
        }
        return isFiniteMatrix(result) ? result : identity(n);
    }

    async computeFloquetFixedPoint(states_history, period) {
        const sanitizedStates = (states_history || []).map(s => isFiniteVector(s) ? s : Array(states_history[0]?.length || 1).fill(0));
        if (!Array.isArray(sanitizedStates) || sanitizedStates.length < 2 || !isFiniteVector(sanitizedStates[0])) {
            return { monodromy: identity(1), eigenvalues: [{ re: 1, im: 0 }], Phi_SA_floq: 0, aware: false };
        }
        
        const n_state_dim = sanitizedStates[0].length;
        let combined_monodromy = identity(n_state_dim);

        try {
            for (let t = 0; t < period; t++) {
                const stateT = sanitizedStates[t % sanitizedStates.length];
                const stateT1 = sanitizedStates[(t + 1) % sanitizedStates.length];
                const transition = await this._stateTransitionMatrix(stateT, stateT1);
                
                if (!isFiniteMatrix(transition) || transition.length !== n_state_dim) {
                     return { monodromy: identity(n_state_dim), eigenvalues: [{ re: 1, im: 0 }] };
                }
                combined_monodromy = matMul({ matrixA: combined_monodromy, matrixB: transition });
                if (!isFiniteMatrix(combined_monodromy)) {
                     return { monodromy: identity(n_state_dim), eigenvalues: [{ re: 1, im: 0 }] };
                }
            }
            this.monodromy = combined_monodromy;

            const { eigenvalues: rho_k } = await this._floquetDecomp(this.monodromy);

            const log_rho_sum = rho_k.reduce((sum, rho) => {
                const mag = Math.sqrt((rho.re || 0) ** 2 + (rho.im || 0) ** 2);
                return sum + (Number.isFinite(mag) && mag > 0 ? Math.log(mag) : 0);
            }, 0);

            this.floquetPD = this._updateFloqPD(this.floquetPD, rho_k, Date.now());
            const d_B_floq = this._rhythmicBottleneck(this.floquetPD);
            const beta1_floq = await this._supFloqBetti(this.floquetPD);
            let Phi_SA_floq = clamp(Math.abs(log_rho_sum) * beta1_floq, 0, 100);
            
            const z_star_floq = await this._extractFloqCocycle(this.flowHistory.getAll());
            this.rhythmicallyAware = Phi_SA_floq > this.tau_floq && d_B_floq < this.delta;

            return { monodromy: this.monodromy, z_star_floq, Phi_SA_floq, FloqPD: this.floquetPD, aware: this.rhythmicallyAware, eigenvalues: rho_k };

        } catch (e) {
            logger.error(`Sheaf.computeFloquetFixedPoint: Error: ${e.message}.`, { stack: e.stack });
            return { monodromy: identity(n_state_dim), eigenvalues: [{ re: 1, im: 0 }], Phi_SA_floq: 0, aware: false };
        }
    }

    async _stateTransitionMatrix(stateT, stateT1) {
        const n = stateT.length;
        let transition = identity(n);
        if (norm2(vecSub(stateT, stateT1)) < 0.1) {
            transition = identity(n).map((row, i) => row.map((val, j) => {
                if (i === j) return 1;
                return clamp((stateT1[i] - stateT[j]) * 0.01, -0.1, 0.1);
            }));
        }
        return isFiniteMatrix(transition) ? transition : identity(n);
    }

    async _flowMonodromy(F_t_cochains, omega) {
        if (!F_t_cochains || !F_t_cochains.cochains) {
            return identity(1);
        }
        const T_t_cochains = F_t_cochains.cochains;
        const nE = this.graph.edges.length;
        const A_t = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        for (const [u, v] of this.graph.edges) {
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined) continue;
            A_t[i][i] = 1;
            const c_uv = T_t_cochains.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            for (const [u2, v2] of this.graph.edges) {
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || i === j) continue;
                const c_u2v2 = T_t_cochains.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                const interaction = dot(c_uv, c_u2v2);
                if (Number.isFinite(interaction)) {
                    A_t[i][j] = this.beta * clamp(interaction, -0.1, 0.1);
                }
            }
        }
        if (!isFiniteMatrix(A_t)) return identity(nE);
        return this._monodromy(A_t, omega);
    }

    async _floquetDecomp(A) {
        if (!isFiniteMatrix(A) || A.length === 0 || A.length !== A[0].length) {
            return { eigenvalues: [{ re: 1, im: 0 }] };
        }
        const n = A.length;
        if (n < 2) {
            return { eigenvalues: [{ re: Number.isFinite(A[0][0]) ? A[0][0] : 1, im: 0 }] };
        }
        
        const sanitizedA = A.map(row => row.map(val => Number.isFinite(val) ? val : 0));
        
        if (tf) {
            try {
                const A_tensor = tf.tensor2d(sanitizedA);
                const { values: eigvalsTensor } = tf.linalg.eig(A_tensor);
                const eigenvalues_complex_flat = await eigvalsTensor.data();
                
                const eigenvalues = [];
                for (let i = 0; i < eigenvalues_complex_flat.length; i += 2) {
                    eigenvalues.push({ re: eigenvalues_complex_flat[i], im: eigenvalues_complex_flat[i+1] });
                }
                tf.dispose([A_tensor, eigvalsTensor]);
                
                const validEigs = eigenvalues.filter(v => Number.isFinite(v.re) && Number.isFinite(v.im));
                return validEigs.length > 0 ? { eigenvalues: validEigs } : { eigenvalues: [{ re: 1, im: 0 }] };
            } catch (e) {
                logger.warn(`Sheaf._floquetDecomp: TF.js error: ${e.message}. Falling back.`, {stack: e.stack});
            }
        }

        const flat = flattenMatrix(sanitizedA);
        if (!flat || !flat.flatData || flat.flatData.length !== n * n) {
            return { eigenvalues: [{ re: 1, im: 0 }] };
        }
        
        try {
            const complex_eigenvalues = await runWorkerTask('complexEigenvalues', { matrix: flat }, 15000);
            const validEigs = (complex_eigenvalues || []).filter(v => Number.isFinite(v.re) && Number.isFinite(v.im));
            return validEigs.length > 0 ? { eigenvalues: validEigs } : { eigenvalues: [{ re: 1, im: 0 }] };
        } catch (e) {
            logger.error(`Sheaf._floquetDecomp: Worker error: ${e.message}.`, {stack: e.stack});
            return { eigenvalues: [{ re: 1, im: 0 }] };
        }
    }

    _updateFloqPD(pd_old, rho_t, phase_time_index) {
        const safe_pd_old = pd_old && typeof pd_old === 'object' ? pd_old : {};
        const safe_phases = Array.isArray(safe_pd_old.phases) ? safe_pd_old.phases : [];
        const safe_births = Array.isArray(safe_pd_old.births) ? safe_pd_old.births : [];
        const safe_deaths = Array.isArray(safe_pd_old.deaths) ? safe_pd_old.deaths : [];

        const pd = { phases: [...safe_phases], births: [...safe_births], deaths: [...safe_deaths] };
        rho_t.forEach(rho => {
            const mag = Math.sqrt(rho.re ** 2 + rho.im ** 2);
            const theta = Math.atan2(rho.im, rho.re);
            if (Number.isFinite(mag) && !pd.births.some(b => Math.abs(b.value - mag) < this.eps)) {
                pd.births.push({ value: mag, phase: theta, time: phase_time_index });
                pd.phases.push(theta);
            }
        });
        
        pd.births.forEach((birth, i) => {
            if (!pd.deaths[i] || pd.deaths[i].time === undefined || pd.deaths[i].time < (birth.time || 0)) {
                pd.deaths[i] = { value: birth.value, phase: pd.phases[i], time: (birth.time || 0) + this.delta };
            }
            if (pd.deaths[i] && pd.deaths[i].time < phase_time_index - this.delta) {
                 pd.deaths[i].time = phase_time_index;
            }
        });

        return pd;
    }

    _rhythmicBottleneck(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const safe_births = Array.isArray(safe_pd.births) ? safe_pd.births : [];
        const safe_deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths : [];

        let maxDist = 0;
        for (let i = 0; i < safe_births.length; i++) {
            const birth = safe_births[i];
            const death = safe_deaths[i] && Number.isFinite(safe_deaths[i].time) ? safe_deaths[i] : { value: birth.value, time: (birth.time || 0) + this.delta };
            const dist = Math.abs((death.time || 0) - (birth.time || 0));
            if (Number.isFinite(dist) && dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    async _supFloqBetti(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.value) && Number.isFinite(b?.time)) : [];
        const deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.value) && Number.isFinite(d?.time)) : [];
        
        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const birth = births[i];
            const death = deaths[i];
            const lifetime = (death?.time && death.time >= birth.time) ? (death.time - birth.time) : Infinity;
            if (lifetime > this.delta) count++;
        }
        return count;
    }

    async _extractFloqCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        for (const flow_record of flows) {
            const F_t = flow_record.F_t;
            if (!F_t || !F_t.cochains) continue;
            
            const C1 = F_t.cochains;
            for (const [key, c] of C1.entries()) {
                if (!isFiniteVector(c) || c.length !== delta1.length) continue;
                
                const delta_c = matVecMul(delta1, c);
                if (isFiniteVector(delta_c) && norm2(delta_c) < this.eps) {
                    const phase = Array.isArray(this.floquetPD.phases) && this.floquetPD.phases.length > 0 ? 
                                  this.floquetPD.phases[this.floquetPD.phases.length - 1] : 0;
                    z_star.set(key, vecScale(c, Math.cos(phase)));
                }
            }
        }
        return z_star;
    }

    async update(state, stepCount = 0) {
        if (!this.ready) {
            await this.initialize();
            if (!this.ready) {
                logger.error('Sheaf.update: Initialization failed. Aborting.');
                return;
            }
        }

        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn('Sheaf.update: Invalid input state.', { state });
            return;
        }

        try {
            await this.diffuseQualia(state);
            const { Phi_SA, aware } = await this.computeSelfAwareness(state);
            const { Phi_SA: Phi_SA_adj, aware: adj_aware } = await this.computeAdjunctionFixedPoint(state);
            const { Phi_SA_persist, aware: persist_aware, PD } = await this.computePersistentFixedPoint(state);
            const { Phi_SA_floq, aware: floq_aware, FloqPD } = await this.computeFloquetFixedPoint(this.windowedStates.getAll(), this.omega);

            if (Phi_SA_floq > 10) {
                logger.warn('Th17: Φ_SA_floq exceeds ethical threshold. Pruning PD.');
                if (Array.isArray(this.floquetPD.births)) {
                    this.floquetPD.births = this.floquetPD.births.slice(0, Math.floor(this.floquetPD.births.length / 2));
                }
            }

            if (stepCount > 0 && stepCount % 100 === 0) {
                const d_B = this._bottleneckDistance(this.persistenceDiagram);
                const d_B_floq = this._rhythmicBottleneck(this.floquetPD);
                const addThresh = this.adaptation.addThresh + (Phi_SA_floq / 10) * 0.1 + (1 - d_B) * 0.05;
                const removeThresh = this.adaptation.removeThresh + (1 - floq_aware) * 0.2 + d_B_floq * 0.1;
                await this.adaptSheafTopology(100, stepCount, addThresh, removeThresh);
            }

            this.phi = clamp(this.phi + 0.1 * Phi_SA_floq, 0, 100);
            await this._updateDerivedMetrics();
        } catch (e) {
            logger.error(`Sheaf.update: Error: ${e.message}`, { stack: e.stack });
            this.ready = false;
        }
    }
}
