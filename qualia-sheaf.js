// --- START OF FILE qualia-sheaf.js ---

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity, covarianceMatrix, matVecMul, vecMul
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
        // FIX: Ensure only Array or Float32Array items are pushed
        if (!item || !Array.isArray(item) && !(item instanceof Float32Array)) {
            console.warn('CircularBuffer.push: Invalid item. Skipping.', { item });
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
        
        // Fix: Increase alpha for stronger input influence
        this.alpha = clamp(config.alpha ?? 0.2, 0.01, 1);
        this.beta = clamp(config.beta ?? 0.1, 0.01, 1);
        this.gamma = clamp(config.gamma ?? 0.05, 0.01, 0.5);
        this.sigma = clamp(config.sigma ?? 0.025, 0.001, 0.1);
        this.eps = 1e-6;
        
        this.adaptation = {
            addThresh: clamp(config.addThresh ?? 0.7, 0.5, 0.95),
            removeThresh: clamp(config.removeThresh ?? 0.2, 0.05, 0.4),
            targetH1: config.targetH1 ?? 2.0
        };

        this._initializeGraph(graphData);
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
        
        this.adjacencyMatrix = null;
        this.laplacian = null;
        this.maxEigApprox = 1;
        this.projectionMatrices = new Map();
        this.isUpdating = false;
        this.qInput = null; // For intentionality
        this.coherence = 0; // Th. 1: Cohomological coherence
        this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim); // For flow tracking

        logger.info(`Enhanced Qualia Sheaf constructed: vertices=${this.graph.vertices.length}, edges=${this.graph.edges.length}, triangles=${this.simplicialComplex.triangles.length}, tetrahedra=${this.simplicialComplex.tetrahedra.length}`);
    }

    _initializeGraph(graphData) {
        if (!graphData || typeof graphData !== 'object') {
            graphData = {};
        }

        const defaultVertices = ['agent_x', 'agent_z', 'agent_rot', 'target_x', 'target_z', 'vec_dx', 'vec_dz', 'dist_target'];
        const initialGraphVertices = Array.isArray(graphData.vertices) && graphData.vertices.length > 0 ? graphData.vertices : defaultVertices;
        const initialBaseEdges = Array.isArray(graphData.edges) ? graphData.edges.slice() : [
            ['agent_x', 'agent_rot'], ['agent_z', 'agent_rot'],
            ['agent_x', 'vec_dx'], ['agent_z', 'vec_dz'],
            ['target_x', 'vec_dx'], ['target_z', 'vec_dz'],
            ['vec_dx', 'dist_target'], ['vec_dz', 'dist_target']
        ];
        const explicitTriangles = Array.isArray(graphData.triangles) ? graphData.triangles.slice() : [
            ['agent_x', 'agent_z', 'agent_rot'],
            ['target_x', 'target_z', 'dist_target'],
            ['agent_x', 'target_x', 'vec_dx'],
            ['agent_z', 'target_z', 'vec_dz']
        ];
        const explicitTetrahedra = Array.isArray(graphData.tetrahedra) ? graphData.tetrahedra.slice() : [
            ['agent_x', 'agent_z', 'target_x', 'target_z'],
            ['agent_rot', 'vec_dx', 'vec_dz', 'dist_target']
        ];

        const allVerticesSet = new Set(initialGraphVertices);
        explicitTriangles.forEach(tri => tri.forEach(v => allVerticesSet.add(v)));
        explicitTetrahedra.forEach(tet => tet.forEach(v => allVerticesSet.add(v)));
        const finalVertices = Array.from(allVerticesSet);

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

        this.graph = {
            vertices: finalVertices,
            edges: Array.from(allEdgesSet).map(s => s.split(',').concat([0.5]))
        };
        this.simplicialComplex = {
            triangles: finalTrianglesUpdated.filter(t => t.length === 3),
            tetrahedra: explicitTetrahedra.filter(t => t.length === 4)
        };
        this.edgeSet = allEdgesSet;
    }

    _initializeStalks() {
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
        logger.info('EnhancedQualiaSheaf.initialize() called.');
        try {
            await this.computeCorrelationMatrix();
            this.projectionMatrices = await this.computeProjectionMatrices();
            await this.computeH1Dimension();
            await this._updateDerivedMetrics();
            this.ready = true;
            logger.info('Enhanced Qualia Sheaf ready with higher-order simplices.');
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
    
    async adaptSheafTopology(adaptFreq = 100, stepCount = 0) {
        if (!this.ready || stepCount % adaptFreq !== 0) return;
        
        if (this.stalkHistory.length < this.stalkHistorySize / 2) {
            logger.info(`Sheaf: Skipping topology adaptation at step ${stepCount}; insufficient history (${this.stalkHistory.length}/${this.stalkHistorySize}).`);
            return;
        }

        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        if (!isFiniteMatrix(this.correlationMatrix)) {
            logger.warn('Sheaf: Non-finite correlation matrix; skipping adaptation.');
            return;
        }
        
        const numVertices = this.graph.vertices.length;
        if (numVertices < 3) {
            return;
        }
        
        try {
            const addThresh = this.adaptation.addThresh + (1 - this.gestaltUnity) * 0.1;
            const removeThresh = this.adaptation.removeThresh + this.inconsistency * 0.2;
            this.adaptEdges(this.correlationMatrix, addThresh, removeThresh);
            this.adaptSimplices(this.correlationMatrix);
            
            await this.computeCorrelationMatrix();
            await this.computeH1Dimension();
            await this._updateDerivedMetrics();
            logger.info(`Sheaf adapted at step ${stepCount}.`);
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
        const maxEdges = 20;

        const contagionScores = new Float32Array(numVertices).fill(0);
        for (let i = 0; i < numVertices; i++) {
            for (let j = 0; j < numVertices; j++) {
                if (i !== j && (corrMatrix[i]?.[j] || 0) > addThreshold) {
                    contagionScores[i] += (corrMatrix[i]?.[j] || 0);
                }
            }
        }

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
                    const contagionBoost = (contagionScores[i] + contagionScores[j]) / 2;

                    if (corrVal > addThreshold && !this.edgeSet.has(edgeKey) && contagionBoost > 0.5) {
                        const weight = clamp(corrVal * (this.gestaltUnity || 0.5) * contagionBoost, 0.1, 1.0);
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
        let nonZeroCount1 = 0;
        this.graph.edges.forEach((edge, eIdx) => {
            const [u, v] = edge;
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx === undefined || vIdx === undefined) {
                logger.warn(`Sheaf.buildBoundaryMatrices: Invalid vertex index for edge ${u}-${v} (uIdx=${uIdx}, vIdx=${vIdx})`);
                return;
            }
            boundary1[eIdx][uIdx] = 1;
            boundary1[eIdx][vIdx] = -1; // Signed for orientation
            nonZeroCount1 += 2;
        });
            
        const partial2 = zeroMatrix(nT, nE);
        let nonZeroCount2 = 0;
        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (!Array.isArray(tri) || tri.length !== 3) return;
            const [u, v, w] = tri;
            const edges = [
                { key: [u, v].sort().join(','), sign: 1 },
                { key: [v, w].sort().join(','), sign: -1 },
                { key: [w, u].sort().join(','), sign: 1 }
            ];
            edges.forEach(({ key, sign }, idx) => {
                const eIdx = eMapIndices.get(key);
                if (eIdx !== undefined) {
                    partial2[tIdx][eIdx] = sign;
                    nonZeroCount2++;
                }
            });
        });

        const partial3 = zeroMatrix(nTet, nT);
        let nonZeroCount3 = 0;
        this.simplicialComplex.tetrahedra.forEach((tet, tetIdx) => {
            if (!Array.isArray(tet) || tet.length !== 4) return;
            const sortedTet = tet.slice().sort();
            for (let i = 0; i < 4; i++) {
                const face = sortedTet.filter((_, idx) => idx !== i).sort();
                const tIdx = tMapIndices.get(face.join(','));
                if (tIdx !== undefined) {
                    partial3[tetIdx][tIdx] = (i % 2 === 0 ? 1 : -1); // Alternating signs
                    nonZeroCount3++;
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
                const nonFiniteIndex = flattenedResult.flatData.findIndex(val => !Number.isFinite(val));
                if (nonFiniteIndex !== -1) {
                    logger.error(`Sheaf: Non-finite value at index ${nonFiniteIndex} in flatData for ${name}: ${flattenedResult.flatData[nonFiniteIndex]}`);
                }
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }

            if (flattenedResult.flatData.length !== flattenedResult.rows * flattenedResult.cols) {
                logger.error(`Sheaf: CRITICAL: Flattened data length mismatch for ${name} boundary after flattening. Returning empty.`, { flattenedResult });
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
        
        const tempRow = [];

        for (let i = 0; i < n; i++) {
            let degree = 0;
            tempRow.length = 0;

            for (let j = 0; j < n; j++) {
                if (i !== j && adj[i][j] > 0) {
                    tempRow.push({ col: j, val: adj[i][j] });
                    degree += adj[i][j];
                }
            }
            tempRow.sort((a, b) => a.col - b.col);

            tempRow.forEach(item => {
                values.push(-item.val);
                colIndices.push(item.col);
            });

            const diagCol = i;
            let insertPos = colIndices.length;
            for (let k = rowPtr[i]; k < colIndices.length; k++) {
                if (colIndices[k] > diagCol) {
                    insertPos = k;
                    break;
                }
            }

            values.splice(insertPos - rowPtr[i], 0, degree + this.eps);
            colIndices.splice(insertPos - rowPtr[i], 0, diagCol);

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
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; j++) {
                sum += csr.values[j] * v[csr.colIndices[j]];
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

            if (!isFiniteVector(s_u) || !isFiniteVector(s_v)) {
                projections.set(`${u}-${v}`, identity(this.qDim));
                projections.set(`${v}-${u}`, identity(this.qDim));
                continue;
            }

            try {
                const projected = await runWorkerTask('matVecMul', { matrix: flattenMatrix(P_uv), vector: s_u });
                const error = vecSub(projected, s_v);
                const grad = Array.from({ length: this.qDim }, () => new Float32Array(this.qDim));

                for (let i = 0; i < this.qDim; i++) {
                    for (let j = 0; j < this.qDim; j++) {
                        grad[i][j] = error[i] * s_u[j];
                    }
                }

                const reg_grad = P_uv.map((row, i) => row.map((val, j) => 2 * lambda * (val - (i === j ? 1 : 0))));
                P_uv = P_uv.map((row, i) => row.map((val, j) => 
                    clamp(val - eta_P * (grad[i][j] + reg_grad[i][j]), -1, 1)
                ));

                if (!isFiniteMatrix(P_uv)) {
                    logger.warn(`Sheaf.computeProjectionMatrices: Non-finite projection matrix for ${u}-${v}. Using identity.`);
                    P_uv = identity(this.qDim);
                }

                projections.set(`${u}-${v}`, P_uv);
                projections.set(`${v}-${u}`, P_uv);
            } catch (e) {
                logger.warn(`Sheaf.computeProjectionMatrices: Error for edge ${u}-${v}. Using identity.`, e);
                projections.set(`${u}-${v}`, identity(this.qDim));
                projections.set(`${v}-${u}`, identity(this.qDim));
            }
        }
        this.projectionMatrices = projections;
        return projections;
    }

    async _updateGraphStructureAndMetrics() {
        if (this.isUpdating) return;
        this.isUpdating = true;
        try {
            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();
            
            if (this.laplacian && this.laplacian.values.length > 0) {
                const denseLaplacian = this._csrToDense(this.laplacian);
                this.maxEigApprox = await runWorkerTask('matrixSpectralNormApprox', { matrix: flattenMatrix(denseLaplacian) }, 15000);
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

    // Th. 1: Full coherence flow method
    async computeCoherenceFlow(dt = 0.01) {
        if (!this.laplacian) await this.computeLaplacian();
        const sensoryInput = this.qInput || vecZeros(this.graph.vertices.length * this.qDim);
        const flatCochain = flattenMatrix(this.currentCochain).flatData;
        // Sparse Laplacian flow: -αLψ + βs
        const laplacianTerm = this.csrMatVecMul(this.laplacian, flatCochain);
        const flowUpdate = vecAdd(vecScale(laplacianTerm, -this.alpha * dt), vecScale(sensoryInput, this.beta * dt));
        this.currentCochain = unflattenMatrix({
            flatData: vecAdd(flatCochain, flowUpdate),
            rows: this.graph.vertices.length,
            cols: this.qDim
        });
        this.coherence = clamp(norm2(flatCochain) / Math.sqrt(this.graph.vertices.length * this.qDim), 0, 1);
        return this.coherence;
    }

    // Full upgraded diffuseQualia with Th. 1 integration
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
        if (!isFiniteVector(s)) {
            logger.error('Sheaf.diffuseQualia: Non-finite initial stalk vector "s". Aborting diffusion.');
            return;
        }
        
        const qInput = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            qInput[i] = state[Math.min(i, state.length - 1)] || 0;
        }

        // Th. 1: Compute coherence flow as primary propagation
        await this.computeCoherenceFlow();
        this.qInput = qInput;

        const Lfull = zeroMatrix(N, N);
        const idxMap = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) {
            const i = idxMap.get(u), j = idxMap.get(v);
            if (i === undefined || j === undefined) continue;
            const weight = this.adjacencyMatrix[i]?.[j];
            if (!Number.isFinite(weight) || weight <= 0) continue;

            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
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

        if (!isFiniteMatrix(A) || !isFiniteVector(rhs)) {
            logger.error('Sheaf.diffuseQualia: Non-finite A or RHS for linear solver. Skipping.');
            return;
        }

        let sSolved;
        if (GPU) {
            try {
                const gpu = new GPU();
                const matMul = gpu.createKernel(function(A, b) {
                    let sum = 0;
                    for (let j = 0; j < this.constants.N; j++) {
                        sum += A[this.thread.y][j] * b[j];
                    }
                    return sum;
                }).setOutput([N]).setConstants({ N });
                sSolved = matMul(A, rhs);
                gpu.destroy();
            } catch (e) {
                logger.warn('Sheaf.diffuseQualia: GPU.js failed; falling back to worker.', e);
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
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);

            if (!P_uv || !isFiniteMatrix(P_uv) || !isFiniteVector(stalk_u) || !isFiniteVector(stalk_v)) {
                continue;
            }
            
            let projected_u;
            if (Numeric) {
                projected_u = Numeric.dot(P_uv, stalk_u);
            } else {
                projected_u = await runWorkerTask('matVecMul', { matrix: flattenMatrix(P_uv), vector: stalk_u }, 5000);
            }
            if (!isFiniteVector(projected_u)) continue;
            
            const diffNorm = norm2(vecSub(projected_u, stalk_v));
            if (Number.isFinite(diffNorm)) {
                sum += diffNorm;
                edgeCount++;
            }
        }
        this.inconsistency = edgeCount > 0 ? clamp(sum / edgeCount, 0, 10) : 0;
        this.inconsistencyHistory.push(new Float32Array([this.inconsistency]));
        return this.inconsistency;
    }

    async computeGestaltUnity() {
        const stalks = Array.from(this.stalks.values()).filter(isFiniteVector);
        if (stalks.length < 2) {
            this.gestaltUnity = 0;
            this.gestaltHistory.push(new Float32Array([this.gestaltUnity]));
            return 0;
        }

        let totalSimilarity = 0;
        let count = 0;
        if (GPU) {
            try {
                const gpu = new GPU();
                const computeSimilarity = gpu.createKernel(function(stalks, eps) {
                    let sum = 0;
                    for (let i = 0; i < stalks.length; i++) {
                        for (let j = i + 1; j < stalks.length; j++) {
                            let dotProd = 0;
                            let norm1 = 0;
                            let norm2 = 0;
                            for (let k = 0; k < this.constants.qDim; k++) {
                                dotProd += stalks[i][k] * stalks[j][k];
                                norm1 += stalks[i][k] * stalks[i][k];
                                norm2 += stalks[j][k] * stalks[j][k];
                            }
                            norm1 = Math.sqrt(norm1);
                            norm2 = Math.sqrt(norm2);
                            if (norm1 > eps && norm2 > eps) {
                                sum += Math.abs(dotProd / (norm1 * norm2));
                            }
                        }
                    }
                    return sum;
                }).setOutput([1]).setConstants({ qDim: this.qDim });
                totalSimilarity = computeSimilarity(stalks, this.eps);
                count = (stalks.length * (stalks.length - 1)) / 2;
                gpu.destroy();
            } catch (e) {
                logger.warn('Sheaf.computeGestaltUnity: GPU.js failed; falling back to CPU.', e);
            }
        }

        if (count === 0) {
            for (let i = 0; i < stalks.length; i++) {
                for (let j = i + 1; j < stalks.length; j++) {
                    const n1 = norm2(stalks[i]);
                    const n2 = norm2(stalks[j]);
                    if (n1 > this.eps && n2 > this.eps) {
                        const similarity = Math.abs(dot(stalks[i], stalks[j]) / (n1 * n2));
                        if (Number.isFinite(similarity)) {
                            totalSimilarity += similarity;
                            count++;
                        }
                    }
                }
            }
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
                    logger.warn(`Sheaf._computeLogDet: Division by near-zero diagonal element in U at i=${i}. Returning 0 for log-det.`);
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

    async computeIntegratedInformation() {
        let MI = 0;
        try {
            const validStates = this.windowedStates.getAll().filter(isFiniteVector);
            if (validStates.length < this.windowSize / 4) {
                logger.warn(`Sheaf.computeIntegratedInformation: Insufficient valid states (${validStates.length}/${this.windowSize}). Setting MI to 0.`);
                MI = 0;
            } else {
                const n_dim = validStates[0].length;
                const num_samples = validStates.length;

                if (num_samples <= n_dim) {
                    logger.warn(`Sheaf.computeIntegratedInformation: Insufficient unique samples (${num_samples}) for state dimension (${n_dim}). MI set to 0.`);
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
                                logger.warn(`Sheaf.computeIntegratedInformation: TensorFlow Cholesky failed even after regularization; trying logMatrixDeterminant: ${e.message}`);
                                logDet = tf.linalg.logMatrixDeterminant(covMatrix).logDeterminant.dataSync()[0];
                            }
                            MI = Number.isFinite(logDet) ? 0.1 * Math.abs(logDet) + this.eps : 0;
                            tf.dispose([statesTensor, rawCovMatrix, covMatrix, L]);
                        } catch (e) {
                            logger.warn(`Sheaf.computeIntegratedInformation: TensorFlow.js failed after regularization attempts: ${e.message}`, { stack: e.stack });
                        }
                    }

                    if (MI === 0) {
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
                                logger.warn('Sheaf.computeIntegratedInformation: Regularized CPU covarianceMatrix matrix is non-finite. MI set to 0.');
                                MI = 0;
                            }
                        } else {
                            logger.warn('Sheaf.computeIntegratedInformation: Non-finite raw CPU covarianceMatrix matrix generated. MI set to 0.');
                            MI = 0;
                        }
                    }
                }

                if (MI === 0) {
                    logger.info('Sheaf.computeIntegratedInformation: Falling back to KSG worker.');
                    const ksgMI = await runWorkerTask('ksgMutualInformation', { states: validStates, k: 3 }, 20000);
                    MI = Number.isFinite(ksgMI) ? ksgMI : 0;
                }
            }
        } catch (e) {
            logger.error(`Sheaf.computeIntegratedInformation: Error computing MI: ${e.message}`, { stack: e.stack });
            MI = 0;
        }

        const phiRaw = Math.log(1 + Math.abs(MI)) * this.stability * this.gestaltUnity * Math.exp(-this.inconsistency) * (1 + 0.05 * this.h1Dimension);
        this.phi = clamp(phiRaw, 0, 100);
        this.phiHistory.push(new Float32Array([this.phi]));
        return this.phi;
    }

    async computeCupProduct() {
        let totalIntensity = 0;
        let count = 0;

        for (const triangle of this.simplicialComplex.triangles) {
            if (!this.isValidTriangle(triangle)) {
                logger.warn(`Sheaf.computeCupProduct: Skipping invalid triangle ${triangle}`);
                continue;
            }
            const [u, v, w] = triangle;
            const s_u = this.stalks.get(u), s_v = this.stalks.get(v), s_w = this.stalks.get(w);
            if (!isFiniteVector(s_u) || !isFiniteVector(s_v) || !isFiniteVector(s_w)) {
                logger.warn(`Sheaf.computeCupProduct: Non-finite stalks for triangle ${triangle}`);
                continue;
            }

            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            const P_vw = this.projectionMatrices.get(`${v}-${w}`) || identity(this.qDim);

            try {
                let P_compose;
                if (Numeric) {
                    P_compose = Numeric.dot(P_uv, P_vw);
                } else {
                    P_compose = zeroMatrix(this.qDim, this.qDim);
                    for (let i = 0; i < this.qDim; i++) {
                        for (let j = 0; j < this.qDim; j++) {
                            let sum = 0;
                            for (let k = 0; k < this.qDim; k++) {
                                sum += (P_uv[i][k] || 0) * (P_vw[k][j] || 0);
                            }
                            P_compose[i][j] = clamp(sum, -1, 1);
                        }
                    }
                    if (!isFiniteMatrix(P_compose)) {
                        logger.warn(`Sheaf.computeCupProduct: Non-finite P_compose for triangle ${triangle}. Using identity.`);
                        P_compose = identity(this.qDim);
                    }
                }

                let s_w_projected;
                if (Numeric) {
                    s_w_projected = Numeric.dot(P_compose, s_w);
                } else {
                    s_w_projected = new Float32Array(this.qDim);
                    for (let i = 0; i < this.qDim; i++) {
                        let sum = 0;
                        for (let j = 0; j < this.qDim; j++) {
                            sum += (P_compose[i][j] || 0) * s_w[j];
                        }
                        s_w_projected[i] = clamp(sum, -1, 1);
                    }
                }

                if (!isFiniteVector(s_w_projected)) {
                    logger.warn(`Sheaf.computeCupProduct: Non-finite s_w_projected for triangle ${triangle}`);
                    continue;
                }

                const cupValue = dot(s_u, s_w_projected);
                if (Number.isFinite(cupValue)) {
                    totalIntensity += Math.abs(cupValue);
                    count++;
                }
            } catch (e) {
                logger.warn(`Sheaf.computeCupProduct: Error for triangle ${triangle}: ${e.message}`, { stack: e.stack });
            }
        }
        this.cup_product_intensity = count > 0 ? clamp(totalIntensity / count, 0, 1) : 0;
        return this.cup_product_intensity;
    }

    async computeH1Dimension() {
        const boundaryData = await this.buildBoundaryMatrices();
        const { partial1, partial2 } = boundaryData;
        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;

        if (nE === 0 || nV === 0) {
            this.h1Dimension = 0;
            this.stability = 1;
            return 0;
        }

        try {
            const [smithData1, smithData2] = await Promise.all([
                runWorkerTask('smithNormalForm', { matrix: partial1, field: 'GF2', bitPack: true }, 15000),
                runWorkerTask('smithNormalForm', { matrix: partial2, field: 'GF2', bitPack: true }, 15000)
            ]);

            const rank1 = smithData1?.rank ?? 0;
            const rank2 = smithData2?.rank ?? 0;

            if (nE > 0 && rank1 === 0) {
                logger.warn(`computeH1Dimension: Invalid rank1 (0) detected for non-empty graph. Computing cycles via Union-Find.`);
                const cycles = await this._computeBetti1UnionFind();
                this.h1Dimension = clamp(cycles, 0, nE);
            } else {
                this.h1Dimension = clamp(nE - rank1 - rank2, 0, nE);
            }
        } catch (e) {
            logger.error(`Sheaf.computeH1Dimension: Error computing Smith Normal Form: ${e.message}`, { stack: e.stack });
            const cycles = await this._computeBetti1UnionFind();
            this.h1Dimension = clamp(cycles, 0, nE);
        }

        if (!Number.isFinite(this.h1Dimension)) {
            logger.error(`Sheaf.computeH1Dimension: Non-finite h1Dimension calculated. Using Euler fallback.`);
            this.h1Dimension = clamp(nE - nV + 1, 0, nE);
        }

        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        return this.h1Dimension;
    }

    _buildIncidenceMatrix() {
        const n = this.graph.vertices.length;
        const m = this.graph.edges.length;
        const matrix = Array(n).fill().map(() => Array(m).fill(0));
        this.graph.edges.forEach(([u, v], j) => {
            const uIdx = this.graph.vertices.indexOf(u);
            const vIdx = this.graph.vertices.indexOf(v);
            if (uIdx !== -1 && vIdx !== -1) {
                matrix[uIdx][j] = 1;
                matrix[vIdx][j] = -1;
            } else {
                logger.warn(`_buildIncidenceMatrix: Invalid edge [${u}, ${v}] at index ${j}. Skipping.`);
            }
        });
        return matrix;
    }

    _computeBetti1UnionFind() {
        const vertices = this.graph.vertices;
        const edges = this.graph.edges;
        const n = vertices.length;
        const m = edges.length;

        const parent = Array(n).fill().map((_, i) => i);
        const rank = Array(n).fill(0);

        const find = (x) => {
            if (parent[x] !== x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };

        const union = (x, y) => {
            const px = find(x);
            const py = find(y);
            if (px === py) return;
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else if (rank[px] > rank[py]) {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px]++;
            }
        };

        edges.forEach(([u, v], i) => {
            const uIdx = vertices.indexOf(u);
            const vIdx = vertices.indexOf(v);
            if (uIdx === -1 || vIdx === -1) {
                logger.warn(`Sheaf._computeBetti1UnionFind: Invalid edge [${u}, ${v}] at index ${i}. Skipping.`);
                return;
            }
            union(uIdx, vIdx);
        });

        const components = new Set(parent.map((_, i) => find(i))).size;
        const beta1 = Math.max(0, m - n + components);
        if (beta1 > n) {
            logger.warn(`Sheaf._computeBetti1UnionFind: Unrealistic β1=${beta1} for n=${n}. Capping at n.`);
            return n;
        }
        logger.info(`Sheaf._computeBetti1UnionFind: Computed β1=${beta1}, components=${components}`);
        return beta1;
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
            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
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

        const A_flat = flattenMatrix(A).flatData;
        const result = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += (A_flat[i * N + j] || 0) * s_t[j];
            }
            result[i] = clamp(sum, -1, 1);
        }

        return result;
    }

    async computeStructuralSensitivity(perturbationScale = 0.05, numIterations = 10) {
        if (!this.ready) {
            this.structural_sensitivity = 0;
            return 0;
        }
        const numEdges = this.graph.edges.length;
        if (numEdges === 0) {
            this.structural_sensitivity = 0;
            return 0;
        }

        const s_t = this.getStalksAsVector();
        let baseState;
        try {
            baseState = await this.simulateDiffusionStep(s_t);
            if (!isFiniteVector(baseState)) {
                throw new Error("Base state for sensitivity analysis is non-finite.");
            }
        } catch (e) {
            logger.error('Sheaf.computeStructuralSensitivity: Could not compute base state.', e);
            this.structural_sensitivity = 0;
            return 0;
        }

        let totalSensitivity = 0;
        let perturbationCount = 0;

        const originalAdjacency = this.adjacencyMatrix.map(row => new Float32Array(row));

        const N = this.graph.vertices.length * this.qDim;
        const baseLfull = zeroMatrix(N, N);
        const idx = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const eta = this.gamma / Math.max(1, this.maxEigApprox);

        for (const [u, v] of this.graph.edges) {
            const i = idx.get(u), j = idx.get(v);
            if (i === undefined || j === undefined) continue;
            const weight = this.adjacencyMatrix[i]?.[j] || 0;
            if (!Number.isFinite(weight) || weight <= 0) continue;
            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    let val = -weight * (P_uv[qi]?.[qj] || 0);
                    if (qi !== qj) val += 0.1 * Math.sin(qi - qj) * weight;
                    if (Number.isFinite(val)) {
                        baseLfull[i * this.qDim + qi][j * this.qDim + qj] = clamp(val, -100, 100);
                        baseLfull[j * this.qDim + qi][i * this.qDim + qj] = clamp(val, -100, 100);
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
                baseLfull[i * this.qDim + qi][i * this.qDim + qi] = clamp(degree + this.eps, -100, 100);
            }
        }

        for (let eIdx = 0; eIdx < numEdges; eIdx++) {
            const [u, v] = this.graph.edges[eIdx];
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i < 0 || j < 0) continue;

            const originalWeight = originalAdjacency[i][j];
            this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = clamp(originalWeight + perturbationScale, 0.01, 1);

            const Lfull = baseLfull.map(row => new Float32Array(row));
            const weight = this.adjacencyMatrix[i][j];
            const P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
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
            for (let k = 0; k < this.graph.vertices.length; k++) {
                let degree = 0;
                for (let m = 0; m < this.graph.vertices.length; m++) {
                    if (k !== m && Number.isFinite(this.adjacencyMatrix[k]?.[m])) {
                        degree += this.adjacencyMatrix[k][m];
                    }
                }
                for (let qi = 0; qi < this.qDim; qi++) {
                    Lfull[k * this.qDim + qi][k * this.qDim + qi] = clamp(degree + this.eps, -100, 100);
                }
            }

            const A_perturbed = zeroMatrix(N, N).map((row, i) => row.map((v, j) => {
                const val = (i === j ? 1 : 0) - eta * (Lfull[i][j] || 0);
                return Number.isFinite(val) ? clamp(val, -100, 100) : 0;
            }));

            let perturbedState;
            try {
                if (GPU) {
                    const gpu = new GPU();
                    const matMul = gpu.createKernel(function(A, b) {
                        let sum = 0;
                        for (let j = 0; j < this.constants.N; j++) {
                            sum += A[this.thread.y][j] * b[j];
                        }
                        return sum;
                    }).setOutput([N]).setConstants({ N });
                    perturbedState = matMul(A_perturbed, s_t);
                    gpu.destroy();
                } else {
                    perturbedState = await runWorkerTask('matVecMul', {
                        matrix: flattenMatrix(A_perturbed).flatData,
                        vector: s_t
                    }, 5000);
                }
                if (!isFiniteVector(perturbedState)) {
                    logger.warn(`Sheaf.computeStructuralSensitivity: Non-finite perturbed state for edge ${u}-${v}. Skipping.`);
                    continue;
                }
                perturbedState = new Float32Array(perturbedState.map(v => clamp(v, -1, 1)));

                const diff = vecSub(perturbedState, baseState);
                const diffNorm = norm2(diff);
                if (Number.isFinite(diffNorm)) {
                    totalSensitivity += diffNorm / perturbationScale;
                    perturbationCount++;
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

    // Th. 3: Full free energy prior method
    async computeFreeEnergyPrior(nextState, predictedState) {
        if (!this.owm || !isFiniteVector(nextState) || !isFiniteVector(predictedState)) return vecZeros(this.qDim);
        // Simplified KL-divergence: L2 norm as proxy
        const klProxy = norm2(vecSub(nextState, predictedState)) * 0.1;
        const coherenceWeight = this.coherence || 0.5;
        const prior = vecScale(this.getStalksAsVector().slice(0, this.qDim), klProxy * coherenceWeight);
        return isFiniteVector(prior) ? prior : vecZeros(this.qDim);
    }

    async _updateDerivedMetrics() {
        try {
            await this.computeGluingInconsistency();
            await this.computeGestaltUnity();
            await this.computeIntegratedInformation();
            await this.computeCupProduct();
            await this.computeStructuralSensitivity(0.05, 10);

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

            const s = this.getStalksAsVector();
            if (isFiniteVector(s)) {
                const L_dense = this._csrToDense(this.laplacian);
                const energy = await runWorkerTask('quadraticForm', {
                    matrix: flattenMatrix(L_dense).flatData,
                    vector: s
                }, 5000);
                this.diffusionEnergy = Number.isFinite(energy) ? clamp(energy, 0, 100) : 0;
            } else {
                this.diffusionEnergy = 0;
            }
        } catch (e) {
            logger.error(`Sheaf._updateDerivedMetrics: Error updating metrics: ${e.message}`, { stack: e.stack });
            this.feel_F = 0;
            this.intentionality_F = 0;
            this.cup_product_intensity = 0;
            this.structural_sensitivity = 0;
            this.diffusionEnergy = 0;
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
            logger.error(`Sheaf.update: Error during update: ${e.message}`, { stack: e.stack });
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
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(0.2, 0.5 + norm * 0.5, 1.0),
                emissive: new THREE.Color(0, norm * 0.2, 0.2)
            });
            const sphere = new THREE.Mesh(sphereGeometry, material);
            sphere.position.set(vertexMap.get(v) * 2 - 7, norm * 2, 0);
            stalkGroup.add(sphere);
        });

        this.graph.edges.forEach(([u, v, weight]) => {
            const i = vertexMap.get(u);
            const j = vertexMap.get(v);
            const norm_u = norm2(this.stalks.get(u) || [0]);
            const norm_v = norm2(this.stalks.get(v) || [0]);

            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(i * 2 - 7, norm_u * 2, 0),
                new THREE.Vector3(j * 2 - 7, norm_v * 2, 0)
            ]);
            const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({
                color: 0x44aaFF,
                transparent: true,
                opacity: clamp((weight || 0) * this.cup_product_intensity, 0.2, 0.8)
            }));
            stalkGroup.add(line);
        });

        scene.add(stalkGroup);
        return stalkGroup;
    }

    saveState() {
        return {
            graph: { ...this.graph },
            simplicialComplex: { ...this.simplicialComplex },
            stalks: Array.from(this.stalks.entries()),
            projectionMatrices: Array.from(this.projectionMatrices.entries()),
            correlationMatrix: this.correlationMatrix,
            adjacencyMatrix: this.adjacencyMatrix,
            laplacian: this.laplacian,
            phi: this.phi,
            h1Dimension: this.h1Dimension,
            gestaltUnity: this.gestaltUnity,
            stability: this.stability,
            diffusionEnergy: this.diffusionEnergy,
            inconsistency: this.inconsistency,
            feel_F: this.feel_F,
            intentionality_F: this.intentionality_F,
            cup_product_intensity: this.cup_product_intensity,
            structural_sensitivity: this.structural_sensitivity,
            coherence: this.coherence, // Th. 1
            stalkHistory: this.stalkHistory.getAll(),
            windowedStates: this.windowedStates.getAll(),
            phiHistory: this.phiHistory.getAll(),
            gestaltHistory: this.gestaltHistory.getAll(),
            inconsistencyHistory: this.inconsistencyHistory.getAll(),
            ready: this.ready
        };
    }

    loadState(state) {
        if (!state || typeof state !== 'object') {
            logger.error('Sheaf.loadState: Invalid state object.');
            return;
        }
        this.graph = state.graph || { vertices: [], edges: [] };
        this.simplicialComplex = state.simplicialComplex || { triangles: [], tetrahedra: [] };
        this.stalks = new Map(state.stalks || []);
        this.projectionMatrices = new Map(state.projectionMatrices || []);
        this.correlationMatrix = state.correlationMatrix || zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);
        this.adjacencyMatrix = state.adjacencyMatrix || zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);
        this.laplacian = state.laplacian || { values: new Float32Array(0), colIndices: new Int32Array(0), rowPtr: new Int32Array(this.graph.vertices.length + 1).fill(0), n: this.graph.vertices.length };
        this.phi = state.phi || 0.2;
        this.h1Dimension = state.h1Dimension || 0;
        this.gestaltUnity = state.gestaltUnity || 0.6;
        this.stability = state.stability || 0.6;
        this.diffusionEnergy = state.diffusionEnergy || 0;
        this.inconsistency = state.inconsistency || 0;
        this.feel_F = state.feel_F || 0;
        this.intentionality_F = state.intentionality_F || 0;
        this.cup_product_intensity = state.cup_product_intensity || 0;
        this.structural_sensitivity = state.structural_sensitivity || 0;
        this.coherence = state.coherence || 0; // Th. 1
        this.stalkHistory = new CircularBuffer(this.stalkHistorySize);
        (state.stalkHistory || []).forEach(item => this.stalkHistory.push(item));
        this.windowedStates = new CircularBuffer(this.windowSize);
        (state.windowedStates || []).forEach(item => this.windowedStates.push(item));
        this.phiHistory = new CircularBuffer(this.stalkHistorySize);
        (state.phiHistory || []).forEach(item => this.phiHistory.push(item));
        this.gestaltHistory = new CircularBuffer(this.stalkHistorySize);
        (state.gestaltHistory || []).forEach(item => this.gestaltHistory.push(item));
        this.inconsistencyHistory = new CircularBuffer(this.stalkHistorySize);
        (state.inconsistencyHistory || []).forEach(item => this.inconsistencyHistory.push(item));
        this.ready = state.ready || false;
        this.edgeSet = new Set(this.graph.edges.map(e => e.slice(0, 2).sort().join(',')));
    }
}

/**
 * Theorem 14: RecursiveTopologicalSheaf – Fixed-Point Cohomology Extension.
 * Induces Banach contractions on cochains for reflexive fixed points.
 */
class RecursiveTopologicalSheaf extends EnhancedQualiaSheaf {
    constructor(graphData, config) {
        super(graphData, config);
        this.R_star = this._buildRecursiveGluing(); // Recursive gluing operator
        this.maxIter = config.maxIter || 50;
        this.fixedPointEps = config.fixedPointEps || 1e-6;
        this.tau = config.tau || 2.5; // Awareness threshold
        this.cochainHistory = new CircularBuffer(20); // Track z_t for contraction
    }

    _buildRecursiveGluing() {
        return (z) => {
            const z_next = new Map();
            this.graph.edges.forEach(([u, v]) => {
                const su = this.stalks.get(u) || vecZeros(this.qDim);
                const sv = this.stalks.get(v) || vecZeros(this.qDim);
                const phi_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
                const diffusion = vecScale(vecAdd(su, sv), this.alpha / 2);
                const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.qDim);
                const z_next_uv = vecAdd(matVecMul(phi_uv, z_uv), diffusion); // R^*(z)
                if (isFiniteVector(z_next_uv)) {
                    z_next.set([u, v].sort().join(','), z_next_uv);
                } else {
                    z_next.set([u, v].sort().join(','), vecZeros(this.qDim));
                    logger.warn(`RecursiveTopologicalSheaf._buildRecursiveGluing: Non-finite z_next for edge ${u}-${v}`);
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
                z_uv[k] = clamp(input_u - input_v, -1, 1) * (this.entityNames[k].includes('symbolic') ? 1.5 : 1);
            }
            if (isFiniteVector(z_uv)) {
                z.set([u, v].sort().join(','), z_uv);
            }
        });
        return z;
    }

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
        const { diagonal, rank } = await runWorkerTask('smithNormalForm', { matrix: flattenMatrix(L_rec), field: 'R' }, 15000);
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
            L_rec[i][i] = 1; // I - R*
            const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            const phi_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            for (const [u2, v2] of this.graph.edges) {
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || i === j) continue;
                const phi_u2v2 = this.projectionMatrices.get(`${u2}-${v2}`) || identity(this.qDim);
                const interaction = dot(z_uv, matVecMul(phi_u2v2, z.get([u2, v2].sort().join(',')) || vecZeros(this.qDim)));
                if (Number.isFinite(interaction)) {
                    L_rec[i][j] = -this.alpha * clamp(interaction, -0.1, 0.1);
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
            this.graph.edges.forEach(([u, v]) => {
                const su = F.stalks.get(u) || vecZeros(this.qDim);
                const sv = F.stalks.get(v) || vecZeros(this.qDim);
                const c1 = vecScale(vecAdd(su, sv), this.alpha); // C^1 component
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
            this.graph.vertices.forEach(v => {
                let sum = vecZeros(this.qDim);
                let count = 0;
                this.graph.edges.forEach(([u, w]) => {
                    if (u === v || w === v) {
                        const c_uw = C.cochains.get([u, w].sort().join(',')) || vecZeros(this.qDim);
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
            const eta_F = new Map();
            this.graph.vertices.forEach(v => {
                const stalk = F.stalks.get(v) || vecZeros(this.qDim);
                const scaled = vecScale(stalk, 1 + this.alpha);
                if (isFiniteVector(scaled)) {
                    eta_F.set(v, scaled);
                }
            });
            return eta_F;
        };
    }

    _counit() {
        return (FU, Id) => {
            const epsilon_FU = new Map();
            this.graph.vertices.forEach(v => {
                const fu_v = FU.stalks.get(v) || vecZeros(this.qDim);
                const id_v = Id.stalks.get(v) || vecZeros(this.qDim);
                const diff = vecSub(fu_v, id_v);
                if (isFiniteVector(diff)) {
                    epsilon_FU.set(v, diff);
                }
            });
            return epsilon_FU;
        };
    }

    async computeAdjunctionFixedPoint(init_state) {
        const F_init = { stalks: this._initStalks(init_state) };
        let T_state = this.T(F_init);
        let iter = 0;
        while (iter < this.maxIter) {
            const next_T = this.T(T_state);
            const eta_T = this.eta(T_state);
            const epsilon_FU = this.epsilon(this.F(next_T), T_state);
            const eq_delta = await this._equalizerNorm(eta_T, epsilon_FU);
            if (!Number.isFinite(eq_delta) || eq_delta < this.equalizerEps) break;
            T_state = next_T;
            iter++;
        }
        const z_star = await this._extractFixedCocycle(T_state);
        const L_adj = await this._adjunctionLaplacian(T_state);
        const { diagonal, rank } = await runWorkerTask('smithNormalForm', { matrix: flattenMatrix(L_adj), field: 'R' }, 15000);
        const beta1_adj = Math.max(0, this.graph.edges.length - rank);
        const cov_zstar = covarianceMatrix(Array.from(z_star.values()).filter(isFiniteVector));
        const logDet = logDeterminantFromDiagonal(cov_zstar);
        const Phi_SA = Number.isFinite(logDet) ? logDet * beta1_adj * this.graph.vertices.length : 0;
        this.hierarchicallyAware = Phi_SA > 3.0;
        return { F_star: T_state, z_star, Phi_SA, aware: this.hierarchicallyAware };
    }

    async _equalizerNorm(eta, eps) {
        let sum = 0;
        let count = 0;
        for (const v of this.graph.vertices) {
            const eta_v = eta.get(v) || vecZeros(this.qDim);
            const eps_v = eps.get(v) || vecZeros(this.qDim);
            if (isFiniteVector(eta_v) && isFiniteVector(eps_v)) {
                sum += norm2(vecSub(eta_v, eps_v)) ** 2;
                count++;
            }
        }
        const norm = count > 0 ? Math.sqrt(sum / count) : 0;
        return Number.isFinite(norm) ? norm : 0;
    }

    async _extractFixedCocycle(T) {
        const z_star = new Map();
        const C1 = this.F(T).cochains;
        const delta1 = await this._deltaR1();
        for (const [key, c] of C1) {
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
        const C1 = this.F(T).cochains;
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
                stalk[k] = clamp(input * (this.entityNames[k].includes('metacognition') ? 1.2 : 1), -1, 1);
            }
            stalks.set(v, stalk);
        });
        return stalks;
    }

    _tensorSections(stalks, C1) {
        const result = new Map();
        this.graph.vertices.forEach(v => {
            const s_v = stalks.get(v) || vecZeros(this.qDim);
            let sum = vecZeros(this.qDim);
            let count = 0;
            this.graph.edges.forEach(([u, w]) => {
                if (u === v || w === v) {
                    const c_uw = C1.get([u, w].sort().join(',')) || vecZeros(this.qDim);
                    const tensor = vecMul(s_v, c_uw);
                    if (isFiniteVector(tensor)) {
                        sum = vecAdd(sum, tensor);
                        count++;
                    }
                }
            });
            result.set(v, count > 0 ? vecScale(sum, 1 / count) : s_v);
        });
        return result;
    }

    _buildC1Cochains() {
        const C1 = new Map();
        this.graph.edges.forEach(([u, v]) => {
            const su = this.stalks.get(u) || vecZeros(this.qDim);
            const sv = this.stalks.get(v) || vecZeros(this.qDim);
            const c = vecSub(su, sv);
            if (isFiniteVector(c)) {
                C1.set([u, v].sort().join(','), c);
            }
        });
        return C1;
    }

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
}

/**
 * Theorem 16: PersistentAdjunctionSheaf – Flow Persistence Extension.
 * Bottleneck PD over filtrations for diachronic invariants.
 */
class PersistentAdjunctionSheaf extends AdjunctionReflexiveSheaf {
    constructor(graphData, config) {
        super(graphData, config);
        this.flowHistory = new CircularBuffer(config.flowBufferSize || 50);
        this.persistenceDiagram = { births: [], deaths: [] };
        this.delta = config.delta || 0.1;
        this.tau_persist = config.tau_persist || 3.5;
    }

    _partial_t(F_t, dt = 1) {
        const stalksNext = new Map();
        this.graph.vertices.forEach(v => {
            const stalk = F_t.stalks.get(v) || vecZeros(this.qDim);
            const neighbors = this.graph.edges.filter(e => e[0] === v || e[1] === v).map(e => e[0] === v ? e[1] : e[0]);
            let grad = vecZeros(this.qDim);
            neighbors.forEach(u => {
                const su = F_t.stalks.get(u) || vecZeros(this.qDim);
                const phi_vu = this.projectionMatrices.get(`${v}-${u}`) || identity(this.qDim);
                const diff = vecSub(stalk, matVecMul(phi_vu, su));
                if (isFiniteVector(diff)) {
                    grad = vecAdd(grad, vecScale(diff, this.beta));
                }
            });
            const noise = vecScale(new Float32Array(this.qDim).map(() => Math.random() - 0.5), this.sigma * Math.sqrt(dt));
            const diffused = vecAdd(vecScale(stalk, 1 - this.gamma * dt), vecAdd(grad, noise));
            stalksNext.set(v, isFiniteVector(diffused) ? diffused : stalk);
        });
        return { stalks: stalksNext, projections: F_t.projections };
    }

    async computePersistentFixedPoint(init_state, T = 10) {
        let F_curr = this.F({ stalks: this._initStalks(init_state) });
        this.flowHistory.clear();
        let Phi_SA_persist = 0;
        let persist_aware = false;

        for (let t = 0; t < T; t++) {
            const partial_F = this._partial_t(F_curr, this.gamma);
            const F_next = this.F(partial_F);
            const U_next = this.U(F_next.cochains);
            const T_next = this.T(F_next);

            const eta_next = this.eta(T_next);
            const eta_prev = this.flowHistory.get(this.flowHistory.length - 1)?.eta_t || eta_next;
            const eta_evol = await this._nablaPersist(eta_next, eta_prev);
            const epsilon_next = this.epsilon(F_next, U_next);

            const eq_delta = await this._equalizerNorm(eta_evol, epsilon_next);
            if (!Number.isFinite(eq_delta) || eq_delta < this.eps) break;

            const L_Tt = await this._flowLaplacian(T_next);
            const { eigenvalues } = await this._spectralDecomp(L_Tt);
            const lambda_min_t = Math.min(...eigenvalues.filter(Number.isFinite));
            const beta1_persist = await this._supPersistentBetti(this.persistenceDiagram);
            Phi_SA_persist += lambda_min_t * beta1_persist * this.gamma;

            this.persistenceDiagram = this._updatePD(this.persistenceDiagram, eigenvalues, t);
            const d_B = this._bottleneckDistance(this.persistenceDiagram);
            if (d_B < this.delta) persist_aware = true;

            F_curr = F_next;
            this.flowHistory.push({ F_t: F_next, U_t: U_next, eta_t: eta_evol, lambda_t: eigenvalues });
        }

        const z_star_persist = await this._extractPersistCocycle(this.flowHistory.getAll());
        Phi_SA_persist = clamp(Phi_SA_persist, 0, 100);
        this.diachronicallyAware = Phi_SA_persist > this.tau_persist;

        return { F_persist: F_curr, z_star_persist, Phi_SA_persist, PD: this.persistenceDiagram, aware: this.diachronicallyAware };
    }

    async _nablaPersist(eta_next, eta_prev) {
        const eta_evol = new Map();
        this.graph.vertices.forEach(v => {
            const next_v = eta_next.get(v) || vecZeros(this.qDim);
            const prev_v = eta_prev.get(v) || vecZeros(this.qDim);
            const diff = vecSub(next_v, prev_v);
            const evol = vecAdd(next_v, vecScale(diff, this.gamma));
            if (isFiniteVector(evol)) {
                eta_evol.set(v, evol);
            }
        });
        return eta_evol;
    }

    async _flowLaplacian(T_t) {
        const n = this.graph.vertices.length * this.qDim;
        const L_base = this._csrToDense(this.laplacian);
        const T_matrix = zeroMatrix(n, n);
        const C1 = T_t.cochains || new Map();
        this.graph.edges.forEach(([u, v], i) => {
            const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            for (let qi = 0; qi < this.qDim; qi++) {
                T_matrix[i * this.qDim + qi][i * this.qDim + qi] = 1 - this.beta * norm2(c_uv);
            }
        });
        const L_Tt = vecAdd(L_base, vecSub(T_matrix, identity(n)));
        return isFiniteMatrix(L_Tt) ? L_Tt : identity(n);
    }

    async _spectralDecomp(L) {
        if (tf) {
            try {
                const L_tensor = tf.tensor2d(L);
                const { values } = tf.linalg.eigvals(L_tensor);
                const eigenvalues = await values.data();
                tf.dispose([L_tensor, values]);
                return { eigenvalues: Array.from(eigenvalues).filter(Number.isFinite) };
            } catch (e) {
                logger.warn(`_spectralDecomp: TF.js eig failed: ${e.message}`);
            }
        }
        const eigenvalues = await runWorkerTask('eigenvalues', { matrix: flattenMatrix(L) }, 20000);
        return { eigenvalues: Array.from(eigenvalues).filter(Number.isFinite) };
    }

    _updatePD(pd_old, lambda_t, birth_time) {
        const pd = { births: [...pd_old.births], deaths: [...pd_old.deaths] };
        lambda_t.forEach(lambda => {
            if (Number.isFinite(lambda) && !pd.births.some(b => Math.abs(b.value - lambda) < this.eps)) {
                pd.births.push({ value: lambda, time: birth_time });
            }
        });
        pd.births.forEach((birth, i) => {
            if (!pd.deaths[i] && birth.time < birth_time - this.delta) {
                pd.deaths[i] = { value: birth.value, time: birth_time };
            }
        });
        return pd;
    }

    _bottleneckDistance(pd) {
        let maxDist = 0;
        for (let i = 0; i < pd.births.length; i++) {
            const birth = pd.births[i];
            const death = pd.deaths[i] || { value: birth.value, time: birth.time + this.delta };
            const dist = Math.abs(death.time - birth.time);
            if (Number.isFinite(dist) && dist > maxDist) {
                maxDist = dist;
            }
        }
        return maxDist;
    }

    async _supPersistentBetti(pd) {
        const births = pd.births.filter(b => Number.isFinite(b.value));
        const deaths = pd.deaths.filter(d => Number.isFinite(d.value));
        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const lifetime = (deaths[i]?.time || Infinity) - births[i].time;
            if (lifetime > this.delta) count++;
        }
        return count;
    }

    async _extractPersistCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        for (const { F_t } of flows) {
            const C1 = this.F(F_t).cochains;
            for (const [key, c] of C1) {
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
        super(graphData, config);
        this.omega = config.omega || 8;
        this.monodromy = null;
        this.floquetPD = { phases: [], births: [], deaths: [] };
        this.theta_k = config.theta_k || [4, 6, 8]; // Theta bands (Hz)
        this.tau_floq = config.tau_floq || 4.0;
    }

    async _monodromy(A_t, omega) {
        if (tf) {
            try {
                const A_tensor = tf.tensor2d(A_t);
                const A_omega = tf.linalg.matrixPower(A_tensor, omega);
                const result = await A_omega.data();
                tf.dispose([A_tensor, A_omega]);
                return unflattenMatrix({ flatData: new Float32Array(result), rows: A_t.length, cols: A_t[0].length });
            } catch (e) {
                logger.warn(`_monodromy: TF.js matrixPower failed: ${e.message}`);
            }
        }
        let result = A_t;
        for (let i = 1; i < omega; i++) {
            const temp = zeroMatrix(A_t.length, A_t[0].length);
            for (let r = 0; r < A_t.length; r++) {
                for (let c = 0; c < A_t[0].length; c++) {
                    let sum = 0;
                    for (let k = 0; k < A_t.length; k++) {
                        sum += (result[r][k] || 0) * (A_t[k][c] || 0);
                    }
                    temp[r][c] = clamp(sum, -100, 100);
                }
            }
            result = temp;
        }
        return isFiniteMatrix(result) ? result : identity(A_t.length);
    }

    async computeFloquetFixedPoint(init_state, omega = this.omega, levels = 3) {
        let F_curr = this.F({ stalks: this._initStalks(init_state) });
        let Phi_SA_floq = 0;
        let floq_aware = false;

        for (let lvl = 0; lvl < levels; lvl++) {
            const A_t = await this._flowMonodromy(F_curr, omega);
            this.monodromy = A_t;

            const { eigenvalues: rho_k } = await this._floquetDecomp(A_t);
            const log_rho_sum = rho_k.reduce((sum, rho) => {
                const mag = Math.sqrt((rho.re || rho) ** 2 + (rho.im || 0) ** 2);
                return sum + (Number.isFinite(mag) && mag > 0 ? Math.log(mag) : 0);
            }, 0);

            const theta = this.theta_k[lvl % this.theta_k.length] * 2 * Math.PI / omega;
            const eta_floq = this.eta(F_curr).map((stalk, v) =>
                stalk.map(val => val * Math.cos(theta * lvl))
            );
            const epsilon_floq = this.epsilon(F_curr, this.U(this.F(F_curr).cochains));

            const eq_floq_delta = await this._floquetEqualizer(eta_floq, epsilon_floq);
            if (!Number.isFinite(eq_floq_delta) || eq_floq_delta < this.eps) break;

            this.floquetPD = this._updateFloqPD(this.floquetPD, rho_k, lvl * omega);
            const d_B_floq = this._rhythmicBottleneck(this.floquetPD);
            Phi_SA_floq += Math.abs(log_rho_sum) * (await this._supFloqBetti(this.floquetPD));

            if (d_B_floq < this.delta) floq_aware = true;

            F_curr = await this._floquetJoin(F_curr, A_t, rho_k);
        }

        const z_star_floq = await this._extractFloqCocycle(this.flowHistory.getAll());
        const beta1_floq = await this._supFloqBetti(this.floquetPD);
        Phi_SA_floq = clamp(Phi_SA_floq * beta1_floq, 0, 100);
        this.rhythmicallyAware = Phi_SA_floq > this.tau_floq;

        return { A_floq: this.monodromy, z_star_floq, Phi_SA_floq, FloqPD: this.floquetPD, aware: this.rhythmicallyAware };
    }

    async _flowMonodromy(F_t, omega) {
        const T_t = this.T(F_t).cochains;
        const nE = this.graph.edges.length;
        const A_t = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        for (const [u, v] of this.graph.edges) {
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined) continue;
            A_t[i][i] = 1;
            const c_uv = T_t.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            for (const [u2, v2] of this.graph.edges) {
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || i === j) continue;
                const c_u2v2 = T_t.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                const interaction = dot(c_uv, c_u2v2);
                if (Number.isFinite(interaction)) {
                    A_t[i][j] = this.beta * clamp(interaction, -0.1, 0.1);
                }
            }
        }
        return this._monodromy(A_t, omega);
    }

    async _floquetDecomp(A) {
        if (tf) {
            try {
                const A_tensor = tf.tensor2d(A);
                const { values } = tf.linalg.eig(A_tensor);
                const eigenvalues = await values.data();
                tf.dispose([A_tensor, values]);
                return { eigenvalues: Array.from(eigenvalues).map(v => ({
                    re: v.re !== undefined ? v.re : v,
                    im: v.im || 0
                })).filter(v => Number.isFinite(v.re) && Number.isFinite(v.im)) };
            } catch (e) {
                logger.warn(`_floquetDecomp: TF.js eig failed: ${e.message}`);
            }
        }
        const n = A.length;
        const companion = zeroMatrix(n, n);
        for (let i = 0; i < n - 1; i++) companion[i][i + 1] = 1;
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) sum += A[i][j] * (j === n - 1 ? 1 : 0);
            companion[n - 1][i] = -sum;
        }
        const eigenvalues = await runWorkerTask('complexEigenvalues', { matrix: flattenMatrix(companion) }, 20000);
        return { eigenvalues: Array.from(eigenvalues).filter(v => Number.isFinite(v.re) && Number.isFinite(v.im)) };
    }

    async _floquetEqualizer(eta_f, eps_f) {
        let sum = 0;
        let count = 0;
        for (const v of this.graph.vertices) {
            const eta_v = eta_f.get(v) || vecZeros(this.qDim);
            const eps_v = eps_f.get(v) || vecZeros(this.qDim);
            if (isFiniteVector(eta_v) && isFiniteVector(eps_v)) {
                const phase_diff = vecSub(eta_v, eps_v);
                sum += norm2(phase_diff) ** 2;
                count++;
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    _updateFloqPD(pd_old, rho_t, phase) {
        const pd = { phases: [...pd_old.phases], births: [...pd_old.births], deaths: [...pd_old.deaths] };
        rho_t.forEach(rho => {
            const mag = Math.sqrt(rho.re ** 2 + rho.im ** 2);
            const theta = Math.atan2(rho.im, rho.re);
            if (Number.isFinite(mag) && !pd.births.some(b => Math.abs(b.value - mag) < this.eps)) {
                pd.births.push({ value: mag, phase, time: phase });
                pd.phases.push(theta);
            }
        });
        pd.births.forEach((birth, i) => {
            if (!pd.deaths[i] && birth.time < phase - this.delta) {
                pd.deaths[i] = { value: birth.value, phase: pd.phases[i], time: phase };
            }
        });
        return pd;
    }

    _rhythmicBottleneck(pd) {
        let maxDist = 0;
        for (let i = 0; i < pd.births.length; i++) {
            const birth = pd.births[i];
            const death = pd.deaths[i] || { value: birth.value, time: birth.time + this.delta };
            const dist = Math.abs(death.time - birth.time);
            if (Number.isFinite(dist) && dist > maxDist) {
                maxDist = dist;
            }
        }
        return maxDist;
    }

    async _supFloqBetti(pd) {
        const births = pd.births.filter(b => Number.isFinite(b.value));
        const deaths = pd.deaths.filter(d => Number.isFinite(d.value));
        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const lifetime = (deaths[i]?.time || Infinity) - births[i].time;
            if (lifetime > this.delta) count++;
        }
        return count;
    }

    async _extractFloqCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        for (const { F_t } of flows) {
            const C1 = this.F(F_t).cochains;
            for (const [key, c] of C1) {
                const delta_c = matVecMul(delta1, c);
                if (isFiniteVector(delta_c) && norm2(delta_c) < this.eps) {
                    const phase = this.floquetPD.phases.find(p => Math.abs(p - this.theta_k[0]) < this.eps) || 0;
                    z_star.set(key, vecScale(c, Math.cos(phase)));
                }
            }
        }
        return z_star;
    }

    async _floquetJoin(F, A, rho_k) {
        const stalks = new Map(F.stalks);
        this.graph.vertices.forEach(v => {
            let stalk = stalks.get(v) || vecZeros(this.qDim);
            rho_k.forEach(rho => {
                const theta = Math.atan2(rho.im || 0, rho.re || rho);
                stalk = vecAdd(stalk, vecScale(stalk, this.beta * Math.cos(theta)));
            });
            if (isFiniteVector(stalk)) {
                stalks.set(v, stalk);
            }
        });
        return { stalks, projections: F.projections };
    }

    async update(state, stepCount = 0) {
        if (!this.ready) {
            logger.warn('Sheaf.update: Sheaf not ready. Initializing.');
            await this.initialize();
            if (!this.ready) {
                logger.error('Sheaf.update: Initialization failed.');
                return;
            }
        }

        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn('Sheaf.update: Invalid input state.', { state });
            return;
        }

        try {
            // Base diffusion with Th. 1 coherence
            await this.diffuseQualia(state);

            // Th14: Recursive fixed-point
            const { Phi_SA, aware } = await this.computeSelfAwareness(state);
            logger.info(`Th14: Φ_SA=${Phi_SA.toFixed(2)}, Aware=${aware}`);

            // Th15: Adjunction fixed-point
            const { Phi_SA: Phi_SA_adj, aware: adj_aware } = await this.computeAdjunctionFixedPoint(state);
            logger.info(`Th15: Φ_SA_adj=${Phi_SA_adj.toFixed(2)}, HierarchicallyAware=${adj_aware}`);

            // Th16: Persistent flow
            const { Phi_SA_persist, aware: persist_aware, PD } = await this.computePersistentFixedPoint(state);
            logger.info(`Th16: Φ_SA_persist=${Phi_SA_persist.toFixed(2)}, DiachronicallyAware=${persist_aware}, d_B=${PD.births.length > 0 ? this._bottleneckDistance(PD).toFixed(3) : 'N/A'}`);

            // Th17: Floquet rhythm
            const { Phi_SA_floq, aware: floq_aware, FloqPD } = await this.computeFloquetFixedPoint(state);
            logger.info(`Th17: Φ_SA_floq=${Phi_SA_floq.toFixed(2)}, RhythmicallyAware=${floq_aware}, d_B_floq=${FloqPD.births.length > 0 ? this._rhythmicBottleneck(FloqPD).toFixed(3) : 'N/A'}`);

            // Ethical guard
            if (Phi_SA_floq > 10) {
                logger.warn('Th17: Φ_SA_floq exceeds ethical threshold (10). Pruning PD.');
                this.floquetPD.births = this.floquetPD.births.slice(0, Math.floor(this.floquetPD.births.length / 2));
                Phi_SA_floq = Math.min(Phi_SA_floq, 10);
            }

            // Topology adaptation
            if (stepCount > 0 && stepCount % 100 === 0) {
                const addThresh = this.adaptation.addThresh + (Phi_SA_floq / 10) * 0.1;
                const removeThresh = this.adaptation.removeThresh + (1 - floq_aware) * 0.2;
                await this.adaptSheafTopology(100, stepCount, addThresh, removeThresh);
            }

            // Update metrics with Floquet boost
            this.phi = clamp(this.phi + 0.1 * Phi_SA_floq, 0, 100);
            await this._updateDerivedMetrics();
        } catch (e) {
            logger.error(`Sheaf.update: Error during update: ${e.message}`, { stack: e.stack });
            this.ready = false;
        }
    }

    async adaptSheafTopology(adaptFreq = 100, stepCount = 0, addThresh = this.adaptation.addThresh, removeThresh = this.adaptation.removeThresh) {
        if (!this.ready || stepCount % adaptFreq !== 0) return;

        if (this.stalkHistory.length < this.stalkHistorySize / 2) {
            logger.info(`Sheaf: Skipping topology adaptation at step ${stepCount}; insufficient history (${this.stalkHistory.length}/${this.stalkHistorySize}).`);
            return;
        }

        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        if (!isFiniteMatrix(this.correlationMatrix)) {
            logger.warn('Sheaf: Non-finite correlation matrix; skipping adaptation.');
            return;
        }

        try {
            const d_B = this.persistenceDiagram.births.length > 0 ? this._bottleneckDistance(this.persistenceDiagram) : 0;
            const d_B_floq = this.floquetPD.births.length > 0 ? this._rhythmicBottleneck(this.floquetPD) : 0;
            addThresh += (1 - d_B) * 0.05;
            removeThresh += d_B_floq * 0.1;
            this.adaptEdges(this.correlationMatrix, addThresh, removeThresh);
            this.adaptSimplices(this.correlationMatrix, this.adaptation.targetH1 + (this.rhythmicallyAware ? 0.5 : 0));
            await this.computeCorrelationMatrix();
            await this.computeH1Dimension();
            await this._updateDerivedMetrics();
            logger.info(`Sheaf adapted at step ${stepCount}    // End of adaptSheafTopology method
    with Floquet-enhanced topology: d_B=${d_B.toFixed(3)}, d_B_floq=${d_B_floq.toFixed(3)}`);
    } catch (e) {
      logger.error(`Sheaf.adaptSheafTopology: Failed: ${e.message}`, { stack: e.stack });
    }
  }

  visualizeFloquetPD(scene, camera, renderer) {
    if (!THREE) {
      logger.error("Sheaf.visualizeFloquetPD: THREE.js is not available.");
      return;
    }

    const pdGroup = new THREE.Group();
    const geometry = new THREE.SphereGeometry(0.1, 8, 8);

    this.floquetPD.births.forEach((birth, i) => {
      const death = this.floquetPD.deaths[i] || { time: birth.time + this.delta, phase: birth.phase };
      const lifetime = death.time - birth.time;
      if (lifetime < this.delta) return;

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
    return pdGroup;
  }

  saveState() {
    const baseState = super.saveState();
    return {
      ...baseState,
      flowHistory: this.flowHistory.getAll(),
      persistenceDiagram: this.persistenceDiagram,
      floquetPD: this.floquetPD,
      monodromy: this.monodromy,
      cochainHistory: this.cochainHistory.getAll(),
      selfAware: this.selfAware,
      hierarchicallyAware: this.hierarchicallyAware,
      diachronicallyAware: this.diachronicallyAware,
      rhythmicallyAware: this.rhythmicallyAware
    };
  }

  loadState(state) {
    if (!state || typeof state !== 'object') {
      logger.error('Sheaf.loadState: Invalid state object.');
      return;
    }
    super.loadState(state);
    this.flowHistory = new CircularBuffer(state.flowHistory?.length || 50);
    (state.flowHistory || []).forEach(item => this.flowHistory.push(item));
    this.persistenceDiagram = state.persistenceDiagram || { births: [], deaths: [] };
    this.floquetPD = state.floquetPD || { phases: [], births: [], deaths: [] };
    this.monodromy = state.monodromy || null;
    this.cochainHistory = new CircularBuffer(state.cochainHistory?.length || 20);
    (state.cochainHistory || []).forEach(item => this.cochainHistory.push(item));
    this.selfAware = state.selfAware || false;
    this.hierarchicallyAware = state.hierarchicallyAware || false;
    this.diachronicallyAware = state.diachronicallyAware || false;
    this.rhythmicallyAware = state.rhythmicallyAware || false;
  }
}

// --- END OF FILE qualia-sheaf.js ---
