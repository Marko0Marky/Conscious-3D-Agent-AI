// --- START OF FILE qualia-sheaf.js ---
import { 
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity
} from './utils.js';

/**
 * Represents an Enhanced Qualia Sheaf, a mathematical structure used to model the AI's "consciousness".
 * It handles qualia diffusion, topological adaptation, and metric computations like Phi, H1 dimension, etc.
 * Phase 1B: Added tetrahedra support, sparse boundary operators, Betti-1 via GF(2) rank, dynamic edges.
 * Phase 2.1: Enhanced dynamic topology adaptation with correlation-based edge/simplex evolution.
 */
export class EnhancedQualiaSheaf {
    /**
     * @param {Object} graphData - Initial graph structure with vertices and edges (optional).
     * @param {OntologicalWorldModel} owm - Reference to OWM for actorLoss (dynamic weights).
     * @param {number} stateDim - Dimension of the input state vector.
     * @param {number} initialQDim - Initial dimension for qualia vectors (should be 7).
     * @param {number} alpha - Parameter controlling sensory input influence.
     * @param {number} beta - Parameter controlling diffusion strength.
     * @param {number} gamma - Parameter controlling qualia update inertia/learning rate.
     */
    constructor(graphData, owm, stateDim = 8, initialQDim = 7, alpha = 0.1, beta = 0.1, gamma = 0.05) {
        this.owm = owm;
        this.stateDim = stateDim;
        this.entityNames = ['being', 'intent', 'existence', 'emergence', 'gestalt', 'context', 'rel_emergence'];
        this.qDim = this.entityNames.length; // Always 7
        
        const defaultVertices = ['agent_x', 'agent_z', 'agent_rot', 'target_x', 'target_z', 'vec_dx', 'vec_dz', 'dist_target'];
        const initialGraphVertices = Array.isArray(graphData?.vertices) && graphData.vertices.length >= defaultVertices.length ? graphData.vertices : defaultVertices;

        const initialBaseEdges = Array.isArray(graphData?.edges) ? graphData.edges.slice() : [
            ['agent_x', 'agent_rot'], ['agent_z', 'agent_rot'],
            ['agent_x', 'vec_dx'], ['agent_z', 'vec_dz'],
            ['target_x', 'vec_dx'], ['target_z', 'vec_dz'],
            ['vec_dx', 'dist_target'], ['vec_dz', 'dist_target']
        ];
        
        const explicitTriangles = Array.isArray(graphData?.triangles) ? graphData.triangles.slice() : [
            ['agent_x', 'agent_z', 'agent_rot'],
            ['target_x', 'target_z', 'dist_target'],
            ['agent_x', 'target_x', 'vec_dx'],
            ['agent_z', 'target_z', 'vec_dz']
        ];

        const explicitTetrahedra = Array.isArray(graphData?.tetrahedra) ? graphData.tetrahedra.slice() : [
            ['agent_x', 'agent_z', 'target_x', 'target_z'],
            ['agent_x', 'target_x', 'vec_dx', 'dist_target'],
            ['agent_rot', 'vec_dx', 'vec_dz', 'dist_target']
        ];

        // Simplicial Complex Closure Logic
        const allVerticesSet = new Set(initialGraphVertices);
        explicitTriangles.forEach(triangle => {
            if (!Array.isArray(triangle) || triangle.length !== 3) {
                logger.warn(`Sheaf: Invalid triangle in input: ${JSON.stringify(triangle)}`);
                return;
            }
            triangle.forEach(v => allVerticesSet.add(v));
        });
        explicitTetrahedra.forEach(tet => {
            if (!Array.isArray(tet) || tet.length !== 4) {
                logger.warn(`Sheaf: Invalid tetrahedron in input: ${JSON.stringify(tet)}`);
                return;
            }
            tet.forEach(v => allVerticesSet.add(v));
        });
        const finalVertices = Array.from(allVerticesSet);
        
        const allEdgesSet = new Set();
        initialBaseEdges.forEach(edge => {
            if (!Array.isArray(edge) || edge.length < 2) {
                logger.warn(`Sheaf: Invalid edge in input: ${JSON.stringify(edge)}`);
                return;
            }
            allEdgesSet.add(edge.slice(0, 2).sort().join(','));
        });

        let finalTrianglesUpdated = [...explicitTriangles];
        let finalTetrahedraUpdated = [...explicitTetrahedra];

        finalTetrahedraUpdated.forEach(tet => {
            if (!Array.isArray(tet) || tet.length !== 4) {
                logger.warn(`Sheaf: Invalid tetrahedron: ${JSON.stringify(tet)}`);
                return;
            }
            for (let i = 0; i < 4; i++) {
                const newTri = tet.filter((_, idx) => idx !== i).sort();
                const triStr = newTri.join(',');
                if (!finalTrianglesUpdated.some(t => t.slice().sort().join(',') === triStr)) {
                    finalTrianglesUpdated.push(newTri);
                }
            }
        });

        finalTrianglesUpdated.forEach(tri => {
            if (!Array.isArray(tri) || tri.length !== 3) {
                logger.warn(`Sheaf: Invalid triangle: ${JSON.stringify(tri)}`);
                return;
            }
            for (let i = 0; i < 3; i++) {
                const newEdge = [tri[i], tri[(i + 1) % 3]].sort();
                const edgeStr = newEdge.join(',');
                if (!allEdgesSet.has(edgeStr)) {
                    allEdgesSet.add(edgeStr);
                }
            }
        });
        const finalEdges = Array.from(allEdgesSet).map(s => s.split(',').concat([0.5])); // Default weight 0.5

        this.graph = {
            vertices: finalVertices,
            edges: finalEdges
        };
        this.simplicialComplex = {
            vertices: finalVertices,
            edges: finalEdges,
            triangles: finalTrianglesUpdated,
            tetrahedra: finalTetrahedraUpdated
        };

        this.alpha = clamp(alpha, 0.01, 1);
        this.beta = clamp(beta, 0.01, 1);
        this.gamma = clamp(gamma, 0.01, 0.5);
        this.eps = 1e-6;

        this.stalks = new Map(this.graph.vertices.map(v =>
            [v, new Float32Array(this.qDim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.5, -1, 1))]
        ));

        this.correlationMatrix = zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);
        this.stalkHistory = [];
        this.stalkHistorySize = 100;

        this.adjacencyMatrix = null;
        this.laplacian = null;
        this.maxEigApprox = 1;
        this.projectionMatrices = new Map();
        this.ready = false;

        this.phi = 0.2;
        this.h1Dimension = 0;
        this.gestaltUnity = 0.6;
        this.stability = 0.6;
        this.diffusionEnergy = 0;
        this.inconsistency = 0;
        this.windowedStates = [];
        const N_total_stalk_dim = this.graph.vertices.length * this.qDim;
        this.windowSize = Math.max(50, N_total_stalk_dim * 2);
        for (let i = 0; i < this.windowSize; i++) {
            this.windowedStates.push(new Float32Array(N_total_stalk_dim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.1, -1, 1)));
        }
        logger.info(`Enhanced Qualia Sheaf initialized: vertices=${this.graph.vertices.length}, edges=${this.graph.edges.length}, triangles=${this.simplicialComplex.triangles.length}, tetrahedra=${this.simplicialComplex.tetrahedra.length}`);
    }

    addEdge(u, v, weight = 0.5) {
        if (!this.graph.vertices.includes(u) || !this.graph.vertices.includes(v)) {
            logger.warn(`Sheaf.addEdge: Vertex not found for ${u} or ${v}`);
            return;
        }
        const sorted = [u, v].sort();
        const key = sorted.join(',');
        if (!this.graph.edges.some(e => e.slice(0, 2).join(',') === key)) {
            this.graph.edges.push([sorted[0], sorted[1], weight]);
            logger.info(`Added edge ${u}-${v} with weight ${weight.toFixed(3)}`);
        }
    }

    removeEdge(u, v) {
        const sorted = [u, v].sort();
        const key = sorted.join(',');
        const idx = this.graph.edges.findIndex(e => e.slice(0, 2).join(',') === key);
        if (idx !== -1) {
            this.graph.edges.splice(idx, 1);
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
        if (!this.simplicialComplex.triangles.some(t => t.join(',') === key)) {
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
        const idx = this.simplicialComplex.triangles.findIndex(t => t.join(',') === key);
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
        if (!this.simplicialComplex.tetrahedra.some(t => t.join(',') === key)) {
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
        const idx = this.simplicialComplex.tetrahedra.findIndex(t => t.join(',') === key);
        if (idx !== -1) {
            this.simplicialComplex.tetrahedra.splice(idx, 1);
            logger.info(`Removed tetrahedron ${a}-${b}-${c}-${d}`);
        }
    }

    computeVertexCorrelationsFromHistory() {
        if (this.stalkHistory.length < this.stalkHistorySize / 2) {
            return identity(this.graph.vertices.length);
        }

        const numVertices = this.graph.vertices.length;
        const corr = zeroMatrix(numVertices, numVertices);

        const means = new Float32Array(numVertices).fill(0);
        for (let i = 0; i < numVertices; i++) {
            for (let t = 0; t < this.stalkHistory.length; t++) {
                const val = this.stalkHistory[t][i] || 0;
                means[i] += Number.isFinite(val) ? val : 0;
            }
            means[i] /= this.stalkHistory.length;
        }

        for (let i = 0; i < numVertices; i++) {
            for (let j = i; j < numVertices; j++) {
                let numerator = 0;
                let denom_i = 0;
                let denom_j = 0;
                for (let t = 0; t < this.stalkHistory.length; t++) {
                    const diff_i = (this.stalkHistory[t][i] || 0) - (means[i] || 0);
                    const diff_j = (this.stalkHistory[t][j] || 0) - (means[j] || 0);
                    numerator += diff_i * diff_j;
                    denom_i += diff_i * diff_i;
                    denom_j += diff_j * diff_j;
                }
                const correlation = (Math.sqrt(denom_i) * Math.sqrt(denom_j)) < 1e-9 ? 0 : numerator / (Math.sqrt(denom_i) * Math.sqrt(denom_j));
                const finalCorr = clamp(correlation, -1, 1);
                corr[i][j] = finalCorr;
                corr[j][i] = finalCorr;
            }
        }

        if (!isFiniteMatrix(corr)) {
            logger.warn('Sheaf: Non-finite correlation matrix; returning identity.');
            return identity(numVertices);
        }

        return corr;
    }

    async initialize() {
        logger.info('EnhancedQualiaSheaf.initialize() called.');
        try {
            await this.computeCorrelationMatrix();
            this.projectionMatrices = await this.computeProjectionMatrices();
            await this.computeH1Dimension();
            await this._updateDerivedMetrics();
            this.ready = true;
            logger.info('Enhanced Qualia Sheaf ready with higher-order simplices.');
        } catch (e) {
            logger.error('Error during Sheaf initialization:', e);
            this.ready = false;
            throw e;
        }
    }

    async adaptSheafTopology(adaptFreq = 100, stepCount = 0) {
        if (stepCount % adaptFreq !== 0) return;

        if (this.stalkHistory.length < this.stalkHistorySize / 2) {
            logger.info(`Sheaf: Skipping topology adaptation at step ${stepCount}; insufficient history (${this.stalkHistory.length}/${this.stalkHistorySize}).`);
            return;
        }

        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        if (!isFiniteMatrix(this.correlationMatrix)) {
            logger.warn('Sheaf: Non-finite correlation matrix; skipping adaptation.');
            return;
        }

        const addThresh = clamp(0.6 + (1 - (this.gestaltUnity || 0.5)) * 0.2, 0.5, 0.9);
        const removeThresh = clamp(0.1 + (this.inconsistency || 0.5) * 0.3, 0.05, 0.4);
        const targetH1 = 2.0;

        const oldEdgeCount = this.graph.edges.length;
        const oldTriangleCount = this.simplicialComplex.triangles.length;

        this.adaptEdges(this.correlationMatrix, addThresh, removeThresh);
        this.adaptSimplices(this.correlationMatrix, targetH1);

        await this.computeCorrelationMatrix();
        await this.computeH1Dimension();
        await this.computeIntegratedInformation();

        const edgeDelta = this.graph.edges.length - oldEdgeCount;
        const triDelta = this.simplicialComplex.triangles.length - oldTriangleCount;
        logger.info(`Sheaf adapted at step ${stepCount}: edges=${this.graph.edges.length} (${edgeDelta > 0 ? '+' : ''}${edgeDelta}), triangles=${this.simplicialComplex.triangles.length} (${triDelta > 0 ? '+' : ''}${triDelta}), h1=${this.h1Dimension.toFixed(2)}, phi=${this.phi.toFixed(3)}`);
    }

    adaptEdges(corrMatrix, addThreshold, removeThreshold) {
        const numVertices = this.graph.vertices.length;
        const currentEdges = new Set(this.graph.edges.map(e => e.slice(0, 2).sort().join(',')));
        let added = 0;
        const maxAdd = 3;
        const maxEdges = 20;

        for (let i = 0; i < numVertices; i++) {
            for (let j = i + 1; j < numVertices; j++) {
                const corrVal = corrMatrix[i][j] || 0;
                const u = this.graph.vertices[i];
                const v = this.graph.vertices[j];
                const edgeKey = [u, v].sort().join(',');
                if (corrVal < removeThreshold && currentEdges.has(edgeKey)) {
                    this.removeEdge(u, v);
                }
            }
        }

        if (this.graph.edges.length < maxEdges) {
            for (let i = 0; i < numVertices && added < maxAdd; i++) {
                for (let j = i + 1; j < numVertices; j++) {
                    const corrVal = corrMatrix[i][j] || 0;
                    const u = this.graph.vertices[i];
                    const v = this.graph.vertices[j];
                    const edgeKey = [u, v].sort().join(',');
                    if (corrVal > addThreshold && !currentEdges.has(edgeKey)) {
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

        if (deltaH1 > 0.5) {
            const toPrune = [];
            this.simplicialComplex.triangles.forEach(tri => {
                const idxs = tri.map(v => this.graph.vertices.indexOf(v));
                if (idxs.some(id => id === -1)) return;
                const pairCorrs = [
                    corrMatrix[idxs[0]][idxs[1]] || 0,
                    corrMatrix[idxs[1]][idxs[2]] || 0,
                    corrMatrix[idxs[2]][idxs[0]] || 0
                ];
                const minCorr = Math.min(...pairCorrs);
                if (minCorr < 0.3) toPrune.push(tri);
            });
            if (toPrune.length > 0) {
                this.removeTriangle(...toPrune[0]);
                changed = true;
                logger.info(`Pruned low-coherence triangle (h1=${this.h1Dimension.toFixed(2)})`);
            }
        }

        if (deltaH1 < -0.5) {
            for (let i = 0; i < numV; i++) {
                for (let j = i + 1; j < numV; j++) {
                    for (let k = j + 1; k < numV; k++) {
                        const c1 = corrMatrix[i][j] || 0;
                        const c2 = corrMatrix[j][k] || 0;
                        const c3 = corrMatrix[k][i] || 0;
                        const avgC = (c1 + c2 + c3) / 3;
                        if (avgC > 0.8) {
                            const tri = [this.graph.vertices[i], this.graph.vertices[j], this.graph.vertices[k]];
                            const key = tri.sort().join(',');
                            if (!this.simplicialComplex.triangles.some(t => t.sort().join(',') === key)) {
                                this.addTriangle(...tri);
                                changed = true;
                                logger.info(`Added high-coherence triangle (h1=${this.h1Dimension.toFixed(2)})`);
                                return;
                            }
                        }
                    }
                }
            }
        }

        if (changed) {
            logger.info(`Simplex adaptation applied (deltaH1=${deltaH1.toFixed(2)})`);
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
            if (eIdx < nE) {
                edge.slice(0, 2).forEach(v => {
                    const vIdx = vMap.get(v);
                    if (vIdx !== undefined && vIdx < nV) {
                        boundary1[eIdx][vIdx] = 1;
                    }
                });
            } else { logger.warn(`Sheaf: Edge index out of bounds for ∂1: ${eIdx}`); }
        });

        const boundary2 = zeroMatrix(nT, nE);
        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (tIdx < nT) {
                const sortedTri = tri.slice().sort();
                const edgesOfTri = [
                    [sortedTri[0], sortedTri[1]].sort().join(','),
                    [sortedTri[1], sortedTri[2]].sort().join(','),
                    [sortedTri[2], sortedTri[0]].sort().join(',')
                ];
                edgesOfTri.forEach(edgeStr => {
                    const eIdx = eMapIndices.get(edgeStr);
                    if (eIdx !== undefined && eIdx < nE) {
                        boundary2[tIdx][eIdx] = 1;
                    } else { logger.warn(`Sheaf: Edge not found or index out of bounds for ∂2: ${edgeStr}`); }
                });
            } else { logger.warn(`Sheaf: Triangle index out of bounds for ∂2: ${tIdx}`); }
        });

        const boundary3 = zeroMatrix(nTet, nT);
        this.simplicialComplex.tetrahedra.forEach((tet, tetIdx) => {
            if (tetIdx < nTet) {
                const sortedTet = tet.slice().sort();
                for (let i = 0; i < 4; i++) {
                    const face = sortedTet.filter((_, idx) => idx !== i).sort();
                    const faceStr = face.join(',');
                    const tIdx = tMapIndices.get(faceStr);
                    if (tIdx !== undefined && tIdx < nT) {
                        boundary3[tetIdx][tIdx] = 1;
                    } else { logger.warn(`Sheaf: Triangle not found or index out of bounds for ∂3: ${faceStr}`); }
                }
            } else { logger.warn(`Sheaf: Tetrahedron index out of bounds for ∂3: ${tetIdx}`); }
        });

        const safeFlatten = (matrix, rows, cols, name) => {
            if (!isFiniteMatrix(matrix)) {
                logger.error(`Sheaf: Non-finite matrix for ${name} boundary detected. Returning empty flattened matrix.`);
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            const flatData = flattenMatrix(matrix).flatData;
            if (!isFiniteVector(flatData)) {
                logger.error(`Sheaf: Non-finite flatData after flattening for ${name} boundary. Returning empty.`);
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            return { flatData, rows, cols };
        };

        return {
            partial1: safeFlatten(boundary1, nE, nV, "boundary1"),
            partial2: safeFlatten(boundary2, nT, nE, "boundary2"),
            partial3: safeFlatten(boundary3, nTet, nT, "boundary3")
        };
    }

    async computeCorrelationMatrix() {
        this.correlationMatrix = this.computeVertexCorrelationsFromHistory();

        const numVertices = this.graph.vertices.length;
        if (numVertices === 0) {
            this.adjacencyMatrix = zeroMatrix(0, 0);
            return;
        }

        let performanceFactor = Number.isFinite(this.owm?.actorLoss) 
            ? (1 - clamp(this.owm.actorLoss / 0.5, 0, 1)) 
            : 0;
        if (!Number.isFinite(performanceFactor)) {
            logger.warn('Sheaf: Non-finite performanceFactor. Setting to 0.');
            performanceFactor = 0;
        }

        let performanceScalar = clamp(1 + performanceFactor, 0.5, 2.0);
        if (!Number.isFinite(performanceScalar)) {
            logger.warn('Sheaf: Non-finite performanceScalar. Setting to 1.');
            performanceScalar = 1;
        }

        let h1Boost = 1 + clamp(this.h1Dimension / Math.max(1, numVertices / 2), 0, 1) * 0.5;
        if (!Number.isFinite(h1Boost)) {
            logger.warn('Sheaf: Non-finite h1Boost. Setting to 1.');
            h1Boost = 1;
        }

        this.adjacencyMatrix = zeroMatrix(numVertices, numVertices);

        this.graph.edges.forEach(([u, v, weight = 0.1]) => {
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i >= 0 && j >= 0) {
                const correlation = this.correlationMatrix[i][j] || 0;
                const dynamicWeight = clamp((weight + 0.9 * ((1 + correlation) / 2)) * performanceScalar * h1Boost, 0.01, 1.0);
                if (!Number.isFinite(dynamicWeight)) {
                    logger.warn(`Sheaf: Non-finite dynamicWeight for edge ${u}-${v}. Setting to ${weight}.`);
                    this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = weight;
                } else {
                    this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = dynamicWeight;
                }
            }
        });

        if (!isFiniteMatrix(this.adjacencyMatrix)) {
            logger.error('Sheaf: Non-finite adjacency matrix after dynamic weighting. Resetting to zero.');
            this.adjacencyMatrix = zeroMatrix(numVertices, numVertices);
        }
    }

    buildLaplacian() {
        const n = this.graph.vertices.length;
        const adj = this.adjacencyMatrix;
        if (!adj || !isFiniteMatrix(adj) || adj.length !== n || (n > 0 && adj[0].length !== n)) {
            logger.error('Sheaf.buildLaplacian: Adjacency matrix is invalid or dimensions mismatch. Cannot build Laplacian.');
            return zeroMatrix(n, n);
        }
        const L = zeroMatrix(n, n);

        for (let i = 0; i < n; i++) {
            let deg = 0;
            for (let j = 0; j < n; j++) {
                const weight = adj[i]?.[j] || 0;
                if (!Number.isFinite(weight)) {
                    logger.warn(`Sheaf.buildLaplacian: Non-finite weight in adjacency matrix at [${i}][${j}]. Setting to 0.`);
                    adj[i][j] = 0;
                }
                if (adj[i][j] > 0) {
                    L[i][j] = -adj[i][j];
                    deg += adj[i][j];
                }
            }
            L[i][i] = deg + this.eps;
        }
        if (!isFiniteMatrix(L)) {
            logger.error('Sheaf: Non-finite Laplacian matrix detected; resetting to zero.');
            return zeroMatrix(n, n);
        }
        return L;
    }

    async computeProjectionMatrices() {
        const projections = new Map();
        const identityMatrix = identity(this.qDim);

        for (const [u, v] of this.graph.edges) {
            projections.set(`${u}-${v}`, identityMatrix);
            projections.set(`${v}-${u}`, identityMatrix);
        }
        return projections;
    }

    async _updateGraphStructureAndMetrics() {
        try {
            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();

            const flatLaplacian = flattenMatrix(this.laplacian);
            if (!isFiniteVector(flatLaplacian.flatData)) {
                logger.error("Sheaf: Non-finite Laplacian matrix detected before spectral norm calculation. Defaulting maxEigApprox to 1.");
                this.maxEigApprox = 1;
            } else {
                this.maxEigApprox = await runWorkerTask('matrixSpectralNormApprox', { matrix: flatLaplacian }, 10000);
                if (!Number.isFinite(this.maxEigApprox) || this.maxEigApprox <= 0) {
                    logger.warn(`Sheaf: maxEigApprox was invalid (${this.maxEigApprox}). Resetting to 1.`);
                    this.maxEigApprox = 1;
                }
            }
            
            this.projectionMatrices = await this.computeProjectionMatrices();
        } catch (e) {
            logger.error("Sheaf: Failed to update graph structure and metrics", e);
            throw e;
        }
    }

    async diffuseQualia(state) {
        if (!this.ready) {
            logger.warn('Sheaf not ready for diffusion. Skipping.');
            return;
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn('Sheaf.diffuseQualia: Invalid input state received. Skipping diffusion.', {state});
            return;
        }

        await this._updateGraphStructureAndMetrics();

        const qInput = new Float32Array(state.slice(0, this.graph.vertices.length).map(v => clamp(v, 0, 1)));
        if (!isFiniteVector(qInput) || qInput.length !== this.graph.vertices.length) {
            logger.warn('Sheaf.diffuseQualia: Invalid qInput generated. Skipping diffusion.', {qInput});
            return;
        }

        const n = this.graph.vertices.length;
        const N = n * this.qDim;

        const s = new Float32Array(N);
        let currentOffset = 0;
        for (const vertexName of this.graph.vertices) {
            let stalkValue = this.stalks.get(vertexName);
            if (!stalkValue || !isFiniteVector(stalkValue) || stalkValue.length !== this.qDim) {
                logger.warn(`Sheaf.diffuseQualia: Found invalid or missing stalk for vertex ${vertexName}. Resetting to zeros.`);
                stalkValue = vecZeros(this.qDim);
                this.stalks.set(vertexName, stalkValue);
            }
            s.set(stalkValue.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0), currentOffset);
            currentOffset += this.qDim;
        }

        if (!isFiniteVector(s)) {
            logger.error('Sheaf.diffuseQualia: Initial concatenated stalk vector "s" contains non-finite values. Resetting to zeros.', {s});
            s.fill(0);
        }

        const Lfull = zeroMatrix(N, N);
        const idx = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) { // Note: 'weight' is now read from adjacencyMatrix
            const i = idx.get(u), j = idx.get(v);
            if (i === undefined || j === undefined) continue;

            const weight = this.adjacencyMatrix[i]?.[j]; // Get dynamic weight from adjacency matrix
            if (!Number.isFinite(weight) || weight <= 0) {
                logger.warn(`Sheaf.diffuseQualia: Non-finite or non-positive weight for edge ${u}-${v}. Skipping block.`);
                continue;
            }

            const P_uv = this.projectionMatrices.get(`${u}-${v}`);
            const P_vu = this.projectionMatrices.get(`${v}-${u}`);

            if (!isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vu)) {
                logger.warn(`Sheaf.diffuseQualia: Non-finite or missing projection matrix for edge ${u}-${v}. Skipping block.`);
                continue;
            }

            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    const val_uv = -weight * (P_uv[qi]?.[qj] || 0);
                    if (Number.isFinite(val_uv)) {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = clamp(val_uv, -100, 100);
                    } else {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = 0;
                    }
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    const val_vu = -weight * (P_vu[qi]?.[qj] || 0);
                    if (Number.isFinite(val_vu)) {
                        Lfull[j * this.qDim + qi][i * this.qDim + qj] = clamp(val_vu, -100, 100);
                    } else {
                        Lfull[j * this.qDim + qi][i * this.qDim + qj] = 0;
                    }
                }
            }
        }

        for (let i = 0; i < n; i++) {
            let degree = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    const adjVal = this.adjacencyMatrix[i]?.[j];
                    if (Number.isFinite(adjVal)) {
                        degree += adjVal;
                    } else {
                        logger.warn(`Sheaf.diffuseQualia: Non-finite adjacency matrix value for degree calculation at [${i}][${j}]. Skipping.`);
                    }
                }
            }
            if (!Number.isFinite(degree)) {
                logger.warn(`Sheaf.diffuseQualia: Non-finite degree for vertex ${this.graph.vertices[i]}. Setting to 0.`);
                degree = 0;
            }

            for (let qi = 0; qi < this.qDim; qi++) {
                Lfull[i * this.qDim + qi][i * this.qDim + qi] = clamp(degree + this.eps, -100, 100);
            }
        }
        if (!isFiniteMatrix(Lfull)) {
            logger.error('Sheaf.diffuseQualia: Lfull matrix contains non-finite values after construction. Resetting to identity.');
            for (let i = 0; i < N; ++i) for (let j = 0; j < N; ++j) Lfull[i][j] = (i === j ? 1 : 0);
        }

        const f_s = new Float32Array(N).fill(0);
        for (let i = 0; i < n; i++) {
            let inputVal = qInput[i % qInput.length];
            if (!Number.isFinite(inputVal)) {
                logger.warn(`Sheaf.diffuseQualia: Non-finite qInput value at index ${i}. Setting to 0.`);
                inputVal = 0;
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                f_s[i * this.qDim + qi] = this.alpha * inputVal * 2.0; // Increased input influence
            }
        }
        if (!isFiniteVector(f_s)) {
            logger.error('Sheaf.diffuseQualia: f_s vector is non-finite. Resetting to zeros.');
            f_s.fill(0);
        }

        let eta = this.gamma / Math.max(1, this.maxEigApprox);
        if (!Number.isFinite(eta)) {
            logger.warn('Sheaf.diffuseQualia: Non-finite eta. Setting to 0.01.');
            eta = 0.01;
        }

        const A = zeroMatrix(N, N).map((row, i) => new Float32Array(row.map((v, j) => {
            const val = (i === j ? 1 : 0) + eta * (Lfull[i]?.[j] || 0);
            return Number.isFinite(val) ? clamp(val, -100, 100) : 0;
        })));

        const rhs = vecAdd(s, vecScale(f_s, eta)).map(v => Number.isFinite(v) ? clamp(v, -100, 100) : 0); // Less restrictive clamping

        if (!isFiniteMatrix(A) || !isFiniteVector(rhs)) {
            logger.error('Sheaf.diffuseQualia: Matrix A or RHS vector contains non-finite values before CG solve. Skipping diffusion.');
            const sNext = new Float32Array(N).fill(0);
            this._updateStalksAndWindow(sNext, n);
            await this._updateDerivedMetrics();
            return;
        }

        let sSolved;
        try {
            sSolved = await runWorkerTask('solveLinearSystemCG', { A: flattenMatrix(A), b: rhs, opts: { tol: 1e-6, maxIter: 15 } }, 5000);
            if (!isFiniteVector(sSolved)) {
                logger.error('Sheaf.diffuseQualia: Worker returned non-finite sSolved from solveLinearSystemCG. Falling back to zeros.');
                sSolved = vecZeros(N);
            }
        } catch (e) {
            logger.error('Sheaf.diffuseQualia: Error solving linear system in worker (CG). Falling back to zero vector:', e);
            sSolved = vecZeros(N);
        }

        if (!isFiniteVector(sSolved)) {
            logger.error('Sheaf.diffuseQualia: CRITICAL: Solver output or fallback contained non-finite values. Resetting sSolved to zero vector.', { sSolved });
            sSolved = vecZeros(N);
        }

        const sNext = new Float32Array(sSolved.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));

        this._updateStalksAndWindow(sNext, n);
        await this._updateDerivedMetrics();
    }

    _updateStalksAndWindow(sNextVector, n) {
        const currentStalkNorms = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const rawStalkSlice = sNextVector.slice(i * this.qDim, (i + 1) * this.qDim);
            const sanitizedStalk = new Float32Array(rawStalkSlice.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
            if (!isFiniteVector(sanitizedStalk)) {
                logger.error(`Sheaf._updateStalksAndWindow: Sanitized stalk for vertex ${this.graph.vertices[i]} is non-finite. Resetting to zeros.`, { sanitizedStalk });
                sanitizedStalk.fill(0);
            }
            this.stalks.set(this.graph.vertices[i], sanitizedStalk);
            currentStalkNorms[i] = norm2(sanitizedStalk);
        }
        this.stalkHistory.push(currentStalkNorms);
        if (this.stalkHistory.length > this.stalkHistorySize) {
            this.stalkHistory.shift();
        }

        const sanitizedNextVector = new Float32Array(sNextVector.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
        if (!isFiniteVector(sanitizedNextVector)) {
            logger.error('Sheaf._updateStalksAndWindow: sanitizedNextVector for windowedStates is non-finite. Resetting to zeros.');
            sanitizedNextVector.fill(0);
        }
        this.windowedStates.push(sanitizedNextVector);
        if (this.windowedStates.length > this.windowSize) this.windowedStates.shift();
    }

    async _updateDerivedMetrics() {
        try {
            await this.computeGluingInconsistency();
            this.computeGestaltUnity();
            await this.computeIntegratedInformation();
        } catch (e) {
            logger.error("Sheaf: Error during derived metrics update:", e);
            this.phi = 0.01;
            this.h1Dimension = 0.5;
            this.gestaltUnity = 0.05;
            this.stability = 0.01;
            this.inconsistency = 1.0;
        }
    }

    async computeH1Dimension() {
        const boundaries = await this.buildBoundaryMatrices();
        
        let rankPartial2 = 0;
        let rankPartial3 = 0;

        try {
            const p2Result = await runWorkerTask('smithNormalForm', boundaries.partial2, 5000);
            rankPartial2 = p2Result?.rank || 0;
            if (!Number.isFinite(rankPartial2)) {
                logger.warn('Sheaf: Non-finite rank for ∂2. Setting to 0.');
                rankPartial2 = 0;
            }

            const p3Result = await runWorkerTask('smithNormalForm', boundaries.partial3, 5000);
            rankPartial3 = p3Result?.rank || 0;
            if (!Number.isFinite(rankPartial3)) {
                logger.warn('Sheaf: Non-finite rank for ∂3. Setting to 0.');
                rankPartial3 = 0;
            }
        } catch (e) {
            logger.error('Sheaf: Homology computation (smithNormalForm) failed:', e);
            this.h1Dimension = clamp(this.graph.edges.length - this.graph.vertices.length + this.simplicialComplex.triangles.length, 0, this.graph.vertices.length);
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }
        
        const numEdges = this.graph.edges.length;
        this.h1Dimension = (numEdges - rankPartial2) - rankPartial3;
        this.h1Dimension = clamp(this.h1Dimension, 0, numEdges);
        if (!Number.isFinite(this.h1Dimension)) {
            logger.error('Sheaf: Final h1Dimension is non-finite. Setting to 0.');
            this.h1Dimension = 0;
        }

        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        if (!Number.isFinite(this.stability)) {
            logger.error('Sheaf: Final stability is non-finite. Setting to 0.5.');
            this.stability = 0.5;
        }
    }

    async computeGluingInconsistency() {
        let sum = 0;
        for (const [u, v] of this.graph.edges) {
            const stalk_u = this.stalks.get(u);
            const stalk_v = this.stalks.get(v);
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);

            if (!P_uv || !isFiniteMatrix(P_uv) || !isFiniteVector(stalk_u) || !isFiniteVector(stalk_v)) {
                logger.warn(`Sheaf.computeGluingInconsistency: Skipping inconsistency calculation for edge ${u}-${v} due to non-finite inputs.`);
                continue;
            }

            let projected_u;
            try {
                projected_u = await runWorkerTask('matVecMul', { matrix: flattenMatrix(P_uv), vector: stalk_u }, 5000);
                if (!isFiniteVector(projected_u)) {
                    logger.warn(`Sheaf.computeGluingInconsistency: Worker returned non-finite projected_u for edge ${u}-${v}. Setting to zeros.`);
                    projected_u = vecZeros(this.qDim);
                }
            } catch (e) {
                logger.error(`Sheaf.computeGluingInconsistency: Error projecting stalk_u for edge ${u}-${v} in worker. Setting to zeros:`, e);
                projected_u = vecZeros(this.qDim);
            }
            
            const safeProjected_u = isFiniteVector(projected_u) ? projected_u.map(v => clamp(v, -1, 1)) : vecZeros(this.qDim);
            
            const diffNorm = norm2(vecSub(safeProjected_u, stalk_v));
            if (Number.isFinite(diffNorm)) {
                sum += diffNorm;
            } else {
                logger.warn(`Sheaf.computeGluingInconsistency: Non-finite diffNorm for edge ${u}-${v}. Skipping addition to sum.`, { projected: safeProjected_u, stalk_v });
            }
        }
        this.inconsistency = this.graph.edges.length > 0 ? clamp(sum / this.graph.edges.length, 0, 1) : 0;
        if (!Number.isFinite(this.inconsistency)) {
            logger.error('Sheaf: Final inconsistency is non-finite. Setting to 1.0.');
            this.inconsistency = 1.0;
        }
    }

    computeGestaltUnity() {
        let sum = 0;
        this.graph.edges.forEach(([u, v]) => {
            const stalk_u = this.stalks.get(u);
            const stalk_v = this.stalks.get(v);
            if (isFiniteVector(stalk_u) && isFiniteVector(stalk_v)) {
                const diffNorm = norm2(vecSub(stalk_u, stalk_v));
                if (Number.isFinite(diffNorm)) {
                    sum += Math.exp(-diffNorm * this.beta);
                } else {
                    logger.warn(`Sheaf.computeGestaltUnity: Non-finite diffNorm for edge ${u}-${v}. Skipping addition to sum.`, { stalk_u, stalk_v });
                }
            }
        });
        if (!Number.isFinite(sum)) {
            logger.warn('Sheaf.computeGestaltUnity: Sum became non-finite. Setting to 0.');
            sum = 0;
        }
        this.gestaltUnity = this.graph.edges.length > 0 ? clamp(sum / this.graph.edges.length, 0.05, 0.99) : 0.05;
        if (!Number.isFinite(this.gestaltUnity)) {
            logger.error('Sheaf: Final gestaltUnity is non-finite. Setting to 0.05.');
            this.gestaltUnity = 0.05;
        }
    }

    async computeIntegratedInformation() {
        const validStates = this.windowedStates.filter(isFiniteVector);
        if (validStates.length < Math.max(4, this.stateDim + this.qDim)) {
            logger.warn('Sheaf: Not enough valid states for meaningful covariance calculation. Using base Phi.');
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }
        let cov;
        try {
            cov = await runWorkerTask('covarianceMatrix', { states: validStates, eps: this.eps }, 10000);
            if (!isFiniteMatrix(cov)) {
                logger.warn("Sheaf.computeIntegratedInformation: Worker returned non-finite covariance matrix. Setting MI to default.");
                cov = zeroMatrix(validStates[0].length, validStates[0].length).map(row => row.fill(this.eps));
            }
        } catch (e) {
            logger.error('Sheaf.computeIntegratedInformation: Error computing covarianceMatrix in worker. Setting Phi to default:', e);
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }

        if (!isFiniteMatrix(cov)) {
            logger.warn("Sheaf.computeIntegratedInformation: Non-finite covariance matrix after worker call. Setting MI to default.");
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }

        let MI_val = logDeterminantFromDiagonal(cov);
        let MI = Number.isFinite(MI_val) ? Math.abs(MI_val) * 0.1 + 1e-6 : 1e-6;
        if (!Number.isFinite(MI)) {
            logger.warn('Sheaf.computeIntegratedInformation: MI value is non-finite. Setting to 1e-6.');
            MI = 1e-6;
        }

        const safeStability = Number.isFinite(this.stability) ? this.stability : 0.1;
        const safeGestaltUnity = Number.isFinite(this.gestaltUnity) ? this.gestaltUnity : 0.1;
        const safeInconsistency = Number.isFinite(this.inconsistency) ? this.inconsistency : 1.0;

        this.phi = clamp(Math.log(1 + MI) * safeStability * safeGestaltUnity * Math.exp(-safeInconsistency) * (1 + 0.05 * this.h1Dimension), 0.01, 5);
        if (!Number.isFinite(this.phi)) {
            logger.error('Sheaf: Final Phi is non-finite. Setting to 0.01.');
            this.phi = 0.01;
        }
    }

    visualizeActivity() {
        this.graph.vertices.forEach((vertexName, idx) => {
            const el = document.getElementById(`vertex-${idx}`);
            if (!el) return;
            const stalk = this.stalks.get(vertexName) || vecZeros(this.qDim);
            const norm = norm2(stalk);
            const intensity = Number.isFinite(norm) ? clamp(norm / Math.sqrt(this.qDim), 0, 1) : 0;
            el.classList.toggle('active', intensity > 0.5);
            const hue = 0;
            const saturation = 100;
            const lightness = 50 + intensity * 40;
            
            el.style.background = `radial-gradient(circle, hsl(${hue}, ${saturation}%, ${lightness}%), hsl(${hue}, ${saturation * 0.8}%, ${lightness * 0.6}%))`;

            if (intensity > 0.5) {
                el.style.background = `radial-gradient(circle, #00ff99, #00cc66)`;
            }
        });
    }

    async tuneParameters() {
        const currentStability = Number.isFinite(this.stability) ? this.stability : 0.5;
        const currentInconsistency = Number.isFinite(this.inconsistency) ? this.inconsistency : 0.5;
        const currentGestaltUnity = Number.isFinite(this.gestaltUnity) ? this.gestaltUnity : 0.5;
        const currentH1Dimension = Number.isFinite(this.h1Dimension) ? this.h1Dimension : 1.0;

        // Alpha: Increase with inconsistency/instability, and H1 dimension to promote new input/change with complexity
        this.alpha = clamp(this.alpha * (1 + 0.02 * (1 - currentStability)) * (1 + 0.01 * currentInconsistency) * (1 + 0.005 * currentH1Dimension), 0.01, 1);
        // Beta: Decrease with high gestalt unity (less need for smoothing), and decrease with H1 (complex systems might need stronger diffusion to integrate)
        this.beta = clamp(this.beta * (1 + 0.02 * (1 - currentGestaltUnity)) * (1 - 0.01 * currentH1Dimension), 0.01, 1);
        // Gamma: Decrease with complexity (H1) and inconsistency (less confident learning), but increase with stability (more confident learning)
        this.gamma = clamp(this.gamma * (1 - 0.05 * currentH1Dimension) * (1 - 0.02 * currentInconsistency) * (1 + 0.01 * currentStability), 0.01, 0.5);
    }

    async testTopologyAdaptation(steps = 100) {
        const initialEdges = this.graph.edges.length;
        const initialTriangles = this.simplicialComplex.triangles.length;
        const initialH1 = this.h1Dimension;

        for (let i = 0; i < steps; i++) {
            const fakeState = new Float32Array(this.stateDim).map(() => (Math.random() - 0.5) * 2);
            await this.diffuseQualia(fakeState);
            await this.adaptSheafTopology(10, i * 10);
        }

        const edgeDelta = this.graph.edges.length - initialEdges;
        const triDelta = this.simplicialComplex.triangles.length - initialTriangles;
        const h1Delta = this.h1Dimension - initialH1;

        logger.info(`Topology test: edges=${this.graph.edges.length} (${edgeDelta > 0 ? '+' : ''}${edgeDelta}), triangles=${this.simplicialComplex.triangles.length} (${triDelta > 0 ? '+' : ''}${triDelta}), h1Delta=${h1Delta.toFixed(2)}`);
        return { edgeDelta, triDelta, h1Delta };
    }
}
// --- END OF FILE qualia-sheaf.js ---
