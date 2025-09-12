// --- START OF FILE qualia-sheaf.js ---
import { 
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, identity,
    isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask
} from './utils.js';

/**
 * Represents an Enhanced Qualia Sheaf, a mathematical structure used to model the AI's "consciousness".
 * It handles qualia diffusion, topological adaptation, and metric computations like Phi, H1 dimension, etc.
 */
export class EnhancedQualiaSheaf {
    /**
     * @param {Object} graph - Initial graph structure with vertices and edges.
     * @param {number} stateDim - Dimension of the input state vector.
     * @param {number} initialQDim - Initial dimension for qualia vectors (should be 7).
     * @param {number} alpha - Parameter controlling sensory input influence.
     * @param {number} beta - Parameter controlling diffusion strength.
     * @param {number} gamma - Parameter controlling qualia update inertia/learning rate.
     */
    constructor(graph, stateDim = 8, initialQDim = 7, alpha = 0.1, beta = 0.1, gamma = 0.05) {
        this.stateDim = stateDim;
        this.entityNames = ['being', 'intent', 'existence', 'emergence', 'gestalt', 'context', 'rel_emergence'];
        this.qDim = this.entityNames.length; // This should always be 7
        
        const defaultVertices = ['agent_x', 'agent_z', 'agent_rot', 'target_x', 'target_z', 'vec_dx', 'vec_dz', 'dist_target'];

        const initialGraphVertices = graph?.vertices && Array.isArray(graph.vertices) && graph.vertices.length >= defaultVertices.length ? graph.vertices : defaultVertices;

        const initialBaseEdges = graph?.edges && Array.isArray(graph.edges) ? graph.edges.slice(0, 15) : [
            ['agent_x', 'agent_rot'], ['agent_z', 'agent_rot'],
            ['agent_x', 'vec_dx'], ['agent_z', 'vec_dz'],
            ['target_x', 'vec_dx'], ['target_z', 'vec_dz'],
            ['vec_dx', 'dist_target'], ['vec_dz', 'dist_target']
        ];
        
        const explicitTriangles = [
             ['agent_x', 'agent_z', 'agent_rot'],
             ['target_x', 'target_z', 'dist_target'],
             ['agent_x', 'target_x', 'vec_dx'],
             ['agent_z', 'target_z', 'vec_dz']
        ];

        const finalTriangles = graph?.triangles && Array.isArray(graph.triangles) ? graph.triangles : explicitTriangles;

        const allVerticesSet = new Set(initialGraphVertices);
        finalTriangles.forEach(triangle => {
            triangle.forEach(v => allVerticesSet.add(v));
        });
        const finalVertices = Array.from(allVerticesSet);

        const allEdgesSet = new Set();
        initialBaseEdges.forEach(edge => {
            allEdgesSet.add(edge.slice().sort().join(','));
        });

        finalTriangles.forEach(triangle => {
            const [v1, v2, v3] = triangle;
            const edgesOfTriangle = [[v1, v2], [v2, v3], [v3, v1]];
            edgesOfTriangle.forEach(edge => {
                allEdgesSet.add(edge.slice().sort().join(','));
            });
        });
        const finalEdges = Array.from(allEdgesSet).map(s => s.split(','));

        this.graph = {
            vertices: finalVertices,
            edges: finalEdges
        };
        this.simplicialComplex = {
            vertices: finalVertices,
            edges: finalEdges,
            triangles: finalTriangles
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
        logger.info(`EnhancedQualiaSheaf initialized: qDim=${this.qDim}, vertices=${this.graph.vertices.length}`);
    }

    async initialize() {
        logger.info('EnhancedQualiaSheaf.initialize() called.');
        try {
            await this._updateGraphStructureAndMetrics();
            await this._updateDerivedMetrics();
            this.ready = true;
            logger.info('EnhancedQualiaSheaf ready.');
        } catch (e) {
            logger.error('Error during Sheaf initialization:', e);
            this.ready = false;
            throw e;
        }
    }

    adaptSheafTopology() {
        if (this.stalkHistory.length < this.stalkHistorySize) return;

        const numVertices = this.graph.vertices.length;
        if (numVertices === 0) return;

        const means = new Float32Array(numVertices).fill(0);

        for (let i = 0; i < numVertices; i++) {
            for (let t = 0; t < this.stalkHistorySize; t++) {
                means[i] += this.stalkHistory[t][i];
            }
            means[i] /= this.stalkHistorySize;
        }

        for (let i = 0; i < numVertices; i++) {
            for (let j = i; j < numVertices; j++) {
                let numerator = 0;
                let denom_i = 0;
                let denom_j = 0;
                for (let t = 0; t < this.stalkHistorySize; t++) {
                    const diff_i = (this.stalkHistory[t][i] || 0) - (means[i] || 0);
                    const diff_j = (this.stalkHistory[t][j] || 0) - (means[j] || 0);
                    numerator += diff_i * diff_j;
                    denom_i += diff_i * diff_i;
                    denom_j += diff_j * diff_j;
                }
                const correlation = (Math.sqrt(denom_i) * Math.sqrt(denom_j)) < 1e-9 ? 0 : numerator / (Math.sqrt(denom_i) * Math.sqrt(denom_j));
                const finalCorr = clamp(correlation, -1, 1);
                this.correlationMatrix[i][j] = finalCorr;
                this.correlationMatrix[j][i] = finalCorr;
            }
        }
    }

    buildBoundaryMatrices() {
        const vIdx = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const edgeKeys = this.graph.edges.map(e => e.slice().sort().join(','));
        const eIdx = new Map(edgeKeys.map((e, i) => [e, i]));
        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;
        const nT = this.simplicialComplex.triangles.length;

        const boundary1 = zeroMatrix(nE, nV);
        this.graph.edges.forEach(([u, v], i) => {
            boundary1[i][vIdx.get(u)] = -1;
            boundary1[i][vIdx.get(v)] = 1;
        });

        const boundary2 = zeroMatrix(nT, nE);
        this.simplicialComplex.triangles.forEach(([u, v, w], i) => {
            const edges = [[u, v], [v, w], [w, u]];
            edges.forEach(([a, b]) => {
                const sortedEdge = [a, b].sort().join(',');
                const eIndex = eIdx.get(sortedEdge);
                if (eIndex !== undefined) {
                    const sign = (a === [a,b].sort()[0]) ? 1 : -1;
                    boundary2[i][eIndex] = sign;
                }
            });
        });
        return { boundary1, boundary2 };
    }

    buildAdjacencyMatrix() {
        const n = this.graph.vertices.length;
        const mapIdx = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const adj = zeroMatrix(n, n);
        
        for (const [u, v] of this.graph.edges) {
            const i = mapIdx.get(u), j = mapIdx.get(v);
            if (i === undefined || j === undefined) continue;

            let baseWeight = 1.0; 
            const correlation = this.correlationMatrix[i][j] || 0;
            const correlationFactor = (1 + correlation) / 2;

            const finalWeight = baseWeight * (0.1 + 0.9 * correlationFactor);
            
            adj[i][j] = clamp(finalWeight, 0.01, 1.0);
            adj[j][i] = clamp(finalWeight, 0.01, 1.0);
        }
        return adj;
    }

    computeClustering() {
        const n = this.graph.vertices.length;
        const adjBinary = zeroMatrix(n, n);
        const mapIdx = new Map(this.graph.vertices.map((v, i) => [v, i]));
        for (const [u, v] of this.graph.edges) {
            const i = mapIdx.get(u), j = mapIdx.get(v);
            if (i !== undefined && j !== undefined) {
                adjBinary[i][j] = 1;
                adjBinary[j][i] = 1;
            }
        }

        const clustering = new Float32Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            const neighbors = [];
            for (let j = 0; j < n; j++) if (adjBinary[i][j]) neighbors.push(j);
            const k = neighbors.length;
            if (k < 2) continue;
            let tri = 0;
            for (let a = 0; a < neighbors.length; a++) {
                for (let b = a + 1; b < neighbors.length; b++) {
                    if (adjBinary[neighbors[a]][neighbors[b]]) tri++;
                }
            }
            const possible = k * (k - 1) / 2;
            clustering[i] = possible > 0 ? tri / possible : 0;
        }
        return clustering;
    }

    buildLaplacian() {
        const n = this.graph.vertices.length;
        const adj = this.adjacencyMatrix || this.buildAdjacencyMatrix();
        const L = zeroMatrix(n, n);

        for (let i = 0; i < n; i++) {
            let deg = 0;
            for (let j = 0; j < n; j++) {
                if (adj[i][j] > 0) {
                    L[i][j] = -adj[i][j];
                    deg += adj[i][j];
                }
            }
            L[i][i] = deg + this.eps;
        }
        return L;
    }

    async computeProjectionMatrices() {
        const projections = new Map();
        const identityMatrix = identity(this.qDim);

        for (const [u, v] of this.graph.edges) {
            const P_uv = identityMatrix;
            const P_vu = identityMatrix;

            projections.set(`${u}-${v}`, P_uv);
            projections.set(`${v}-${u}`, P_vu);
        }
        return projections;
    }

    async _updateGraphStructureAndMetrics() {
        try {
            this.adaptSheafTopology();
            this.adjacencyMatrix = this.buildAdjacencyMatrix();
            this.laplacian = this.buildLaplacian();

            // Ensure flattenMatrix receives finite data and returns finite data
            const flatLaplacian = flattenMatrix(this.laplacian);
            if (!isFiniteMatrix(this.laplacian) || !isFiniteVector(flatLaplacian.flatData)) {
                logger.error("Non-finite Laplacian matrix detected before spectral norm calculation. Defaulting maxEigApprox to 1.");
                this.maxEigApprox = 1;
            } else {
                this.maxEigApprox = await runWorkerTask('matrixSpectralNormApprox', { matrix: flatLaplacian }, 10000) || 1;
            }
            
            this.projectionMatrices = await this.computeProjectionMatrices();
            if (!Number.isFinite(this.maxEigApprox) || this.maxEigApprox <= 0) {
                logger.warn(`maxEigApprox was invalid (${this.maxEigApprox}). Resetting to 1.`);
                this.maxEigApprox = 1;
            }
        } catch (e) {
            logger.error("Failed to update graph structure and metrics", e);
            throw e;
        }
    }

    async diffuseQualia(state) {
        if (!this.ready) {
            logger.warn('Sheaf not ready for diffusion. Skipping.');
            return;
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.warn('diffuseQualia: Invalid input state received. Skipping diffusion.', state);
            return;
        }

        await this._updateGraphStructureAndMetrics(); 

        const qInput = new Float32Array(state.slice(0, this.graph.vertices.length).map(v => clamp(v, 0, 1)));
        if (!isFiniteVector(qInput) || qInput.length !== this.graph.vertices.length) {
            logger.warn('diffuseQualia: Invalid qInput generated. Skipping diffusion.', qInput);
            return;
        }

        const n = this.graph.vertices.length;
        const N = n * this.qDim;

        const s = new Float32Array(N);
        let currentOffset = 0;
        for (const vertexName of this.graph.vertices) {
            let stalkValue = this.stalks.get(vertexName);
            if (!stalkValue || !isFiniteVector(stalkValue) || stalkValue.length !== this.qDim) {
                logger.warn(`Found invalid or missing stalk for vertex ${vertexName}. Resetting to zeros.`);
                stalkValue = vecZeros(this.qDim);
                this.stalks.set(vertexName, stalkValue);
            }
            s.set(stalkValue.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0), currentOffset); // Ensure elements are finite and clamped
            currentOffset += this.qDim;
        }

        if (!isFiniteVector(s)) {
            logger.error('diffuseQualia: Initial concatenated stalk vector "s" contains non-finite values. Resetting to zeros for current diffusion step.', s);
            s.fill(0);
        }

        const Lfull = zeroMatrix(N, N);
        const idx = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) {
            const i = idx.get(u), j = idx.get(v);
            if (i === undefined || j === undefined) continue;

            const weight = this.adjacencyMatrix[i][j];
            if (!Number.isFinite(weight) || weight <= 0) continue;

            const P_uv = this.projectionMatrices.get(`${u}-${v}`);
            const P_vu = this.projectionMatrices.get(`${v}-${u}`);

            if (!isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vu)) {
                logger.warn(`Non-finite projection matrix for edge ${u}-${v}. Skipping block.`);
                continue;
            }

            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    const val_uv = -weight * (P_uv[qi]?.[qj] || 0);
                    if (Number.isFinite(val_uv)) {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = clamp(val_uv, -100, 100); // Clamp Lfull components
                    } else {
                        Lfull[i * this.qDim + qi][j * this.qDim + qj] = 0;
                    }
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                     const val_vu = -weight * (P_vu[qi]?.[qj] || 0);
                     if (Number.isFinite(val_vu)) {
                        Lfull[j * this.qDim + qi][i * this.qDim + qj] = clamp(val_vu, -100, 100); // Clamp Lfull components
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
                    degree += (this.adjacencyMatrix[i]?.[j] || 0);
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                Lfull[i * this.qDim + qi][i * this.qDim + qi] = clamp((Number.isFinite(degree) ? degree : 0) + this.eps, -100, 100); // Clamp Lfull components
            }
        }

        const f_s = new Float32Array(N).fill(0);
        for (let i = 0; i < n; i++) {
            const inputVal = qInput[i % qInput.length];
            for (let qi = 0; qi < this.qDim; qi++) {
                f_s[i * this.qDim + qi] = this.alpha * inputVal * 0.7;
            }
        }

        const eta = this.gamma / Math.max(1, this.maxEigApprox);
        
        // Ensure A is constructed robustly and its elements are clamped
        const A = identity(N).map((row, i) => new Float32Array(row.map((v, j) => {
            const val = v + eta * (Lfull[i]?.[j] || 0);
            return Number.isFinite(val) ? clamp(val, -100, 100) : 0; // Sanitize and clamp A elements
        })));
        
        const rhs = vecAdd(s, vecScale(f_s, eta)).map(v => Number.isFinite(v) ? clamp(v, -100, 100) : 0); // Explicitly sanitize rhs

        if (!isFiniteMatrix(A) || !isFiniteVector(rhs)) {
            logger.error('diffuseQualia: Matrix A or RHS vector contains non-finite values before CG solve. Skipping diffusion or using fallback.');
            const sNext = new Float32Array(N).fill(0); 
            this._updateStalksAndWindow(sNext, n);
            await this._updateDerivedMetrics();
            return;
        }

        let sSolved;
        try {
            sSolved = await runWorkerTask('solveLinearSystemCG', { A: flattenMatrix(A), b: rhs, opts: { tol: 1e-6, maxIter: 15 } }, 5000);
            if (!isFiniteVector(sSolved)) { // Check immediately after worker call
                 logger.error('Worker returned non-finite sSolved from solveLinearSystemCG. Falling back to zeros.');
                 sSolved = vecZeros(N);
            }
        } catch (e) {
            logger.error('Error solving linear system in worker (CG). Falling back to zero vector:', e);
            sSolved = vecZeros(N);
        }

        if (!isFiniteVector(sSolved)) {
            logger.error('CRITICAL: Solver output or fallback contained non-finite values. Resetting sSolved to zero vector.', { sSolved });
            sSolved = vecZeros(N); 
        }
        
        const sNext = new Float32Array(sSolved.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0)); // Ensure final stalk values are tightly clamped

        this._updateStalksAndWindow(sNext, n);
        await this._updateDerivedMetrics();
    }

    _updateStalksAndWindow(sNextVector, n) {
        const currentStalkNorms = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const rawStalkSlice = sNextVector.slice(i * this.qDim, (i + 1) * this.qDim);
            const sanitizedStalk = new Float32Array(rawStalkSlice.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
            this.stalks.set(this.graph.vertices[i], sanitizedStalk);
            currentStalkNorms[i] = norm2(sanitizedStalk);
        }
        this.stalkHistory.push(currentStalkNorms);
        if (this.stalkHistory.length > this.stalkHistorySize) {
            this.stalkHistory.shift();
        }

        const sanitizedNextVector = new Float32Array(sNextVector.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
        this.windowedStates.push(sanitizedNextVector);
        if (this.windowedStates.length > this.windowSize) this.windowedStates.shift();
    }

    async _updateDerivedMetrics() {
        try {
            await this.computeH1Dimension();
            await this.computeGluingInconsistency();
            this.computeGestaltUnity();
            await this.computeIntegratedInformation();
        } catch (e) {
            logger.error("Error during derived metrics update:", e);
            this.phi = 0.01;
            this.h1Dimension = 0.5;
            this.gestaltUnity = 0.05;
            this.stability = 0.01;
            this.inconsistency = 1.0;
        }
    }

    async computeH1Dimension() {
        const { boundary1, boundary2 } = this.buildBoundaryMatrices();
        if (!isFiniteMatrix(boundary1) || !isFiniteMatrix(boundary2)) {
            logger.warn("Non-finite boundary matrices detected. Setting H1 to default max/min.");
            this.h1Dimension = 1;
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }

        const flatBoundary1 = flattenMatrix(boundary1);
        const flatBoundary2 = flattenMatrix(boundary2);

        if (!flatBoundary1?.flatData || !isFiniteVector(flatBoundary1.flatData)) {
            logger.error("Flat boundary1 is invalid before sending to worker for rank calculation.");
            this.h1Dimension = 1;
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }
        if (!flatBoundary2?.flatData || !isFiniteVector(flatBoundary2.flatData)) {
            logger.error("Flat boundary2 is invalid before sending to worker for rank calculation.");
            this.h1Dimension = 1;
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }

        let rankB1, rankB2;
        try {
            rankB1 = await runWorkerTask('matrixRank', { matrix: flatBoundary1 }, 10000);
            if (!Number.isFinite(rankB1)) rankB1 = 0;
        } catch (e) {
            logger.error('Error computing rankB1 in worker:', e);
            this.h1Dimension = 1;
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }
        try {
            rankB2 = await runWorkerTask('matrixRank', { matrix: flatBoundary2 }, 10000);
            if (!Number.isFinite(rankB2)) rankB2 = 0;
        } catch (e) {
            logger.error('Error computing rankB2 in worker:', e);
            this.h1Dimension = 1;
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }
        
        const safeRankB1 = Number.isFinite(rankB1) && rankB1 >= 0 ? rankB1 : 0;
        const safeRankB2 = Number.isFinite(rankB2) && rankB2 >= 0 ? rankB2 : 0;

        const rawH1 = this.graph.edges.length - safeRankB1 - safeRankB2;
        this.h1Dimension = clamp(rawH1, 0, 3);
        if (!Number.isFinite(this.h1Dimension)) this.h1Dimension = 1;

        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        if (!Number.isFinite(this.stability)) this.stability = 0.5;
    }

    async computeGluingInconsistency() {
        let sum = 0;
        for (const [u, v] of this.graph.edges) {
            const stalk_u = this.stalks.get(u);
            const stalk_v = this.stalks.get(v);
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);

            if (!P_uv || !isFiniteVector(stalk_u) || !isFiniteVector(stalk_v) || !isFiniteMatrix(P_uv)) {
                logger.warn(`Skipping inconsistency calculation for edge ${u}-${v} due to non-finite inputs.`);
                continue;
            }

            let projected_u;
            try {
                projected_u = await runWorkerTask('matVecMul', { matrix: flattenMatrix(P_uv), vector: stalk_u }, 5000);
                if (!isFiniteVector(projected_u)) {
                    logger.warn(`Worker returned non-finite projected_u for edge ${u}-${v}. Setting to zeros.`);
                    projected_u = vecZeros(this.qDim);
                }
            } catch (e) {
                logger.error(`Error projecting stalk_u for edge ${u}-${v} in worker:`, e);
                projected_u = vecZeros(this.qDim);
            }
            
            const safeProjected_u = isFiniteVector(projected_u) ? projected_u.map(v => clamp(v, -1, 1)) : vecZeros(this.qDim); // Clamp
            
            sum += norm2(vecSub(safeProjected_u, stalk_v));
        }
        this.inconsistency = this.graph.edges.length > 0 ? clamp(sum / this.graph.edges.length, 0, 1) : 0;
        if (!Number.isFinite(this.inconsistency)) this.inconsistency = 1.0;
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
                }
            }
        });
        this.gestaltUnity = this.graph.edges.length > 0 ? clamp(sum / this.graph.edges.length, 0.05, 0.99) : 0.05;
        if (!Number.isFinite(this.gestaltUnity)) this.gestaltUnity = 0.05;
    }

    async computeIntegratedInformation() {
        const validStates = this.windowedStates.filter(isFiniteVector);
        if (validStates.length < Math.max(4, this.stateDim + this.qDim)) {
            logger.warn('Not enough valid states for meaningful covariance calculation. Using base Phi.');
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }
        let cov;
        try {
            cov = await runWorkerTask('covarianceMatrix', { states: validStates, eps: this.eps }, 10000);
            if (!isFiniteMatrix(cov)) {
                 logger.warn("Worker returned non-finite covariance matrix. Setting MI to default.");
                 cov = [[this.eps]]; // Fallback to minimal valid matrix
            }
        } catch (e) {
            logger.error('Error computing covarianceMatrix in worker:', e);
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }

        if (!isFiniteMatrix(cov)) {
            logger.warn("Non-finite covariance matrix after worker call. Setting MI to default.");
            this.phi = clamp(0.5 + (this.gestaltUnity || 0.1) * (this.stability || 0.1), 0.01, 5);
            return;
        }

        const MI_val = logDeterminantFromDiagonal(cov);
        const MI = Number.isFinite(MI_val) ? Math.abs(MI_val) * 0.1 + 1e-6 : 1e-6;
        
        const safeStability = Number.isFinite(this.stability) ? this.stability : 0.1;
        const safeGestaltUnity = Number.isFinite(this.gestaltUnity) ? this.gestaltUnity : 0.1;
        const safeInconsistency = Number.isFinite(this.inconsistency) ? this.inconsistency : 1.0;

        this.phi = clamp(Math.log(1 + MI) * safeStability * safeGestaltUnity * Math.exp(-safeInconsistency), 0.01, 5);
        if (!Number.isFinite(this.phi)) this.phi = 0.01;
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

    tuneParameters() {
        const currentStability = Number.isFinite(this.stability) ? this.stability : 0.5;
        const currentInconsistency = Number.isFinite(this.inconsistency) ? this.inconsistency : 0.5;
        const currentGestaltUnity = Number.isFinite(this.gestaltUnity) ? this.gestaltUnity : 0.5;
        const currentH1Dimension = Number.isFinite(this.h1Dimension) ? this.h1Dimension : 1.0;

        this.alpha = clamp(this.alpha * (1 + 0.02 * (1 - currentStability)) * (1 + 0.01 * currentInconsistency) * (1 + 0.005 * currentH1Dimension), 0.01, 1);
        this.beta = clamp(this.beta * (1 + 0.02 * (1 - currentGestaltUnity)) * (1 - 0.01 * currentH1Dimension), 0.01, 1);
        this.gamma = clamp(this.gamma * (1 - 0.05 * currentH1Dimension) * (1 - 0.02 * currentInconsistency) * (1 + 0.01 * currentStability), 0.01, 0.5);
        
        logger.info(`Tuned parameters: Alpha=${this.alpha.toFixed(3)}, Beta=${this.beta.toFixed(3)}, Gamma=${this.gamma.toFixed(3)}`);
    }
}
// --- END OF FILE qualia-sheaf.js ---
