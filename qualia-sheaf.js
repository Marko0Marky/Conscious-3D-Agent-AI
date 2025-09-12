// --- START OF FILE qualia-sheaf.js ---
import { 
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity // Ensure identity is imported from utils
} from './utils.js';

/**
 * Represents an Enhanced Qualia Sheaf, a mathematical structure used to model the AI's "consciousness".
 * It handles qualia diffusion, topological adaptation, and metric computations like Phi, H1 dimension, etc.
 * Phase 1B: Added tetrahedra support, sparse boundary operators, Betti-1 via GF(2) rank, dynamic edges.
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
        this.owm = owm; // For dynamic weights via actorLoss
        this.stateDim = stateDim;
        this.entityNames = ['being', 'intent', 'existence', 'emergence', 'gestalt', 'context', 'rel_emergence'];
        this.qDim = this.entityNames.length; // Always 7
        
        const defaultVertices = ['agent_x', 'agent_z', 'agent_rot', 'target_x', 'target_z', 'vec_dx', 'vec_dz', 'dist_target'];

        const initialGraphVertices = graphData?.vertices && Array.isArray(graphData.vertices) && graphData.vertices.length >= defaultVertices.length ? graphData.vertices : defaultVertices;

        const initialBaseEdges = graphData?.edges && Array.isArray(graphData.edges) ? graphData.edges.slice(0, 15) : [
            ['agent_x', 'agent_rot'], ['agent_z', 'agent_rot'],
            ['agent_x', 'vec_dx'], ['agent_z', 'vec_dz'],
            ['target_x', 'vec_dx'], ['target_z', 'vec_dz'],
            ['vec_dx', 'dist_target'], ['vec_dz', 'dist_target']
        ];
        
        const explicitTriangles = graphData?.triangles && Array.isArray(graphData.triangles) ? graphData.triangles : [
             ['agent_x', 'agent_z', 'agent_rot'],
             ['target_x', 'target_z', 'dist_target'],
             ['agent_x', 'target_x', 'vec_dx'],
             ['agent_z', 'target_z', 'vec_dz']
        ];

        // Phase 1B: Tetrahedra support
        const explicitTetrahedra = graphData?.tetrahedra && Array.isArray(graphData.tetrahedra) ? graphData.tetrahedra : [
            ['agent_x', 'agent_z', 'target_x', 'target_z'],  // Example: Spatial frame
            ['agent_x', 'target_x', 'vec_dx', 'dist_target'], // Example: Directional pursuit context
            ['agent_rot', 'vec_dx', 'vec_dz', 'dist_target']  // Example: Orientational loop with direction vectors
        ];

        // --- Simplicial Complex Closure Logic ---
        const allVerticesSet = new Set(initialGraphVertices);
        explicitTriangles.forEach(triangle => {
            triangle.forEach(v => allVerticesSet.add(v));
        });
        explicitTetrahedra.forEach(tet => {
            tet.forEach(v => allVerticesSet.add(v));
        });
        const finalVertices = Array.from(allVerticesSet);
        
        const allEdgesSet = new Set();
        initialBaseEdges.forEach(edge => {
            allEdgesSet.add(edge.slice().sort().join(','));
        });

        const finalTrianglesUpdated = [...explicitTriangles];
        const finalTetrahedraUpdated = [...explicitTetrahedra];

        // Ensure higher-order simplices imply lower-order ones
        // From Tetrahedra, generate implied Triangles and Edges
        finalTetrahedraUpdated.forEach(tet => {
            if (tet.length !== 4) { logger.warn(`Invalid tetrahedron: ${tet}`); return; }
            for (let i = 0; i < 4; i++) {
                const newTri = tet.filter((_, idx) => idx !== i).sort(); 
                const triStr = newTri.join(',');
                if (!finalTrianglesUpdated.some(t => t.slice().sort().join(',') === triStr)) {
                    finalTrianglesUpdated.push(newTri);
                }
            }
        });

        // From Triangles, generate implied Edges
        finalTrianglesUpdated.forEach(tri => {
            if (tri.length !== 3) { logger.warn(`Invalid triangle: ${tri}`); return; }
            for (let i = 0; i < 3; i++) {
                const newEdge = [tri[i], tri[(i + 1) % 3]].sort();
                const edgeStr = newEdge.join(',');
                if (!allEdgesSet.has(edgeStr)) {
                    allEdgesSet.add(edgeStr);
                }
            }
        });
        const finalEdges = Array.from(allEdgesSet).map(s => s.split(','));
        // --- End Simplicial Complex Closure Logic ---


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
        this.h1Dimension = 0;  // Phase 1B: Betti-1
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

    async initialize() {
        logger.info('EnhancedQualiaSheaf.initialize() called.');
        try {
            // Initial computation of adjacency and H1
            await this.computeCorrelationMatrix(); // This also builds adjacencyMatrix
            this.projectionMatrices =             this.projectionMatrices = await this.computeProjectionMatrices(); // ENSURE THIS IS AWAITED AND ASSIGNED AND ASSIGNED
            await this.computeH1Dimension();  // Phase 1B
            await this._updateDerivedMetrics(); // Compute other metrics
            this.ready = true;
            logger.info('Enhanced Qualia Sheaf ready with higher-order simplices.');
        } catch (e) {
            logger.error('Error during Sheaf initialization:', e);
            this.ready = false;
            throw e;
        }
    }

    // --- Phase 1B: Topology Adaptation & Homology ---
    adaptSheafTopology() {
        if (this.stalkHistory.length < this.stalkHistorySize) return;

        const numVertices = this.graph.vertices.length;
        if (numVertices === 0) return;

        // Using average stalk magnitude as a proxy for activity
        const means = new Float32Array(numVertices).fill(0);
        for (let i = 0; i < numVertices; i++) {
            for (let t = 0; t < this.stalkHistorySize; t++) {
                // stalkHistory now stores norms, so directly use
                const val = this.stalkHistory[t][i];
                means[i] += Number.isFinite(val) ? val : 0;
            }
            means[i] /= this.stalkHistorySize;
        }

        // Compute correlation between vertex activities
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
        if (!isFiniteMatrix(this.correlationMatrix)) {
            logger.warn('Non-finite correlation matrix detected; resetting to zero.');
            this.correlationMatrix = zeroMatrix(numVertices, numVertices);
        }
    }

    async buildBoundaryMatrices() {
        // Phase 1B: Sparse boundary matrices over GF(2)
        if (!this.graph.vertices.length) {
            return {
                partial1: { flatData: new Float32Array(0), rows: 0, cols: 0 },
                partial2: { flatData: new Float32Array(0), rows: 0, cols: 0 },
                partial3: { flatData: new Float32Array(0), rows: 0, cols: 0 }
            };
        }

        const vMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        // Map strings to indices directly for efficient lookup
        const eMapIndices = new Map(this.graph.edges.map((e, i) => [e.slice().sort().join(','), i]));
        const tMapIndices = new Map(this.simplicialComplex.triangles.map((t, i) => [t.slice().sort().join(','), i]));

        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;
        const nT = this.simplicialComplex.triangles.length;
        const nTet = this.simplicialComplex.tetrahedra.length;

        // ∂₁: C₁ → C₀ (Edges to Vertices) - (Incidence matrix)
        // In GF(2), it simply connects vertices to edges.
        // For homology, we typically use ∂_k: C_k -> C_{k-1}. So ∂1: C1 -> C0
        // and ∂2: C2 -> C1 (triangles to edges), ∂3: C3 -> C2 (tetrahedra to triangles)
        const boundary1 = zeroMatrix(nE, nV); // Edges by Vertices
        this.graph.edges.forEach((edge, eIdx) => {
            if (eIdx < nE) { // Bounds check
                edge.forEach(v => {
                    const vIdx = vMap.get(v);
                    if (vIdx !== undefined && vIdx < nV) {
                        boundary1[eIdx][vIdx] = 1; // GF(2)
                    }
                });
            } else { logger.warn(`Edge index out of bounds for ∂1: ${eIdx}`); }
        });


        // ∂₂: C₂ → C₁ (Triangles to Edges) - (Faces of triangles)
        // In GF(2), it's the cycle matrix
        const boundary2 = zeroMatrix(nT, nE); // Triangles by Edges
        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (tIdx < nT) { // Bounds check
                const sortedTri = tri.slice().sort();
                const edgesOfTri = [
                    [sortedTri[0], sortedTri[1]].sort().join(','),
                    [sortedTri[1], sortedTri[2]].sort().join(','),
                    [sortedTri[2], sortedTri[0]].sort().join(',')
                ];
                edgesOfTri.forEach(edgeStr => {
                    const eIdx = eMapIndices.get(edgeStr); // Direct index lookup
                    if (eIdx !== undefined && eIdx < nE) {
                        boundary2[tIdx][eIdx] = 1; // GF(2)
                    } else { logger.warn(`Edge not found or index out of bounds for ∂2: ${edgeStr}`); }
                });
            } else { logger.warn(`Triangle index out of bounds for ∂2: ${tIdx}`); }
        });

        // ∂₃: C₃ → C₂ (Tetrahedra to Triangles) - (Faces of tetrahedra)
        const boundary3 = zeroMatrix(nTet, nT); // Tetrahedra by Triangles
        this.simplicialComplex.tetrahedra.forEach((tet, tetIdx) => {
            if (tetIdx < nTet) { // Bounds check
                const sortedTet = tet.slice().sort();
                for (let i = 0; i < 4; i++) {
                    const face = sortedTet.filter((_, idx) => idx !== i).sort(); // A face is a triangle
                    const faceStr = face.join(',');
                    const tIdx = tMapIndices.get(faceStr); // Direct index lookup
                    if (tIdx !== undefined && tIdx < nT) {
                        boundary3[tetIdx][tIdx] = 1; // GF(2)
                    } else { logger.warn(`Triangle not found or index out of bounds for ∂3: ${faceStr}`); }
                    // Log the face generation for debugging
                    // logger.info(`Tet ${tet.join(',')} -> Face ${faceStr} -> tIdx ${tIdx}`);
                }
            } else { logger.warn(`Tetrahedron index out of bounds for ∂3: ${tetIdx}`); }
        });


        // Flatten for worker, ensuring finite matrices
        const safeFlatten = (matrix, rows, cols, name) => {
            if (!isFiniteMatrix(matrix)) {
                logger.error(`Non-finite matrix for ${name} boundary detected. Returning empty flattened matrix.`);
                return { flatData: new Float32Array(0), rows: 0, cols: 0 };
            }
            return flattenMatrix(matrix);
        };

        return {
            // Note: Homology typically uses ∂_k: C_k -> C_{k-1}. We need rank(∂_k)
            // For H1 = Ker(∂2) / Im(∂3)
            // so we need rank(∂2) and rank(∂3). The matrix representing ∂1 (edges to vertices) isn't directly used for H1 Betti-number calculation here, but included for completeness of the complex.
            partial1: safeFlatten(boundary1, nE, nV, "boundary1"),
            partial2: safeFlatten(boundary2, nT, nE, "boundary2"),
            partial3: safeFlatten(boundary3, nTet, nT, "boundary3")
        };
    }

    async computeCorrelationMatrix() {
        // Phase 1B: Dynamic weights
        // OWM.actorLoss is typically a positive value, so normalize it.
        // Lower actorLoss means better performance (less error), so we want to boost weights.
        const performanceFactor = Number.isFinite(this.owm?.actorLoss) 
            ? (1 - clamp(this.owm.actorLoss / 0.5, 0, 1)) // Scale actorLoss (e.g., 0.5 is max expected)
            : 0; // If actorLoss is NaN, assume worst performance (0 boost)
        const performanceScalar = clamp(1 + performanceFactor, 0.5, 2.0); // Boost range 0.5 to 2.0

        // Heuristic: boost connectivity if the AI is 'confused' (high h1Dimension)
        // This encourages exploring new conceptual links to resolve topological 'holes'
        const h1Boost = 1 + clamp(this.h1Dimension / (this.graph.vertices.length / 2), 0, 1) * 0.5; // Max 50% boost

        this.adaptSheafTopology(); // Update based on stalk history
        this.adjacencyMatrix = zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);

        this.graph.edges.forEach(([u, v]) => {
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i >= 0 && j >= 0) {
                const correlation = this.correlationMatrix[i][j];
                
                // Combine correlation with dynamic factors
                // Base weight starts from 0.1 to always allow some connection
                const dynamicWeight = clamp((0.1 + 0.9 * ((1 + correlation) / 2)) * performanceScalar * h1Boost, 0.01, 1.0);
                
                this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = dynamicWeight;
            }
        });

        if (!isFiniteMatrix(this.adjacencyMatrix)) {
            logger.error('Non-finite adjacency matrix after dynamic weighting; resetting to zero.');
            this.adjacencyMatrix = zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);
        }

        // logger.info(`Correlation matrix computed. PerfScalar=${performanceScalar.toFixed(3)}, H1Boost=${h1Boost.toFixed(3)}, Adj[0][1]=${this.adjacencyMatrix[0]?.[1]?.toFixed(3) || 'N/A'}`);
    }

    // This method is now effectively replaced by computeCorrelationMatrix for adjacency logic
    buildAdjacencyMatrix() { /* Legacy, now handled by computeCorrelationMatrix */ }

    buildLaplacian() {
        const n = this.graph.vertices.length;
        const adj = this.adjacencyMatrix; // Use the dynamically computed adjacency
        if (!adj || !isFiniteMatrix(adj)) {
            logger.error('buildLaplacian: Adjacency matrix is invalid. Cannot build Laplacian.');
            return zeroMatrix(n, n);
        }
        const L = zeroMatrix(n, n);

        for (let i = 0; i < n; i++) {
            let deg = 0;
            for (let j = 0; j < n; j++) {
                const weight = adj[i]?.[j] || 0;
                if (!Number.isFinite(weight)) { 
                    logger.warn(`Non-finite weight in adjacency matrix at [${i}][${j}]. Setting to 0.`);
                    adj[i][j] = 0; 
                }
                if (adj[i][j] > 0) {
                    L[i][j] = -adj[i][j];
                    deg += adj[i][j];
                }
            }
            L[i][i] = deg + this.eps; // Add epsilon for numerical stability
        }
        if (!isFiniteMatrix(L)) {
            logger.error('Non-finite Laplacian matrix detected; resetting to zero.');
            return zeroMatrix(n, n);
        }
        return L;
    }

    async computeProjectionMatrices() {
        const projections = new Map();
        // Identity matrix for now, can be replaced by learned projections later
        const identityMatrix = identity(this.qDim); // Use the utility's identity which creates Float32Array rows

        for (const [u, v] of this.graph.edges) {
            projections.set(`${u}-${v}`, identityMatrix);
            projections.set(`${v}-${u}`, identityMatrix);
        }
        return projections;
    }

    async _updateGraphStructureAndMetrics() {
        try {
            await this.computeCorrelationMatrix(); // This will call adaptSheafTopology internally
            this.laplacian = this.buildLaplacian();

            const flatLaplacian = flattenMatrix(this.laplacian);
            if (!isFiniteVector(flatLaplacian.flatData)) {
                logger.error("Non-finite Laplacian matrix detected before spectral norm calculation. Defaulting maxEigApprox to 1.");
                this.maxEigApprox = 1;
            } else {
                this.maxEigApprox = await runWorkerTask('matrixSpectralNormApprox', { matrix: flatLaplacian }, 10000);
                if (!Number.isFinite(this.maxEigApprox) || this.maxEigApprox <= 0) {
                    logger.warn(`maxEigApprox was invalid (${this.maxEigApprox}). Resetting to 1.`);
                    this.maxEigApprox = 1;
                }
            }
            
            this.projectionMatrices =             this.projectionMatrices = await this.computeProjectionMatrices(); // ENSURE THIS IS AWAITED AND ASSIGNED AND ASSIGNED
            await this.computeH1Dimension(); // Update H1 after graph structure changes
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

        // --- Build Lfull: the block Laplacian matrix ---
        const Lfull = zeroMatrix(N, N);
        const idx = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) {
            const i = idx.get(u), j = idx.get(v);
            if (i === undefined || j === undefined) continue;

            const weight = this.adjacencyMatrix[i][j];
            if (!Number.isFinite(weight) || weight <= 0) continue;

            const P_uv = this.projectionMatrices.get(`${u}-${v}`);
            const P_vu = this.projectionMatrices.get(`${v}-${u}`);

            if (!isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vu)) { // This check should now pass if P_uv is correctly populated
                logger.warn(`Non-finite or missing projection matrix for edge ${u}-${v}. Skipping block. P_uv isFiniteMatrix: ${isFiniteMatrix(P_uv)}.`);
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
        // --- End Build Lfull ---

        const f_s = new Float32Array(N).fill(0);
        for (let i = 0; i < n; i++) {
            const inputVal = qInput[i % qInput.length];
            for (let qi = 0; qi < this.qDim; qi++) {
                f_s[i * this.qDim + qi] = this.alpha * inputVal * 0.7;
            }
        }

        const eta = this.gamma / Math.max(1, this.maxEigApprox);
        
        // Ensure A is constructed robustly and its elements are clamped
        const A = zeroMatrix(N, N).map((row, i) => new Float32Array(row.map((v, j) => {
            const val = (i === j ? 1 : 0) + eta * (Lfull[i]?.[j] || 0); // Identity(N) + eta * Lfull
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

    // --- Phase 1B: Betti-1 Computation ---
    async computeH1Dimension() {
        if (!this.adjacencyMatrix) {
            await this.computeCorrelationMatrix();
        }
        const boundaries = await this.buildBoundaryMatrices();
        
        let rankPartial2 = 0; // rank(∂2: C2 -> C1)
        let rankPartial3 = 0; // rank(∂3: C3 -> C2)

        try {
            // Compute rank of ∂₂ (triangles to edges)
            const p2Result = await runWorkerTask('smithNormalForm', boundaries.partial2, 5000);
            rankPartial2 = p2Result.rank || 0;
            if (!Number.isFinite(rankPartial2)) { rankPartial2 = 0; logger.warn('Non-finite rank for ∂2'); }

            // Compute rank of ∂₃ (tetrahedra to triangles)
            const p3Result = await runWorkerTask('smithNormalForm', boundaries.partial3, 5000);
            rankPartial3 = p3Result.rank || 0;
            if (!Number.isFinite(rankPartial3)) { rankPartial3 = 0; logger.warn('Non-finite rank for ∂3'); }

        } catch (e) {
            logger.error('Homology computation (smithNormalForm) failed:', e);
            // Fallback to simpler heuristic for Betti-1 if worker fails
            this.h1Dimension = this.graph.edges.length - this.graph.vertices.length + this.simplicialComplex.triangles.length;
            this.h1Dimension = clamp(this.h1Dimension, 0, this.graph.vertices.length);
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
            return;
        }
        
        const numEdges = this.graph.edges.length;
        const numTriangles = this.simplicialComplex.triangles.length;

        // Betti-1 = dim Ker(∂₂) - dim Im(∂₃)
        // dim Ker(∂₂) = num_edges - rank(∂₂)
        // dim Im(∂₃) = rank(∂₃)
        this.h1Dimension = (numEdges - rankPartial2) - rankPartial3;
        this.h1Dimension = clamp(this.h1Dimension, 0, numEdges); // Clamp within reasonable bounds
        if (!Number.isFinite(this.h1Dimension)) this.h1Dimension = 0;

        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        if (!Number.isFinite(this.stability)) this.stability = 0.5;

        // logger.info(`H1 dim computed: ${this.h1Dimension.toFixed(2)} (numEdges=${numEdges}, rank(∂2)=${rankPartial2}, rank(∂3)=${rankPartial3})`);
    }

    async computeGluingInconsistency() {
        let sum = 0;
        for (const [u, v] of this.graph.edges) {
            const stalk_u = this.stalks.get(u);
            const stalk_v = this.stalks.get(v);
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);

            // Explicitly check for non-finite values at each point of access
            if (!P_uv || !isFiniteMatrix(P_uv) || !isFiniteVector(stalk_u) || !isFiniteVector(stalk_v)) {
                logger.warn(`Skipping inconsistency calculation for edge ${u}-${v} due to non-finite inputs (stalk_u: ${isFiniteVector(stalk_u)}, stalk_v: ${isFiniteVector(stalk_v)}, P_uv: ${isFiniteMatrix(P_uv) ? 'true' : 'false' /* more informative */}).`);
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
                 cov = zeroMatrix(validStates[0].length, validStates[0].length).map(row => row.fill(this.eps)); // Fallback
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

        // Phase 1B: Incorporate h1Dimension into Phi for higher-order integration
        this.phi = clamp(Math.log(1 + MI) * safeStability * safeGestaltUnity * Math.exp(-safeInconsistency) * (1 + 0.05 * this.h1Dimension), 0.01, 5);
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
        
        // logger.info(`Tuned parameters: Alpha=${this.alpha.toFixed(3)}, Beta=${this.beta.toFixed(3)}, Gamma=${this.gamma.toFixed(3)}, H1Dim=${currentH1Dimension.toFixed(2)}`);
    }
}
// --- END OF FILE qualia-sheaf.js ---
