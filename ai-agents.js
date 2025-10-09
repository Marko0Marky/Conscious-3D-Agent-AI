
/**
 * Combined, Refined, and Unified Qualia Sheaf Module (TUC Framework)
 * Version: 7.2 (Corrected Update Cycle) - Condensed Version
 *
 * This condensed version separates the core sheaf dynamics from higher-order
 * cognitive processing into two cohesive classes: QualiaSheafBase and QualiaCognitionLayer.
 * This ensures a clear distinction between the foundational state and its advanced interpretations.
 */

// Importing utilities from a separate module, as per original structure.
import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity, transpose, covarianceMatrix, matVecMul, matMul, vecMul, vectorAsRow, vectorAsCol, isFiniteNumber, safeEigenDecomposition, randomMatrix
} from './utils.js';


// Helper functions for matrix/vector operations (remaining external or in utils)
function safeVecScale(vector, scalar) {
    if (!isFiniteVector(vector) || !isFiniteNumber(scalar) || !Number.isFinite(scalar)) {
        logger.warn('safeVecScale: Invalid input. Returning zero vector.');
        return vecZeros(vector?.length || 1);
    }
    const result = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
        result[i] = vector[i] * scalar;
    }
    return result;
}

function _matMul(matrixA, matrixB) {
    if (!isFiniteMatrix(matrixA) || !isFiniteMatrix(matrixB) || matrixA[0]?.length !== matrixB.length) {
        const rows = matrixA?.length || 1;
        const cols = matrixB?.[0]?.length || 1;
        logger.warn(`_matMul: Invalid matrix multiplication. Returning ${rows}x${cols} zero matrix.`);
        return zeroMatrix(rows, cols);
    }
    const C = zeroMatrix(matrixA.length, matrixB[0].length);
    for (let i = 0; i < matrixA.length; i++) {
        for (let j = 0; j < matrixB[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < matrixA[0].length; k++) {
                sum += (matrixA[i][k] || 0) * (matrixB[k][j] || 0);
            }
            C[i][j] = clamp(sum, -1e6, 1e6);
        }
    }
    return C;
}

function _transpose(matrix) {
    if (!isFiniteMatrix(matrix) || matrix.length === 0) return [];
    const result = zeroMatrix(matrix[0].length, matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

function _matVecMul(matrix, vector) {
    if (!isFiniteMatrix(matrix) || !isFiniteVector(vector) || matrix[0]?.length !== vector.length) {
        logger.warn('_matVecMul: Invalid matrix or vector input. Returning zero vector.');
        return vecZeros(matrix?.length || 1);
    }
    const result = new Float32Array(matrix.length).fill(0);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

export async function runWorkerTaskWithRetry(type, data, timeout = 20000, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            return await runWorkerTask(type, data, timeout);
        } catch (err) {
            logger.warn(`Attempt ${i + 1} failed for task ${type}: ${err.message}`);
            if (i === retries - 1) {
                logger.error(`All ${retries} attempts failed for task ${type}.`, { error: err.message, stack: err.stack });
                return type === 'matVecMul' ? vecZeros(data.expectedDim || 7) :
                       type === 'matMul' ? identity(data.rows || 7) :
                       type === 'complexEigenvalues' ? Array(data.dim || 1).fill({ re: 1, im: 0 }) :
                       type === 'ksg_mi' ? 0 :
                       type === 'topologicalScore' ? 0 : null;
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
}

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
            logger.warn('CircularBuffer.push: Invalid item (null/undefined). Skipping.', { item });
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
            const item = this.get(i);
            if (item !== null) {
                result.push(item);
            }
        }
        return result;
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
 * QualiaSheafBase: Core structural and dynamic representation of the Sheaf.
 * Handles graph topology, stalk states, projection matrices, and fundamental diffusion.
 */
export class QualiaSheafBase {
    constructor(graphData, config = {}) {
        this.owm = null; // Reference to an external OWM instance
        this.ready = false;
        this.entityNames = config.entityNames || ['shape', 'emotion', 'symbolic', 'synesthesia', 'metacognition', 'social', 'temporal'];
        this.qDim = this.entityNames.length;
        this.stateDim = config.stateDim || 13; // Dimension of external state input
        this.leakRate = config.leakRate ?? 0.01;
        this.maxForceNorm = config.maxForceNorm ?? 0.2;

        this.alpha = clamp(config.alpha ?? 0.1, 0.01, 1); // Diffusion/learning rates
        this.beta = clamp(config.beta ?? 0.05, 0.01, 1);
        this.gamma = clamp(config.gamma ?? 0.005, 0.01, 0.5);
        this.sigma = clamp(config.sigma ?? 0.01, 0.001, 0.1); // Noise
        this.eps = config.eps ?? 1e-6; // Small epsilon for numerical stability
        this.stabEps = config.stabEps ?? 1e-8; // Stability epsilon for matrix operations

        this.adaptation = { // Parameters for adaptive graph topology
            addThresh: clamp(config.addThresh ?? 0.7, 0.5, 0.95),
            removeThresh: clamp(config.removeThresh ?? 0.2, 0.05, 0.4),
            targetH1: config.targetH1 ?? 2.0,
            maxEdges: config.maxEdges ?? 50,
        };

        this.resonanceOmega = config.resonanceOmega ?? 1.2; // Spectral resonance parameters
        this.resonanceEps = config.resonanceEps ?? 0.08;

        // Graph and simplicial complex structure
        this.graph = { vertices: [], edges: [] };
        this.simplicialComplex = { triangles: [], tetrahedra: [] };
        this.edgeSet = new Set();
        this._initializeGraph(graphData);

        // Stalks (qualia states at vertices) and their history
        this.stalks = new Map();
        this._initializeStalks();
        this.stalkHistorySize = config.stalkHistorySize ?? 100;
        this.stalkHistory = new CircularBuffer(this.stalkHistorySize);
        this.stalkNormHistory = new CircularBuffer(this.stalkHistorySize);

        // State projection matrix (learns how external state maps to qualia)
        const projData = randomMatrix(this.qDim, this.stateDim, 1.0);
        this.stateToQualiaProjection = unflattenMatrix(projData);
        this.stateToQualiaProjection.rows = this.qDim;
        this.stateToQualiaProjection.cols = this.stateDim;

        // Projection matrices between stalks on edges
        this.projectionMatrices = new Map();
        this.graph.edges.forEach(edge => {
            const [u, v] = edge;
            const P_identity = identity(this.qDim);
            this.projectionMatrices.set(`${u}-${v}`, P_identity);
            this.projectionMatrices.set(`${v}-${u}`, P_identity);
        });

        // Windowed states for correlation and higher-order computations
        const N_total_stalk_dim = this.graph.vertices.length * this.qDim;
        this.windowSize = Math.max(100, N_total_stalk_dim * 3);
        this.windowedStates = new CircularBuffer(this.windowSize);
        this._initializeWindowedStates(N_total_stalk_dim);

        // Dynamical matrices
        this.laplacian = null;
        this.correlationMatrix = null;
        this.adjacencyMatrix = null;
        this.maxEigApprox = 1; // Approximation of max eigenvalue for stable diffusion

        this.lastGoodEigenResult = null; // Cache for eigenvalue decomposition

        logger.info(`QualiaSheafBase constructed: vertices=${this.graph.vertices.length}, edges=${this.graph.edges.length}, triangles=${this.simplicialComplex.triangles.length}, tetrahedra=${this.simplicialComplex.tetrahedra.length}`);
    }

    setOWM(owmInstance) { this.owm = owmInstance; }

    /**
     * Initializes or re-initializes the graph structure based on provided data or defaults.
     * Ensures minimum valid structure.
     *
     */
    _initializeGraph(graphData) {
        let safeGraphData = graphData && typeof graphData === 'object' ? graphData : {};

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

        // Ensure graphData has minimum valid structure, otherwise use defaults.
        if (
            !Array.isArray(safeGraphData.vertices) || safeGraphData.vertices.length < 2 ||
            !Array.isArray(safeGraphData.edges) || safeGraphData.edges.length < 1
        ) {
            safeGraphData = {
                vertices: defaultVertices,
                edges: defaultEdges,
                triangles: defaultTriangles,
                tetrahedra: defaultTetrahedra
            };
        }

        const allVerticesSet = new Set(safeGraphData.vertices);
        (safeGraphData.triangles || []).forEach(tri => tri.forEach(v => allVerticesSet.add(v)));
        (safeGraphData.tetrahedra || []).forEach(tet => tet.forEach(v => allVerticesSet.add(v)));
        let finalVertices = Array.from(allVerticesSet);
        if (finalVertices.length === 0) {
            logger.warn('Sheaf._initializeGraph: No vertices derived from graphData or defaults. Forcing default vertices.');
            finalVertices = [...defaultVertices];
        }

        const allEdgesSet = new Set(safeGraphData.edges.map(e => e.slice(0, 2).sort().join(',')));
        let finalTrianglesUpdated = [...(safeGraphData.triangles || [])];
        (safeGraphData.tetrahedra || []).forEach(tet => {
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

        if (finalVertices.length < 2) {
            if (!finalVertices.includes('fallback_v1')) finalVertices.push('fallback_v1');
            if (!finalVertices.includes('fallback_v2')) finalVertices.push('fallback_v2');
            logger.warn('Sheaf._initializeGraph: Less than 2 vertices, added fallbacks.');
        }
        if (finalVertices.length >= 2 && allEdgesSet.size === 0) {
            allEdgesSet.add([finalVertices[0], finalVertices[1]].sort().join(','));
            logger.warn('Sheaf._initializeGraph: No edges, added a fallback edge.');
        }

        this.graph = {
            vertices: finalVertices,
            edges: Array.from(allEdgesSet).map(s => s.split(',').concat([0.5])) // Add default weight
        };
        this.simplicialComplex = {
            triangles: finalTrianglesUpdated.filter(t => t.length === 3),
            tetrahedra: (safeGraphData.tetrahedra || []).filter(t => t.length === 4)
        };
        this.edgeSet = allEdgesSet;
    }

    /**
     * Initializes stalks for each vertex in the graph with random qualia vectors.
     *
     */
    _initializeStalks() {
        if (!this.graph || !Array.isArray(this.graph.vertices) || this.graph.vertices.length === 0) {
            logger.error('Sheaf._initializeStalks: Graph is invalid. Cannot initialize stalks.');
            return;
        }
        if (!Number.isInteger(this.qDim) || this.qDim <= 0) {
             logger.error(`Sheaf._initializeStalks: Invalid qDim (${this.qDim}). Cannot create stalks.`);
             return;
        }
        this.graph.vertices.forEach(v => {
            const stalk = new Float32Array(this.qDim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.5, -1, 1));
            if (!isFiniteVector(stalk) || stalk.length !== this.qDim) {
                logger.error(`Non-finite stalk generated for vertex ${v}; setting to zeros.`);
                this.stalks.set(v, vecZeros(this.qDim));
            } else {
                this.stalks.set(v, stalk);
            }
        });
    }

    /**
     * Initializes the circular buffer with random states for the windowed history.
     * @param {number} N_total_stalk_dim - The total dimension of the flattened stalk vector.
     *
     */
    _initializeWindowedStates(N_total_stalk_dim) {
        const effectiveN_total_stalk_dim = Number.isFinite(N_total_stalk_dim) && N_total_stalk_dim > 0 ? N_total_stalk_dim : 1;
        for (let i = 0; i < this.windowSize; i++) {
            const randomState = new Float32Array(effectiveN_total_stalk_dim).fill(0).map(() => clamp((Math.random() - 0.5) * 0.1, -1, 1));
            if (!isFiniteVector(randomState)) {
                logger.warn('_initializeWindowedStates: Non-finite random state generated. Filling with zeros.');
                randomState.fill(0);
            }
            this.windowedStates.push(randomState);
        }
    }

    /**
     * Retrieves the current stalk values as a single flattened vector.
     *
     */
    getStalksAsVector() {
        const nVertices = this.graph.vertices.length;
        const expectedLength = nVertices * this.qDim;
        const result = new Float32Array(expectedLength);
        let offset = 0;

        for (const vertex of this.graph.vertices) {
            let stalk = this.stalks.get(vertex);
            if (!isFiniteVector(stalk) || stalk.length !== this.qDim) {
                logger.warn(`Sheaf.getStalksAsVector: Invalid stalk for ${vertex}. Resetting to zeros.`, {
                    length: stalk?.length,
                    expected: this.qDim
                });
                stalk = vecZeros(this.qDim);
                this.stalks.set(vertex, stalk);
            }
            result.set(stalk, offset);
            offset += this.qDim;
        }

        if (!isFiniteVector(result)) {
            logger.error(`Sheaf.getStalksAsVector: CRITICAL: Assembled vector is non-finite. Returning a valid zero vector to prevent crash.`);
            return vecZeros(expectedLength);
        }
        return result;
    }

    /**
     * Updates the internal stalk states and records them in history buffers.
     *
     */
    _updateStalksAndWindow(sNext, nV, qInput) {
        if (!sNext || !(sNext instanceof Float32Array) || !isFiniteVector(sNext) || sNext.length !== nV * this.qDim) {
            logger.error(`CRITICAL: _updateStalksAndWindow received an invalid sNext vector. State will NOT be updated.`, {
                expected_len: nV * this.qDim, received_len: sNext?.length
            });
            return;
        }

        const currentStalks = [];
        const currentStalkNorms = new Float32Array(nV);
        for (let i = 0; i < nV; i++) {
            const vertex = this.graph.vertices[i];
            const start = i * this.qDim;
            const end = start + this.qDim;
            const newStalk = sNext.slice(start, end);

            this.stalks.set(vertex, newStalk);
            currentStalks.push(newStalk);
            currentStalkNorms[i] = norm2(newStalk);
        }

        this.stalkHistory.push(currentStalks);
        this.stalkNormHistory.push(currentStalkNorms);
        this.windowedStates.push(new Float32Array(sNext));
        this.qInput = qInput; // Store qualia input for other computations
    }

    /**
     * Initializes the sheaf, setting up graph, stalks, and matrices.
     *
     */
    async initialize() {
        if (this.ready) return;
        logger.info('QualiaSheafBase.initialize() called.');

        try {
            this._initializeGraph({}); // Re-initialize with default if needed, ensure consistency
            this._initializeStalks();

            // Projection matrices are already initialized in constructor; ensuring validity:
            this.projectionMatrices.forEach((matrix, key) => {
                if (!isFiniteMatrix(matrix) || matrix.length !== this.qDim) {
                    this.projectionMatrices.set(key, identity(this.qDim));
                }
            });

            await this.computeCorrelationMatrix();
            this.laplacian = this.buildLaplacian();

            const N = this.graph.vertices.length * this.qDim;
            if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N) {
                logger.warn('QualiaSheafBase.initialize: Laplacian was invalid after build. Using identity fallback.');
                this.laplacian = identity(N);
            }

            this.ready = true;
            logger.info('QualiaSheafBase ready.');
        } catch (e) {
            logger.error('CRITICAL ERROR: QualiaSheafBase initialization failed:', { message: e.message, stack: e.stack });
            this.ready = false;
            throw e;
        }
    }

    /**
     * Normalizes the state-to-qualia projection matrix row-wise.
     *
     */
    _normalizeProjectionMatrix() {
        if (!isFiniteMatrix(this.stateToQualiaProjection) || this.stateToQualiaProjection.length !== this.qDim) {
            logger.warn('_normalizeProjectionMatrix: Invalid or non-finite matrix upon entry. Re-randomizing.');
            const newProjData = randomMatrix(this.qDim, this.stateDim, 0.1);
            this.stateToQualiaProjection = unflattenMatrix(newProjData);
            this.stateToQualiaProjection.rows = this.qDim;
            this.stateToQualiaProjection.cols = this.stateDim;
            return;
        }

        const P = this.stateToQualiaProjection;
        for (let q = 0; q < this.qDim; q++) {
            let row = P[q];
            const norm = norm2(row);
            if (isFiniteNumber(norm) && norm > this.eps) {
                P[q] = vecScale(row, 1.0 / norm);
            } else {
                logger.warn(`_normalizeProjectionMatrix: Row ${q} has zero norm. Re-initializing it.`);
                const randomVector = new Float32Array(this.stateDim).map(() => (Math.random() - 0.5) * 0.1);
                P[q] = vecScale(randomVector, 1.0 / norm2(randomVector));
            }
        }
        if (!isFiniteMatrix(this.stateToQualiaProjection)) {
            logger.error('_normalizeProjectionMatrix: Matrix became non-finite after row normalization. This should not happen.');
            this.stateToQualiaProjection = identity(this.qDim);
        }
    }

    /**
     * Updates the state-to-qualia projection matrix using Hebbian learning with Gram-Schmidt orthogonalization.
     *
     */
    _updateStateToQualiaProjection(rawState, qualiaInput) {
        const learningRate = 0.002;
        const decay = 0.0001;

        if (!isFiniteVector(rawState) || !isFiniteVector(qualiaInput)) {
            logger.warn('_updateStateToQualiaProjection: Invalid rawState or qualiaInput. Skipping update.');
            return;
        }

        const P = this.stateToQualiaProjection;
        for (let q = 0; q < this.qDim; q++) {
            let row_q = P[q];
            for (let i = 0; i < this.stateDim; i++) {
                const hebbianUpdate = learningRate * (qualiaInput[q] || 0) * (rawState[i] || 0);
                row_q[i] = (row_q[i] || 0) + hebbianUpdate - (row_q[i] || 0) * decay;
            }

            // Gram-Schmidt orthogonalization
            for (let j = 0; j < q; j++) {
                const row_j = P[j];
                const dot_qj = dot(row_q, row_j);
                const dot_jj = dot(row_j, row_j);
                if (isFiniteNumber(dot_qj) && isFiniteNumber(dot_jj) && dot_jj > this.eps) {
                    const scale = dot_qj / dot_jj;
                    const projection = safeVecScale(row_j, scale);
                    row_q = vecSub(row_q, projection);
                }
            }

            const norm_q = norm2(row_q);
            if (isFiniteNumber(norm_q) && norm_q > this.eps) {
                P[q] = safeVecScale(row_q, 1.0 / norm_q);
            } else {
                const randomVector = new Float32Array(this.stateDim).map(() => Math.random() - 0.5);
                P[q] = safeVecScale(randomVector, 1.0 / norm2(randomVector));
            }
        }
        this._normalizeProjectionMatrix();
    }

    /**
     * Jacobi Eigenvalue Decomposition for a real symmetric matrix.
     *
     */
    _jacobiEigenvalueDecomposition(matrix, maxIterations = 50) {
        const n = matrix.length;
        if (n === 0) return { eigenvalues: [], eigenvectors: [] };

        let A = matrix.map(row => [...row]); // Deep copy
        let V = identity(n); // Eigenvector matrix

        for (let iter = 0; iter < maxIterations; iter++) {
            let p = 0, q = 1, maxOffDiagonal = 0;
            // Find pivot element (largest off-diagonal element)
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const absAij = Math.abs(A[i][j]);
                    if (absAij > maxOffDiagonal) {
                        maxOffDiagonal = absAij;
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxOffDiagonal < this.eps) break; // Convergence check

            // Compute rotation angle
            const apq = A[p][q];
            const app = A[p][p];
            const aqq = A[q][q];
            const tau = (aqq - app) / (2 * (apq !== 0 ? apq : this.eps));
            const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
            const c = 1 / Math.sqrt(1 + t * t); // cos(theta)
            const s = c * t; // sin(theta)

            // Apply rotation to A (A' = R^T A R)
            for (let i = 0; i < n; i++) {
                if (i !== p && i !== q) {
                    const aip = A[i][p];
                    const aiq = A[i][q];
                    A[i][p] = c * aip - s * aiq;
                    A[i][q] = s * aip + c * aiq;
                    A[p][i] = A[i][p]; // Symmetric update
                    A[q][i] = A[i][q]; // Symmetric update
                }
            }
            const app_new = app * c * c - 2 * apq * s * c + aqq * s * s;
            const aqq_new = app * s * s + 2 * apq * s * c + aqq * c * c;
            A[p][p] = app_new;
            A[q][q] = aqq_new;
            A[p][q] = 0; // Off-diagonal elements become zero
            A[q][p] = 0;

            // Apply rotation to V (V' = V R)
            for (let i = 0; i < n; i++) {
                const vip = V[i][p];
                const viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }

        const eigenvalues = A.map((row, i) => row[i]);
        return { eigenvalues, eigenvectors: V };
    }


    /**
     * Validates if a given vector is finite.
     * @param {Float32Array|Array<number>} vec - The vector to validate.
     * @returns {boolean} - True if the vector is finite, false otherwise.
     */
    _isValidStateVec(vec) {
        if (!vec || !Array.isArray(vec) && !(vec instanceof Float32Array)) return false;
        return vec.every(v => Number.isFinite(v));
    }

    /**
     * Adds an edge to the graph.
     *
     */
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

    /**
     * Removes an edge from the graph.
     *
     */
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

    /**
     * Adds a triangle (2-simplex) to the simplicial complex.
     * Automatically adds constituent edges if they don't exist.
     *
     */
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

    /**
     * Removes a triangle (2-simplex) from the simplicial complex.
     *
     */
    removeTriangle(a, b, c) {
        const sorted = [a, b, c].sort();
        const key = sorted.join(',');
        const idx = this.simplicialComplex.triangles.findIndex(t => t.sort().join(',') === key);
        if (idx !== -1) {
            this.simplicialComplex.triangles.splice(idx, 1);
            logger.info(`Removed triangle ${a}-${b}-${c}`);
        }
    }

    /**
     * Adds a tetrahedron (3-simplex) to the simplicial complex.
     * Automatically adds constituent triangles and edges if they don't exist.
     *
     */
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

    /**
     * Removes a tetrahedron (3-simplex) from the simplicial complex.
     *
     */
    removeTetrahedron(a, b, c, d) {
        const sorted = [a, b, c, d].sort();
        const key = sorted.join(',');
        const idx = this.simplicialComplex.tetrahedra.findIndex(t => t.sort().join(',') === key);
        if (idx !== -1) {
            this.simplicialComplex.tetrahedra.splice(idx, 1);
            logger.info(`Removed tetrahedron ${a}-${b}-${c}-${d}`);
        }
    }

    /**
     * Checks if a triangle is valid, meaning its vertices exist and its edges are present.
     *
     */
    isValidTriangle(tri) {
        if (!Array.isArray(tri) || tri.length !== 3) return false;
        const [a, b, c] = tri;
        if (!this.graph.vertices.includes(a) || !this.graph.vertices.includes(b) || !this.graph.vertices.includes(c)) return false;

        const edge_ab = [a, b].sort().join(',');
        const edge_bc = [b, c].sort().join(',');
        const edge_ca = [c, a].sort().join(',');

        return this.edgeSet.has(edge_ab) && this.edgeSet.has(edge_bc) && this.edgeSet.has(edge_ca);
    }

    /**
     * Computes the correlation matrix between vertex stalk norms from historical data.
     *
     */
    async computeVertexCorrelationsFromHistory() {
        const n = this.graph.vertices.length;
        if (n === 0) return identity(0);

        const validHistory = this.stalkNormHistory.getAll().filter(item => isFiniteVector(item) && item.length === n);

        if (validHistory.length < 2) {
            logger.warn('computeVertexCorrelationsFromHistory: Insufficient valid history for correlation. Returning identity.');
            return identity(n);
        }

        try {
            const covMatrixRaw = await runWorkerTaskWithRetry('covarianceMatrix', { states: validHistory, dim: n, eps: this.eps }, 10000);
            if (isFiniteMatrix(covMatrixRaw) && covMatrixRaw.length === n) {
                return covMatrixRaw;
            }
            logger.warn('computeVertexCorrelationsFromHistory: All correlation matrix computations failed; returning identity matrix as a final fallback.');
            return identity(n);
        } catch (e) {
            logger.error(`computeVertexCorrelationsFromHistory: An unexpected error occurred.`, { error: e.message, stack: e.stack });
            return identity(n);
        }
    }

    /**
     * Computes and updates the correlation and adjacency matrices.
     *
     */
    async computeCorrelationMatrix() {
        const nV = this.graph.vertices.length;
        if (nV === 0) {
            this.correlationMatrix = identity(0);
            this.adjacencyMatrix = identity(0);
            return;
        }
        this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
        if (!isFiniteMatrix(this.correlationMatrix) || this.correlationMatrix.length !== nV || (this.correlationMatrix[0]?.length || 0) !== nV) {
            logger.warn('Sheaf.computeCorrelationMatrix: Invalid correlation matrix from history. Using identity.');
            this.correlationMatrix = identity(nV);
        }

        this.adjacencyMatrix = zeroMatrix(nV, nV);
        this.graph.edges.forEach(([u, v, weight = 0.1]) => {
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i !== -1 && j !== -1 && i < nV && j < nV) {
                const correlation = this.correlationMatrix[i]?.[j] || 0;
                const dynamicWeight = clamp(weight + 0.5 * correlation, 0.01, 1.0);
                this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = dynamicWeight;
            }
        });
        if (!isFiniteMatrix(this.adjacencyMatrix)) {
            logger.error('Sheaf.computeCorrelationMatrix: Generated adjacency matrix is non-finite. Resetting to identity.');
            this.adjacencyMatrix = identity(nV);
        }
    }

    /**
     * Builds the combinatorial Laplacian matrix for the sheaf.
     *
     */
    buildLaplacian() {
        const nV = this.graph.vertices.length;
        const N = nV * this.qDim;
        if (nV === 0 || this.qDim === 0) {
            return identity(0);
        }

        if (!isFiniteMatrix(this.adjacencyMatrix) || this.adjacencyMatrix.length !== nV) {
            logger.error('buildLaplacian: Called with an invalid adjacency matrix. Returning identity.');
            return identity(N);
        }

        const L = zeroMatrix(N, N);
        const idxMap = new Map(this.graph.vertices.map((v, i) => [v, i]));

        for (const [u, v] of this.graph.edges) {
            const i = idxMap.get(u);
            const j = idxMap.get(v);
            if (i === undefined || j === undefined) continue;

            const weight = this.adjacencyMatrix[i][j] || 0;
            let P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);
            if (!isFiniteMatrix(P_uv) || P_uv.length !== this.qDim) {
                P_uv = identity(this.qDim);
            }

            for (let qi = 0; qi < this.qDim; qi++) {
                for (let qj = 0; qj < this.qDim; qj++) {
                    const val = -weight * (P_uv[qi][qj] || 0);
                    if (Number.isFinite(val)) {
                        L[i * this.qDim + qi][j * this.qDim + qj] = val;
                        L[j * this.qDim + qi][i * this.qDim + qj] = val;
                    }
                }
            }
        }

        for (let i = 0; i < nV; i++) {
            let degree = 0;
            for (let j = 0; j < nV; j++) {
                if (i !== j) {
                    degree += this.adjacencyMatrix[i][j] || 0;
                }
            }
            for (let qi = 0; qi < this.qDim; qi++) {
                // Add degree and leak rate/stability term to diagonal
                L[i * this.qDim + qi][i * this.qDim + qi] = degree + this.stabEps + this.leakRate;
            }
        }

        return isFiniteMatrix(L) ? L : identity(N);
    }

    /**
     * Dynamically adapts the sheaf's graph topology (edges and simplices) based on correlations.
     *
     */
    async adaptSheafTopology(adaptFreq = 100, stepCount = 0, addThresh = this.adaptation.addThresh, removeThresh = this.adaptation.removeThresh) {
        if (!this.ready || stepCount % adaptFreq !== 0) return;
        if (this.stalkNormHistory.length < this.stalkHistorySize / 2) {
            return;
        }

        try {
            this.correlationMatrix = await this.computeVertexCorrelationsFromHistory();
            if (!isFiniteMatrix(this.correlationMatrix)) {
                logger.warn('Sheaf: Non-finite correlation matrix; skipping adaptation.');
                return;
            }
            this.adaptEdges(this.correlationMatrix, addThresh, removeThresh);
            // Simplices adaptation might rely on higher-level metrics (H1 dimension, gestalt unity, inconsistency)
            // which are computed in QualiaCognitionLayer. For now, keep as a placeholder or simplified.
            this.adaptSimplices(this.correlationMatrix, this.adaptation.targetH1);

            await this.computeCorrelationMatrix(); // Recompute after structural changes
            this.laplacian = this.buildLaplacian();

            logger.info(`Sheaf adapted at step ${stepCount}. All dependent matrices rebuilt.`);

        } catch(e) {
            logger.error(`Sheaf.adaptSheafTopology: Failed: ${e.message}`, { stack: e.stack });
        }
    }

    /**
     * Adapts edges based on correlation thresholds.
     *
     */
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

            if (i !== -1 && j !== -1 && corrMatrix[i] && Number.isFinite(corrMatrix[i][j])) {
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
                    if (!corrMatrix[i] || !Number.isFinite(corrMatrix[i][j])) continue;
                    const corrVal = corrMatrix[i][j];
                    const u = this.graph.vertices[i];
                    const v = this.graph.vertices[j];
                    const edgeKey = [u, v].sort().join(',');

                    if (corrVal > addThreshold && !this.edgeSet.has(edgeKey)) {
                        const weight = clamp(corrVal * 0.5, 0.1, 1.0); // Simplified weight
                        this.addEdge(u, v, weight);
                        added++;
                    }
                }
            }
        }
    }

    /**
     * Adapts simplices (triangles, tetrahedra) based on coherence and higher-order metrics.
     * This is a simplified version; full logic might depend on `QualiaCognitionLayer` metrics.
     *
     */
    adaptSimplices(corrMatrix, targetH1 = 2.0) {
        const numV = this.graph.vertices.length;
        // Simplified adaptation: Add triangles for highly correlated nodes, remove for low.
        // For full H1, gestalt unity, inconsistency based adaptation, see QualiaCognitionLayer.
        const maxAddTri = 2;
        let addedTri = 0;
        for (let i = 0; i < numV && addedTri < maxAddTri; i++) {
            for (let j = i + 1; j < numV; j++) {
                for (let k = j + 1; k < numV; k++) {
                    if (!corrMatrix[i] || !corrMatrix[j] || !corrMatrix[k] ||
                        !Number.isFinite(corrMatrix[i][j]) || !Number.isFinite(corrMatrix[j][k]) || !Number.isFinite(corrMatrix[k][i])) continue;

                    const avgC = (corrMatrix[i][j] + corrMatrix[j][k] + corrMatrix[k][i]) / 3;
                    const u = this.graph.vertices[i], v = this.graph.vertices[j], w = this.graph.vertices[k];
                    if (avgC > 0.8 && this.isValidTriangle([u, v, w])) {
                        const tri = [u, v, w];
                        const key = tri.sort().join(',');
                        if (!this.simplicialComplex.triangles.some(t => t.sort().join(',') === key)) {
                            this.addTriangle(...tri);
                            addedTri++;
                            return;
                        }
                    }
                }
            }
        }
    }

    /**
     * Computes the current projection matrices for all edges based on Hebbian-like learning.
     *
     */
    async computeProjectionMatrices() {
        const newProjections = new Map();
        const learningRate = 0.01;
        const regularization = 0.002;

        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const s_u = this.stalks.get(u);
            const s_v = this.stalks.get(v);
            let P_uv = this.projectionMatrices.get(`${u}-${v}`) || identity(this.qDim);

            if (!isFiniteVector(s_u) || !isFiniteVector(s_v)) {
                newProjections.set(`${u}-${v}`, P_uv);
                newProjections.set(`${v}-${u}`, _transpose(P_uv));
                continue;
            }

            const projected_u = _matVecMul(P_uv, s_u);
            const error = vecSub(projected_u, s_v);

            const gradient = zeroMatrix(this.qDim, this.qDim);
            for (let i = 0; i < this.qDim; i++) {
                for (let j = 0; j < this.qDim; j++) {
                    gradient[i][j] = (error[i] || 0) * (s_u[j] || 0);
                }
            }

            const updated_P_uv = zeroMatrix(this.qDim, this.qDim);
            for (let i = 0; i < this.qDim; i++) {
                for (let j = 0; j < this.qDim; j++) {
                    const currentWeight = P_uv[i][j];
                    const gradUpdate = learningRate * gradient[i][j];
                    const regUpdate = regularization * (currentWeight - (i === j ? 1 : 0));
                    updated_P_uv[i][j] = clamp(currentWeight - gradUpdate - regUpdate, -2, 2);
                }
            }
            newProjections.set(`${u}-${v}`, updated_P_uv);
            newProjections.set(`${v}-${u}`, _transpose(updated_P_uv));
        }
        this.projectionMatrices = newProjections;
        return newProjections;
    }

    /**
     * Diffuses qualia states across the sheaf based on the Laplacian and external input.
     *
     */
    async diffuseQualia(state, qualiaInput) {
        if (!this.ready) return;

        const nV = this.graph.vertices.length;
        const N = nV * this.qDim;
        const s = this.getStalksAsVector();

        if (!isFiniteVector(s) || s.length !== N) {
            logger.error(`Sheaf.diffuseQualia: Invalid initial stalk vector 's'. Aborting.`);
            return;
        }

        // Apply perception filters if they exist (from original code, assuming initialization elsewhere if needed)
        // For condensation, this is kept as a potential future extension or simplified.
        if (!this.perceptionFilters) {
            this.perceptionFilters = new Map();
            for (const vertex of this.graph.vertices) {
                const filter = identity(this.qDim).map((row, i) =>
                    row.map((val, j) => val + (i === j ? 0 : (Math.random() - 0.5) * 0.1))
                );
                this.perceptionFilters.set(vertex, filter);
            }
        }

        const vertexInputs = new Float32Array(nV);
        for (let i = 0; i < nV; i++) {
            vertexInputs[i] = state[Math.min(i, state.length - 1)] || 0;
        }

        let f_s = vecZeros(N); // External force on stalks
        for (let i = 0; i < nV; i++) {
            const vertexInputScalar = vertexInputs[i];
            const forceVectorForThisVertex = new Float32Array(this.qDim).fill(this.alpha * vertexInputScalar);
            f_s.set(forceVectorForThisVertex, i * this.qDim);
        }

        const f_s_norm = norm2(f_s);
        if (f_s_norm > this.maxForceNorm) {
            f_s = safeVecScale(f_s, this.maxForceNorm / (f_s_norm > this.eps ? f_s_norm : this.eps));
        }

        const Lfull = this.laplacian;
        if (!isFiniteMatrix(Lfull) || Lfull.length !== N) {
            logger.warn('Sheaf.diffuseQualia: Invalid Laplacian. Skipping diffusion step.');
            return;
        }

        let eta = this.gamma / Math.max(1, this.maxEigApprox); // Time step for diffusion
        if (!isFiniteNumber(eta) || eta <= 0) {
            eta = 0.01;
        }

        const laplacianEffect = _matVecMul(Lfull, s);
        let sNext = vecSub(s, safeVecScale(laplacianEffect, eta));
        sNext = vecAdd(sNext, safeVecScale(f_s, eta));

        const noise = new Float32Array(N).map(() => (Math.random() - 0.5) * this.sigma * Math.sqrt(eta));
        sNext = vecAdd(sNext, noise);

        const clampedSNext = new Float32Array(sNext.map(v => clamp(v, -1, 1)));

        if (!isFiniteVector(clampedSNext)) {
            logger.error("Sheaf.diffuseQualia: Diffusion resulted in a non-finite state. Reverting to previous state.");
            return;
        }

        this._updateStalksAndWindow(clampedSNext, nV, qualiaInput);
        this._updateStateToQualiaProjection(state, qualiaInput);
    }

    /**
     * Performs one step of linear diffusion based on the Laplacian.
     *
     */
    simulateDiffusionStep(state, tDelta = 0.1) {
        const nV = this.graph.vertices.length;
        const N = nV * this.qDim;

        if (!isFiniteVector(state) || state.length !== N) {
            logger.error(`simulateDiffusionStep received an invalid state vector. Returning original vector.`);
            return state;
        }

        const Lfull = this.laplacian;
        if (!isFiniteMatrix(Lfull) || Lfull.length !== N) {
            logger.error(`simulateDiffusionStep cannot proceed with an invalid Laplacian. Returning original state.`);
            return state;
        }

        const eta = 0.01;
        const laplacianEffect = _matVecMul(Lfull, state);
        const sNext = vecSub(state, safeVecScale(laplacianEffect, eta));

        if (!isFiniteVector(sNext)) {
            logger.error('CRITICAL: simulateDiffusionStep produced a non-finite vector. Reverting to pre-simulation state.');
            return state;
        }

        return new Float32Array(sNext.map(v => clamp(v, -1, 1)));
    }


    /**
     * Computes harmonic state of the sheaf by averaging and normalizing stalk vectors.
     *
     */
    async computeHarmonicState() {
        const nV = this.graph.vertices.length;
        if (nV === 0 || this.stalks.size === 0 || this.qDim === 0) return vecZeros(1);

        let avgStalk = vecZeros(this.qDim);
        let count = 0;
        for (const stalk of this.stalks.values()) {
            if (isFiniteVector(stalk) && stalk.length === this.qDim) {
                avgStalk = vecAdd(avgStalk, stalk);
                count++;
            } else {
                logger.warn('computeHarmonicState: Invalid stalk found. Skipping.');
            }
        }

        if (count === 0) return vecZeros(this.qDim);
        avgStalk = safeVecScale(avgStalk, 1 / (count !== 0 ? count : this.eps));
        const norm = norm2(avgStalk);
        return safeVecScale(avgStalk, 1 / (norm > this.eps ? norm : this.eps));
    }


    /**
     * Builds boundary matrices (∂1, ∂2, ∂3) for the simplicial complex.
     *
     */
    async buildBoundaryMatrices() {
        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;
        const nT = this.simplicialComplex.triangles.length;
        const nTet = this.simplicialComplex.tetrahedra.length;

        if (nV === 0) {
            return {
                partial1: flattenMatrix(zeroMatrix(0,0)),
                partial2: flattenMatrix(zeroMatrix(0,0)),
                partial3: flattenMatrix(zeroMatrix(0,0))
            };
        }

        const vMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const eMapIndices = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        const tMapIndices = new Map(this.simplicialComplex.triangles.map((t, i) => [t.slice().sort().join(','), i]));

        const boundary1 = zeroMatrix(nE, nV);
        this.graph.edges.forEach((edge, eIdx) => {
            const [u, v] = edge;
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx === undefined || vIdx === undefined || eIdx >= nE || uIdx >= nV || vIdx >= nV) {
                logger.warn(`Sheaf.buildBoundaryMatrices: Invalid vertex or edge index for edge ${u}-${v}. Skipping.`);
                return;
            }
            boundary1[eIdx][uIdx] = 1;
            boundary1[eIdx][vIdx] = -1;
        });

        const partial2 = zeroMatrix(nT, nE);
        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (!Array.isArray(tri) || tri.length !== 3 || tIdx >= nT) return;
            const [u, v, w] = tri;
            const edges = [
                { key: [u, v].sort().join(','), sign: 1 },
                { key: [v, w].sort().join(','), sign: -1 },
                { key: [w, u].sort().join(','), sign: 1 }
            ];
            edges.forEach(({ key, sign }) => {
                const eIdx = eMapIndices.get(key);
                if (eIdx !== undefined && eIdx < nE) {
                    partial2[tIdx][eIdx] = sign;
                }
            });
        });

        const partial3 = zeroMatrix(nTet, nT);
        this.simplicialComplex.tetrahedra.forEach((tet, tetIdx) => {
            if (!Array.isArray(tet) || tet.length !== 4 || tetIdx >= nTet) return;
            const sortedTet = tet.slice().sort();
            for (let i = 0; i < 4; i++) {
                const face = sortedTet.filter((_, idx) => idx !== i).sort();
                const tIdx = tMapIndices.get(face.join(','));
                if (tIdx !== undefined && tIdx < nT) {
                    partial3[tetIdx][tIdx] = (i % 2 === 0 ? 1 : -1);
                }
            }
        });

        return {
            partial1: flattenMatrix(boundary1),
            partial2: flattenMatrix(partial2),
            partial3: flattenMatrix(partial3)
        };
    }

    /**
     * Builds the combinatorial graph Laplacian matrix.
     *
     */
    _buildGraphLaplacian() {
        const nV = this.graph.vertices.length;
        if (nV === 0) return zeroMatrix(0, 0);
        const adj = zeroMatrix(nV, nV);
        const degree = new Float32Array(nV).fill(0);
        const vMap = new Map(this.graph.vertices.map((v, i) => [v, i]));

        this.graph.edges.forEach(([u, v]) => {
            const i = vMap.get(u);
            const j = vMap.get(v);
            if (i !== undefined && j !== undefined) {
                adj[i][j] = 1;
                adj[j][i] = 1;
                degree[i]++;
                degree[j]++;
            }
        });

        const L = zeroMatrix(nV, nV);
        for (let i = 0; i < nV; i++) {
            L[i][i] = degree[i];
            for (let j = 0; j < nV; j++) {
                if (i !== j) {
                    L[i][j] = -adj[i][j];
                }
            }
        }
        return L;
    }

    /**
     * Computes the H1 Betti number (number of 'holes' or cycles) using a Union-Find algorithm.
     *
     */
    computeH1Dimension() {
        const vertices = this.graph.vertices;
        const edges = this.graph.edges;
        const n = vertices.length;
        const m = edges.length;

        if (n === 0) return 0;

        const parent = Array(n).fill().map((_, i) => i);
        const rank = Array(n).fill(0);

        const find = (x) => {
            if (parent[x] !== x) return parent[x] = find(parent[x]);
            return parent[x];
        };

        const union = (x, y) => {
            const px = find(x);
            const py = find(y);
            if (px === py) return false; // Already in the same component
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else {
                parent[py] = px;
                if (rank[px] === rank[py]) rank[px]++;
            }
            return true;
        };

        const vMap = new Map(vertices.map((v, i) => [v, i]));
        let connectedComponents = n;

        edges.forEach(edge => {
            const u = edge[0];
            const v = edge[1];
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx !== undefined && vIdx !== undefined && uIdx < n && vIdx < n) {
                if (union(uIdx, vIdx)) {
                    connectedComponents--;
                }
            } else {
                logger.warn(`_computeBetti1UnionFind: Invalid vertex index or edge format for edge starting with ${u}-${v}. Skipping.`, { edge, uIdx, vIdx, n });
            }
        });

        const calculatedH1 = Math.max(0, m - n + connectedComponents);
        return calculatedH1;
    }

    /**
     * Computes eigenvalues and eigenvectors of a matrix. Prefers custom Jacobi, falls back if needed.
     * Caches results for performance.
     *
     */
    async _spectralDecomp(matrix) {
        if (!isFiniteMatrix(matrix) || matrix.length === 0 || matrix.length !== (matrix[0]?.length || 0)) {
            const n = matrix?.length || 1;
            logger.warn('_spectralDecomp: Invalid matrix. Returning safe default.');
            return { eigenvalues: Array(n).fill(1), eigenvectors: identity(n) };
        }

        try {
            const result = this._jacobiEigenvalueDecomposition(matrix);

            if (result && isFiniteVector(result.eigenvalues) && isFiniteMatrix(result.eigenvectors)) {
                this.lastGoodEigenResult = result;
                return result;
            }
            throw new Error('Jacobi method returned invalid data or failed.');
        } catch (err) {
            logger.warn(`_spectralDecomp failed: ${err.message}. Using last known good result or fallback.`);
            const n = matrix.length;
            if (this.lastGoodEigenResult && this.lastGoodEigenResult.eigenvalues.length === n) {
                return this.lastGoodEigenResult;
            }
            return { eigenvalues: Array(n).fill(1), eigenvectors: identity(n) };
        }
    }


    /**
     * Saves the current state of the QualiaSheafBase instance.
     *
     */
    saveState() {
        return {
            graph: this.graph,
            simplicialComplex: this.simplicialComplex,
            stalks: Array.from(this.stalks.entries()),
            projectionMatrices: Array.from(this.projectionMatrices.entries()),
            adaptation: this.adaptation,
            qDim: this.qDim,
            stateDim: this.stateDim,
            alpha: this.alpha, beta: this.beta, gamma: this.gamma, sigma: this.sigma, eps: this.eps, stabEps: this.stabEps,
            resonanceOmega: this.resonanceOmega, resonanceEps: this.resonanceEps,
            stalkHistorySize: this.stalkHistorySize,
            windowSize: this.windowSize,
            stalkHistory: this.stalkHistory.getAll(),
            stalkNormHistory: this.stalkNormHistory.getAll(),
            windowedStates: this.windowedStates.getAll(),
        };
    }

    /**
     * Loads a previously saved state into the QualiaSheafBase instance.
     *
     */
    loadState(state) {
        if (!state) {
            logger.warn('QualiaSheafBase.loadState: No state provided to load.');
            return;
        }

        try {
            this.graph = state.graph || this.graph;
            this.simplicialComplex = state.simplicialComplex || this.simplicialComplex;
            this.edgeSet = new Set(this.graph.edges.map(e => e.slice(0, 2).sort().join(',')));

            this.stalks = new Map(state.stalks || []);
            this.projectionMatrices = new Map(state.projectionMatrices || []);

            this.adaptation = state.adaptation || this.adaptation;
            this.qDim = state.qDim ?? this.qDim;
            this.stateDim = state.stateDim ?? this.stateDim;
            this.alpha = state.alpha ?? this.alpha;
            this.beta = state.beta ?? this.beta;
            this.gamma = state.gamma ?? this.gamma;
            this.sigma = state.sigma ?? this.sigma;
            this.eps = state.eps ?? this.eps;
            this.stabEps = state.stabEps ?? this.stabEps;
            this.resonanceOmega = state.resonanceOmega ?? this.resonanceOmega;
            this.resonanceEps = state.resonanceEps ?? this.resonanceEps;
            this.stalkHistorySize = state.stalkHistorySize ?? this.stalkHistorySize;
            this.windowSize = state.windowSize ?? this.windowSize;

            this.stalkHistory = new CircularBuffer(this.stalkHistorySize);
            if (Array.isArray(state.stalkHistory)) state.stalkHistory.forEach(item => this.stalkHistory.push(item));

            this.stalkNormHistory = new CircularBuffer(this.stalkHistorySize);
            if (Array.isArray(state.stalkNormHistory)) state.stalkNormHistory.forEach(item => this.stalkNormHistory.push(item));

            this.windowedStates = new CircularBuffer(this.windowSize);
            if (Array.isArray(state.windowedStates)) state.windowedStates.forEach(item => this.windowedStates.push(item));

            // Rebuild matrices after loading structural data
            this.correlationMatrix = null;
            this.adjacencyMatrix = null;
            this.laplacian = null;
            this.computeCorrelationMatrix().then(() => {
                this.laplacian = this.buildLaplacian();
                this.ready = true;
                logger.info('QualiaSheafBase.loadState: State loaded and matrices rebuilt successfully.');
            });

        } catch (e) {
            logger.error(`QualiaSheafBase.loadState: Error loading state: ${e.message}`, { stack: e.stack });
            this.ready = false;
        }
    }

    /**
     * Visualizes the current activity of the sheaf using THREE.js.
     *
     */
    visualizeActivity(scene, camera, renderer, cognitionLayer = null) {
        if (!THREE) {
            logger.error("Sheaf.visualizeActivity: THREE.js is not available.");
            return;
        }
        if (!scene || !camera || !renderer) {
            logger.error("Sheaf.visualizeActivity: THREE.js scene, camera, or renderer not provided.");
            return;
        }

        const stalkGroup = new THREE.Group();
        const vertexMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);

        this.stalks.forEach((stalk, v) => {
            const norm = isFiniteVector(stalk) ? norm2(stalk) : 0;
            const rhythmicallyAware = cognitionLayer?.rhythmicallyAware ?? false;
            const phase = Array.isArray(cognitionLayer?.floquetPD?.phases) && cognitionLayer.floquetPD.phases.length > 0 ? cognitionLayer.floquetPD.phases[cognitionLayer.floquetPD.phases.length - 1] : 0;

            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(0.2, 0.5 + norm * 0.5, 1.0),
                emissive: new THREE.Color(0, norm * 0.2 * (rhythmicallyAware ? 1.5 * Math.cos(phase) : 1), 0.2)
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
            const rhythmicallyAware = cognitionLayer?.rhythmicallyAware ?? false;
            const phase = Array.isArray(cognitionLayer?.floquetPD?.phases) && cognitionLayer.floquetPD.phases.length > 0 ? cognitionLayer.floquetPD.phases[cognitionLayer.floquetPD.phases.length - 1] : 0;

            const cupProductIntensity = cognitionLayer?.cup_product_intensity ?? 0;
            const opacity = clamp((weight || 0) * cupProductIntensity * (rhythmicallyAware ? Math.cos(phase) : 1), 0.2, 0.8);

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

        if (Array.isArray(cognitionLayer?.floquetPD?.births) && cognitionLayer.floquetPD.births.length > 0) {
            const barcodeGroup = new THREE.Group();
            cognitionLayer.floquetPD.births.forEach((birth, idx) => {
                const t = birth.time || idx;
                const death_t = (Array.isArray(cognitionLayer.floquetPD.deaths) && cognitionLayer.floquetPD.deaths[idx]?.time !== undefined) ? cognitionLayer.floquetPD.deaths[idx].time : t + 1;
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
}

/**
 * QualiaCognitionLayer: Interprets the state of QualiaSheafBase to derive
 * higher-order cognitive properties and emergent awareness.
 * This class applies the "theorems" from the original hierarchy.
 */
export class QualiaCognitionLayer {
    constructor(baseSheaf, config = {}) {
        if (!(baseSheaf instanceof QualiaSheafBase)) {
            throw new Error("QualiaCognitionLayer must be initialized with an instance of QualiaSheafBase.");
        }
        this.baseSheaf = baseSheaf;
        this.logger = logger; // Use the global logger

        // Parameters from advanced theorems
        this.maxIter = config.maxIter || 50;
        this.fixedPointEps = config.fixedPointEps || 1e-6;
        this.tau = config.tau || 2.5; // Threshold for self-awareness
        this.equalizerEps = config.equalizerEps || 1e-5;
        this.flowBufferSize = config.flowBufferSize || 50;
        this.delta = config.delta || 0.1; // Persistence diagram threshold
        this.tau_persist = config.tau_persist || 3.5; // Threshold for diachronic awareness
        this.omega = config?.omega || 8; // Floquet period
        this.theta_k = config?.theta_k || [4, 6, 8]; // Floquet phases
        this.tau_floq = config?.tau_floq || 4.0; // Threshold for rhythmic awareness

        // Internal states and histories for cognitive processes
        this.phiBase = config.phiBase ?? 0.2; // Base for integrated information
        this.gestaltBase = config.gestaltBase ?? 0.6; // Base for gestalt unity

        this.phiHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
        this.gestaltHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
        this.inconsistencyHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
        this.cochainHistory = new CircularBuffer(20); // For RecursiveTopologicalSheaf fixed points
        this.flowHistory = new CircularBuffer(this.flowBufferSize); // For PersistentAdjunctionSheaf
        this.persistenceDiagram = { births: [], deaths: [] }; // For PersistentAdjunctionSheaf
        this.floquetPD = { births: [], phases: [], deaths: [] }; // For FloquetPersistentSheaf

        // Derived metrics and awareness flags
        this.phi = this.phiBase;
        this.h1Dimension = 0;
        this.gestaltUnity = this.gestaltBase;
        this.stability = 0.6;
        this.inconsistency = 0;
        this.feel_F = 0;
        this.intentionality_F = 0;
        this.cup_product_intensity = 0;
        this.structural_sensitivity = 0;
        this.coherence = 0; // Placeholder, derived later
        this.overallCoherence = 0;

        // Awareness flags
        this.selfAware = false;
        this.hierarchicallyAware = false;
        this.diachronicallyAware = false;
        this.rhythmicallyAware = false;
        this.emergentAware = false; // Overall awareness

        this.qInput = null; // Store last qualia input for computations
        this.currentCochain = []; // Store C0 cochains for some computations

        this.R_star = this._buildRecursiveGluing(); // Initialize R_star operator
        logger.info(`QualiaCognitionLayer constructed.`);
    }

    /**
     * Compute metric gain (simple norm of next state)
     *
     */
    computeMetricGain(sNext) {
        if (!this.baseSheaf._isValidStateVec(sNext)) {
            logger.warn('QualiaCognitionLayer.computeMetricGain: Invalid sNext vector. Returning 0.');
            return 0;
        }
        const gain = norm2(sNext);
        logger.debug('QualiaCognitionLayer.computeMetricGain: Computed gain', { gain });
        return Number.isFinite(gain) ? gain : 0;
    }

    /**
     * Computes the gluing inconsistency across the sheaf's edges.
     *
     */
    async computeGluingInconsistency() {
        try {
            const edges = this.baseSheaf.graph?.edges;
            if (!edges || edges.length === 0) {
                this.inconsistency = 0;
                return 0;
            }

            let totalInconsistency = 0;
            let edgeCount = 0;

            for (const [u, v] of edges) {
                const s_u = this.baseSheaf.stalks.get(u);
                const s_v = this.baseSheaf.stalks.get(v);
                const P_uv = this.baseSheaf.projectionMatrices.get(`${u}-${v}`);

                if (!this.baseSheaf._isValidStateVec(s_u) || !this.baseSheaf._isValidStateVec(s_v) || !isFiniteMatrix(P_uv)) {
                    continue;
                }

                try {
                    const s_u_projected = await runWorkerTaskWithRetry('matVecMul', {
                        matrix: flattenMatrix(P_uv),
                        vector: s_u,
                        expectedDim: this.baseSheaf.qDim
                    });

                    if (this.baseSheaf._isValidStateVec(s_u_projected)) {
                        const inconsistency = norm2(vecSub(s_u_projected, s_v));
                        if (isFiniteNumber(inconsistency)) {
                            totalInconsistency += inconsistency;
                            edgeCount++;
                        }
                    }
                } catch (e) {
                    this.logger?.error(`QualiaCognitionLayer.computeGluingInconsistency: Error processing edge ${u}-${v}.`, { error: e.message });
                }
            }

            this.inconsistency = edgeCount > 0 ? clamp(totalInconsistency / edgeCount, 0, 1) : 0;
            this.inconsistencyHistory.push(new Float32Array([this.inconsistency]));
            return this.inconsistency;
        } catch (err) {
            this.logger?.error(`QualiaCognitionLayer.computeGluingInconsistency: unexpected error: ${err.message}`);
            this.inconsistency = 0;
            return 0;
        }
    }

    /**
     * Computes the Gestalt Unity of the sheaf from the coherence of windowed states.
     * Uses GPU.js if available, falls back to CPU.
     *
     */
    async computeGestaltUnity() {
        let validStates = this.baseSheaf.windowedStates?.getAll?.()?.filter(s => this.baseSheaf._isValidStateVec(s) && s.length > 0) || [];

        if (validStates.length < 2) {
            this.gestaltUnity = 0;
            return 0;
        }

        let totalSimilarity = 0;
        let count = 0;
        const totalStateDim = validStates[0].length;

        // Try GPU.js
        if (this.baseSheaf.gpu && this.baseSheaf.gestaltKernel && totalStateDim > 0) {
            try {
                this.baseSheaf.gestaltKernel.setConstants({
                    eps_const: this.baseSheaf.eps || 1e-6,
                    state_dim_const: totalStateDim,
                    num_states_const: validStates.length
                });

                const flattenedStates = new Float32Array(validStates.flat());
                if (!this.baseSheaf._isValidStateVec(flattenedStates)) {
                    throw new Error('Flattened states for GPU.js are non-finite.');
                }

                const result = this.baseSheaf.gestaltKernel(flattenedStates);
                totalSimilarity = Number.isFinite(result[0]) ? result[0] : 0;
            } catch (e) {
                logger.warn('QualiaCognitionLayer.computeGestaltUnity: GPU.js execution failed; falling back to CPU.', { error: e.message });
                this.baseSheaf.gpu = null;
                this.baseSheaf.gestaltKernel = null;
                totalSimilarity = 0;
            }
        }

        // Fallback to CPU calculation if GPU failed or not available
        if (totalSimilarity === 0) {
            for (let i = 0; i < validStates.length; i++) {
                for (let j = i + 1; j < validStates.length; j++) {
                    const n1 = norm2(validStates[i]);
                    const n2 = norm2(validStates[j]);
                    if (n1 > this.baseSheaf.eps && n2 > this.baseSheaf.eps) {
                        const similarity = Math.abs(dot(validStates[i], validStates[j]) / (n1 * n2));
                        if (Number.isFinite(similarity)) {
                            totalSimilarity += similarity;
                            count++;
                        }
                    }
                }
            }
        } else { // If GPU calculated, we need to determine count for normalization
             count = (validStates.length * (validStates.length - 1)) / 2;
        }

        this.gestaltUnity = count > 0 ? clamp(totalSimilarity / count, 0, 1) : 0;
        this.gestaltHistory.push(new Float32Array([this.gestaltUnity]));
        return this.gestaltUnity;
    }

    /**
     * Computes the logarithmic determinant of a matrix, falling back to custom LU if numeric.js fails.
     *
     */
    _computeLogDet(A) {
        const n = A?.length || 0;
        if (n === 0 || !isFiniteMatrix(A)) {
            logger.warn('QualiaCognitionLayer._computeLogDet: Invalid or non-finite matrix. Returning 0.', { matrixRows: n });
            return 0;
        }

        // Custom LU decomposition
        const L = zeroMatrix(n, n);
        const U = zeroMatrix(n, n);
        for (let i = 0; i < n; i++) {
            L[i][i] = 1;
        }

        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                let sum = A[i]?.[j] || 0;
                for (let k = 0; k < i; k++) {
                    sum -= (L[i]?.[k] || 0) * (U[k]?.[j] || 0);
                }
                U[i][j] = Number.isFinite(sum) ? sum : 0;
            }
            for (let j = i + 1; j < n; j++) {
                let sum = A[j]?.[i] || 0;
                for (let k = 0; k < i; k++) {
                    sum -= (L[j]?.[k] || 0) * (U[k]?.[i] || 0);
                }
                const U_ii = U[i]?.[i] || 0;
                if (!Number.isFinite(U_ii) || Math.abs(U_ii) < (this.baseSheaf.eps || 1e-6) * 10) {
                    logger.warn(`QualiaCognitionLayer._computeLogDet: Division by near-zero diagonal element in U at i=${i}. Returning 0.`);
                    return 0;
                }
                L[j][i] = Number.isFinite(sum / U_ii) ? sum / U_ii : 0;
            }
        }

        let logDet = 0;
        for (let i = 0; i < n; i++) {
            const U_ii = U[i]?.[i] || 0;
            if (!Number.isFinite(U_ii) || Math.abs(U_ii) <= (this.baseSheaf.eps || 1e-6) * 10) {
                logger.warn(`QualiaCognitionLayer._computeLogDet: Non-finite or near-zero diagonal element in U at i=${i}. Returning 0.`);
                return 0;
            }
            logDet += Math.log(Math.abs(U_ii));
        }

        if (!Number.isFinite(logDet)) {
            logger.warn('QualiaCognitionLayer._computeLogDet: Non-finite logDet from custom LU. Returning 0.');
            return 0;
        }
        return logDet;
    }

    /**
     * Computes directional mutual information (Transfer Entropy) between two halves of a state vector.
     *
     */
    async _computeDirectionalMI(states) {
        if (!Array.isArray(states) || states.length < 2 || !this.baseSheaf._isValidStateVec(states[0]) || states[0].length < 2) {
            logger.warn('QualiaCognitionLayer._computeDirectionalMI: Insufficient or invalid states. Returning 0.');
            return 0;
        }

        const n_dim = states[0].length;
        const n_half = Math.floor(n_dim / 2);

        const Y_future = [];
        const Y_past = [];
        const X_past = [];
        const YX_past = [];

        for (let i = 0; i < states.length - 1; i++) {
            const s_t = states[i];
            const s_t1 = states[i + 1];

            if (!this.baseSheaf._isValidStateVec(s_t) || !this.baseSheaf._isValidStateVec(s_t1) || s_t.length !== n_dim || s_t1.length !== n_dim) {
                logger.warn(`QualiaCognitionLayer._computeDirectionalMI: Non-finite or mismatched state vector at index ${i}. Skipping sample.`);
                continue;
            }

            const x_t = s_t.slice(0, n_half);
            const y_t = s_t.slice(n_half);
            const y_t1 = s_t1.slice(n_half);

            if (!this.baseSheaf._isValidStateVec(x_t) || !this.baseSheaf._isValidStateVec(y_t) || !this.baseSheaf._isValidStateVec(y_t1)) {
                logger.warn(`QualiaCognitionLayer._computeDirectionalMI: Non-finite slices at index ${i}. Skipping sample.`);
                continue;
            }

            Y_future.push(y_t1);
            Y_past.push(y_t);
            X_past.push(x_t);
            YX_past.push([...y_t, ...x_t]);
        }

        if (Y_future.length < 2) {
            logger.warn('QualiaCognitionLayer._computeDirectionalMI: Insufficient valid time-lagged samples. Falling back to variance-based estimate.');
            try {
                const mean = vecZeros(n_dim);
                states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
                let sumVar = 0;
                states.forEach(state => {
                    const centered = vecSub(state, mean);
                    centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
                });
                const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.baseSheaf.eps || 1e-6);
                return clamp(miEstimate, 0, 5);
            } catch (e) {
                logger.warn('QualiaCognitionLayer._computeDirectionalMI: Fallback MI failed. Returning 0.', { error: e.message });
                return 0;
            }
        }

        try {
            const mi_full_input = Y_future.map((yf, i) => {
                const combined = [...yf, ...(YX_past[i] || vecZeros(n_dim))];
                return this.baseSheaf._isValidStateVec(combined) ? combined : null;
            }).filter(Boolean);
            const mi_partial_input = Y_future.map((yf, i) => {
                const combined = [...yf, ...(Y_past[i] || vecZeros(n_half))];
                return this.baseSheaf._isValidStateVec(combined) ? combined : null;
            }).filter(Boolean);

            if (mi_full_input.length < 2 || mi_partial_input.length < 2) {
                logger.warn('QualiaCognitionLayer._computeDirectionalMI: Insufficient valid input for KSG MI. Falling back to variance-based estimate.');
                // Fallback handled by previous block, re-implementing for robustness
                const mean = vecZeros(n_dim);
                states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
                let sumVar = 0;
                states.forEach(state => {
                    const centered = vecSub(state, mean);
                    centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
                });
                const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.baseSheaf.eps || 1e-6);
                return clamp(miEstimate, 0, 5);
            }

            const [mi_full_raw, mi_partial_raw] = await Promise.all([
                runWorkerTaskWithRetry('ksg_mi', { states: mi_full_input, k: Math.min(3, mi_full_input.length - 1) }, 15000),
                runWorkerTaskWithRetry('ksg_mi', { states: mi_partial_input, k: Math.min(3, mi_partial_input.length - 1) }, 15000)
            ]);

            const mi_full = Number.isFinite(mi_full_raw) ? mi_full_raw : 0;
            const mi_partial = Number.isFinite(mi_partial_raw) ? mi_partial_raw : 0;

            const transferEntropy = mi_full - mi_partial;
            return clamp(transferEntropy, 0, 10);
        } catch (e) {
            logger.warn(`QualiaCognitionLayer._computeDirectionalMI: KSG computation failed: ${e.message}. Falling back to variance-based estimate.`, { stack: e.stack });
            // Fallback to variance-based estimate in case of KSG failure
            const mean = vecZeros(n_dim);
            states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
            let sumVar = 0;
            states.forEach(state => {
                const centered = vecSub(state, mean);
                centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
            });
            const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.baseSheaf.eps || 1e-6);
            return clamp(miEstimate, 0, 5);
        }
    }

    /**
     * Computes integrated information (Phi) and related "feel" and "intentionality" metrics.
     *
     */
    async computeIntegratedInformation() {
        try {
            const stateSource = this.baseSheaf.stalkNormHistory || this.baseSheaf.windowedStates;
            const allStates = stateSource.getAll();
            if (allStates.length < 10) {
                logger.info('QualiaCognitionLayer.computeIntegratedInformation: Too few states; delaying computation.', { stateCount: allStates.length });
                this.integrated_information = 0;
                this.phi = 0.001;
                return { integrated_information: 0, phi: 0.001, feelIntensity: 0.001, intentionality: 0.001 };
            }

            const vertexCount = this.baseSheaf.graph?.vertices?.length || 8;
            const stateDim = allStates.length > 0 ? allStates[0].length : 0;
            const validStates = allStates.filter(item => this.baseSheaf._isValidStateVec(item) && item.length === stateDim);

            if (validStates.length < Math.max(vertexCount, 10)) {
                 logger.warn('QualiaCognitionLayer.computeIntegratedInformation: Insufficient valid states after filtering. Returning defaults.');
                 this.integrated_information = 0;
                 this.phi = 0.001;
                 return { integrated_information: 0, phi: 0.001, feelIntensity: 0.001, intentionality: 0.001 };
            }

            const n_dim = validStates[0].length;
            const num_samples = validStates.length;

            let MI = 0;
            let covMatrix = null;

            const tfVersion = tf?.version?.core || '0.0.0';
            const useTF = tf && parseFloat(tfVersion) >= 2.0 && tf.linalg?.determinant;

            if (useTF) {
                try {
                    const statesArray = validStates.map(s => Array.from(s));
                    if (!statesArray.every(this.baseSheaf._isValidStateVec)) throw new Error('Non-finite states in TF input.');
                    const statesTensor = tf.tensor2d(statesArray);
                    const regularizer = tf.eye(n_dim).mul(this.baseSheaf.eps * 100);
                    const rawCovMatrix = tf.matMul(statesTensor.transpose(), statesTensor).div(Math.max(1, num_samples - 1));
                    covMatrix = rawCovMatrix.add(regularizer);

                    let logDet;
                    try {
                        const L = tf.linalg.cholesky(covMatrix);
                        logDet = tf.sum(L.log().mul(2)).dataSync()[0];
                    } catch (e) {
                        logDet = tf.linalg.logMatrixDeterminant(covMatrix).logDeterminant.dataSync()[0];
                    }
                    if (Number.isFinite(logDet)) MI = 0.1 * Math.abs(logDet) + this.baseSheaf.eps;
                    tf.dispose([statesTensor, rawCovMatrix, covMatrix]);
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer.computeIntegratedInformation: TF path failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0) {
                try {
                    covMatrix = await runWorkerTaskWithRetry('covarianceMatrix', { states: validStates, dim: n_dim, eps: this.baseSheaf.eps }, 5000);
                    if (!isFiniteMatrix(covMatrix)) {
                        logger.warn('QualiaCognitionLayer.computeIntegratedInformation: Non-finite covariance. Manual CPU fallback.');
                        const mean = vecZeros(n_dim);
                        validStates.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / num_samples));
                        covMatrix = zeroMatrix(n_dim, n_dim);
                        validStates.forEach(state => {
                            const centered = vecSub(state, mean);
                            for (let i = 0; i < n_dim; i++) for (let j = i; j < n_dim; j++) {
                                covMatrix[i][j] += (centered[i] || 0) * (centered[j] || 0) / Math.max(1, num_samples - 1);
                                if (i !== j) covMatrix[j][i] = covMatrix[i][j];
                            }
                        });
                    }

                    if (isFiniteMatrix(covMatrix)) {
                        const regularizedCovMatrix = covMatrix.map((row, i) => row.map((val, j) => i === j ? val + this.baseSheaf.eps * 100 : val));
                        if (isFiniteMatrix(regularizedCovMatrix)) {
                            const logDet = this._computeLogDet(regularizedCovMatrix);
                            if (Number.isFinite(logDet)) MI = 0.1 * Math.abs(logDet) + this.baseSheaf.eps;
                        }
                    }
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer.computeIntegratedInformation: CPU path failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0 && validStates.length >= 2 && this.baseSheaf._isValidStateVec(validStates[0])) {
                try {
                    const ksgMI = await runWorkerTaskWithRetry('ksg_mi', { states: validStates, k: Math.min(3, validStates.length - 1) }, 20000);
                    MI = Number.isFinite(ksgMI) ? ksgMI : 0;
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer.computeIntegratedInformation: KSG fallback failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0 && validStates.length >= 2) {
                logger.info('QualiaCognitionLayer.computeIntegratedInformation: Partial MI with limited states.', { validStatesLength: validStates.length });
                try {
                    const mean = vecZeros(n_dim);
                    validStates.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / num_samples));
                    let sumVar = 0;
                    validStates.forEach(state => {
                        const centered = vecSub(state, mean);
                        centered.forEach(v => sumVar += v * v / Math.max(1, num_samples - 1));
                    });
                    MI = 0.05 * Math.log(1 + sumVar / n_dim) + this.baseSheaf.eps;
                    MI = clamp(MI, 0, 5);
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer.computeIntegratedInformation: Partial MI failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            this.integrated_information = clamp(MI, 0, 10);

            const safeFloquetBirths = Array.isArray(this.floquetPD?.births) ? this.floquetPD.births : [];
            const betaFloq = await this._supFloqBetti(this.floquetPD);
            const avgBirthTime = safeFloquetBirths.reduce((sum, b) => sum + (b.time || 0), 0) / Math.max(1, safeFloquetBirths.length);
            const persistenceBoost = 0.05 * betaFloq * Math.log(1 + (this.baseSheaf.stalkHistory?.length || 1) / Math.max(1, avgBirthTime + 1));

            const phiRaw = (Math.log(1 + Math.abs(MI)) + persistenceBoost) *
                           (this.stability || 1) * (this.gestaltUnity || 1) *
                           Math.exp(-(this.inconsistency || 0)) * (1 + 0.05 * (this.h1Dimension || 0));
            this.phi = clamp(phiRaw, 0.001, 100);
            this.phiHistory.push(new Float32Array([this.phi]));

            this.feel_F = clamp((MI + 0.02 * betaFloq) * (this.stability || 1) * Math.exp(-(this.inconsistency || 0)), 0.001, 10);

            const MI_dir = await this._computeDirectionalMI(validStates);
            this.intentionality_F = clamp((MI_dir + 0.01 * betaFloq) * (this.stability || 1) * Math.exp(-(this.inconsistency || 0)), 0.001, 10);

            return {
                integrated_information: this.integrated_information,
                phi: this.phi,
                feelIntensity: this.feel_F,
                intentionality: this.intentionality_F
            };
        } catch (err) {
            logger.error('QualiaCognitionLayer.computeIntegratedInformation: Computation error.', {
                error: err.message,
                stack: err.stack
            });
            this.integrated_information = 0;
            this.phi = 0.001;
            this.feel_F = 0.001;
            this.intentionality_F = 0.001;
            return {
                integrated_information: 0,
                phi: 0.001,
                feelIntensity: 0.001,
                intentionality: 0.001
            };
        }
    }

    /**
     * Computes cochains (0-cochains, 1-cochains, 2-cochains) for the sheaf.
     *
     */
    async _computeCochains(qualiaState) {
        try {
            this.logger?.debug('QualiaCognitionLayer._computeCochains: Starting', {
                vertices: this.baseSheaf.graph?.vertices?.length,
                edges: this.baseSheaf.graph?.edges?.length,
                qDim: this.baseSheaf.qDim
            });

            if (!this.baseSheaf.ready || this.baseSheaf.stalks.size === 0 || !this.baseSheaf.graph?.vertices?.length || !this.baseSheaf.graph?.edges || !Number.isFinite(this.baseSheaf.qDim) || this.baseSheaf.qDim <= 0) {
                this.logger?.warn('QualiaCognitionLayer._computeCochains: Invalid state, stalks, graph, or qDim', {
                    ready: this.baseSheaf.ready,
                    stalksSize: this.baseSheaf.stalks.size,
                    vertices: this.baseSheaf.graph?.vertices?.length,
                    edges: this.baseSheaf.graph?.edges?.length,
                    qDim: this.baseSheaf.qDim
                });
                const fallbackC1 = new Map();
                if (this.baseSheaf.graph?.vertices?.length >= 2) {
                    const [u, v] = this.baseSheaf.graph.vertices.slice(0, 2);
                    fallbackC1.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim || 7));
                }
                this.currentCochain = [];
                return { C0: [], C1: fallbackC1, C2: new Map() };
            }

            if (!this.baseSheaf._isValidStateVec(qualiaState) && qualiaState.length !== 20) {
                this.logger?.warn(`QualiaCognitionLayer._computeCochains: Invalid qualiaState vector. Using zeros.`, { length: qualiaState?.length });
                qualiaState = vecZeros(this.baseSheaf.qDim);
            }

            const C0 = [];
            for (const [vertex, stalk] of this.baseSheaf.stalks) {
                if (!this.baseSheaf._isValidStateVec(stalk)) {
                    this.logger?.warn(`QualiaCognitionLayer._computeCochains: Invalid stalk for ${vertex}. Resetting to zeros.`);
                    this.baseSheaf.stalks.set(vertex, vecZeros(this.baseSheaf.qDim));
                } else {
                    C0.push(stalk);
                }
            }
            if (C0.length === 0) {
                this.logger?.warn('QualiaCognitionLayer._computeCochains: No valid C0 vectors');
            }

            const C1 = new Map();
            for (const edge of this.baseSheaf.graph.edges) {
                const [u, v] = edge.slice(0, 2);
                let s_u = this.baseSheaf.stalks.get(u) || vecZeros(this.baseSheaf.qDim);
                let s_v = this.baseSheaf.stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                let P_vu = this.baseSheaf.projectionMatrices.get(`${v}-${u}`) || identity(this.baseSheaf.qDim);

                if (!this.baseSheaf._isValidStateVec(s_u)) { s_u = vecZeros(this.baseSheaf.qDim); this.baseSheaf.stalks.set(u, s_u); }
                if (!this.baseSheaf._isValidStateVec(s_v)) { s_v = vecZeros(this.baseSheaf.qDim); this.baseSheaf.stalks.set(v, s_v); }
                if (!isFiniteMatrix(P_vu) || P_vu.length !== this.baseSheaf.qDim || (P_vu[0]?.length || 0) !== this.baseSheaf.qDim) {
                    this.logger?.warn(`QualiaCognitionLayer._computeCochains: Invalid P_vu for edge ${v}-${u}. Using identity.`);
                    P_vu = identity(this.baseSheaf.qDim);
                    this.baseSheaf.projectionMatrices.set(`${v}-${u}`, P_vu);
                }

                let projected_u;
                try {
                    projected_u = _matVecMul(P_vu, s_u);
                    if (!this.baseSheaf._isValidStateVec(projected_u)) {
                        this.logger?.warn(`QualiaCognitionLayer._computeCochains: Invalid projected_u for edge ${v}-${u}. Using zeros.`);
                        projected_u = vecZeros(this.baseSheaf.qDim);
                    }
                } catch {
                    projected_u = vecZeros(this.baseSheaf.qDim);
                }

                const difference = vecSub(s_v, projected_u);
                C1.set([u, v].sort().join(','), this.baseSheaf._isValidStateVec(difference) ? difference : vecZeros(this.baseSheaf.qDim));
            }

            const C2 = new Map();
            const triangles = this.baseSheaf.simplicialComplex?.triangles || [];
            for (const tri of triangles) {
                if (!this.baseSheaf.isValidTriangle(tri)) continue;
                const [u, v, w] = tri;

                const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.baseSheaf.qDim);
                const c_vw = C1.get([v, w].sort().join(',')) || vecZeros(this.baseSheaf.qDim);
                const c_wu = C1.get([w, u].sort().join(',')) || vecZeros(this.baseSheaf.qDim);

                let curl = vecAdd(vecAdd(c_uv, c_vw), c_wu);
                if (!this.baseSheaf._isValidStateVec(curl)) continue;
                C2.set([u, v, w].sort().join(','), curl);
            }

            if (C1.size === 0 && this.baseSheaf.graph.edges.length > 0) {
                const [u, v] = this.baseSheaf.graph.edges[0].slice(0, 2);
                C1.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
            }

            this.currentCochain = C0; // C0 cochains for self-awareness
            this.logger?.debug('QualiaCognitionLayer._computeCochains: Completed', { C0Length: C0.length, C1Size: C1.size, C2Size: C2.size });

            return { C0, C1, C2 };
        } catch (err) {
            this.logger?.error(`QualiaCognitionLayer._computeCochains: Unexpected error`, { error: err.message, stack: err.stack });
            this.currentCochain = [];
            return { C0: [], C1: new Map(), C2: new Map() };
        }
    }

    /**
     * Computes the "Cup Product" intensity as a measure of higher-order coherence.
     *
     */
    async computeCupProduct() {
        try {
            const triangles = this.baseSheaf.simplicialComplex?.triangles;
            if (!triangles || triangles.length === 0) {
                this.cup_product_intensity = 0;
                return 0;
            }

            let totalIntensity = 0;
            let count = 0;

            for (const triangle of triangles) {
                if (!this.baseSheaf.isValidTriangle(triangle)) continue;

                const [u, v, w] = triangle;
                const s_u = this.baseSheaf.stalks.get(u);
                const s_w = this.baseSheaf.stalks.get(w);
                const P_uv = this.baseSheaf.projectionMatrices.get(`${u}-${v}`);
                const P_vw = this.baseSheaf.projectionMatrices.get(`${v}-${w}`);

                if (!this.baseSheaf._isValidStateVec(s_u) || !this.baseSheaf._isValidStateVec(s_w) || !isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vw)) {
                    continue;
                }

                try {
                    const P_compose_result = _matMul(P_uv, P_vw);
                    if (!isFiniteMatrix(P_compose_result)) continue;

                    const s_w_projected = await runWorkerTaskWithRetry('matVecMul', {
                        matrix: flattenMatrix(P_compose_result),
                        vector: s_w,
                        expectedDim: this.baseSheaf.qDim
                    });

                    if (this.baseSheaf._isValidStateVec(s_w_projected)) {
                        const cupValue = dot(s_u, s_w_projected);
                        if (isFiniteNumber(cupValue)) {
                            totalIntensity += Math.abs(cupValue);
                            count++;
                        }
                    }
                } catch (e) {
                    this.logger?.error(`QualiaCognitionLayer.computeCupProduct: Error processing triangle ${triangle.join(',')}.`, { error: e.message });
                }
            }

            const betaFloq = await this._supFloqBetti(this.floquetPD);
            this.cup_product_intensity = count > 0 ? clamp((totalIntensity / count) + (0.01 * betaFloq), 0, 10) : 0;

            return this.cup_product_intensity;
        } catch (err) {
            this.logger?.error(`QualiaCognitionLayer.computeCupProduct: unexpected error: ${err.message}`);
            this.cup_product_intensity = 0;
            return 0;
        }
    }

    /**
     * Computes the structural sensitivity of the sheaf to small perturbations in edge weights.
     *
     */
    async computeStructuralSensitivity(perturbationScale = 0.05) {
        if (!this.baseSheaf.ready || this.baseSheaf.graph.edges.length === 0 || this.baseSheaf.graph.vertices.length === 0 || this.baseSheaf.qDim === 0) {
            this.structural_sensitivity = 0;
            return 0;
        }

        const s_t = this.baseSheaf.getStalksAsVector();
        if (!this.baseSheaf._isValidStateVec(s_t) || s_t.length === 0) {
            logger.warn('QualiaCognitionLayer.computeStructuralSensitivity: Initial stalk vector is invalid or empty. Returning 0.');
            this.structural_sensitivity = 0;
            return 0;
        }

        let baseState;
        try {
            baseState = await this.baseSheaf.simulateDiffusionStep(s_t);
            if (!this.baseSheaf._isValidStateVec(baseState) || baseState.length !== s_t.length) throw new Error("Base state is non-finite or mismatched length.");
        } catch (e) {
            logger.error('QualiaCognitionLayer.computeStructuralSensitivity: Could not compute base state.', e);
            this.structural_sensitivity = 0;
            return 0;
        }

        let totalSensitivity = 0;
        let perturbationCount = 0;
        const originalAdjacency = this.baseSheaf.adjacencyMatrix ? this.baseSheaf.adjacencyMatrix.map(row => new Float32Array(row)) : zeroMatrix(this.baseSheaf.graph.vertices.length, this.baseSheaf.graph.vertices.length);

        if (!isFiniteMatrix(originalAdjacency)) {
            logger.error('QualiaCognitionLayer.computeStructuralSensitivity: Original adjacency matrix is non-finite. Aborting.');
            this.structural_sensitivity = 0;
            return 0;
        }

        for (const edge of this.baseSheaf.graph.edges) {
            const [u, v] = edge;
            const i = this.baseSheaf.graph.vertices.indexOf(u);
            const j = this.baseSheaf.graph.vertices.indexOf(v);
            if (i < 0 || j < 0 || i >= originalAdjacency.length || j >= originalAdjacency.length) {
                logger.warn(`QualiaCognitionLayer.computeStructuralSensitivity: Invalid vertex index for edge ${u}-${v}. Skipping perturbation.`);
                continue;
            }

            const originalWeight = originalAdjacency[i]?.[j] || 0;
            this.baseSheaf.adjacencyMatrix[i][j] = this.baseSheaf.adjacencyMatrix[j][i] = clamp(originalWeight + perturbationScale, 0.01, 1);
            if (!Number.isFinite(this.baseSheaf.adjacencyMatrix[i][j])) {
                 logger.warn(`QualiaCognitionLayer.computeStructuralSensitivity: Perturbed adjacency weight for edge ${u}-${v} is non-finite. Resetting.`);
                 this.baseSheaf.adjacencyMatrix[i][j] = this.baseSheaf.adjacencyMatrix[j][i] = originalWeight;
                 continue;
            }

            try {
                const perturbedState = await this.baseSheaf.simulateDiffusionStep(s_t);
                if (this.baseSheaf._isValidStateVec(perturbedState) && perturbedState.length === baseState.length) {
                    const diffNorm = norm2(vecSub(perturbedState, baseState));
                    if (Number.isFinite(diffNorm)) {
                        totalSensitivity += diffNorm / (perturbationScale !== 0 ? perturbationScale : this.baseSheaf.eps);
                        perturbationCount++;
                    } else {
                        logger.warn(`QualiaCognitionLayer.computeStructuralSensitivity: Non-finite difference norm for edge ${u}-${v}.`);
                    }
                } else {
                    logger.warn(`QualiaCognitionLayer.computeStructuralSensitivity: Perturbed state for edge ${u}-${v} is non-finite or mismatched. Skipping.`);
                }
            } catch (e) {
                logger.warn(`QualiaCognitionLayer.computeStructuralSensitivity: Error during perturbation for edge ${u}-${v}: ${e.message}`);
            } finally {
                this.baseSheaf.adjacencyMatrix[i][j] = this.baseSheaf.adjacencyMatrix[j][i] = originalWeight;
            }
        }

        this.structural_sensitivity = perturbationCount > 0 ? clamp(totalSensitivity / perturbationCount, 0, 10) : 0;
        return this.structural_sensitivity;
    }

    /**
     * Computes the "Geodesic Free Energy" based on KL-divergence proxy and cochain norms.
     *
     */
    async _computeGeodesicFreeEnergy(qualiaState) {
        logger.debug('QualiaCognitionLayer._computeGeodesicFreeEnergy (Robust Version): Starting');

        const kl_proxy = Number.isFinite(this.baseSheaf.owm?.actorLoss) ? this.baseSheaf.owm.actorLoss : 0.1;
        const nV = this.baseSheaf.graph?.vertices?.length || 0;

        if (nV < 2) {
            logger.warn('QualiaCognitionLayer._computeGeodesicFreeEnergy: Insufficient vertices.', { nV });
            return { F_int: kl_proxy, geodesic_divergence: 0 };
        }

        let C1;
        try {
            ({ C1 } = await this._computeCochains(qualiaState));
        } catch (e) {
            logger.error('QualiaCognitionLayer._computeGeodesicFreeEnergy: Error in _computeCochains', { error: e.message });
            return { F_int: kl_proxy, geodesic_divergence: 0 };
        }

        if (!this.currentCochain || !Array.isArray(this.currentCochain)) {
            logger.warn('QualiaCognitionLayer._computeGeodesicFreeEnergy: currentCochain is undefined or invalid. Using stalks.');
            this.currentCochain = Array.from(this.baseSheaf.stalks.values());
        }

        let totalCochainNormSq = 0;
        let edgeCount = 0;

        for (const cochain of C1.values()) {
            if (this.baseSheaf._isValidStateVec(cochain)) {
                totalCochainNormSq += norm2(cochain) ** 2;
                edgeCount++;
            }
        }

        const geodesic_divergence = clamp(this.inconsistency * 0.5, 0, 10);
        const F_base = kl_proxy + (edgeCount > 0 ? totalCochainNormSq / edgeCount : 0);
        const F_int = F_base - geodesic_divergence;

        const result = {
            F_int: clamp(Number.isFinite(F_int) ? F_int : kl_proxy, 0, 10),
            geodesic_divergence: geodesic_divergence
        };

        logger.debug('QualiaCognitionLayer._computeGeodesicFreeEnergy (Robust Version): Completed', { F_int: result.F_int, geodesic_divergence: result.geodesic_divergence });
        return result;
    }

    /**
     * Updates all derived metrics related to awareness and coherence.
     *
     */
    async _updateDerivedMetrics(qualiaState) {
        const last_intentionality_F = this.intentionality_F;
        const last_overallCoherence = this.overallCoherence;

        try {
            // Update H1 dimension and stability based on graph topology
            this.h1Dimension = this.baseSheaf.computeH1Dimension();
            this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);

            // Compute awareness-related metrics
            await this.computeGluingInconsistency();
            await this.computeGestaltUnity();
            await this.computeIntegratedInformation();
            await this.computeCupProduct();
            await this.computeStructuralSensitivity(0.05);

            const { F_int, geodesic_divergence } = await this._computeGeodesicFreeEnergy(qualiaState);
            this.intentionality_F = Number.isFinite(F_int) ? F_int : 0;

            let totalNorm = 0;
            let validVertices = 0;
            for (const stalk of this.baseSheaf.stalks.values()) {
                if (this.baseSheaf._isValidStateVec(stalk)) {
                    totalNorm += norm2(stalk);
                    validVertices++;
                }
            }
            this.feel_F = validVertices > 0 ? clamp(totalNorm / validVertices, 0, 1) : 0;

            const coherenceScore = (this.coherence + (1 - this.inconsistency) + this.gestaltUnity + this.cup_product_intensity * 0.5);
            const divergencePenalty = Number.isFinite(geodesic_divergence) ? geodesic_divergence * 0.1 : 0;
            const overallCoherenceRaw = (coherenceScore / 3.5) - divergencePenalty;

            this.overallCoherence = clamp(Number.isFinite(overallCoherenceRaw) ? overallCoherenceRaw : 0, 0, 1);
            this.emergentAware = this.overallCoherence > 0.7 && this.intentionality_F > 0.5;

        } catch (e) {
            logger.error(`QualiaCognitionLayer._updateDerivedMetrics: Error: ${e.message}`, { stack: e.stack });
            this.intentionality_F = 0;
            this.feel_F = 0;
            this.overallCoherence = 0;
            this.emergentAware = false;
        }
        return {
            intentionality_F_gain: this.intentionality_F - last_intentionality_F,
            overallCoherence_gain: this.overallCoherence - last_overallCoherence
        };
    }

    // --- Recursive Topological Sheaf (Self-Awareness) functions ---

    /**
     * Builds the recursive gluing operator R*.
     *
     */
    _buildRecursiveGluing() {
        return (z) => {
            const z_next = new Map();
            this.baseSheaf.graph.edges.forEach(([u, v]) => {
                const su = this.baseSheaf.stalks.get(u) || vecZeros(this.baseSheaf.qDim);
                const sv = this.baseSheaf.stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                let phi_uv = this.baseSheaf.projectionMatrices.get(`${u}-${v}`);

                if (!this.baseSheaf._isValidStateVec(su) || !this.baseSheaf._isValidStateVec(sv) || su.length !== this.baseSheaf.qDim || sv.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._buildRecursiveGluing: Invalid stalk for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
                    return;
                }
                const diffusion = safeVecScale(vecAdd(su, sv), this.baseSheaf.alpha / 2);

                const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.baseSheaf.qDim);
                if (!this.baseSheaf._isValidStateVec(z_uv) || z_uv.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._buildRecursiveGluing: Invalid z_uv for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
                    return;
                }

                if (!isFiniteMatrix(phi_uv) || phi_uv.length !== this.baseSheaf.qDim || (phi_uv[0]?.length || 0) !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._buildRecursiveGluing: Invalid projection matrix for edge ${u}-${v}. Using identity.`);
                    phi_uv = identity(this.baseSheaf.qDim);
                }
                const z_next_uv = vecAdd(_matVecMul(phi_uv, z_uv), diffusion);
                if (this.baseSheaf._isValidStateVec(z_next_uv)) {
                    z_next.set([u, v].sort().join(','), z_next_uv);
                } else {
                    logger.warn(`QualiaCognitionLayer._buildRecursiveGluing: Non-finite z_next_uv for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
                }
            });
            return z_next;
        };
    }

    /**
     * Computes initial linguistic cocycles for self-awareness.
     *
     */
    computeLinguisticCocycles(state) {
        const z = new Map();
        const nV = this.baseSheaf.graph.vertices.length;
        if (nV === 0 || this.baseSheaf.qDim === 0) return z;

        const idxMap = new Map(this.baseSheaf.graph.vertices.map((v, i) => [v, i]));
        this.baseSheaf.graph.edges.forEach(([u, v]) => {
            const i = idxMap.get(u), j = idxMap.get(v);
            if (i === undefined || j === undefined || i >= nV || j >= nV) return;

            const z_uv = new Float32Array(this.baseSheaf.qDim);
            const input_u = Number.isFinite(state[Math.min(i, state.length - 1)]) ? state[Math.min(i, state.length - 1)] : 0;
            const input_v = Number.isFinite(state[Math.min(j, state.length - 1)]) ? state[Math.min(j, state.length - 1)] : 0;

            for (let k = 0; k < this.baseSheaf.qDim; k++) {
                z_uv[k] = clamp(input_u - input_v, -1, 1) * (this.baseSheaf.entityNames[k]?.includes('symbolic') ? 1.5 : 1);
            }
            if (this.baseSheaf._isValidStateVec(z_uv)) {
                z.set([u, v].sort().join(','), z_uv);
            } else {
                logger.warn(`QualiaCognitionLayer.computeLinguisticCocycles: Non-finite cocycle for edge ${u}-${v}. Using zeros.`);
                z.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
            }
        });
        return z;
    }

    /**
     * Computes self-awareness based on fixed points of the recursive gluing operator.
     *
     */
    async computeSelfAwareness(fullQualiaState) {
        const fallbackDim = this.baseSheaf.graph.vertices.length * this.baseSheaf.qDim || 7;
        const safeDefault = { Phi_SA: 0, aware: false };

        try {
            const expectedDim = this.baseSheaf.graph.vertices.length * this.baseSheaf.qDim;
            if (!this.baseSheaf._isValidStateVec(fullQualiaState) || fullQualiaState.length !== expectedDim) {
                logger.warn(`QualiaCognitionLayer.computeSelfAwareness: Invalid input state vector. Expected dim ${expectedDim}, got ${fullQualiaState?.length}. Returning safe default.`);
                return safeDefault;
            }

            const z_initial = this.computeLinguisticCocycles(fullQualiaState);
            let z_fixed = z_initial;
            for (let i = 0; i < this.maxIter; i++) {
                const z_next = this.R_star(z_fixed);
                if (this._cocycleNormDiff(z_next, z_fixed) < this.fixedPointEps) break;
                z_fixed = z_next;
            }
            this.cochainHistory.push(z_fixed);

            const L_rec = await this._recursiveLaplacian(z_fixed);
            const { eigenvalues } = await this.baseSheaf._spectralDecomp(L_rec);
            const lambda_min = eigenvalues.length > 0 ? Math.min(...eigenvalues.filter(Number.isFinite)) : 0;

            const Phi_SA = clamp(lambda_min > 0 ? Math.log(1 + lambda_min) : 0, 0, 10); // Simplified Phi_SA
            this.selfAware = Phi_SA > this.tau;

            return { Phi_SA, aware: this.selfAware };

        } catch (e) {
            logger.error(`QualiaCognitionLayer.computeSelfAwareness: Computation failed critically.`, { error: e.message });
            return safeDefault;
        }
    }

    /**
     * Computes the norm difference between two cocycle maps.
     *
     */
    _cocycleNormDiff(z1, z2) {
        let sum = 0;
        let count = 0;
        if (! (z1 instanceof Map) || ! (z2 instanceof Map)) {
            logger.warn('QualiaCognitionLayer._cocycleNormDiff: Invalid cocycle map input. Returning 0.');
            return 0;
        }
        for (const key of z1.keys()) {
            const v1 = z1.get(key);
            const v2 = z2.get(key);
            if (this.baseSheaf._isValidStateVec(v1) && this.baseSheaf._isValidStateVec(v2) && v1.length === this.baseSheaf.qDim && v2.length === this.baseSheaf.qDim) {
                sum += norm2(vecSub(v1, v2)) ** 2;
                count++;
            } else {
                logger.warn(`QualiaCognitionLayer._cocycleNormDiff: Invalid or mismatched cocycle vector for key ${key}. Skipping.`);
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    /**
     * Builds a recursive Laplacian matrix based on cocycles.
     *
     */
    async _recursiveLaplacian(z) {
        const nE = this.baseSheaf.graph.edges.length;
        if (nE === 0) return identity(0);

        const L_rec = zeroMatrix(nE, nE);
        const eMap = new Map(this.baseSheaf.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));

        for (const edge of this.baseSheaf.graph.edges) {
            const [u, v] = edge;
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined || i >= nE) {
                logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid edge index ${i} for edge ${u}-${v}. Skipping.`);
                continue;
            }

            L_rec[i][i] = 1;

            const z_uv = z.get([u, v].sort().join(',')) ?? vecZeros(this.baseSheaf.qDim);
            if (!this.baseSheaf._isValidStateVec(z_uv) || z_uv.length !== this.baseSheaf.qDim) {
                logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid z_uv for edge ${u}-${v}. Using zeros.`);
                continue;
            }

            let phi_uv = this.baseSheaf.projectionMatrices.get(`${u}-${v}`);
            if (!isFiniteMatrix(phi_uv) || phi_uv.length !== this.baseSheaf.qDim || (phi_uv[0]?.length || 0) !== this.baseSheaf.qDim) {
                logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid projection matrix for edge ${u}-${v}. Using identity.`);
                phi_uv = identity(this.baseSheaf.qDim);
            }

            for (const edge2 of this.baseSheaf.graph.edges) {
                const [u2, v2] = edge2;
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || j >= nE || i === j) {
                    if (i === j) continue;
                    logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid edge index ${j} for edge ${u2}-${v2}. Skipping interaction.`);
                    continue;
                }

                let phi_u2v2 = this.baseSheaf.projectionMatrices.get(`${u2}-${v2}`);
                if (!isFiniteMatrix(phi_u2v2) || phi_u2v2.length !== this.baseSheaf.qDim || (phi_u2v2[0]?.length || 0) !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid projection matrix for edge ${u2}-${v2}. Using identity.`);
                    phi_u2v2 = identity(this.baseSheaf.qDim);
                }

                const z_u2v2 = z.get([u2, v2].sort().join(',')) ?? vecZeros(this.baseSheaf.qDim);
                if (!this.baseSheaf._isValidStateVec(z_u2v2) || z_u2v2.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._recursiveLaplacian: Invalid z_u2v2 for edge ${u2}-${v2}. Using zeros.`);
                    continue;
                }

                let mat_vec_result;
                try {
                    mat_vec_result = _matVecMul(phi_u2v2, z_u2v2);
                } catch (err) {
                    logger.warn(`[QualiaCognitionLayer._recursiveLaplacian] matVecMul failed for edge ${u2}-${v2}, using zero vector. Error: ${err.message}`);
                    mat_vec_result = vecZeros(this.baseSheaf.qDim);
                }

                if (!this.baseSheaf._isValidStateVec(mat_vec_result) || mat_vec_result.length !== this.baseSheaf.qDim) mat_vec_result = vecZeros(this.baseSheaf.qDim);

                let interaction = dot(z_uv, mat_vec_result);
                if (!Number.isFinite(interaction)) interaction = 0;

                L_rec[i][j] = -this.baseSheaf.alpha * clamp(interaction, -0.1, 0.1);
            }
        }

        if (!isFiniteMatrix(L_rec)) {
            logger.error('QualiaCognitionLayer._recursiveLaplacian: Generated recursive Laplacian is non-finite. Returning identity.');
            return identity(nE);
        }
        return L_rec;
    }

    // --- Adjunction Reflexive Sheaf (Hierarchical Awareness) functions ---

    /**
     * Left adjoint functor F. Maps a sheaf object to a cochain complex.
     *
     */
    _leftAdjoint() {
        return (F_in) => {
            const cochains = new Map();
            const F_stalks = F_in?.stalks || new Map();
            this.baseSheaf.graph.edges.forEach(([u, v]) => {
                const su = F_stalks.get(u) || vecZeros(this.baseSheaf.qDim);
                const sv = F_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                if (!this.baseSheaf._isValidStateVec(su) || !this.baseSheaf._isValidStateVec(sv) || su.length !== this.baseSheaf.qDim || sv.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._leftAdjoint: Invalid stalk for edge ${u}-${v}. Using zero vector for cochain.`);
                    cochains.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
                    return;
                }
                const c1 = safeVecScale(vecAdd(su, sv), this.baseSheaf.alpha / 2);
                if (this.baseSheaf._isValidStateVec(c1)) {
                    cochains.set([u, v].sort().join(','), c1);
                } else {
                    logger.warn(`QualiaCognitionLayer._leftAdjoint: Non-finite cochain c1 for edge ${u}-${v}. Using zero vector.`);
                    cochains.set([u, v].sort().join(','), vecZeros(this.baseSheaf.qDim));
                }
            });
            return {
                cochains,
                transport: (phi) => (v_vec) => _matVecMul(phi, v_vec)
            };
        };
    }

    /**
     * Right adjoint functor U. Maps a cochain complex back to a sheaf object.
     *
     */
    _rightAdjoint() {
        return (C_in) => {
            const stalks = new Map();
            const C_cochains = C_in?.cochains || new Map();
            this.baseSheaf.graph.vertices.forEach(v => {
                let sum = vecZeros(this.baseSheaf.qDim);
                let count = 0;
                this.baseSheaf.graph.edges.forEach(([u, w]) => {
                    if (u === v || w === v) {
                        const c_uw = C_cochains.get([u, w].sort().join(',')) || vecZeros(this.baseSheaf.qDim);
                        if (this.baseSheaf._isValidStateVec(c_uw) && c_uw.length === this.baseSheaf.qDim) {
                            sum = vecAdd(sum, c_uw);
                            count++;
                        } else {
                            logger.warn(`QualiaCognitionLayer._rightAdjoint: Invalid cochain for edge incident to ${v}. Skipping.`);
                        }
                    }
                });
                const newStalk = count > 0 ? safeVecScale(sum, 1 / count) : vecZeros(this.baseSheaf.qDim);
                if (this.baseSheaf._isValidStateVec(newStalk)) {
                    stalks.set(v, newStalk);
                } else {
                    logger.warn(`QualiaCognitionLayer._rightAdjoint: Non-finite new stalk for vertex ${v}. Using zero vector.`);
                    stalks.set(v, vecZeros(this.baseSheaf.qDim));
                }
            });
            return {
                stalks,
                projections: this.baseSheaf.projectionMatrices
            };
        };
    }

    /**
     * The monad T = UF (composition of adjoints).
     *
     */
    _monadUF() {
        return (F_in) => this._rightAdjoint()(this._leftAdjoint()(F_in));
    }

    /**
     * The unit of the adjunction.
     *
     */
    _unit() {
        return (F_in) => {
            const eta_F_stalks = new Map();
            const F_stalks = F_in?.stalks || new Map();
            this.baseSheaf.graph.vertices.forEach(v => {
                const stalk = F_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                if (!this.baseSheaf._isValidStateVec(stalk) || stalk.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._unit: Invalid stalk for vertex ${v}. Using zero vector.`);
                    eta_F_stalks.set(v, vecZeros(this.baseSheaf.qDim));
                    return;
                }
                const scaled = safeVecScale(stalk, 1 + this.baseSheaf.alpha);
                if (this.baseSheaf._isValidStateVec(scaled)) {
                    eta_F_stalks.set(v, scaled);
                } else {
                    logger.warn(`QualiaCognitionLayer._unit: Non-finite scaled stalk for vertex ${v}. Using zero vector.`);
                    eta_F_stalks.set(v, vecZeros(this.baseSheaf.qDim));
                }
            });
            return { stalks: eta_F_stalks };
        };
    }

    /**
     * The counit of the adjunction.
     *
     */
    _counit() {
        return (FU, Id) => {
            const epsilon_FU_stalks = new Map();
            const FU_stalks = FU?.stalks || new Map();
            const Id_stalks = Id?.stalks || new Map();
            this.baseSheaf.graph.vertices.forEach(v => {
                const fu_v = FU_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                const id_v = Id_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
                if (!this.baseSheaf._isValidStateVec(fu_v) || !this.baseSheaf._isValidStateVec(id_v) || fu_v.length !== this.baseSheaf.qDim || id_v.length !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._counit: Invalid stalks for vertex ${v}. Using zero vector for difference.`);
                    epsilon_FU_stalks.set(v, vecZeros(this.baseSheaf.qDim));
                    return;
                }
                const diff = vecSub(fu_v, id_v);
                if (this.baseSheaf._isValidStateVec(diff)) {
                    epsilon_FU_stalks.set(v, diff);
                } else {
                    logger.warn(`QualiaCognitionLayer._counit: Non-finite difference for vertex ${v}. Using zero vector.`);
                    epsilon_FU_stalks.set(v, vecZeros(this.baseSheaf.qDim));
                }
            });
            return { stalks: epsilon_FU_stalks };
        };
    }

    /**
     * Computes the "adjunction fixed point" to determine hierarchical awareness.
     *
     */
    async computeAdjunctionFixedPoint(fullQualiaState) {
        const N = this.baseSheaf.graph.vertices.length * this.baseSheaf.qDim;
        if (N === 0) {
            this.hierarchicallyAware = false;
            return { fixedPoint: vecZeros(1), aware: false, Phi_SA: 0 };
        }

        try {
            this.baseSheaf.laplacian = this.baseSheaf.buildLaplacian();
            if (!isFiniteMatrix(this.baseSheaf.laplacian) || this.baseSheaf.laplacian.length !== N) {
                throw new Error("Cannot compute fixed point with an invalid Laplacian.");
            }

            const { eigenvalues, eigenvectors } = await this.baseSheaf._spectralDecomp(this.baseSheaf.laplacian);

            const fixedPoint = vecZeros(N);
            let contributingVectors = 0;
            eigenvalues.forEach((lambda, i) => {
                if (isFiniteNumber(lambda) && Math.abs(lambda) < this.baseSheaf.eps) {
                    const eigenvector = eigenvectors.map(row => row[i] || 0);
                    if (this.baseSheaf._isValidStateVec(eigenvector)) {
                        for (let j = 0; j < N; j++) {
                            fixedPoint[j] += eigenvector[j];
                        }
                        contributingVectors++;
                    }
                }
            });

            if (contributingVectors > 0) {
                const norm = norm2(fixedPoint);
                if (norm > this.baseSheaf.eps) {
                    for (let i = 0; i < N; i++) {
                        fixedPoint[i] /= norm;
                    }
                }
            }
            const Phi_SA = clamp(norm2(fixedPoint), 0, 10); // Simplified Phi_SA for this context
            const aware = Phi_SA > 0.1;
            this.hierarchicallyAware = aware;

            return { fixedPoint, aware, Phi_SA };

        } catch (e) {
            logger.error(`QualiaCognitionLayer.computeAdjunctionFixedPoint: Computation failed. Returning a safe zero-vector.`, { error: e.message });
            this.hierarchicallyAware = false;
            return { fixedPoint: vecZeros(N), aware: false, Phi_SA: 0 };
        }
    }

    /**
     * Computes the L2 norm difference between two stalk maps.
     *
     */
    async _equalizerNorm(eta_stalks, eps_stalks) {
        let sum = 0;
        let count = 0;
        if (!(eta_stalks instanceof Map) || !(eps_stalks instanceof Map)) {
            logger.warn('QualiaCognitionLayer._equalizerNorm: Invalid stalk map input. Returning 0.');
            return 0;
        }
        for (const v of this.baseSheaf.graph.vertices) {
            const eta_v = eta_stalks.get(v);
            const eps_v = eps_stalks.get(v);
            if (this.baseSheaf._isValidStateVec(eta_v) && this.baseSheaf._isValidStateVec(eps_v) && eta_v.length === this.baseSheaf.qDim && eps_v.length === this.baseSheaf.qDim) {
                sum += norm2(vecSub(eta_v, eps_v)) ** 2;
                count++;
            } else {
                logger.warn(`QualiaCognitionLayer._equalizerNorm: Invalid or mismatched stalk vector for vertex ${v}. Skipping.`);
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    /**
     * Extracts a "fixed cocycle" based on trivial curls (from H2).
     *
     */
    async _extractFixedCocycle(T) {
        const z_star = new Map();
        const F_T_result = this._leftAdjoint()(T);
        if (!F_T_result || !(F_T_result.cochains instanceof Map)) {
            logger.warn('QualiaCognitionLayer._extractFixedCocycle: F(T) result or cochains invalid. Returning empty map.');
            return z_star;
        }

        const { C1, C2 } = await this._computeCochains(this.qInput || vecZeros(this.baseSheaf.qDim)); // Use qInput from baseSheaf update
        if (!(C1 instanceof Map) || !(C2 instanceof Map)) {
            logger.warn('QualiaCognitionLayer._extractFixedCocycle: C1 or C2 cochains are invalid. Returning empty map.');
            return z_star;
        }

        for (const [key, c_curl] of C2.entries()) {
            if (this.baseSheaf._isValidStateVec(c_curl) && norm2(c_curl) < this.baseSheaf.eps) {
                const triangle_vertices = key.split(',');
                if (triangle_vertices.length !== 3) {
                    logger.warn(`QualiaCognitionLayer._extractFixedCocycle: Invalid triangle key ${key}. Skipping.`);
                    continue;
                }
                const edges_of_triangle = [
                    [triangle_vertices[0], triangle_vertices[1]].sort().join(','),
                    [triangle_vertices[1], triangle_vertices[2]].sort().join(','),
                    [triangle_vertices[2], triangle_vertices[0]].sort().join(',')
                ];
                edges_of_triangle.forEach(edge_key => {
                    const edge_cocycle = C1.get(edge_key);
                    if (edge_cocycle && !z_star.has(edge_key) && this.baseSheaf._isValidStateVec(edge_cocycle) && edge_cocycle.length === this.baseSheaf.qDim) {
                        z_star.set(edge_key, edge_cocycle);
                    }
                });
            } else if (!this.baseSheaf._isValidStateVec(c_curl)) {
                logger.warn(`QualiaCognitionLayer._extractFixedCocycle: Non-finite curl for triangle key ${key}. Skipping.`);
            }
        }
        return z_star;
    }

    /**
     * Builds the delta-1 boundary matrix.
     *
     */
    async _deltaR1() {
        const nE = this.baseSheaf.graph.edges.length;
        const nT = this.baseSheaf.simplicialComplex.triangles.length;
        if (nE === 0 || nT === 0) return zeroMatrix(nT, nE);

        const delta = zeroMatrix(nT, nE);
        const eMap = new Map(this.baseSheaf.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));

        this.baseSheaf.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (!this.baseSheaf.isValidTriangle(tri) || tIdx >= nT) {
                logger.warn(`QualiaCognitionLayer._deltaR1: Invalid triangle ${tri.join(',')} or index ${tIdx}. Skipping.`);
                return;
            }
            const sortedTri = tri.slice().sort();
            const [v0, v1, v2] = sortedTri;

            const edge01_key = [v0, v1].sort().join(',');
            const edge12_key = [v1, v2].sort().join(',');
            const edge02_key = [v0, v2].sort().join(',');

            const eIdx01 = eMap.get(edge01_key);
            const eIdx12 = eMap.get(edge12_key);
            const eIdx02 = eMap.get(edge02_key);

            if (eIdx01 !== undefined && eIdx01 < nE) {
                delta[tIdx][eIdx01] = 1;
            }
            if (eIdx12 !== undefined && eIdx12 < nE) {
                delta[tIdx][eIdx12] = 1;
            }
            if (eIdx02 !== undefined && eIdx02 < nE) {
                delta[tIdx][eIdx02] = -1;
            }
        });

        if (!isFiniteMatrix(delta)) {
            logger.error('QualiaCognitionLayer._deltaR1: Generated matrix is non-finite. Returning zero matrix.');
            return zeroMatrix(nT, nE);
        }
        return delta;
    }

    // --- Persistent Adjunction Sheaf (Diachronic Awareness) functions ---

    /**
     * Computes the "temporal derivative" of sheaf stalks.
     *
     */
    _partial_t(F_t, dt = 1) {
        const stalksNext = new Map();
        const F_t_stalks = F_t?.stalks || new Map();
        this.baseSheaf.graph.vertices.forEach(v => {
            const stalk = F_t_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
            if (!this.baseSheaf._isValidStateVec(stalk) || stalk.length !== this.baseSheaf.qDim) {
                logger.warn(`QualiaCognitionLayer._partial_t: Invalid stalk for vertex ${v}. Using zero vector.`);
                stalksNext.set(v, vecZeros(this.baseSheaf.qDim));
                return;
            }

            const neighbors = this.baseSheaf.graph.edges.filter(e => e[0] === v || e[1] === v).map(e => e[0] === v ? e[1] : e[0]);
            let grad = vecZeros(this.baseSheaf.qDim);
            neighbors.forEach(u => {
                const su = F_t_stalks.get(u) || vecZeros(this.baseSheaf.qDim);
                let phi_vu = this.baseSheaf.projectionMatrices.get(`${v}-${u}`);
                if (!isFiniteMatrix(phi_vu) || phi_vu.length !== this.baseSheaf.qDim || (phi_vu[0]?.length || 0) !== this.baseSheaf.qDim) {
                    logger.warn(`QualiaCognitionLayer._partial_t: Invalid projection matrix for edge ${v}-${u}. Using identity.`);
                    phi_vu = identity(this.baseSheaf.qDim);
                }
                const mat_vec_result = _matVecMul(phi_vu, su);
                if (this.baseSheaf._isValidStateVec(mat_vec_result)) {
                    const diff = vecSub(stalk, mat_vec_result);
                    if (this.baseSheaf._isValidStateVec(diff)) grad = vecAdd(grad, safeVecScale(diff, this.baseSheaf.beta));
                } else {
                    logger.warn(`QualiaCognitionLayer._partial_t: Non-finite mat_vec_result for edge ${v}-${u}. Skipping gradient contribution.`);
                }
            });
            const noise = safeVecScale(new Float32Array(this.baseSheaf.qDim).map(() => Math.random() - 0.5), this.baseSheaf.sigma * Math.sqrt(dt));
            const diffused = vecAdd(safeVecScale(stalk, 1 - this.baseSheaf.gamma * dt), vecAdd(grad, noise));
            stalksNext.set(v, this.baseSheaf._isValidStateVec(diffused) ? diffused : vecZeros(this.baseSheaf.qDim));
        });
        return { stalks: stalksNext, projections: F_t?.projections || new Map() };
    }

    /**
     * Initializes stalks based on the input state for persistent fixed point computations.
     *
     */
    _initStalks(state) {
        const stalks = new Map();
        const nV = this.baseSheaf.graph.vertices.length;
        if (nV === 0 || this.baseSheaf.qDim === 0) return stalks;

        this.baseSheaf.graph.vertices.forEach((v, i) => {
            const stalk = new Float32Array(this.baseSheaf.qDim).fill(0);
            const input = Number.isFinite(state[Math.min(i, state.length - 1)]) ? state[Math.min(i, state.length - 1)] : 0;
            for (let k = 0; k < this.baseSheaf.qDim; k++) {
                stalk[k] = clamp(input * (this.baseSheaf.entityNames[k]?.includes('metacognition') ? 1.2 : 1), -1, 1);
            }
            if (this.baseSheaf._isValidStateVec(stalk)) {
                stalks.set(v, stalk);
            } else {
                logger.warn(`QualiaCognitionLayer._initStalks: Non-finite stalk for vertex ${v}. Using zeros.`);
                stalks.set(v, vecZeros(this.baseSheaf.qDim));
            }
        });
        return stalks;
    }

    /**
     * Computes the "Persistent Fixed Point" over a temporal flow, for diachronic awareness.
     *
     */
    async computePersistentFixedPoint(init_state, T = 10) {
        const nV = this.baseSheaf.graph.vertices.length;
        const nE = this.baseSheaf.graph.edges.length;
        if (nV === 0 || nE === 0 || this.baseSheaf.qDim === 0) {
            logger.warn('QualiaCognitionLayer.computePersistentFixedPoint: Graph has no vertices/edges or qDim is zero. Returning default.');
            return { F_persist: null, z_star_persist: new Map(), Phi_SA_persist: 0, PD: { births: [], deaths: [] }, aware: false };
        }

        let F_curr = { stalks: this._initStalks(init_state) };
        this.flowHistory.clear();
        let Phi_SA_persist = 0;

        for (let t = 0; t < T; t++) {
            const partial_F = this._partial_t(F_curr, this.baseSheaf.gamma);
            const F_next = this._leftAdjoint()(partial_F);
            const U_next = this._rightAdjoint()(F_next);
            const T_next = this._monadUF()(partial_F);
            const eta_next = this._unit()(T_next);

            // Fetch previous eta stalks for nablaPersist
            const eta_prev_obj = this.flowHistory.get(this.flowHistory.length - 1);
            const eta_prev_stalks = (eta_prev_obj && eta_prev_obj.eta_t?.stalks) ? eta_prev_obj.eta_t.stalks : eta_next.stalks;

            const eta_evol = await this._nablaPersist(eta_next.stalks, eta_prev_stalks);
            const epsilon_next = this._counit()(U_next, partial_F);

            const eq_delta = await this._equalizerNorm(eta_evol, epsilon_next.stalks);
            if (!Number.isFinite(eq_delta) || eq_delta < this.baseSheaf.eps) break;

            let L_Tt;
            try {
                L_Tt = await this._flowLaplacian();
            } catch (e) {
                logger.error(`QualiaCognitionLayer.computePersistentFixedPoint: _flowLaplacian failed: ${e.message}. Using identity.`, {stack: e.stack});
                L_Tt = identity(nV * this.baseSheaf.qDim > 0 ? nV * this.baseSheaf.qDim : 1);
            }

            let spectralResult;
            try {
                spectralResult = await this.baseSheaf._spectralDecomp(L_Tt);
            } catch (e) {
                logger.error(`QualiaCognitionLayer.computePersistentFixedPoint: _spectralDecomp failed: ${e.message}. Using empty eigenvalues.`, {stack: e.stack});
                spectralResult = { eigenvalues: [] };
            }
            const eigenvalues = spectralResult?.eigenvalues || [];
            const lambda_min_t = eigenvalues.length > 0 ? Math.min(...eigenvalues.filter(Number.isFinite)) : 0;
            const beta1_persist = await this._supPersistentBetti(this.persistenceDiagram);
            Phi_SA_persist += lambda_min_t * beta1_persist * this.baseSheaf.gamma;

            this.persistenceDiagram = this._updatePD(this.persistenceDiagram, eigenvalues, t);
            const d_B = this._bottleneckDistance(this.persistenceDiagram);
            this.diachronicallyAware = (d_B < this.delta && Phi_SA_persist > this.tau_persist); // Set awareness here

            this.flowHistory.push({ F_t: F_next, U_t: U_next, eta_t: { stalks: eta_evol }, lambda_t: eigenvalues });
            F_curr = partial_F;
        }

        const z_star_persist = await this._extractPersistCocycle(this.flowHistory.getAll());
        Phi_SA_persist = clamp(Number.isFinite(Phi_SA_persist) ? Phi_SA_persist : 0, 0, 100);

        return { F_persist: F_curr, z_star_persist, Phi_SA_persist, PD: this.persistenceDiagram, aware: this.diachronicallyAware };
    }

    /**
     * Computes the temporal gradient of the unit component for persistence.
     *
     */
    async _nablaPersist(eta_next_stalks, eta_prev_stalks) {
        const eta_evol = new Map();
        if (!(eta_next_stalks instanceof Map) || !(eta_prev_stalks instanceof Map)) {
            logger.warn('QualiaCognitionLayer._nablaPersist: Invalid stalk map input. Returning empty map.');
            return eta_evol;
        }
        this.baseSheaf.graph.vertices.forEach(v => {
            const next_v = eta_next_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
            const prev_v = eta_prev_stalks.get(v) || vecZeros(this.baseSheaf.qDim);
            if (!this.baseSheaf._isValidStateVec(next_v) || !this.baseSheaf._isValidStateVec(prev_v) || next_v.length !== this.baseSheaf.qDim || prev_v.length !== this.baseSheaf.qDim) {
                logger.warn(`QualiaCognitionLayer._nablaPersist: Invalid stalk vector for vertex ${v}. Using zero vector.`);
                eta_evol.set(v, vecZeros(this.baseSheaf.qDim));
                return;
            }
            const diff = vecSub(next_v, prev_v);
            const evol = vecAdd(next_v, safeVecScale(diff, this.baseSheaf.gamma));
            if (this.baseSheaf._isValidStateVec(evol)) eta_evol.set(v, evol);
             else {
                logger.warn(`QualiaCognitionLayer._nablaPersist: Non-finite evolved stalk for vertex ${v}. Using zero vector.`);
                eta_evol.set(v, vecZeros(this.baseSheaf.qDim));
            }
        });
        return eta_evol;
    }

    /**
     * Builds the flow Laplacian based on the base sheaf's Laplacian.
     *
     */
    async _flowLaplacian() {
        const nV = this.baseSheaf.graph.vertices.length;
        const N = nV * this.baseSheaf.qDim;
        if (N === 0) return identity(0);

        this.baseSheaf.laplacian = this.baseSheaf.buildLaplacian();

        if (!isFiniteMatrix(this.baseSheaf.laplacian) || this.baseSheaf.laplacian.length !== N || (this.baseSheaf.laplacian[0]?.length || 0) !== N) {
            logger.warn('QualiaCognitionLayer._flowLaplacian: Base Laplacian is invalid even after a rebuild attempt. Returning identity.');
            return identity(N > 0 ? N : 1);
        }

        const L_base = this.baseSheaf.laplacian;
        const L_Tt = zeroMatrix(N, N);

        for (let i = 0; i < nV; i++) {
            for (let j = 0; j < nV; j++) {
                for (let qi = 0; qi < this.baseSheaf.qDim; qi++) {
                    const baseVal = L_base[i * this.baseSheaf.qDim + qi]?.[j * this.baseSheaf.qDim + qi];
                    if (Number.isFinite(baseVal)) {
                        L_Tt[i * this.baseSheaf.qDim + qi][j * this.baseSheaf.qDim + qi] = baseVal;
                    }
                }
            }
        }

        if (!isFiniteMatrix(L_Tt)) {
            logger.error('QualiaCognitionLayer._flowLaplacian: Generated full flow Laplacian became non-finite. Returning identity.');
            return identity(N);
        }
        return L_Tt;
    }

    /**
     * Updates the persistence diagram with new eigenvalues.
     *
     */
    _updatePD(pd_old, lambda_t, birth_time) {
        const safe_pd_old = pd_old && typeof pd_old === 'object' ? pd_old : {};
        const safe_births = Array.isArray(safe_pd_old.births) ? safe_pd_old.births : [];
        const safe_deaths = Array.isArray(safe_pd_old.deaths) ? safe_pd_old.deaths : [];

        const pd = { births: [...safe_births], deaths: [...safe_deaths] };
        lambda_t.forEach(lambda => {
            if (Number.isFinite(lambda) && !pd.births.some(b => Number.isFinite(b?.value) && Math.abs(b.value - lambda) < this.baseSheaf.eps)) {
                pd.births.push({ value: lambda, time: birth_time });
            }
        });
        pd.births.forEach((birth, i) => {
            if (!Number.isFinite(birth?.time)) birth.time = 0;
            if (!pd.deaths[i] || !Number.isFinite(pd.deaths[i].time) || pd.deaths[i].time < birth.time) {
                pd.deaths[i] = { value: birth.value, time: birth.time + this.delta };
            }
            if (pd.deaths[i] && pd.deaths[i].time < birth_time - this.delta) {
                 pd.deaths[i].time = birth_time;
            }
            if (pd.deaths[i] && pd.deaths[i].time < birth.time) {
                 pd.deaths[i].time = birth.time + this.delta;
            }
        });
        return pd;
    }

    /**
     * Computes the bottleneck distance between persistence diagrams.
     *
     */
    _bottleneckDistance(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const safe_births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.time)) : [];
        const safe_deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.time)) : [];

        let maxDist = 0;
        for (let i = 0; i < safe_births.length; i++) {
            const birth = safe_births[i];
            const death = safe_deaths[i] && Number.isFinite(safe_deaths[i].time) && safe_deaths[i].time >= birth.time
                          ? safe_deaths[i]
                          : { value: birth.value, time: birth.time + this.delta };
            const dist = Math.abs(death.time - birth.time);
            if (Number.isFinite(dist) && dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    /**
     * Computes a simplified Betti number from a persistence diagram.
     *
     */
    async _supPersistentBetti(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.value) && Number.isFinite(b?.time)) : [];
        const deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.value) && Number.isFinite(d?.time)) : [];

        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const birth = births[i];
            const death = deaths[i];
            const lifetime = (death?.time && death.time >= birth.time) ? (death.time - birth.time) : Infinity;
            if (lifetime >= this.delta && Number.isFinite(lifetime)) count++;
        }
        return count;
    }

    /**
     * Extracts a persistent cocycle from flow records.
     *
     */
    async _extractPersistCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        if (!isFiniteMatrix(delta1) || delta1.length === 0 || (delta1[0]?.length || 0) === 0) {
            logger.warn('QualiaCognitionLayer._extractPersistCocycle: Invalid delta1 matrix. Returning empty map.');
            return z_star;
        }

        const nT = delta1.length;
        const nE = delta1[0]?.length || 0;
        if (nE === 0) return z_star;


        for (const flow_record of flows) {
            const F_t = flow_record?.F_t;
            if (!F_t || !(F_t.cochains instanceof Map)) {
                logger.warn('QualiaCognitionLayer._extractPersistCocycle: Invalid flow_record or cochains. Skipping.');
                continue;
            }

            const C1_map = F_t.cochains;

            for (let q = 0; q < this.baseSheaf.qDim; q++) {
                const C1_q_coeffs = new Float32Array(nE);
                this.baseSheaf.graph.edges.forEach((edge, eIdx) => {
                    if (eIdx >= nE) {
                        logger.warn(`QualiaCognitionLayer._extractPersistCocycle: Edge index ${eIdx} out of bounds for C1_q_coeffs. Skipping.`);
                        return;
                    }
                    const edgeKey = [edge[0], edge[1]].sort().join(',');
                    const c_edge = C1_map.get(edgeKey) || vecZeros(this.baseSheaf.qDim);
                    if (this.baseSheaf._isValidStateVec(c_edge) && c_edge.length === this.baseSheaf.qDim && Number.isFinite(c_edge[q])) {
                        C1_q_coeffs[eIdx] = c_edge[q];
                    } else {
                        logger.warn(`QualiaCognitionLayer._extractPersistCocycle: Non-finite cochain or invalid dimension for edge ${edgeKey}, dim ${q}. Using 0.`);
                        C1_q_coeffs[eIdx] = 0;
                    }
                });

                if (!this.baseSheaf._isValidStateVec(C1_q_coeffs)) {
                    logger.warn(`QualiaCognitionLayer._extractPersistCocycle: Non-finite C1_q_coeffs for qualia dimension ${q}. Skipping.`);
                    continue;
                }

                let d1_C1_q;
                try {
                    d1_C1_q = _matVecMul(delta1, C1_q_coeffs);
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer._extractPersistCocycle: matVecMul with delta1 failed for dim ${q}: ${e.message}. Skipping.`);
                    continue;
                }


                if (this.baseSheaf._isValidStateVec(d1_C1_q) && norm2(d1_C1_q) < this.baseSheaf.eps * nT) {
                    this.baseSheaf.graph.edges.forEach((edge, eIdx) => {
                        const edgeKey = [edge[0], edge[1]].sort().join(',');
                        const current_cocycle = C1_map.get(edgeKey);
                        if (current_cocycle && !z_star.has(edgeKey) && this.baseSheaf._isValidStateVec(current_cocycle) && current_cocycle.length === this.baseSheaf.qDim) {
                            z_star.set(edgeKey, current_cocycle);
                        }
                    });
                } else if (!this.baseSheaf._isValidStateVec(d1_C1_q)) {
                    logger.warn(`QualiaCognitionLayer._extractPersistCocycle: matVecMul returned non-finite d1_C1_q for dim ${q}. Skipping.`);
                }
            }
        }
        return z_star;
    }

    // --- Floquet Persistent Sheaf (Rhythmic Awareness) functions ---

    /**
     * Converts a dense matrix to Compressed Sparse Row (CSR) format.
     *
     */
    _denseToCSR(dense) {
        const n = dense.length;
        const values = [];
        const colIndices = [];
        const rowPtr = new Int32Array(n + 1);
        rowPtr[0] = 0;

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = Number.isFinite(dense[i]?.[j]) ? dense[i][j] : 0;
                if (val !== 0) {
                    values.push(clamp(val, -1e6, 1e6));
                    colIndices.push(j);
                }
            }
            rowPtr[i + 1] = values.length;
        }

        return {
            values: new Float32Array(values),
            colIndices: new Int32Array(colIndices),
            rowPtr,
            n
        };
    }

    /**
     * Updates the Floquet Persistence Diagram internally.
     *
     */
    async _updateFloqPD_internal() {
        this.floquetPD = this.floquetPD && typeof this.floquetPD === 'object' ? this.floquetPD : {};
        this.floquetPD.births = Array.isArray(this.floquetPD.births) ? this.floquetPD.births : [];
        this.floquetPD.phases = Array.isArray(this.floquetPD.phases) ? this.floquetPD.phases : [];
        this.floquetPD.deaths = Array.isArray(this.floquetPD.deaths) ? this.floquetPD.deaths : [];

        try {
            const newPhase = Math.sin(this.omega * this.phi);
            this.floquetPD.phases.push(Number.isFinite(newPhase) ? newPhase : 0);

            const currentStalkNorm = norm2(this.baseSheaf.getStalksAsVector());
            this.floquetPD.births.push({ value: Number.isFinite(currentStalkNorm) ? currentStalkNorm : 0, phase: newPhase, time: Date.now() });
        } catch (e) {
            logger.error('QualiaCognitionLayer._updateFloqPD_internal: Failed to update Floquet PD.', e);
            this.floquetPD = { births: [], phases: [], deaths: [] };
        }
    }

    /**
     * Computes the monodromy matrix for Floquet analysis.
     *
     */
    async _monodromy(A_t, omega) {
        if (!isFiniteMatrix(A_t) || A_t.length === 0 || A_t.length !== (A_t[0]?.length || 0)) {
            logger.warn('QualiaCognitionLayer._monodromy: Input matrix A_t is invalid. Returning identity.');
            return identity(A_t?.length || 1);
        }
        const n = A_t.length;
        if (n === 0) return identity(0);
        if (n === 1) {
            const val = A_t[0][0];
            return [[Number.isFinite(val) ? (val ** omega) : 1]];
        }

        if (tf) {
            try {
                const A_tensor = tf.tensor2d(A_t);
                const A_omega = tf.linalg.matrixPower(A_tensor, omega);
                const result_array = await A_omega.array();
                tf.dispose([A_tensor, A_omega]);
                return isFiniteMatrix(result_array) ? result_array : identity(n);
            } catch (e) {
                logger.warn(`QualiaCognitionLayer._monodromy: TF.js failed: ${e.message}. Falling back to CPU matrix multiplication.`, {stack: e.stack});
            }
        }

        let result = A_t;
        for (let i = 1; i < omega; i++) {
            try {
                result = _matMul(result, A_t);
                if (!isFiniteMatrix(result)) {
                    throw new Error("CPU matrix multiplication resulted in non-finite matrix.");
                }
            } catch (e) {
                logger.warn(`QualiaCognitionLayer._monodromy: CPU matrix multiplication failed: ${e.message}. Returning identity.`, {stack: e.stack});
                return identity(n);
            }
        }
        return isFiniteMatrix(result) ? result : identity(n);
    }

    /**
     * Computes the Floquet Fixed Point for rhythmic awareness.
     *
     */
    async computeFloquetFixedPoint(states_history, period) {
        const n_total_stalk_dim = Math.max(1, this.baseSheaf.graph.vertices.length * this.baseSheaf.qDim);

        const sanitizedStates = (states_history || [])
            .filter(s => this.baseSheaf._isValidStateVec(s) && s.length === n_total_stalk_dim);


        if (sanitizedStates.length < 2) {
            logger.warn('QualiaCognitionLayer.computeFloquetFixedPoint: Not enough valid states for analysis. Returning defaults.');
            return {
                monodromy: identity(n_total_stalk_dim),
                eigenvalues: Array(n_total_stalk_dim).fill({ re: 1, im: 0 }),
                Phi_SA_floq: 0,
                aware: false,
                FloqPD: { births: [], phases: [], deaths: [] }
            };
        }

        let combined_monodromy = identity(n_total_stalk_dim);

        try {
            for (let t = 0; t < period; t++) {
                const stateT = sanitizedStates[t % sanitizedStates.length];
                const stateT1 = sanitizedStates[(t + 1) % sanitizedStates.length];

                let transition = await this._stateTransitionMatrix(stateT, stateT1);

                if (!isFiniteMatrix(transition) || transition.length !== n_total_stalk_dim) {
                    logger.warn(`Invalid transition matrix at step ${t}, replacing with identity.`);
                    transition = identity(n_total_stalk_dim);
                }

                combined_monodromy = _matMul(combined_monodromy, transition);

                if (!isFiniteMatrix(combined_monodromy) || combined_monodromy.length !== n_total_stalk_dim) {
                    logger.warn(`Combined monodromy became invalid at step ${t}, resetting to identity.`);
                    combined_monodromy = identity(n_total_stalk_dim);
                }
            }

            this.monodromy = combined_monodromy;
            const rho_k_result = await this._floquetDecomp(this.monodromy);
            const rho_k = rho_k_result.eigenvalues || Array(n_total_stalk_dim).fill({ re: 1, im: 0 });

            const log_rho_sum = rho_k.reduce((sum, rho) => {
                const mag = Math.sqrt((rho.re || 0) ** 2 + (rho.im || 0) ** 2);
                return sum + (isFiniteNumber(mag) && mag > 0 ? Math.log(mag) : 0);
            }, 0);

            const FloqPD = this._updateFloqPD(this.floquetPD || { births: [], phases: [], deaths: [] }, rho_k, Date.now());
            const d_B_floq = this._rhythmicBottleneck(FloqPD);
            const beta1_floq = await this._supFloqBetti(FloqPD);
            const Phi_SA_floq = clamp(Math.abs(log_rho_sum) * beta1_floq, 0, 100);
            this.rhythmicallyAware = (Phi_SA_floq > this.tau_floq && d_B_floq < this.delta);

            return { monodromy: this.monodromy, Phi_SA_floq, FloqPD, aware: this.rhythmicallyAware, eigenvalues: rho_k };

        } catch (e) {
            logger.error(`QualiaCognitionLayer.computeFloquetFixedPoint: Critical error during computation: ${e.message}`, { stack: e.stack });
            return {
                monodromy: identity(n_total_stalk_dim),
                eigenvalues: Array(n_total_stalk_dim).fill({ re: 1, im: 0 }),
                Phi_SA_floq: 0,
                aware: false,
                FloqPD: { births: [], phases: [], deaths: [] }
            };
        }
    }

    /**
     * Computes a state transition matrix between two successive states.
     *
     */
    async _stateTransitionMatrix(stateT, stateT1) {
        const n = stateT?.length || 0;

        if (!this.baseSheaf._isValidStateVec(stateT) || !this.baseSheaf._isValidStateVec(stateT1) || stateT1.length !== n || n === 0) {
            logger.warn('QualiaCognitionLayer._stateTransitionMatrix: Invalid input states. Returning identity.', { stateT, stateT1 });
            return identity(n || 1);
        }

        const diffNorm = norm2(vecSub(stateT, stateT1));
        if (!Number.isFinite(diffNorm) || diffNorm > 0.1 * n) {
            logger.warn('QualiaCognitionLayer._stateTransitionMatrix: States too far apart. Using perturbed identity.', { diffNorm });
            return identity(n).map(row => row.map(val => val + (Math.random() - 0.5) * 0.005));
        }

        const transition = identity(n).map((row, i) => row.map((val, j) => {
            if (i === j) return 1;
            const diff_val = (stateT1[i] || 0) - (stateT[j] || 0);
            const safeVal = clamp(diff_val * 0.01, -0.1, 0.1);
            return Number.isFinite(safeVal) ? safeVal : 0;
        }));

        return isFiniteMatrix(transition) ? transition : identity(n);
    }

    /**
     * Computes Floquet multipliers (complex eigenvalues) from a monodromy matrix.
     *
     */
    async _floquetDecomp(A) {
        const n = A?.length || 0;
        if (!isFiniteMatrix(A) || n === 0 || A[0]?.length !== n) {
            logger.warn('QualiaCognitionLayer._floquetDecomp: Invalid input matrix. Returning fallback (identity eigenvalues).', { rows: n, cols: A[0]?.length });
            return { eigenvalues: Array(n).fill({ re: 1, im: 0 }) };
        }

        const sanitizedA = A.map(row => row.map(val => Number.isFinite(val) ? val : 0));

        try {
            const flat = flattenMatrix(sanitizedA);
            if (!this.baseSheaf._isValidStateVec(flat.flatData) || flat.flatData.length !== n * n) {
                 throw new Error("Flattened matrix is invalid for worker.");
            }
            const complex_eigenvalues = await runWorkerTaskWithRetry('complexEigenvalues', { matrix: flat }, 15000);
            const validEigs = (complex_eigenvalues || []).filter(v => Number.isFinite(v.re) && Number.isFinite(v.im));

            if (validEigs.length > 0) {
                logger.debug('QualiaCognitionLayer._floquetDecomp: Successfully obtained complex eigenvalues from worker.');
                return { eigenvalues: validEigs };
            } else {
                logger.warn('QualiaCognitionLayer._floquetDecomp: Complex eigenvalues worker returned an empty or invalid result. Falling back to magnitudes.');
                throw new Error("Worker returned no valid complex eigenvalues.");
            }
        } catch (e) {
            logger.error(`QualiaCognitionLayer._floquetDecomp: Complex eigenvalues worker or data validation failed (${e.message}). Falling back to Jacobi on A^T * A for magnitudes.`, {stack: e.stack});

            try {
                const At = _transpose(sanitizedA);
                const ATA = _matMul(At, sanitizedA);
                if (!isFiniteMatrix(ATA)) {
                    throw new Error("A^T * A computation resulted in a non-finite matrix in fallback.");
                }
                const { eigenvalues: realEigsATA } = this.baseSheaf._jacobiEigenvalueDecomposition(ATA);
                const floquetMagnitudes = realEigsATA.map(val => ({
                    re: Math.sqrt(Math.max(0, val)),
                    im: 0
                }));
                logger.debug('QualiaCognitionLayer._floquetDecomp: Successfully obtained singular value magnitudes from Jacobi fallback.');
                return { eigenvalues: floquetMagnitudes.filter(v => Number.isFinite(v.re) && Number.isFinite(v.im)) };
            } catch (innerError) {
                logger.error(`QualiaCognitionLayer._floquetDecomp: Final fallback (Jacobi on ATA) also failed (${innerError.message}). Returning default eigenvalues.`, {stack: innerError.stack});
                return { eigenvalues: Array(n).fill({ re: 1, im: 0 }) };
            }
        }
    }

    /**
     * Updates the Floquet Persistence Diagram with Floquet multipliers.
     *
     */
    _updateFloqPD(pd_old, rho_t, phase_time_index) {
        const safe_pd_old = pd_old && typeof pd_old === 'object' ? pd_old : {};
        const safe_phases = Array.isArray(safe_pd_old.phases) ? safe_pd_old.phases : [];
        const safe_births = Array.isArray(safe_pd_old.births) ? safe_pd_old.births : [];
        const safe_deaths = Array.isArray(safe_pd_old.deaths) ? safe_pd_old.deaths : [];

        const pd = { phases: [...safe_phases], births: [...safe_births], deaths: [...safe_deaths] };
        rho_t.forEach(rho => {
            const re = Number.isFinite(rho.re) ? rho.re : 0;
            const im = Number.isFinite(rho.im) ? rho.im : 0;
            const mag = Math.sqrt(re ** 2 + im ** 2);
            const theta = Math.atan2(im, re);

            if (Number.isFinite(mag) && !pd.births.some(b => Number.isFinite(b?.value) && Math.abs(b.value - mag) < this.baseSheaf.eps)) {
                pd.births.push({ value: mag, phase: theta, time: phase_time_index });
                pd.phases.push(theta);
            } else if (!Number.isFinite(mag)) {
                 logger.warn(`QualiaCognitionLayer._updateFloqPD: Non-finite Floquet multiplier magnitude detected. Skipping birth event.`);
            }
        });

        pd.births.forEach((birth, i) => {
            if (!Number.isFinite(birth?.time)) birth.time = 0;
            if (!pd.deaths[i] || !Number.isFinite(pd.deaths[i].time) || pd.deaths[i].time < birth.time) {
                pd.deaths[i] = { value: birth.value, phase: pd.phases[i], time: birth.time + this.delta };
            }
            if (pd.deaths[i] && pd.deaths[i].time < phase_time_index - this.delta) {
                 pd.deaths[i].time = phase_time_index;
            }
            if (pd.deaths[i] && pd.deaths[i].time < birth.time) {
                 pd.deaths[i].time = birth.time + this.delta;
            }
            if (!Number.isFinite(pd.phases[i])) pd.phases[i] = 0;
            if (!Number.isFinite(pd.deaths[i]?.phase)) pd.deaths[i].phase = pd.phases[i];
        });

        return pd;
    }

    /**
     * Computes the rhythmic bottleneck distance for Floquet persistence.
     *
     */
    _rhythmicBottleneck(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const safe_births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.time)) : [];
        const safe_deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.time)) : [];

        let maxDist = 0;
        for (let i = 0; i < safe_births.length; i++) {
            const birth = safe_births[i];
            const death = safe_deaths[i] && Number.isFinite(safe_deaths[i].time) && safe_deaths[i].time >= birth.time
                          ? safe_deaths[i]
                          : { value: birth.value, time: birth.time + this.delta };
            const dist = Math.abs(death.time - birth.time);
            if (Number.isFinite(dist) && dist > maxDist) maxDist = dist;
        }
        return maxDist;
    }

    /**
     * Computes a simplified Betti number for Floquet persistence.
     *
     */
    async _supFloqBetti(pd) {
        const safe_pd = pd && typeof pd === 'object' ? pd : {};
        const births = Array.isArray(safe_pd.births) ? safe_pd.births.filter(b => Number.isFinite(b?.value) && Number.isFinite(b?.time)) : [];
        const deaths = Array.isArray(safe_pd.deaths) ? safe_pd.deaths.filter(d => Number.isFinite(d?.value) && Number.isFinite(d?.time)) : [];

        let count = 0;
        for (let i = 0; i < births.length; i++) {
            const birth = births[i];
            const death = deaths[i];
            const lifetime = (death?.time && death.time >= birth.time) ? (death.time - birth.time) : Infinity;
            if (lifetime >= this.delta && Number.isFinite(lifetime)) count++;
        }
        return count;
    }

    /**
     * Extracts a Floquet cocycle from flow records.
     *
     */
    async _extractFloqCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        if (!isFiniteMatrix(delta1) || delta1.length === 0 || (delta1[0]?.length || 0) === 0) {
            logger.warn('QualiaCognitionLayer._extractFloqCocycle: Invalid delta1 matrix. Returning empty map.');
            return z_star;
        }

        const nT = delta1.length;
        const nE = delta1[0]?.length || 0;
        if (nE === 0) return z_star;

        for (const flow_record of flows) {
            const F_t = flow_record?.F_t;
            if (!F_t || !(F_t.cochains instanceof Map)) {
                logger.warn('QualiaCognitionLayer._extractFloqCocycle: Invalid flow_record or cochains. Skipping.');
                continue;
            }

            const C1_map = F_t.cochains;

            for (let q = 0; q < this.baseSheaf.qDim; q++) {
                const C1_q_coeffs = new Float32Array(nE);
                this.baseSheaf.graph.edges.forEach((edge, eIdx) => {
                    if (eIdx >= nE) {
                        logger.warn(`QualiaCognitionLayer._extractFloqCocycle: Edge index ${eIdx} out of bounds for C1_q_coeffs. Skipping.`);
                        return;
                    }
                    const edgeKey = [edge[0], edge[1]].sort().join(',');
                    const c_edge = C1_map.get(edgeKey) || vecZeros(this.baseSheaf.qDim);
                    if (this.baseSheaf._isValidStateVec(c_edge) && c_edge.length === this.baseSheaf.qDim && Number.isFinite(c_edge[q])) {
                        C1_q_coeffs[eIdx] = c_edge[q];
                    } else {
                        logger.warn(`QualiaCognitionLayer._extractFloqCocycle: Non-finite cochain or invalid dimension for edge ${edgeKey}, dim ${q}. Using 0.`);
                        C1_q_coeffs[eIdx] = 0;
                    }
                } );

                if (!this.baseSheaf._isValidStateVec(C1_q_coeffs)) {
                    logger.warn(`QualiaCognitionLayer._extractFloqCocycle: Non-finite C1_q_coeffs for qualia dimension ${q}. Skipping.`);
                    continue;
                }

                let d1_C1_q;
                try {
                    d1_C1_q = _matVecMul(delta1, C1_q_coeffs);
                } catch (e) {
                    logger.warn(`QualiaCognitionLayer._extractFloqCocycle: matVecMul with delta1 failed for dim ${q}: ${e.message}. Skipping.`);
                    continue;
                }


                if (this.baseSheaf._isValidStateVec(d1_C1_q) && norm2(d1_C1_q) < this.baseSheaf.eps * nT) {
                    this.baseSheaf.graph.edges.forEach((edge, eIdx) => {
                        const edgeKey = [edge[0], edge[1]].sort().join(',');
                        const current_cocycle = C1_map.get(edgeKey);
                        if (current_cocycle && !z_star.has(edgeKey) && this.baseSheaf._isValidStateVec(current_cocycle) && current_cocycle.length === this.baseSheaf.qDim) {
                            const phase = Array.isArray(this.floquetPD?.phases) && this.floquetPD.phases.length > 0 ?
                                          this.floquetPD.phases[this.floquetPD.phases.length - 1] : 0;
                            const weightedCocycle = safeVecScale(current_cocycle, Number.isFinite(Math.cos(phase)) ? Math.cos(phase) : 0);
                            if (this.baseSheaf._isValidStateVec(weightedCocycle)) {
                                z_star.set(edgeKey, weightedCocycle);
                            } else {
                                logger.warn(`QualiaCognitionLayer._extractFloqCocycle: Non-finite weighted cocycle for edge ${edgeKey}. Storing original (unweighted) instead.`);
                                z_star.set(edgeKey, current_cocycle);
                            }
                        }
                    });
                } else if (!this.baseSheaf._isValidStateVec(d1_C1_q)) {
                    logger.warn(`QualiaCognitionLayer._extractFloqCocycle: matVecMul returned non-finite d1_C1_q for dim ${q}. Skipping.`);
                }
            }
        }
        return z_star;
    }

    /**
     * The main update loop orchestrating all sheaf dynamics and cognitive computations.
     *
     */
    async update(state, stepCount = 0) {
        if (!this.baseSheaf.ready) {
            await this.baseSheaf.initialize();
            if (!this.baseSheaf.ready) {
                logger.error('QualiaCognitionLayer.update: BaseSheaf initialization failed. Aborting update.');
                return;
            }
        }

        if (!this.baseSheaf._isValidStateVec(state) || state.length !== this.baseSheaf.stateDim) {
            logger.warn('QualiaCognitionLayer.update: Invalid input state. Skipping update.', { state });
            return;
        }

        try {
            // 1. Update core sheaf structure and dynamics (Laplacian, Projections, Diffusion)
            await this.baseSheaf.computeCorrelationMatrix();
            this.baseSheaf.laplacian = this.baseSheaf.buildLaplacian();
            await this.baseSheaf.computeProjectionMatrices();

            // Prepare qualia input from external state
            const rawQualiaState = _matVecMul(this.baseSheaf.stateToQualiaProjection, state);
            const spontaneousActivation = new Float32Array(this.baseSheaf.qDim).map(() => (Math.random() - 0.5) * 0.05);
            this.qInput = vecAdd(rawQualiaState, spontaneousActivation);

            // Diffuse qualia states
            await this.baseSheaf.diffuseQualia(state, this.qInput);
            const fullQualiaState = this.baseSheaf.getStalksAsVector();

            // 2. Compute higher-order awareness metrics and states
            const { Phi_SA, aware: self_aware } = await this.computeSelfAwareness(fullQualiaState);
            this.selfAware = self_aware;

            const { Phi_SA: Phi_SA_adj, aware: adj_aware } = await this.computeAdjunctionFixedPoint(fullQualiaState);
            this.hierarchicallyAware = adj_aware;

            const { Phi_SA_persist, aware: persist_aware, PD } = await this.computePersistentFixedPoint(fullQualiaState, this.omega);
            this.diachronicallyAware = persist_aware;
            this.persistenceDiagram = PD;

            const { Phi_SA_floq, aware: floq_aware, FloqPD } = await this.computeFloquetFixedPoint(this.baseSheaf.windowedStates.getAll(), this.omega);
            this.rhythmicallyAware = floq_aware;
            this.floquetPD = FloqPD;

            // 3. Adapt graph topology if needed (uses metrics calculated above)
            if (stepCount > 0 && stepCount % 100 === 0) {
                // Pass current cognitive state for more informed adaptation decisions if desired
                await this.baseSheaf.adaptSheafTopology(100, stepCount, this.baseSheaf.adaptation.addThresh, this.baseSheaf.adaptation.removeThresh);
            }

            // 4. Update overall derived metrics and awareness status
            await this._updateDerivedMetrics(fullQualiaState);

        } catch (e) {
            logger.error(`QualiaCognitionLayer.update: Error: ${e.message}`, { stack: e.stack });
            // Reset awareness flags and metrics on critical error
            this.selfAware = false;
            this.hierarchicallyAware = false;
            this.diachronicallyAware = false;
            this.rhythmicallyAware = false;
            this.emergentAware = false;
            this.intentionality_F = 0;
            this.feel_F = 0;
            this.overallCoherence = 0;
            this.baseSheaf.ready = false; // Indicate base sheaf might be in an invalid state
        }
    }

    /**
     * Saves the current state of the QualiaCognitionLayer instance.
     *
     */
    saveState() {
        return {
            maxIter: this.maxIter, fixedPointEps: this.fixedPointEps, tau: this.tau,
            equalizerEps: this.equalizerEps, flowBufferSize: this.flowBufferSize,
            delta: this.delta, tau_persist: this.tau_persist, omega: this.omega,
            theta_k: this.theta_k, tau_floq: this.tau_floq,
            phiBase: this.phiBase, gestaltBase: this.gestaltBase,

            phiHistory: this.phiHistory.getAll(),
            gestaltHistory: this.gestaltHistory.getAll(),
            inconsistencyHistory: this.inconsistencyHistory.getAll(),
            cochainHistory: this.cochainHistory.getAll(),
            flowHistory: this.flowHistory.getAll(),
            persistenceDiagram: this.persistenceDiagram,
            floquetPD: this.floquetPD,

            phi: this.phi, h1Dimension: this.h1Dimension, gestaltUnity: this.gestaltUnity,
            stability: this.stability, inconsistency: this.inconsistency, feel_F: this.feel_F,
            intentionality_F: this.intentionality_F, cup_product_intensity: this.cup_product_intensity,
            structural_sensitivity: this.structural_sensitivity, coherence: this.coherence,
            overallCoherence: this.overallCoherence, emergentAware: this.emergentAware,

            selfAware: this.selfAware, hierarchicallyAware: this.hierarchicallyAware,
            diachronicallyAware: this.diachronicallyAware, rhythmicallyAware: this.rhythmicallyAware,

            qInput: this.qInput,
            currentCochain: this.currentCochain,
        };
    }

    /**
     * Loads a previously saved state into the QualiaCognitionLayer instance.
     *
     */
    loadState(state) {
        if (!state) {
            logger.warn('QualiaCognitionLayer.loadState: No state provided to load.');
            return;
        }

        try {
            this.maxIter = state.maxIter ?? this.maxIter;
            this.fixedPointEps = state.fixedPointEps ?? this.fixedPointEps;
            this.tau = state.tau ?? this.tau;
            this.equalizerEps = state.equalizerEps ?? this.equalizerEps;
            this.flowBufferSize = state.flowBufferSize ?? this.flowBufferSize;
            this.delta = state.delta ?? this.delta;
            this.tau_persist = state.tau_persist ?? this.tau_persist;
            this.omega = state.omega ?? this.omega;
            this.theta_k = state.theta_k ?? this.theta_k;
            this.tau_floq = state.tau_floq ?? this.tau_floq;
            this.phiBase = state.phiBase ?? this.phiBase;
            this.gestaltBase = state.gestaltBase ?? this.gestaltBase;

            this.phiHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
            if (Array.isArray(state.phiHistory)) state.phiHistory.forEach(item => this.phiHistory.push(item));

            this.gestaltHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
            if (Array.isArray(state.gestaltHistory)) state.gestaltHistory.forEach(item => this.gestaltHistory.push(item));

            this.inconsistencyHistory = new CircularBuffer(this.baseSheaf.stalkHistorySize);
            if (Array.isArray(state.inconsistencyHistory)) state.inconsistencyHistory.forEach(item => this.inconsistencyHistory.push(item));

            this.cochainHistory = new CircularBuffer(20);
            if (Array.isArray(state.cochainHistory)) state.cochainHistory.forEach(item => this.cochainHistory.push(item));

            this.flowHistory = new CircularBuffer(this.flowBufferSize);
            if (Array.isArray(state.flowHistory)) state.flowHistory.forEach(item => this.flowHistory.push(item));

            this.persistenceDiagram = state.persistenceDiagram || { births: [], deaths: [] };
            this.floquetPD = state.floquetPD || { births: [], phases: [], deaths: [] };

            this.phi = state.phi ?? this.phi;
            this.h1Dimension = state.h1Dimension ?? this.h1Dimension;
            this.gestaltUnity = state.gestaltUnity ?? this.gestaltUnity;
            this.stability = state.stability ?? this.stability;
            this.inconsistency = state.inconsistency ?? this.inconsistency;
            this.feel_F = state.feel_F ?? this.feel_F;
            this.intentionality_F = state.intentionality_F ?? this.intentionality_F;
            this.cup_product_intensity = state.cup_product_intensity ?? this.cup_product_intensity;
            this.structural_sensitivity = state.structural_sensitivity ?? this.structural_sensitivity;
            this.coherence = state.coherence ?? this.coherence;
            this.overallCoherence = state.overallCoherence ?? this.overallCoherence;
            this.emergentAware = state.emergentAware ?? this.emergentAware;

            this.selfAware = state.selfAware ?? this.selfAware;
            this.hierarchicallyAware = state.hierarchicallyAware ?? this.hierarchicallyAware;
            this.diachronicallyAware = state.diachronicallyAware ?? this.diachronicallyAware;
            this.rhythmicallyAware = state.rhythmicallyAware ?? this.rhythmicallyAware;

            this.qInput = state.qInput ?? this.qInput;
            const expectedCochainDim = this.baseSheaf.graph.vertices.length * this.baseSheaf.qDim;
            if (this.baseSheaf._isValidStateVec(state.currentCochain) && state.currentCochain.length === expectedCochainDim) {
                this.currentCochain = state.currentCochain;
            } else {
                logger.warn('QualiaCognitionLayer.loadState: currentCochain from state is invalid or mismatched. Re-initializing to zeros.');
                this.currentCochain = zeroMatrix(this.baseSheaf.graph.vertices.length, this.baseSheaf.qDim);
            }

            this.R_star = this._buildRecursiveGluing(); // Re-initialize R_star
            logger.info('QualiaCognitionLayer.loadState: State loaded successfully.');
        } catch (e) {
            logger.error(`QualiaCognitionLayer.loadState: Error loading state: ${e.message}`, { stack: e.stack });
        }
    }
}
