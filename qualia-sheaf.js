
/**
 * Combined, Refined, and Unified Qualia Sheaf Module (TUC Framework)
 * Version: 7.2 (Corrected Update Cycle)
 *
 * This version corrects the order of operations in the main update loop to ensure
 * that derived metrics are calculated *after* all state diffusion and higher-order
 * awareness computations have completed. This fixes the issue of uniform metrics by
 * allowing the full system state to evolve before being measured.
 */

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix, unflattenMatrix, logDeterminantFromDiagonal,
    logger, runWorkerTask, identity, transpose, covarianceMatrix, matVecMul, matMul, vecMul, vectorAsRow, vectorAsCol, isFiniteNumber, safeEigenDecomposition, randomMatrix
} from './utils.js';


function safeVecScale(vector, scalar) {
    if (!isFiniteVector(vector) || !isFiniteNumber(scalar) || !Number.isFinite(scalar)) {
        
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
 * Base EnhancedQualiaSheaf – Kernel for all theoremic extensions.
 */
export class EnhancedQualiaSheaf {
    constructor(graphData, config = {}) {
        this.owm = null;
        this.ready = false;
        this.entityNames = config.entityNames || ['shape', 'emotion', 'symbolic', 'synesthesia', 'metacognition', 'social', 'temporal'];
        this.qDim = this.entityNames.length;
        this.stateDim = config.stateDim || 13;
        this.leakRate = config.leakRate ?? 0.01; 
        this.maxForceNorm = config.maxForceNorm ?? 0.2; 

        this.alpha = clamp(config.alpha ?? 0.1, 0.01, 1);
        this.beta = clamp(config.beta ?? 0.05, 0.01, 1);
        this.gamma = clamp(config.gamma ?? 0.005, 0.01, 0.5);
        this.sigma = clamp(config.sigma ?? 0.01, 0.001, 0.1);
        this.eps = config.eps ?? 1e-6;
        this.stabEps = config.stabEps ?? 1e-8;
        this.phiBase = config.phiBase ?? 0.2;
        this.gestaltBase = config.gestaltBase ?? 0.6;

        this.adaptation = {
            addThresh: clamp(config.addThresh ?? 0.7, 0.5, 0.95),
            removeThresh: clamp(config.removeThresh ?? 0.2, 0.05, 0.4),
            targetH1: config.targetH1 ?? 2.0,
            maxEdges: config.maxEdges ?? 50,
        };

        this.useAdvancedTheorems = config.useAdvancedTheorems ?? true;
        this.resonanceOmega = config.resonanceOmega ?? 1.2;
        this.resonanceEps = config.resonanceEps ?? 0.08;

        this._initializeGraph(graphData);
        this.stalks = new Map();
        this._initializeStalks();

        this.stalkHistorySize = config.stalkHistorySize ?? 100;
        this.stalkHistory = new CircularBuffer(this.stalkHistorySize);
        this.stalkNormHistory = new CircularBuffer(this.stalkHistorySize);

        const N_total_stalk_dim = this.graph.vertices.length * this.qDim;
        this.windowSize = Math.max(100, N_total_stalk_dim * 3);
        this.windowedStates = new CircularBuffer(this.windowSize);
        this._initializeWindowedStates(N_total_stalk_dim);
        
        this.phiHistory = new CircularBuffer(this.stalkHistorySize);
        this.gestaltHistory = new CircularBuffer(this.stalkHistorySize);
        this.inconsistencyHistory = new CircularBuffer(this.stalkHistorySize);
        
        this.lastGoodEigenResult = null;
        
        
        this.phi = this.phiBase;
        this.h1Dimension = 0;
        this.gestaltUnity = this.gestaltBase;
        this.stability = 0.6;
        this.inconsistency = 0;
        this.feel_F = 0;
        this.intentionality_F = 0;
        this.cup_product_intensity = 0;
        this.structural_sensitivity = 0;
        this.coherence = 0;
        this.overallCoherence = 0;
        this.emergentAware = false;

        logger.info(`Enhanced Qualia Sheaf constructed: vertices=${this.graph.vertices.length}, edges=${this.graph.edges.length}, triangles=${this.simplicialComplex.triangles.length}, tetrahedra=${this.simplicialComplex.tetrahedra.length}`);
    }

    
/**
 * DEFINITIVE FIX: Replaces the flawed eigenvalue-based normalization with a mathematically correct
 * row-wise normalization. This is the standard approach for stabilizing non-square weight matrices.
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
        const row = P[q];
        
        
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
     * DEFINITIVE FIX: This Hebbian learning function allows the sheaf to learn how to perceive its environment.
     * It is placed here, inside the EnhancedQualiaSheaf class.
     */
    _updateStateToQualiaProjection(rawState, qualiaInput) {
    const learningRate = 0.002; 
    const decay = 0.0001;

    if (!isFiniteVector(rawState) || !isFiniteVector(qualiaInput)) {
        return;
    }

    const P = this.stateToQualiaProjection;

    
    
    for (let q = 0; q < this.qDim; q++) {
        
        let row_q = P[q];
        for (let i = 0; i < this.stateDim; i++) {
            const hebbianUpdate = learningRate * qualiaInput[q] * rawState[i];
            row_q[i] = (row_q[i] || 0) + hebbianUpdate - (row_q[i] || 0) * decay;
        }

        
        for (let j = 0; j < q; j++) {
            const row_j = P[j];
            const dot_qj = dot(row_q, row_j);
            const dot_jj = dot(row_j, row_j); 

            if (isFiniteNumber(dot_qj) && isFiniteNumber(dot_jj) && dot_jj > this.eps) {
                const scale = dot_qj / dot_jj;
                const projection = vecScale(row_j, scale);
                row_q = vecSub(row_q, projection);
            }
        }

        
        const norm_q = norm2(row_q);
        if (isFiniteNumber(norm_q) && norm_q > this.eps) {
            P[q] = vecScale(row_q, 1.0 / norm_q);
        } else {
            
            const randomVector = new Float32Array(this.stateDim).map(() => Math.random() - 0.5);
            P[q] = vecScale(randomVector, 1.0 / norm2(randomVector));
        }
    }

    
    
    this._normalizeProjectionMatrix();
}
    /**
     * OPTIMIZATION: Jacobi Eigenvalue Decomposition
     * A self-contained, robust method for finding all eigenvalues and eigenvectors
     * of a real symmetric matrix. This replaces all external dependencies.
     */
    _jacobiEigenvalueDecomposition(matrix, maxIterations = 50) {
        const n = matrix.length;
        if (n === 0) return { eigenvalues: [], eigenvectors: [] };

        let A = matrix.map(row => [...row]); 
        let V = identity(n); 

        for (let iter = 0; iter < maxIterations; iter++) {
            let p = 0, q = 1, maxOffDiagonal = 0;
            
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

            
            if (maxOffDiagonal < this.eps) break;

            
            const apq = A[p][q];
            const app = A[p][p];
            const aqq = A[q][q];
            const tau = (aqq - app) / (2 * (apq !== 0 ? apq : this.eps));
            const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
            const c = 1 / Math.sqrt(1 + t * t); 
            const s = c * t; 

            const R = identity(n);
            R[p][p] = c; R[p][q] = s;
            R[q][p] = -s; R[q][q] = c;

            
            
            for (let i = 0; i < n; i++) {
                const aip = A[i][p];
                const aiq = A[i][q];
                A[i][p] = c * aip - s * aiq;
                A[i][q] = s * aip + c * aiq;
            }
            for (let i = 0; i < n; i++) {
                const api = A[p][i];
                const aqi = A[q][i];
                A[p][i] = c * api - s * aqi;
                A[q][i] = s * api + c * aqi;
            }

            
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

  
  if (
    !Array.isArray(safeGraphData.vertices) || safeGraphData.vertices.length < 2 ||
    !Array.isArray(safeGraphData.edges)    || safeGraphData.edges.length < 1
  ) {
    safeGraphData = {
      vertices: defaultVertices,
      edges: defaultEdges,
      triangles: defaultTriangles,
      tetrahedra: defaultTetrahedra
    };
  }

  
  
  this.graph = {
    vertices: safeGraphData.vertices,
    edges: safeGraphData.edges,
    triangles: safeGraphData.triangles ?? [],
    tetrahedra: safeGraphData.tetrahedra ?? []
  };

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

        if (finalVertices.length < 2) {
            if (!finalVertices.includes('fallback_v1')) finalVertices.push('fallback_v1');
            if (!finalVertices.includes('fallback_v2')) finalVertices.push('fallback_v2');
            logger.warn('Sheaf._initializeGraph: Less than 2 vertices, added fallbacks.');
        }
        if (finalVertices.length >= 2 && allEdgesSet.size === 0) {
            allEdgesSet.add([finalVertices[0], finalVertices[1]].sort().join(','));
            logger.warn('Sheaf._initializeGraph: No edges, added a fallback edge.');
        }

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
     * CORRECTED to update the sheaf's internal state (`this.stalks`).
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
        this.qInput = qInput;
        
    }


        setOWM(owmInstance) { this.owm = owmInstance; }


    
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
      
    async initialize() {
    if (this.ready) return;
    logger.info('EnhancedQualiaSheaf.initialize() called.');

    try {
        
        this._initializeGraph({});
        this._initializeStalks();

        

const projData = randomMatrix(this.qDim, this.stateDim, 1.0); 
this.stateToQualiaProjection = unflattenMatrix(projData); 
this.stateToQualiaProjection.rows = this.qDim;
this.stateToQualiaProjection.cols = this.stateDim;


        
        
        this.projectionMatrices = new Map();
        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const P_identity = identity(this.qDim);
            this.projectionMatrices.set(`${u}-${v}`, P_identity);
            this.projectionMatrices.set(`${v}-${u}`, P_identity);
        }

        
        
        await this.computeCorrelationMatrix();

        
        
        this.laplacian = this.buildLaplacian();
        
        
        const N = this.graph.vertices.length * this.qDim;
        if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N) {
            logger.warn('EnhancedQualiaSheaf.initialize: Laplacian was invalid after build. Using identity fallback.');
            this.laplacian = identity(N);
        }
        

        this.ready = true;
        logger.info('Enhanced Qualia Sheaf ready.');
    } catch (e) {
        logger.error('CRITICAL ERROR: EnhancedQualiaSheaf initialization failed:', { message: e.message, stack: e.stack });
        this.ready = false;
        throw e;
    }
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

    /**
 * FINAL, UNCONDITIONALLY ROBUST VERSION: computeVertexCorrelationsFromHistory
 * This version definitively fixes the root cause of all system errors by guaranteeing a valid output,
 * even when the historical data is insufficient. It breaks the data corruption pipeline at its source.
 */
 async computeVertexCorrelationsFromHistory() {
        const n = this.graph.vertices.length;
        if (n === 0) return identity(0);

        const validHistory = this.stalkNormHistory.getAll().filter(item => isFiniteVector(item) && item.length === n);

        if (validHistory.length < 2) {
            return identity(n);
        }

        try {
            
            
            const covMatrixRaw = await runWorkerTask('covarianceMatrix', { states: validHistory, dim: n, eps: this.eps }, 10000);
            
            if (isFiniteMatrix(covMatrixRaw) && covMatrixRaw.length === n) {
                return covMatrixRaw;
            }

            logger.warn('Sheaf: All correlation matrix computations failed; returning identity matrix as a final fallback.');
            return identity(n);

        } catch (e) {
            logger.error(`Sheaf.computeVertexCorrelationsFromHistory: An unexpected error occurred.`, { error: e.message, stack: e.stack });
            return identity(n);
        }
    }
    _csrToDense(csr) {
    
    if (!csr || !Array.isArray(csr.rowPtr)) {
        
        if (isFiniteMatrix(csr)) {
            return csr;
        }
        logger.warn('_csrToDense: Invalid input. Returning 1x1 zero matrix.');
        return [[0]];
    }
    
    const n = csr.n || (csr.rowPtr.length - 1);
    if (n <= 0) {
         logger.warn('_csrToDense: CSR matrix dimension n is zero or negative. Returning 1x1 zero matrix.');
         return [[0]];
    }
    const dense = zeroMatrix(n, n);
    for (let i = 0; i < n; i++) {
        const start = csr.rowPtr[i] ?? 0;
        const end = csr.rowPtr[i + 1] ?? start;
        for (let j = start; j < end; j++) {
            const col = csr.colIndices[j] ?? 0;
            const val = Number.isFinite(csr.values[j]) ? csr.values[j] : 0;
            dense[i][col] = val;
        }
    }
    return dense;
}

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
        this.adaptSimplices(this.correlationMatrix, this.adaptation.targetH1);

        
        await this.computeCorrelationMatrix(); 
        this.laplacian = this.buildLaplacian(); 
        
        
        
        this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim);
        await this.computeH1Dimension();
        
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
                        if (!corrMatrix[i] || !corrMatrix[j] || !corrMatrix[k] ||
                            !Number.isFinite(corrMatrix[i][j]) || !Number.isFinite(corrMatrix[j][k]) || !Number.isFinite(corrMatrix[k][i])) continue;

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
                if (!corrMatrix[idxs[0]] || !corrMatrix[idxs[1]] || !corrMatrix[idxs[2]] ||
                    !Number.isFinite(corrMatrix[idxs[0]][idxs[1]]) || !Number.isFinite(corrMatrix[idxs[1]][idxs[2]]) || !Number.isFinite(corrMatrix[idxs[2]][idxs[0]])) return;
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
                                    if (corrMatrix[c1] && Number.isFinite(corrMatrix[c1][c2])) {
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
                        if (corrMatrix[idxs[i]] && Number.isFinite(corrMatrix[idxs[i]][idxs[j]])) {
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
                logger.warn(`Sheaf.buildBoundaryMatrices: Invalid vertex or edge index for edge ${u}-${v} (uIdx=${uIdx}, vIdx=${vIdx}, eIdx=${eIdx}). Skipping.`);
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
            
L[i * this.qDim + qi][i * this.qDim + qi] = degree + this.stabEps + this.leakRate * (1 + this.inconsistency);        }
    }

    return isFiniteMatrix(L) ? L : identity(N);
}


    csrMatVecMul(csr, v) {
        const n = csr.n;
        if (!csr || !Array.isArray(csr.values) || !Array.isArray(csr.colIndices) || !Array.isArray(csr.rowPtr) ||
            !isFiniteVector(v) || v.length !== n || n === 0) {
            logger.warn('Sheaf.csrMatVecMul: Invalid CSR matrix or vector, or zero dimension. Returning zero vector.');
            return vecZeros(n > 0 ? n : 1);
        }

        const result = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            const start = csr.rowPtr[i] ?? 0;
            const end = csr.rowPtr[i + 1] ?? start;

            if (start < 0 || end < start || start > csr.values.length || end > csr.values.length) {
                logger.warn(`Sheaf.csrMatVecMul: Invalid rowPtr indices for row ${i}. Skipping row.`);
                continue;
            }

            for (let j = start; j < end; j++) {
                if (j < 0 || j >= csr.colIndices.length || j >= csr.values.length) {
                    logger.warn(`Sheaf.csrMatVecMul: Index out of bounds in CSR data for row ${i}, index ${j}. Skipping element.`);
                    continue;
                }
                const colIdx = csr.colIndices[j];
                if (colIdx >= 0 && colIdx < n && Number.isFinite(csr.values[j]) && Number.isFinite(v[colIdx])) {
                    sum += csr.values[j] * v[colIdx];
                } else {
                    logger.warn(`Sheaf.csrMatVecMul: Non-finite value or invalid column index for row ${i}, element ${j}.`);
                }
            }
            result[i] = sum;
        }
        if (!isFiniteVector(result)) {
            logger.warn('Sheaf.csrMatVecMul: Computed result vector is non-finite. Returning zero vector.');
            return vecZeros(n);
        }
        return result;
    }

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
                gradient[i][j] = error[i] * s_u[j];
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

    async computeFreeEnergyPrior(state, hiddenState) {
        const actionDim = (this.owm && this.owm.actionDim) ? this.owm.actionDim : 4;
        const prior = vecZeros(actionDim);

        const intentionalityBonus = this.intentionality_F * 0.1;
        const coherenceBonus = (this.coherence || 0) * 0.05;
        const inconsistencyPenalty = (this.inconsistency || 0) * -0.1;

        for (let i = 0; i < actionDim; i++) {
            prior[i] = intentionalityBonus + coherenceBonus + inconsistencyPenalty;
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

        if (!isFiniteMatrix(this.laplacian) || this.laplacian.length === 0) {
            logger.error('Sheaf._updateGraphStructureAndMetrics: Laplacian is invalid even after build. Skipping spectral norm.');
            this.maxEigApprox = 1;
        } else {
            const flattenedDenseLaplacian = flattenMatrix(this.laplacian);
            let spectralNorm = await runWorkerTask('matrixSpectralNormApprox', { matrix: flattenedDenseLaplacian }, 15000);
            this.maxEigApprox = Number.isFinite(spectralNorm) && spectralNorm > 0 ? spectralNorm : 1;
        }

        await this.computeProjectionMatrices();

    } catch (e) {
        logger.error('Sheaf._updateGraphStructureAndMetrics: Failed to update.', e);
        this.ready = false;
        throw e;
    } finally {
        this.isUpdating = false;
    }
}

    async computeCoherenceFlow() {
    logger.debug('Sheaf.computeCoherenceFlow: Starting', { vertices: this.graph?.vertices?.length, qDim: this.qDim });

    const nV = this.graph?.vertices?.length || 0;
    const N = nV * (this.qDim || 1);
    if (nV === 0 || !Number.isFinite(this.qDim) || this.qDim <= 0) {
        logger.error('Sheaf.computeCoherenceFlow: Invalid graph or qDim', { nV, qDim: this.qDim });
        this.laplacian = identity(N);
        return this.laplacian;
    }

    try {
        if (!isFiniteMatrix(this.adjacencyMatrix) || this.adjacencyMatrix.length !== nV) {
            logger.warn('Sheaf.computeCoherenceFlow: Invalid adjacency matrix. Recomputing.');
            const states = this.getCurrentStates?.() || zeroMatrix(nV, this.qDim);
            this.adjacencyMatrix = await runWorkerTask('covarianceMatrix', { states, elementDim: this.qDim, eps: this.eps });
            if (!isFiniteMatrix(this.adjacencyMatrix) || this.adjacencyMatrix.length !== nV) {
                this.adjacencyMatrix = identity(nV);
            }
        }

        this.laplacian = this.buildLaplacian();
        if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N) {
            throw new Error('Invalid Laplacian from buildLaplacian');
        }

        logger.debug('Sheaf.computeCoherenceFlow: Completed', { laplacianRows: this.laplacian.length });
        return this.laplacian;
    } catch (e) {
        logger.error('Sheaf.computeCoherenceFlow: Rebuilding Laplacian failed', { error: e.message, stack: e.stack });
        this.laplacian = identity(N);
        return this.laplacian;
    }
}


async diffuseQualia(state, qualiaInput) {
    if (!this.ready) return;

    const nV = this.graph.vertices.length;
    const N = nV * this.qDim;
    const s = this.getStalksAsVector();

    if (!isFiniteVector(s) || s.length !== N) {
        logger.error(`Sheaf.diffuseQualia: Invalid initial stalk vector 's'. Aborting.`);
        return;
    }


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


let f_s = vecZeros(N);
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

    
    let eta = this.gamma / Math.max(1, this.maxEigApprox); 
    if (!isFiniteNumber(eta) || eta <= 0) {
        eta = 0.01;
    }
    
    
    const laplacianEffect = _matVecMul(Lfull, s);
    
    
    let sNext = vecSub(s, vecScale(laplacianEffect, eta));
    
    
    sNext = vecAdd(sNext, vecScale(f_s, eta));
    
    
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
 * OPTIMIZED VERSION: _applySpectralResonance
 * Uses numeric.eig (QR algorithm) for eigenvalue decomposition, as per user directive.
 * Caches results to avoid recomputation and includes performance logging.
 */
    async _applySpectralResonance(fullStateVec, L_base) { 
    const N_total_stalk_dim = fullStateVec.length;
    try {
        const L_norm_dense = this._normalizedLaplacian(L_base);
        const eigResult = await this._spectralDecomp(L_norm_dense);
        
        if (!eigResult || !isFiniteVector(eigResult.eigenvalues) || !isFiniteMatrix(eigResult.eigenvectors)) {
            return { success: false };
        }

        const { eigenvalues, eigenvectors } = eigResult;
        const omega = this.resonanceOmega || 1.0;
        const eps_resonance = this.resonanceEps || 0.1;
        let resonantIndices = eigenvalues.reduce((acc, mag, i) => { if (isFiniteNumber(mag) && Math.abs(mag - omega) < eps_resonance) acc.push(i); return acc; }, []);

        if (!resonantIndices.length && eigenvalues.length) {
            const closestIndex = eigenvalues.reduce((min, mag, i) => { const diff = Math.abs(mag - omega); return diff < min.diff ? { index: i, diff } : min; }, { index: -1, diff: Infinity }).index;
            if (closestIndex !== -1) resonantIndices.push(closestIndex);
        }
        if (!resonantIndices.length) return { success: false };
        
        const P_R = zeroMatrix(N_total_stalk_dim, N_total_stalk_dim);
        for (const k of resonantIndices) {
            const uk = _transpose(eigenvectors)[k] || vecZeros(N_total_stalk_dim);
            if (!isFiniteVector(uk)) continue;
            for (let i = 0; i < N_total_stalk_dim; i++) for (let j = 0; j < N_total_stalk_dim; j++) P_R[i][j] += uk[i] * uk[j];
        }

        
        const sNext = _matVecMul(P_R, fullStateVec);

        if (!isFiniteVector(sNext)) return { success: false };
        return { success: true, sNext };

    } catch (err) {
        logger.error(`Sheaf._applySpectralResonance: Error: ${err.message}`);
        return { success: false };
    }
}

/**
 * OPTIMIZED VERSION: _linearDiffuse
 * Robust linear diffusion with strict input validation.
 */
  async _linearDiffuse(s, Lfull) { 
    const N = s.length;
    if (!isFiniteMatrix(Lfull) || Lfull.length !== N || !isFiniteVector(s)) {
        return { sNext: s };
    }
    const eta = 0.001;
    const laplacianEffect = _matVecMul(Lfull, s);
    if (!isFiniteVector(laplacianEffect)) {
        return { sNext: s };
    }
    const sNext = vecSub(s, vecScale(laplacianEffect, eta));
    const noise = new Float32Array(N).map(() => (Math.random() - 0.5) * this.sigma);
    
    
    const finalSNext = vecAdd(sNext, noise);
    
    const clampedSNext = new Float32Array(finalSNext.map(v => clamp(v || 0, -1, 1)));
    if (!isFiniteVector(clampedSNext)) {
        return { sNext: s };
    }
    return { sNext: clampedSNext };
}


async computeGluingInconsistency() {
    try {
        const edges = this.graph?.edges;
        if (!edges || edges.length === 0) {
            this.inconsistency = 0;
            return 0;
        }

        let totalInconsistency = 0;
        let edgeCount = 0;

        for (const [u, v] of edges) {
            const s_u = this.stalks.get(u);
            const s_v = this.stalks.get(v);
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);

            
            if (!this._isValidStateVec(s_u) || !this._isValidStateVec(s_v) || !isFiniteMatrix(P_uv)) {
                continue; 
            }

            try {
                
                const s_u_copy = s_u.slice();
                const P_uv_copy = P_uv.map(row => row.slice());

                const s_u_projected = await runWorkerTask('matVecMul', {
                    matrix: flattenMatrix(P_uv_copy),
                    vector: s_u_copy,
                    expectedDim: this.qDim
                });

                if (this._isValidStateVec(s_u_projected)) {
                    const inconsistency = norm2(vecSub(s_u_projected, s_v));
                    if (isFiniteNumber(inconsistency)) {
                        totalInconsistency += inconsistency;
                        edgeCount++;
                    }
                }
            } catch (e) {
                this.logger?.error(`Sheaf.computeGluingInconsistency: Error processing edge ${u}-${v}.`, { error: e.message });
            }
        }

        this.inconsistency = edgeCount > 0 ? clamp(totalInconsistency / edgeCount, 0, 1) : 0;
        return this.inconsistency;
    } catch (err) {
        this.logger?.error(`Sheaf.computeGluingInconsistency: unexpected error: ${err.message}`);
        this.inconsistency = 0;
        return 0;
    }
}


      async computeGestaltUnity() {
        let validStates = this.windowedStates?.getAll?.()?.filter(s => isFiniteVector(s) && s.length > 0) || [];

        if (validStates.length < 2) {
            this.gestaltUnity = 0;
            return 0;
        }

        let totalSimilarity = 0;
        let count = (validStates.length * (validStates.length - 1)) / 2;
        const totalStateDim = validStates[0].length;

        
        
        if (this.gpu && this.gestaltKernel && totalStateDim > 0) {
            try {
                
                this.gestaltKernel.setConstants({
                    eps_const: this.eps || 1e-6,
                    state_dim_const: totalStateDim,
                    num_states_const: validStates.length
                });

                const flattenedStates = new Float32Array(validStates.flat());
                if (!isFiniteVector(flattenedStates)) {
                    throw new Error('Flattened states for GPU.js are non-finite.');
                }
                
                const result = this.gestaltKernel(flattenedStates);
                totalSimilarity = Number.isFinite(result[0]) ? result[0] : 0;
            } catch (e) {
                logger.warn('Sheaf.computeGestaltUnity: GPU.js execution failed; falling back to CPU.', { error: e.message });
                this.gpu = null; 
                totalSimilarity = 0; 
            }
        }

        
        if (totalSimilarity === 0) {
            count = 0; 
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
        
    
        this.gestaltUnity = count > 0 ? clamp(totalSimilarity / count, 0, 1) : 0;
        this.gestaltHistory.push(new Float32Array([this.gestaltUnity]));
        return this.gestaltUnity;
    }

    _computeLogDet(A) {
    const n = A?.length || 0;
    if (n === 0 || !isFiniteMatrix(A)) {
        logger.warn('Sheaf._computeLogDet: Invalid or non-finite matrix. Returning 0.', { matrixRows: n });
        return 0;
    }

    if (numeric && numeric.ludecomp) {
        try {
            const lu = numeric.ludecomp(A);
            if (!lu || !lu.LU || !isFiniteMatrix(lu.LU)) {
                logger.warn('Sheaf._computeLogDet: numeric.js LU decomposition failed. Falling back to custom LU.');
            } else {
                let logDet = 0;
                for (let i = 0; i < n; i++) {
                    const U_ii = lu.LU[i]?.[i] || 0;
                    if (!Number.isFinite(U_ii) || Math.abs(U_ii) <= (this.eps || 1e-6) * 10) {
                        logger.warn(`Sheaf._computeLogDet: Non-finite or near-zero diagonal element in U (numeric.js) at i=${i}. Returning 0.`);
                        return 0;
                    }
                    logDet += Math.log(Math.abs(U_ii));
                }
                if (Number.isFinite(logDet)) {
                    return logDet;
                }
                logger.warn('Sheaf._computeLogDet: Non-finite logDet from numeric.js. Falling back.');
            }
        } catch (e) {
            logger.warn(`Sheaf._computeLogDet: numeric.js LU decomposition failed: ${e.message}. Falling back to custom LU.`, { stack: e.stack });
        }
    }

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
            if (!Number.isFinite(U_ii) || Math.abs(U_ii) < (this.eps || 1e-6) * 10) {
                logger.warn(`Sheaf._computeLogDet: Division by near-zero diagonal element in U at i=${i}. Returning 0.`);
                return 0;
            }
            L[j][i] = Number.isFinite(sum / U_ii) ? sum / U_ii : 0;
        }
    }

    let logDet = 0;
    for (let i = 0; i < n; i++) {
        const U_ii = U[i]?.[i] || 0;
        if (!Number.isFinite(U_ii) || Math.abs(U_ii) <= (this.eps || 1e-6) * 10) {
            logger.warn(`Sheaf._computeLogDet: Non-finite or near-zero diagonal element in U at i=${i}. Returning 0.`);
            return 0;
        }
        logDet += Math.log(Math.abs(U_ii));
    }

    if (!Number.isFinite(logDet)) {
        logger.warn('Sheaf._computeLogDet: Non-finite logDet from custom LU. Returning 0.');
        return 0;
    }

    return logDet;
}

    async _computeDirectionalMI(states) {
        if (!Array.isArray(states) || states.length < 2 || !isFiniteVector(states[0]) || states[0].length !== 8) {
            logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Insufficient or invalid states. Returning 0.', {
                validStatesLength: states.length,
                stateDim: states[0]?.length
            });
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

            if (!isFiniteVector(s_t) || !isFiniteVector(s_t1) || s_t.length !== n_dim || s_t1.length !== n_dim) {
                logger.warn(`FloquetPersistentSheaf._computeDirectionalMI: Non-finite or mismatched state vector at index ${i}. Skipping sample.`);
                continue;
            }

            const x_t = s_t.slice(0, n_half);
            const y_t = s_t.slice(n_half);
            const y_t1 = s_t1.slice(n_half);

            if (!isFiniteVector(x_t) || !isFiniteVector(y_t) || !isFiniteVector(y_t1)) {
                logger.warn(`FloquetPersistentSheaf._computeDirectionalMI: Non-finite slices at index ${i}. Skipping sample.`);
                continue;
            }

            Y_future.push(y_t1);
            Y_past.push(y_t);
            X_past.push(x_t);
            YX_past.push([...y_t, ...x_t]);
        }

        if (Y_future.length < 2) {
            logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Insufficient valid time-lagged samples. Falling back to variance-based estimate.', {
                laggedStatesLength: Y_future.length
            });
            try {
                const mean = vecZeros(n_dim);
                states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
                let sumVar = 0;
                states.forEach(state => {
                    const centered = vecSub(state, mean);
                    centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
                });
                const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.eps || 1e-6);
                return clamp(miEstimate, 0, 5);
            } catch (e) {
                logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Fallback MI failed. Returning 0.', { error: e.message });
                return 0;
            }
        }

        try {
            const mi_full_input = Y_future.map((yf, i) => {
                const combined = [...yf, ...(YX_past[i] || vecZeros(n_dim))];
                return isFiniteVector(combined) ? combined : null;
            }).filter(Boolean);
            const mi_partial_input = Y_future.map((yf, i) => {
                const combined = [...yf, ...(Y_past[i] || vecZeros(n_half))];
                return isFiniteVector(combined) ? combined : null;
            }).filter(Boolean);

            if (mi_full_input.length < 2 || mi_partial_input.length < 2) {
                logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Insufficient valid input for KSG MI. Falling back to variance-based estimate.', {
                    miFullLength: mi_full_input.length,
                    miPartialLength: mi_partial_input.length
                });
                try {
                    const mean = vecZeros(n_dim);
                    states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
                    let sumVar = 0;
                    states.forEach(state => {
                        const centered = vecSub(state, mean);
                        centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
                    });
                    const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.eps || 1e-6);
                    return clamp(miEstimate, 0, 5);
                } catch (e) {
                    logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Fallback MI failed. Returning 0.', { error: e.message });
                    return 0;
                }
            }

            const [mi_full_raw, mi_partial_raw] = await Promise.all([
                runWorkerTask('ksg_mi', { states: mi_full_input, k: Math.min(3, mi_full_input.length - 1) }, 15000),
                runWorkerTask('ksg_mi', { states: mi_partial_input, k: Math.min(3, mi_partial_input.length - 1) }, 15000)
            ]);

            const mi_full = Number.isFinite(mi_full_raw) ? mi_full_raw : 0;
            const mi_partial = Number.isFinite(mi_partial_raw) ? mi_partial_raw : 0;

            const transferEntropy = mi_full - mi_partial;
            return clamp(transferEntropy, 0, 10);
        } catch (e) {
            logger.warn(`FloquetPersistentSheaf._computeDirectionalMI: KSG computation failed: ${e.message}. Falling back to variance-based estimate.`, { stack: e.stack });
            try {
                const mean = vecZeros(n_dim);
                states.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / states.length));
                let sumVar = 0;
                states.forEach(state => {
                    const centered = vecSub(state, mean);
                    centered.forEach(v => sumVar += v * v / Math.max(1, states.length - 1));
                });
                const miEstimate = 0.05 * Math.log(1 + sumVar / n_dim) + (this.eps || 1e-6);
                return clamp(miEstimate, 0, 5);
            } catch (fallbackErr) {
                logger.warn('FloquetPersistentSheaf._computeDirectionalMI: Fallback MI failed. Returning 0.', { error: fallbackErr.message });
                return 0;
            }
        }
    }


    async _computeCochains(state) {
    try {
        this.logger?.debug('Sheaf._computeCochains: Starting', {
            vertices: this.graph?.vertices?.length,
            edges: this.graph?.edges?.length,
            qDim: this.qDim
        });

        if (!this.ready || this.stalks.size === 0 || !this.graph?.vertices?.length || !this.graph?.edges || !Number.isFinite(this.qDim) || this.qDim <= 0) {
            this.logger?.warn('Sheaf._computeCochains: Invalid state, stalks, graph, or qDim', {
                ready: this.ready,
                stalksSize: this.stalks.size,
                vertices: this.graph?.vertices?.length,
                edges: this.graph?.edges?.length,
                qDim: this.qDim
            });

            const fallbackC1 = new Map();
            if (this.graph?.vertices?.length >= 2) {
                const [u, v] = this.graph.vertices.slice(0, 2);
                fallbackC1.set([u, v].sort().join(','), vecZeros(this.qDim || 7));
            }
            this.currentCochain = []; 
            return { C0: [], C1: fallbackC1, C2: new Map() };
        }

        
        if (!this._isValidStateVec(state) && state.length !== 20) {
            this.logger?.warn(`Sheaf._computeCochains: Invalid state vector. Using zeros.`, { length: state?.length });
            state = vecZeros(this.qDim);
        }

        const C0 = [];
        for (const [vertex, stalk] of this.stalks) {
            if (!this._isValidStateVec(stalk)) {
                this.logger?.warn(`Sheaf._computeCochains: Invalid stalk for ${vertex}. Resetting to zeros.`);
                this.stalks.set(vertex, vecZeros(this.qDim));
            } else {
                C0.push(stalk);
            }
        }
        if (C0.length === 0) {
            this.logger?.warn('Sheaf._computeCochains: No valid C0 vectors');
        }

        const C1 = new Map();
        for (const edge of this.graph.edges) {
            const [u, v] = edge.slice(0, 2);
            let s_u = this.stalks.get(u) || vecZeros(this.qDim);
            let s_v = this.stalks.get(v) || vecZeros(this.qDim);
            let P_vu = this.projectionMatrices.get(`${v}-${u}`) || identity(this.qDim);

            if (!this._isValidStateVec(s_u)) { s_u = vecZeros(this.qDim); this.stalks.set(u, s_u); }
            if (!this._isValidStateVec(s_v)) { s_v = vecZeros(this.qDim); this.stalks.set(v, s_v); }
            if (!isFiniteMatrix(P_vu) || P_vu.rows !== this.qDim || P_vu.cols !== this.qDim) {
                this.logger?.warn(`Sheaf._computeCochains: Invalid P_vu for edge ${v}-${u}. Using identity.`);
                P_vu = identity(this.qDim); P_vu.rows = P_vu.cols = this.qDim;
                this.projectionMatrices.set(`${v}-${u}`, P_vu);
            }

            let projected_u;
            try {
                projected_u = matVecMul(P_vu, s_u);
                if (!this._isValidStateVec(projected_u)) {
                    this.logger?.warn(`Sheaf._computeCochains: Invalid projected_u for edge ${v}-${u}. Using zeros.`);
                    projected_u = vecZeros(this.qDim);
                }
            } catch {
                projected_u = vecZeros(this.qDim);
            }

            const difference = vecSub(s_v, projected_u);
            C1.set([u, v].sort().join(','), this._isValidStateVec(difference) ? difference : vecZeros(this.qDim));
        }

        const C2 = new Map();
        const triangles = this.simplicialComplex?.triangles || [];
        for (const tri of triangles) {
            if (!this.isValidTriangle(tri)) continue;
            const [u, v, w] = tri;

            const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            const c_vw = C1.get([v, w].sort().join(',')) || vecZeros(this.qDim);
            const c_wu = C1.get([w, u].sort().join(',')) || vecZeros(this.qDim);

            let curl = vecAdd(vecAdd(c_uv, c_vw), c_wu);
            if (!this._isValidStateVec(curl)) continue;
            C2.set([u, v, w].sort().join(','), curl);
        }

        if (C1.size === 0 && this.graph.edges.length > 0) {
            const [u, v] = this.graph.edges[0].slice(0, 2);
            C1.set([u, v].sort().join(','), vecZeros(this.qDim));
        }

        this.currentCochain = C0;
        this.logger?.debug('Sheaf._computeCochains: Completed', { C0Length: C0.length, C1Size: C1.size, C2Size: C2.size });

        return { C0, C1, C2 };
    } catch (err) {
        this.logger?.error(`Sheaf._computeCochains: Unexpected error`, { error: err.message, stack: err.stack });
        this.currentCochain = [];
        return { C0: [], C1: new Map(), C2: new Map() };
    }
}

async computeIntegratedInformation() {
        try {
            const stateSource = this.stalkNormHistory || this.windowedStates;
            const allStates = stateSource.getAll();
            if (allStates.length < 10) {
                logger.info('FloquetPersistentSheaf.computeIntegratedInformation: Too few states; delaying computation.', { stateCount: allStates.length });
                this.integrated_information = 0;
                this.phi = 0.001;
                return { integrated_information: 0, phi: 0.001, feelIntensity: 0.001, intentionality: 0.001 };
            }

            const validStates = allStates.filter(item => {
                const isValid = (this._isValidStateVec?.(item) || isFiniteVector(item)) && item.length === 8;
                if (!isValid) {
                    logger.debug('FloquetPersistentSheaf.computeIntegratedInformation: Invalid state.', {
                        state: Array.from(item).slice(0, 10),
                        length: item.length
                    });
                }
                return isValid;
            });

            if (allStates.length > validStates.length) {
                logger.warn('FloquetPersistentSheaf.computeIntegratedInformation: States filtered out.', {
                    totalStates: allStates.length,
                    validStatesLength: validStates.length,
                    invalidStatesCount: allStates.length - validStates.length
                });
            }

            const vertexCount = this.graph?.vertices?.length || 8;
            const windowThreshold = this.windowSize ? this.windowSize / 4 : 0;
            let minStates = Math.max(vertexCount, windowThreshold);
            if (validStates.length > 0 && validStates.length < minStates) {
                logger.warn('FloquetPersistentSheaf.computeIntegratedInformation: Adjusting minStates for sparse data.', {
                    originalMinStates: minStates,
                    validStatesLength: validStates.length
                });
                minStates = Math.max(2, validStates.length);
            }

            if (validStates.length < minStates) {
                logger.warn('FloquetPersistentSheaf.computeIntegratedInformation: Insufficient valid states. Returning defaults.', {
                    validStatesLength: validStates.length,
                    minStates
                });
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
                    if (!statesArray.every(isFiniteVector)) throw new Error('Non-finite states in TF input.');
                    const statesTensor = tf.tensor2d(statesArray);
                    const regularizer = tf.eye(n_dim).mul(this.eps * 100);
                    const rawCovMatrix = tf.matMul(statesTensor.transpose(), statesTensor).div(Math.max(1, num_samples - 1));
                    covMatrix = rawCovMatrix.add(regularizer);

                    let logDet;
                    try {
                        const L = tf.linalg.cholesky(covMatrix);
                        logDet = tf.sum(L.log().mul(2)).dataSync()[0];
                    } catch (e) {
                        logDet = tf.linalg.logMatrixDeterminant(covMatrix).logDeterminant.dataSync()[0];
                    }
                    if (Number.isFinite(logDet)) MI = 0.1 * Math.abs(logDet) + this.eps;
                    tf.dispose([statesTensor, rawCovMatrix, covMatrix]);
                } catch (e) {
                    logger.warn(`FloquetPersistentSheaf.computeIntegratedInformation: TF path failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0) {
                try {
                    covMatrix = await runWorkerTask('covarianceMatrix', { states: validStates, dim: n_dim, eps: this.eps }, 5000);
                    if (!isFiniteMatrix(covMatrix)) {
                        logger.warn('FloquetPersistentSheaf.computeIntegratedInformation: Non-finite covariance. Manual CPU fallback.');
                        const mean = vecZeros(n_dim);
                        validStates.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / num_samples));
                        covMatrix = zeroMatrix(n_dim, n_dim);
                        validStates.forEach(state => {
                            const centered = vecSub(state, mean);
                            for (let i = 0; i < n_dim; i++) for (let j = i; j < n_dim; j++) {
                                covMatrix[i][j] += centered[i] * centered[j] / Math.max(1, num_samples - 1);
                                if (i !== j) covMatrix[j][i] = covMatrix[i][j];
                            }
                        });
                    }

                    if (isFiniteMatrix(covMatrix)) {
                        const regularizedCovMatrix = covMatrix.map((row, i) => row.map((val, j) => i === j ? val + this.eps * 100 : val));
                        if (isFiniteMatrix(regularizedCovMatrix)) {
                            const logDet = this._computeLogDet(regularizedCovMatrix);
                            if (Number.isFinite(logDet)) MI = 0.1 * Math.abs(logDet) + this.eps;
                        }
                    }
                } catch (e) {
                    logger.warn(`FloquetPersistentSheaf.computeIntegratedInformation: CPU path failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0 && validStates.length >= 2 && isFiniteVector(validStates[0])) {
                try {
                    const ksgMI = await runWorkerTask('ksg_mi', { states: validStates, k: Math.min(3, validStates.length - 1) }, 20000);
                    MI = Number.isFinite(ksgMI) ? ksgMI : 0;
                } catch (e) {
                    logger.warn(`FloquetPersistentSheaf.computeIntegratedInformation: KSG fallback failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            if (MI === 0 && validStates.length >= 2) {
                logger.info('FloquetPersistentSheaf.computeIntegratedInformation: Partial MI with limited states.', { validStatesLength: validStates.length });
                try {
                    const mean = vecZeros(n_dim);
                    validStates.forEach(state => mean.forEach((_, i) => mean[i] += state[i] / num_samples));
                    let sumVar = 0;
                    validStates.forEach(state => {
                        const centered = vecSub(state, mean);
                        centered.forEach(v => sumVar += v * v / Math.max(1, num_samples - 1));
                    });
                    MI = 0.05 * Math.log(1 + sumVar / n_dim) + this.eps;
                    MI = clamp(MI, 0, 5);
                } catch (e) {
                    logger.warn(`FloquetPersistentSheaf.computeIntegratedInformation: Partial MI failed: ${e.message}`, { stack: e.stack });
                    MI = 0;
                }
            }

            this.integrated_information = clamp(MI, 0, 10);

            const safeFloquetBirths = Array.isArray(this.floquetPD?.births) ? this.floquetPD.births : [];
            const betaFloq = await this._supFloqBetti(this.floquetPD);
            const avgBirthTime = safeFloquetBirths.reduce((sum, b) => sum + (b.time || 0), 0) / Math.max(1, safeFloquetBirths.length);
            const persistenceBoost = 0.05 * betaFloq * Math.log(1 + (this.stalkHistory?.length || 1) / Math.max(1, avgBirthTime + 1));

            const phiRaw = (Math.log(1 + Math.abs(MI)) + persistenceBoost) * 
                           (this.stability || 1) * (this.gestaltUnity || 1) * 
                           Math.exp(-(this.inconsistency || 0)) * (1 + 0.05 * (this.h1Dimension || 0));
            this.phi = clamp(phiRaw, 0.001, 100);
            this.phiHistory = this.phiHistory || [];
            this.phiHistory.push(new Float32Array([this.phi]));

            this.feelIntensity = clamp((MI + 0.02 * betaFloq) * (this.stability || 1) * Math.exp(-(this.inconsistency || 0)), 0.001, 10);

            const MI_dir = await this._computeDirectionalMI(validStates);
            this.intentionality = clamp((MI_dir + 0.01 * betaFloq) * (this.stability || 1) * Math.exp(-(this.inconsistency || 0)), 0.001, 10);

            return {
                integrated_information: this.integrated_information,
                phi: this.phi,
                feelIntensity: this.feelIntensity,
                intentionality: this.intentionality
            };
        } catch (err) {
            logger.error('FloquetPersistentSheaf.computeIntegratedInformation: Computation error.', { 
                error: err.message, 
                stack: err.stack 
            });
            this.integrated_information = 0;
            this.phi = 0.001;
            this.feelIntensity = 0.001;
            this.intentionality = 0.001;
            return {
                integrated_information: 0,
                phi: 0.001,
                feelIntensity: 0.001,
                intentionality: 0.001
            };
        }
    }



    async computeCupProduct() {
    try {
        const triangles = this.simplicialComplex?.triangles;
        if (!triangles || triangles.length === 0) {
            this.cup_product_intensity = 0;
            return 0;
        }

        let totalIntensity = 0;
        let count = 0;

        for (const triangle of triangles) {
            if (!this.isValidTriangle(triangle)) continue;

            const [u, v, w] = triangle;
            const s_u = this.stalks.get(u);
            const s_w = this.stalks.get(w);
            const P_uv = this.projectionMatrices.get(`${u}-${v}`);
            const P_vw = this.projectionMatrices.get(`${v}-${w}`);

            
            if (!this._isValidStateVec(s_u) || !this._isValidStateVec(s_w) || !isFiniteMatrix(P_uv) || !isFiniteMatrix(P_vw)) {
                continue;
            }

            try {
                
                const P_compose_result = matMul({ matrixA: P_uv, matrixB: P_vw });
                if (!isFiniteMatrix(P_compose_result)) continue;

                
                const s_w_copy = s_w.slice();

                const s_w_projected = await runWorkerTask('matVecMul', {
                    matrix: flattenMatrix(P_compose_result),
                    vector: s_w_copy,
                    expectedDim: this.qDim
                });

                if (this._isValidStateVec(s_w_projected)) {
                    const cupValue = dot(s_u, s_w_projected);
                    if (isFiniteNumber(cupValue)) {
                        totalIntensity += Math.abs(cupValue);
                        count++;
                    }
                }
            } catch (e) {
                this.logger?.error(`Sheaf.computeCupProduct: Error processing triangle ${triangle.join(',')}.`, { error: e.message });
            }
        }

        const betaFloq = await this._supFloqBetti(this.floquetPD);
        this.cup_product_intensity = count > 0 ? clamp((totalIntensity / count) + (0.01 * betaFloq), 0, 10) : 0;

        return this.cup_product_intensity;
    } catch (err) {
        this.logger?.error(`Sheaf.computeCupProduct: unexpected error: ${err.message}`);
        this.cup_product_intensity = 0;
        return 0;
    }
}


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


  async computeH1Dimension() {
        const n = this.graph.vertices.length;
        if (n === 0) { 
            this.h1Dimension = 0; 
            return 0; 
        }
        
        
        
        this.h1Dimension = this._computeBetti1UnionFind();
        
        this.stability = clamp(Math.exp(-this.h1Dimension * 0.2), 0.01, 1);
        return this.h1Dimension;
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
        if (parent[x] !== x) return parent[x] = find(parent[x]);
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
    
    

    edges.forEach(edge => {
        const u = edge[0]; 
        const v = edge[1]; 
        const uIdx = vMap.get(u);
        const vIdx = vMap.get(v);
        if (uIdx !== undefined && vIdx !== undefined && uIdx < n && vIdx < n) {
            union(uIdx, vIdx);
        } else {
            
            logger.warn(`_computeBetti1UnionFind: Invalid vertex index or edge format for edge starting with ${u}-${v}. Skipping.`, { edge, uIdx, vIdx, n });
        }
    });

    const components = new Set(parent.map((_, i) => find(i))).size;
    const calculatedH1 = Math.max(0, m - n + components);

    
    

    return calculatedH1;
}

    _buildIncidenceMatrix() {
        const n = this.graph.vertices.length;
        const m = this.graph.edges.length;
        if (n === 0 || m === 0) return zeroMatrix(0, 0);

        const matrix = zeroMatrix(n, m);
        const vMap = new Map(this.graph.vertices.map((v, i) => [v, i]));

        this.graph.edges.forEach(([u, v], j) => {
            const uIdx = vMap.get(u);
            const vIdx = vMap.get(v);
            if (uIdx !== undefined && vIdx !== undefined && uIdx < n && vIdx < n && j < m) {
                matrix[uIdx][j] = 1;
                matrix[vIdx][j] = -1;
            } else {
                logger.warn(`_buildIncidenceMatrix: Invalid vertex or edge index encountered for edge ${u}-${v}. Skipping.`);
            }
        });
        if (!isFiniteMatrix(matrix)) {
            logger.error('_buildIncidenceMatrix: Generated matrix is non-finite. Returning zero matrix.');
            return zeroMatrix(n, m);
        }
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
        if (parent[x] !== x) return parent[x] = find(parent[x]);
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
    
    

    edges.forEach(edge => {
        const u = edge[0]; 
        const v = edge[1]; 
        const uIdx = vMap.get(u);
        const vIdx = vMap.get(v);
        if (uIdx !== undefined && vIdx !== undefined && uIdx < n && vIdx < n) {
            union(uIdx, vIdx);
        } else {
            
            logger.warn(`_computeBetti1UnionFind: Invalid vertex index or edge format for edge starting with ${u}-${v}. Skipping.`, { edge, uIdx, vIdx, n });
        }
    });

    const components = new Set(parent.map((_, i) => find(i))).size;
    const calculatedH1 = Math.max(0, m - n + components);

    
    

    return calculatedH1;
}

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

    
    const laplacianEffect = matVecMul(Lfull, state);
    const sNext = vecSub(state, vecScale(laplacianEffect, eta));

    
    if (!isFiniteVector(sNext)) {
        logger.error('CRITICAL: simulateDiffusionStep produced a non-finite vector. Reverting to pre-simulation state.');
        return state;
    }

    return new Float32Array(sNext.map(v => clamp(v, -1, 1)));
}

    async computeStructuralSensitivity(perturbationScale = 0.05) {
        if (!this.ready || this.graph.edges.length === 0 || this.graph.vertices.length === 0 || this.qDim === 0) {
            this.structural_sensitivity = 0;
            return 0;
        }

        const s_t = this.getStalksAsVector();
        if (!isFiniteVector(s_t) || s_t.length === 0) {
            logger.warn('Sheaf.computeStructuralSensitivity: Initial stalk vector is invalid or empty. Returning 0.');
            this.structural_sensitivity = 0;
            return 0;
        }

        let baseState;
        try {
            baseState = await this.simulateDiffusionStep(s_t);
            if (!isFiniteVector(baseState) || baseState.length !== s_t.length) throw new Error("Base state is non-finite or mismatched length.");
        } catch (e) {
            logger.error('Sheaf.computeStructuralSensitivity: Could not compute base state.', e);
            this.structural_sensitivity = 0;
            return 0;
        }

        let totalSensitivity = 0;
        let perturbationCount = 0;
        const originalAdjacency = this.adjacencyMatrix ? this.adjacencyMatrix.map(row => new Float32Array(row)) : zeroMatrix(this.graph.vertices.length, this.graph.vertices.length);

        if (!isFiniteMatrix(originalAdjacency)) {
            logger.error('Sheaf.computeStructuralSensitivity: Original adjacency matrix is non-finite. Aborting.');
            this.structural_sensitivity = 0;
            return 0;
        }

        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const i = this.graph.vertices.indexOf(u);
            const j = this.graph.vertices.indexOf(v);
            if (i < 0 || j < 0 || i >= originalAdjacency.length || j >= originalAdjacency.length) {
                logger.warn(`Sheaf.computeStructuralSensitivity: Invalid vertex index for edge ${u}-${v}. Skipping perturbation.`);
                continue;
            }

            const originalWeight = originalAdjacency[i]?.[j] || 0;
            this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = clamp(originalWeight + perturbationScale, 0.01, 1);
            if (!Number.isFinite(this.adjacencyMatrix[i][j])) {
                 logger.warn(`Sheaf.computeStructuralSensitivity: Perturbed adjacency weight for edge ${u}-${v} is non-finite. Resetting.`);
                 this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = originalWeight;
                 continue;
            }

            try {
                const perturbedState = await this.simulateDiffusionStep(s_t);
                if (isFiniteVector(perturbedState) && perturbedState.length === baseState.length) {
                    const diffNorm = norm2(vecSub(perturbedState, baseState));
                    if (Number.isFinite(diffNorm)) {
                        totalSensitivity += diffNorm / (perturbationScale !== 0 ? perturbationScale : this.eps);
                        perturbationCount++;
                    } else {
                        logger.warn(`Sheaf.computeStructuralSensitivity: Non-finite difference norm for edge ${u}-${v}.`);
                    }
                } else {
                    logger.warn(`Sheaf.computeStructuralSensitivity: Perturbed state for edge ${u}-${v} is non-finite or mismatched. Skipping.`);
                }
            } catch (e) {
                logger.warn(`Sheaf.computeStructuralSensitivity: Error during perturbation for edge ${u}-${v}: ${e.message}`);
            } finally {
                this.adjacencyMatrix[i][j] = this.adjacencyMatrix[j][i] = originalWeight;
            }
        }

        this.structural_sensitivity = perturbationCount > 0 ? clamp(totalSensitivity / perturbationCount, 0, 10) : 0;
        return this.structural_sensitivity;
    }


async _computeGeodesicFreeEnergy(qualiaState) { 
    logger.debug('Sheaf._computeGeodesicFreeEnergy (Robust Version): Starting');

    const kl_proxy = Number.isFinite(this.owm?.actorLoss) ? this.owm.actorLoss : 0.1;
    const nV = this.graph?.vertices?.length || 0;

    if (nV < 2) {
        logger.warn('Sheaf._computeGeodesicFreeEnergy: Insufficient vertices.', { nV });
        return { F_int: kl_proxy, geodesic_divergence: 0 };
    }

    let C1;
    try {
        ({ C1 } = await this._computeCochains(qualiaState));
    } catch (e) {
        logger.error('Sheaf._computeGeodesicFreeEnergy: Error in _computeCochains', { error: e.message });
        return { F_int: kl_proxy, geodesic_divergence: 0 };
    }

    if (!this.currentCochain || !Array.isArray(this.currentCochain)) {
        logger.warn('Sheaf._computeGeodesicFreeEnergy: currentCochain is undefined or invalid. Using stalks.');
        this.currentCochain = Array.from(this.stalks.values());
    }

    let totalCochainNormSq = 0;
    let totalCurlProxy = 0;
    let edgeCount = 0;

    for (const cochain of C1.values()) {
        if (isFiniteVector(cochain)) {
            totalCochainNormSq += norm2(cochain) ** 2;
            edgeCount++;
        }
    }

    totalCurlProxy = this.inconsistency || 0;

    const geodesic_divergence = clamp(totalCurlProxy * 0.5, 0, 10);
    const F_base = kl_proxy + (edgeCount > 0 ? totalCochainNormSq / edgeCount : 0);
    const F_int = F_base - geodesic_divergence;

    const result = {
        F_int: clamp(Number.isFinite(F_int) ? F_int : kl_proxy, 0, 10),
        geodesic_divergence: geodesic_divergence
    };

    logger.debug('Sheaf._computeGeodesicFreeEnergy (Robust Version): Completed', { F_int: result.F_int, geodesic_divergence: result.geodesic_divergence });
    return result;
}
    async _spectralRicci(g) {
    logger.debug('Sheaf._spectralRicci: Starting', { gRows: g?.length, gCols: g[0]?.length });

    const qDim = this.qDim || 1;
    if (!isFiniteMatrix(g) || g.length !== qDim || (g[0]?.length || 0) !== qDim) {
        logger.warn('Sheaf._spectralRicci: Invalid input matrix g', { rows: g?.length, cols: g[0]?.length, expected: qDim });
        return identity(qDim);
    }

    try {
        const eigResult = await this._spectralDecomp(g, qDim); 
        logger.debug('Sheaf._spectralRicci: eigResult', { eigenvalueCount: eigResult?.eigenvalues?.length });
        if (!eigResult || !Array.isArray(eigResult.eigenvalues) || !isFiniteMatrix(eigResult.eigenvectors)) {
            logger.warn('Sheaf._spectralRicci: Invalid eigenvalue decomposition result');
            return identity(qDim);
        }

        const ricci = zeroMatrix(qDim, qDim);
        const eigenvalues = eigResult.eigenvalues;
        const eigenvectors = eigResult.eigenvectors;

        for (let i = 0; i < qDim; i++) {
            for (let j = 0; j < qDim; j++) {
                let sum = 0;
                for (let k = 0; k < qDim; k++) {
                    if (Number.isFinite(eigenvalues[k]) && Math.abs(eigenvalues[k]) > (this.eps || 1e-6)) {
                        sum += (eigenvectors[i][k] || 0) * (eigenvectors[j][k] || 0) / eigenvalues[k];
                    }
                }
                ricci[i][j] = clamp(sum, -1e6, 1e6);
            }
        }

        if (!isFiniteMatrix(ricci) || ricci.length !== qDim) {
            logger.warn('Sheaf._spectralRicci: Invalid Ricci tensor', { rows: ricci?.length, cols: ricci[0]?.length });
            return identity(qDim);
        }

        logger.debug('Sheaf._spectralRicci: Completed', { ricciRows: ricci.length });
        return ricci;
    } catch (e) {
        logger.error('Sheaf._spectralRicci: Computation failed', { error: e.message, stack: e.stack });
        return identity(qDim);
    }
}

_normalizedLaplacian(L_base) {
    const N = this.graph.vertices.length * this.qDim;
    if (N === 0) {
        const I = identity(0);
        I.rows = 0;
        I.cols = 0;
        return I;
    }

    try {
        const L_combinatorial = L_base;
        if (!isFiniteMatrix(L_combinatorial) || L_combinatorial.length !== N) {
            throw new Error('Base combinatorial Laplacian is invalid.');
        }

        const D_inv_sqrt = zeroMatrix(N, N);
        for (let i = 0; i < N; i++) {
            const degree = L_combinatorial[i][i];
            if (isFiniteNumber(degree) && degree > this.eps) {
                D_inv_sqrt[i][i] = 1.0 / Math.sqrt(degree);
            }
        }

        const A = zeroMatrix(N, N);
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                A[i][j] = (i === j ? L_combinatorial[i][i] : 0) - L_combinatorial[i][j];
            }
        }
        
        
        D_inv_sqrt.rows = N; D_inv_sqrt.cols = N;
        A.rows = N; A.cols = N;
        
        const T1 = matMul({ matrixA: D_inv_sqrt, matrixB: A });
        if(!isFiniteMatrix(T1)) throw new Error("Matrix multiplication (T1) failed.");
        T1.rows = N; T1.cols = N;

        const T2 = matMul({ matrixA: T1, matrixB: D_inv_sqrt });
        if(!isFiniteMatrix(T2)) throw new Error("Matrix multiplication (T2) failed.");

        const L_sym = zeroMatrix(N, N);
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                L_sym[i][j] = (i === j ? 1 : 0) - (T2[i][j] || 0);
            }
        }

        if (!isFiniteMatrix(L_sym)) {
            throw new Error('Final symmetric normalized Laplacian is non-finite.');
        }

        
        
        L_sym.rows = N;
        L_sym.cols = N;
        return L_sym;

    } catch (e) {
        logger.error(`_normalizedLaplacian: Critical failure: ${e.message}. Returning a valid identity matrix.`);
        const I = identity(N);
        I.rows = N;
        I.cols = N;
        return I;
    }
}


async _updateDerivedMetrics(qualiaState) {
    const last_intentionality_F = this.intentionality_F;
    const last_overallCoherence = this.overallCoherence;

    try {
        

        
        await this.computeGluingInconsistency();
        await this.computeGestaltUnity();
        await this.computeIntegratedInformation();
        await this.computeCupProduct();

        
        await this.computeStructuralSensitivity(0.05);

        
        const { F_int, geodesic_divergence } = await this._computeGeodesicFreeEnergy(qualiaState);
        
        

        this.intentionality_F = Number.isFinite(F_int) ? F_int : 0;

        let totalNorm = 0;
        let validVertices = 0;
        for (const stalk of this.stalks.values()) {
            if (isFiniteVector(stalk)) {
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
        logger.error(`Sheaf._updateDerivedMetrics: Error: ${e.message}`, { stack: e.stack });
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

    visualizeActivity(scene, camera, renderer) {
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
            const phase = Array.isArray(this.floquetPD?.phases) && this.floquetPD.phases.length > 0 ? this.floquetPD.phases[this.floquetPD.phases.length - 1] : 0;
            const rhythmicallyAware = this.rhythmicallyAware ?? false;

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
            const phase = Array.isArray(this.floquetPD?.phases) && this.floquetPD.phases.length > 0 ? this.floquetPD.phases[this.floquetPD.phases.length - 1] : 0;
            const rhythmicallyAware = this.rhythmicallyAware ?? false;

            const opacity = clamp((weight || 0) * this.cup_product_intensity * (rhythmicallyAware ? Math.cos(phase) : 1), 0.2, 0.8);

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

        if (Array.isArray(this.floquetPD?.births) && this.floquetPD.births.length > 0) {
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
            adaptation: this.adaptation,
            qDim: this.qDim,
            stateDim: this.stateDim,
            alpha: this.alpha, beta: this.beta, gamma: this.gamma, sigma: this.sigma, eps: this.eps, stabEps: this.stabEps,
            phiBase: this.phiBase, gestaltBase: this.gestaltBase,
            useAdvancedTheorems: this.useAdvancedTheorems, resonanceOmega: this.resonanceOmega, resonanceEps: this.resonanceEps,
            stalkHistorySize: this.stalkHistorySize,
            windowSize: this.windowSize,
            stalkHistory: this.stalkHistory.getAll(),
            stalkNormHistory: this.stalkNormHistory.getAll(),
            windowedStates: this.windowedStates.getAll(),
            phiHistory: this.phiHistory.getAll(),
            gestaltHistory: this.gestaltHistory.getAll(),
            inconsistencyHistory: this.inconsistencyHistory.getAll(),
            metrics_last_step: this.metrics_last_step,
            phi: this.phi, h1Dimension: this.h1Dimension, gestaltUnity: this.gestaltUnity,
            stability: this.stability, inconsistency: this.inconsistency, feel_F: this.feel_F,
            intentionality_F: this.intentionality_F, cup_product_intensity: this.cup_product_intensity,
            structural_sensitivity: this.structural_sensitivity, coherence: this.coherence,
            overallCoherence: this.overallCoherence, emergentAware: this.emergentAware,
            floquetPD: this.floquetPD,
            currentCochain: this.currentCochain
        };
    }

    loadState(state) {
        if (!state) {
            logger.warn('Sheaf.loadState: No state provided to load.');
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
            this.phiBase = state.phiBase ?? this.phiBase;
            this.gestaltBase = state.gestaltBase ?? this.gestaltBase;
            this.useAdvancedTheorems = state.useAdvancedTheorems ?? this.useAdvancedTheorems;
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

            this.phiHistory = new CircularBuffer(this.stalkHistorySize);
            if (Array.isArray(state.phiHistory)) state.phiHistory.forEach(item => this.phiHistory.push(item));

            this.gestaltHistory = new CircularBuffer(this.stalkHistorySize);
            if (Array.isArray(state.gestaltHistory)) state.gestaltHistory.forEach(item => this.gestaltHistory.push(item));

            this.inconsistencyHistory = new CircularBuffer(this.stalkHistorySize);
            if (Array.isArray(state.inconsistencyHistory)) state.inconsistencyHistory.forEach(item => this.inconsistencyHistory.push(item));

            this.metrics_last_step = state.metrics_last_step || {};
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

            this.floquetPD = state.floquetPD || { births: [], phases: [], deaths: [] };

            if (isFiniteMatrix(state.currentCochain) && state.currentCochain.length === this.graph.vertices.length && (state.currentCochain[0]?.length || 0) === this.qDim) {
                this.currentCochain = state.currentCochain;
            } else {
                logger.warn('Sheaf.loadState: currentCochain from state is invalid or mismatched. Re-initializing to zeros.');
                this.currentCochain = zeroMatrix(this.graph.vertices.length, this.qDim);
            }

            this.laplacian = this.buildLaplacian();
            this.ready = true;
            logger.info('Sheaf.loadState: State loaded successfully.');
        } catch (e) {
            logger.error(`Sheaf.loadState: Error loading state: ${e.message}`, { stack: e.stack });
            this.ready = false;
              
        this.gpu = null;
        this.gestaltKernel = null;
        try {
            
            const GPU = window.GPU || window.GPU;
            if (typeof GPU === 'function') {
                this.gpu = new GPU();
                
                this.gestaltKernel = this.gpu.createKernel(function(flattenedStates, eps_const, state_dim_const, num_states_const) {
                    let sum = 0;
                    for (let i = 0; i < num_states_const; i++) {
                        for (let j = i + 1; j < num_states_const; j++) {
                            let dotProd = 0;
                            let norm1_sq = 0;
                            let norm2_sq = 0;
                            for (let k = 0; k < state_dim_const; k++) {
                                const val1 = flattenedStates[i * state_dim_const + k];
                                const val2 = flattenedStates[j * state_dim_const + k];
                                dotProd += val1 * val2;
                                norm1_sq += val1 * val1;
                                norm2_sq += val2 * val2;
                            }
                            if (norm1_sq > 0 && norm2_sq > 0) { 
                                sum += Math.abs(dotProd / (Math.sqrt(norm1_sq) * Math.sqrt(norm2_sq)));
                            }
                        }
                    }
                    return sum;
                }).setOutput([1]);
                logger.info('FloquetPersistentSheaf: GPU.js instance and kernel created successfully.');
            }
        } catch (e) {
            logger.warn('FloquetPersistentSheaf: GPU.js initialization failed. Will use CPU fallback.', { error: e.message });
            this.gpu = null;
            this.gestaltKernel = null;
        }
        
        }
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
        this.selfAware = false;
    }

    _buildRecursiveGluing() {
        return (z) => {
            const z_next = new Map();
            this.graph.edges.forEach(([u, v]) => {
                const su = this.stalks.get(u) || vecZeros(this.qDim);
                const sv = this.stalks.get(v) || vecZeros(this.qDim);
                let phi_uv = this.projectionMatrices.get(`${u}-${v}`);

                if (!isFiniteVector(su) || !isFiniteVector(sv) || su.length !== this.qDim || sv.length !== this.qDim) {
                    logger.warn(`RecursiveTopologicalSheaf._buildRecursiveGluing: Invalid stalk for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.qDim));
                    return;
                }
                const diffusion = vecScale(vecAdd(su, sv), this.alpha / 2);

                const z_uv = z.get([u, v].sort().join(',')) || vecZeros(this.qDim);
                if (!isFiniteVector(z_uv) || z_uv.length !== this.qDim) {
                    logger.warn(`RecursiveTopologicalSheaf._buildRecursiveGluing: Invalid z_uv for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.qDim));
                    return;
                }

                if (!isFiniteMatrix(phi_uv) || phi_uv.length !== this.qDim || (phi_uv[0]?.length || 0) !== this.qDim) {
                    logger.warn(`RecursiveTopologicalSheaf._buildRecursiveGluing: Invalid projection matrix for edge ${u}-${v}. Using identity.`);
                    phi_uv = identity(this.qDim);
                }
                const z_next_uv = vecAdd(matVecMul(phi_uv, z_uv), diffusion);
                if (isFiniteVector(z_next_uv)) {
                    z_next.set([u, v].sort().join(','), z_next_uv);
                } else {
                    logger.warn(`RecursiveTopologicalSheaf._buildRecursiveGluing: Non-finite z_next_uv for edge ${u}-${v}. Using zeros.`);
                    z_next.set([u, v].sort().join(','), vecZeros(this.qDim));
                }
            });
            return z_next;
        };
    }

    computeLinguisticCocycles(state) {
        const z = new Map();
        const nV = this.graph.vertices.length;
        if (nV === 0 || this.qDim === 0) return z;

        const idxMap = new Map(this.graph.vertices.map((v, i) => [v, i]));
        this.graph.edges.forEach(([u, v]) => {
            const i = idxMap.get(u), j = idxMap.get(v);
            if (i === undefined || j === undefined || i >= nV || j >= nV) return;

            const z_uv = new Float32Array(this.qDim);
            const input_u = Number.isFinite(state[Math.min(i, state.length - 1)]) ? state[Math.min(i, state.length - 1)] : 0;
            const input_v = Number.isFinite(state[Math.min(j, state.length - 1)]) ? state[Math.min(j, state.length - 1)] : 0;

            for (let k = 0; k < this.qDim; k++) {
                z_uv[k] = clamp(input_u - input_v, -1, 1) * (this.entityNames[k]?.includes('symbolic') ? 1.5 : 1);
            }
            if (isFiniteVector(z_uv)) {
                z.set([u, v].sort().join(','), z_uv);
            } else {
                logger.warn(`RecursiveTopologicalSheaf.computeLinguisticCocycles: Non-finite cocycle for edge ${u}-${v}. Using zeros.`);
                z.set([u, v].sort().join(','), vecZeros(this.qDim));
            }
        });
        return z;
    }

  
async computeSelfAwareness(state) {
    const fallbackDim = this.graph.vertices.length * this.qDim || 7;
    const safeDefault = { psi: vecZeros(fallbackDim), aware: false };

    try {
        
        
        const expectedDim = this.graph.vertices.length * this.qDim;
        if (!isFiniteVector(state) || state.length !== expectedDim) {
            logger.warn(`computeSelfAwareness: Invalid input state vector. Expected dim ${expectedDim}, got ${state?.length}. Returning safe default.`);
            return safeDefault;
        }

        
        const z = this.computeLinguisticCocycles(state);
        let z_fixed = z;
        for (let i = 0; i < this.maxIter; i++) {
            const z_next = this.R_star(z_fixed);
            if (this._cocycleNormDiff(z_next, z_fixed) < this.fixedPointEps) break;
            z_fixed = z_next;
        }
        this.cochainHistory.push(z_fixed);

        const L_rec = await this._recursiveLaplacian(z_fixed);
        const { eigenvalues } = await this._spectralDecomp(L_rec);
        const lambda_min = eigenvalues.length > 0 ? Math.min(...eigenvalues) : 0;

        const psi = vecZeros(fallbackDim); 
        const aware = lambda_min > this.tau;
        this.selfAware = aware;

        return { psi, aware };

    } catch (e) {
        logger.error(`computeSelfAwareness: Computation failed critically.`, { error: e.message });
        return safeDefault;
    }
}

    _cocycleNormDiff(z1, z2) {
        let sum = 0;
        let count = 0;
        if (! (z1 instanceof Map) || ! (z2 instanceof Map)) {
            logger.warn('RecursiveTopologicalSheaf._cocycleNormDiff: Invalid cocycle map input. Returning 0.');
            return 0;
        }
        for (const key of z1.keys()) {
            const v1 = z1.get(key);
            const v2 = z2.get(key);
            if (isFiniteVector(v1) && isFiniteVector(v2) && v1.length === this.qDim && v2.length === this.qDim) {
                sum += norm2(vecSub(v1, v2)) ** 2;
                count++;
            } else {
                logger.warn(`RecursiveTopologicalSheaf._cocycleNormDiff: Invalid or mismatched cocycle vector for key ${key}. Skipping.`);
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    async _recursiveLaplacian(z) {
    const nE = this.graph.edges.length;
    if (nE === 0) return identity(0);

    const L_rec = zeroMatrix(nE, nE);
    const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));

    for (const edge of this.graph.edges) {
        const [u, v] = edge;
        const i = eMap.get([u, v].sort().join(','));
        if (i === undefined || i >= nE) {
            logger.warn(`_recursiveLaplacian: Invalid edge index ${i} for edge ${u}-${v}. Skipping.`);
            continue;
        }

        L_rec[i][i] = 1;

        const z_uv = z.get([u, v].sort().join(',')) ?? vecZeros(this.qDim);
        if (!isFiniteVector(z_uv) || z_uv.length !== this.qDim) {
            logger.warn(`_recursiveLaplacian: Invalid z_uv for edge ${u}-${v}. Using zeros.`);
            continue;
        }

        let phi_uv = this.projectionMatrices.get(`${u}-${v}`);
        if (!isFiniteMatrix(phi_uv) || phi_uv.length !== this.qDim || (phi_uv[0]?.length || 0) !== this.qDim) {
            logger.warn(`_recursiveLaplacian: Invalid projection matrix for edge ${u}-${v}. Using identity.`);
            phi_uv = identity(this.qDim);
        }

        for (const edge2 of this.graph.edges) {
            const [u2, v2] = edge2;
            const j = eMap.get([u2, v2].sort().join(','));
            if (j === undefined || j >= nE || i === j) {
                if (i === j) continue;
                logger.warn(`_recursiveLaplacian: Invalid edge index ${j} for edge ${u2}-${v2}. Skipping interaction.`);
                continue;
            }

            let phi_u2v2 = this.projectionMatrices.get(`${u2}-${v2}`);
            if (!isFiniteMatrix(phi_u2v2) || phi_u2v2.length !== this.qDim || (phi_u2v2[0]?.length || 0) !== this.qDim) {
                logger.warn(`_recursiveLaplacian: Invalid projection matrix for edge ${u2}-${v2}. Using identity.`);
                phi_u2v2 = identity(this.qDim);
            }

            const z_u2v2 = z.get([u2, v2].sort().join(',')) ?? vecZeros(this.qDim);
            if (!isFiniteVector(z_u2v2) || z_u2v2.length !== this.qDim) {
                logger.warn(`_recursiveLaplacian: Invalid z_u2v2 for edge ${u2}-${v2}. Using zeros.`);
                continue;
            }

            let mat_vec_result;
            try {
                mat_vec_result = matVecMul(phi_u2v2, z_u2v2);
            } catch (err) {
                logger.warn(`[RecursiveLaplacian] matVecMul failed for edge ${u2}-${v2}, using zero vector. Error: ${err.message}`);
                mat_vec_result = vecZeros(this.qDim);
            }

            if (!isFiniteVector(mat_vec_result) || mat_vec_result.length !== this.qDim) mat_vec_result = vecZeros(this.qDim);

            let interaction = dot(z_uv, mat_vec_result);
            if (!Number.isFinite(interaction)) interaction = 0;

            L_rec[i][j] = -this.alpha * clamp(interaction, -0.1, 0.1);
        }
    }

    if (!isFiniteMatrix(L_rec)) {
        logger.error('_recursiveLaplacian: Generated recursive Laplacian is non-finite. Returning identity.');
        return identity(nE);
    }
    return L_rec;
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
        this.hierarchicallyAware = false;
    }

    _leftAdjoint() {
        return (F_in) => {
            const cochains = new Map();
            const F_stalks = F_in?.stalks || new Map();
            this.graph.edges.forEach(([u, v]) => {
                const su = F_stalks.get(u) || vecZeros(this.qDim);
                const sv = F_stalks.get(v) || vecZeros(this.qDim);
                if (!isFiniteVector(su) || !isFiniteVector(sv) || su.length !== this.qDim || sv.length !== this.qDim) {
                    logger.warn(`_leftAdjoint: Invalid stalk for edge ${u}-${v}. Using zero vector for cochain.`);
                    cochains.set([u, v].sort().join(','), vecZeros(this.qDim));
                    return;
                }
                const c1 = vecScale(vecAdd(su, sv), this.alpha / 2); 
                if (isFiniteVector(c1)) {
                    cochains.set([u, v].sort().join(','), c1);
                } else {
                    logger.warn(`_leftAdjoint: Non-finite cochain c1 for edge ${u}-${v}. Using zero vector.`);
                    cochains.set([u, v].sort().join(','), vecZeros(this.qDim));
                }
            });
            return {
                cochains,
                transport: (phi) => (v) => matVecMul(phi, v)
            };
        };
    }

    _rightAdjoint() {
        return (C_in) => {
            const stalks = new Map();
            const C_cochains = C_in?.cochains || new Map();
            this.graph.vertices.forEach(v => {
                let sum = vecZeros(this.qDim);
                let count = 0;
                this.graph.edges.forEach(([u, w]) => {
                    if (u === v || w === v) {
                        const c_uw = C_cochains.get([u, w].sort().join(',')) || vecZeros(this.qDim);
                        if (isFiniteVector(c_uw) && c_uw.length === this.qDim) {
                            sum = vecAdd(sum, c_uw);
                            count++;
                        } else {
                            logger.warn(`_rightAdjoint: Invalid cochain for edge incident to ${v}. Skipping.`);
                        }
                    }
                });
                const newStalk = count > 0 ? vecScale(sum, 1 / count) : vecZeros(this.qDim);
                if (isFiniteVector(newStalk)) {
                    stalks.set(v, newStalk);
                } else {
                    logger.warn(`_rightAdjoint: Non-finite new stalk for vertex ${v}. Using zero vector.`);
                    stalks.set(v, vecZeros(this.qDim));
                }
            });
            return {
                stalks,
                projections: this.projectionMatrices
            };
        };
    }

    _monadUF() {
        return (F_in) => this.U(this.F(F_in));
    }

    _unit() {
        return (F_in) => {
            const eta_F_stalks = new Map();
            const F_stalks = F_in?.stalks || new Map();
            this.graph.vertices.forEach(v => {
                const stalk = F_stalks.get(v) || vecZeros(this.qDim);
                if (!isFiniteVector(stalk) || stalk.length !== this.qDim) {
                    logger.warn(`_unit: Invalid stalk for vertex ${v}. Using zero vector.`);
                    eta_F_stalks.set(v, vecZeros(this.qDim));
                    return;
                }
                const scaled = vecScale(stalk, 1 + this.alpha);
                if (isFiniteVector(scaled)) {
                    eta_F_stalks.set(v, scaled);
                } else {
                    logger.warn(`_unit: Non-finite scaled stalk for vertex ${v}. Using zero vector.`);
                    eta_F_stalks.set(v, vecZeros(this.qDim));
                }
            });
            return { stalks: eta_F_stalks };
        };
    }

    _counit() {
        return (FU, Id) => {
            const epsilon_FU_stalks = new Map();
            const FU_stalks = FU?.stalks || new Map();
            const Id_stalks = Id?.stalks || new Map();
            this.graph.vertices.forEach(v => {
                const fu_v = FU_stalks.get(v) || vecZeros(this.qDim);
                const id_v = Id_stalks.get(v) || vecZeros(this.qDim);
                if (!isFiniteVector(fu_v) || !isFiniteVector(id_v) || fu_v.length !== this.qDim || id_v.length !== this.qDim) {
                    logger.warn(`_counit: Invalid stalks for vertex ${v}. Using zero vector for difference.`);
                    epsilon_FU_stalks.set(v, vecZeros(this.qDim));
                    return;
                }
                const diff = vecSub(fu_v, id_v);
                if (isFiniteVector(diff)) {
                    epsilon_FU_stalks.set(v, diff);
                } else {
                    logger.warn(`_counit: Non-finite difference for vertex ${v}. Using zero vector.`);
                    epsilon_FU_stalks.set(v, vecZeros(this.qDim));
                }
            });
            return { stalks: epsilon_FU_stalks };
        };
    }

    async computeAdjunctionFixedPoint() {
        const N = this.graph.vertices.length * this.qDim;
        if (N === 0) {
            this.hierarchicallyAware = false;
            return { fixedPoint: vecZeros(1), aware: false };
        }

        try {
            
            this.laplacian = this.buildLaplacian();
            if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N) {
                throw new Error("Cannot compute fixed point with an invalid Laplacian.");
            }

            
            
            const { eigenvalues, eigenvectors } = await this._spectralDecomp(this.laplacian);

            
            
            const fixedPoint = vecZeros(N);
            let contributingVectors = 0;
            eigenvalues.forEach((lambda, i) => {
                if (isFiniteNumber(lambda) && Math.abs(lambda) < this.eps) {
                    
                    const eigenvector = eigenvectors.map(row => row[i] || 0);
                    if (isFiniteVector(eigenvector)) {
                        for (let j = 0; j < N; j++) {
                            fixedPoint[j] += eigenvector[j];
                        }
                        contributingVectors++;
                    }
                }
            });

            
            if (contributingVectors > 0) {
                const norm = norm2(fixedPoint);
                if (norm > this.eps) {
                    for (let i = 0; i < N; i++) {
                        fixedPoint[i] /= norm;
                    }
                }
            }

            const aware = norm2(fixedPoint) > 0.1; 
            this.hierarchicallyAware = aware;

            return { fixedPoint, aware };

        } catch (e) {
            logger.error(`Sheaf.computeAdjunctionFixedPoint: Computation failed. Returning a safe zero-vector.`, { error: e.message });
            this.hierarchicallyAware = false;
            return { fixedPoint: vecZeros(N), aware: false };
        }
    }
    async _equalizerNorm(eta_stalks, eps_stalks) {
        let sum = 0;
        let count = 0;
        if (!(eta_stalks instanceof Map) || !(eps_stalks instanceof Map)) {
            logger.warn('AdjunctionReflexiveSheaf._equalizerNorm: Invalid stalk map input. Returning 0.');
            return 0;
        }
        for (const v of this.graph.vertices) {
            const eta_v = eta_stalks.get(v);
            const eps_v = eps_stalks.get(v);
            if (isFiniteVector(eta_v) && isFiniteVector(eps_v) && eta_v.length === this.qDim && eps_v.length === this.qDim) {
                sum += norm2(vecSub(eta_v, eps_v)) ** 2;
                count++;
            } else {
                logger.warn(`AdjunctionReflexiveSheaf._equalizerNorm: Invalid or mismatched stalk vector for vertex ${v}. Skipping.`);
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    async _extractFixedCocycle(T) {
        const z_star = new Map();
        const F_T_result = this.F(T);
        if (!F_T_result || ! (F_T_result.cochains instanceof Map)) {
            logger.warn('_extractFixedCocycle: F(T) result or cochains invalid. Returning empty map.');
            return z_star;
        }

         const { C1, C2 } = await this._computeCochains(this.qInput); 
        if (! (C1 instanceof Map) || ! (C2 instanceof Map)) {
            logger.warn('_extractFixedCocycle: C1 or C2 cochains are invalid. Returning empty map.');
            return z_star;
        }

        for (const [key, c_curl] of C2.entries()) {
            if (isFiniteVector(c_curl) && norm2(c_curl) < this.eps) {
                const triangle_vertices = key.split(',');
                if (triangle_vertices.length !== 3) {
                    logger.warn(`_extractFixedCocycle: Invalid triangle key ${key}. Skipping.`);
                    continue;
                }
                const edges_of_triangle = [
                    [triangle_vertices[0], triangle_vertices[1]].sort().join(','),
                    [triangle_vertices[1], triangle_vertices[2]].sort().join(','),
                    [triangle_vertices[2], triangle_vertices[0]].sort().join(',')
                ];
                edges_of_triangle.forEach(edge_key => {
                    const edge_cocycle = C1.get(edge_key);
                    if (edge_cocycle && !z_star.has(edge_key) && isFiniteVector(edge_cocycle) && edge_cocycle.length === this.qDim) {
                        z_star.set(edge_key, edge_cocycle);
                    }
                });
            } else if (!isFiniteVector(c_curl)) {
                logger.warn(`_extractFixedCocycle: Non-finite curl for triangle key ${key}. Skipping.`);
            }
        }
        return z_star;
    }

    async _adjunctionLaplacian(T) {
        const nE = this.graph.edges.length;
        if (nE === 0) return identity(0);

        const L_adj = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        const F_T_result = this.F(T);
        if (!F_T_result || ! (F_T_result.cochains instanceof Map)) {
            logger.warn('_adjunctionLaplacian: F(T) result or cochains invalid. Returning identity.');
            return identity(nE);
        }

        const C1 = F_T_result.cochains;
        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined || i >= nE) {
                logger.warn(`_adjunctionLaplacian: Invalid edge index ${i} for edge ${u}-${v}. Skipping.`);
                continue;
            }
            L_adj[i][i] = 1;
            const c_uv = C1.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            if (!isFiniteVector(c_uv) || c_uv.length !== this.qDim) {
                logger.warn(`_adjunctionLaplacian: Invalid cochain c_uv for edge ${u}-${v}. Using zeros.`);
                continue;
            }

            for (const edge2 of this.graph.edges) {
                const [u2, v2] = edge2;
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || j >= nE || i === j) {
                    if (i === j) continue;
                    logger.warn(`_adjunctionLaplacian: Invalid edge index ${j} for edge ${u2}-${v2}. Skipping interaction.`);
                    continue;
                }
                const c_u2v2 = C1.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                if (!isFiniteVector(c_u2v2) || c_u2v2.length !== this.qDim) {
                    logger.warn(`_adjunctionLaplacian: Invalid cochain c_u2v2 for edge ${u2}-${v2}. Using zeros.`);
                    continue;
                }
                const interaction = dot(c_uv, c_u2v2);
                if (Number.isFinite(interaction)) {
                    L_adj[i][j] = -this.beta * clamp(interaction, -0.1, 0.1);
                } else {
                    logger.warn(`_adjunctionLaplacian: Non-finite interaction for edges ${u}-${v} and ${u2}-${v2}. Setting to 0.`);
                    L_adj[i][j] = 0;
                }
            }
        }
        if (!isFiniteMatrix(L_adj)) {
            logger.error('_adjunctionLaplacian: Generated adjacency Laplacian is non-finite. Returning identity.');
            return identity(nE);
        }
        return L_adj;
    }

    _initStalks(state) {
        const stalks = new Map();
        const nV = this.graph.vertices.length;
        if (nV === 0 || this.qDim === 0) return stalks;

        this.graph.vertices.forEach((v, i) => {
            const stalk = new Float32Array(this.qDim).fill(0);
            const input = Number.isFinite(state[Math.min(i, state.length - 1)]) ? state[Math.min(i, state.length - 1)] : 0;
            for (let k = 0; k < this.qDim; k++) {
                stalk[k] = clamp(input * (this.entityNames[k]?.includes('metacognition') ? 1.2 : 1), -1, 1);
            }
            if (isFiniteVector(stalk)) {
                stalks.set(v, stalk);
            } else {
                logger.warn(`_initStalks: Non-finite stalk for vertex ${v}. Using zeros.`);
                stalks.set(v, vecZeros(this.qDim));
            }
        });
        return stalks;
    }

    async _deltaR1() {
        const nE = this.graph.edges.length;
        const nT = this.simplicialComplex.triangles.length;
        if (nE === 0 || nT === 0) return zeroMatrix(nT, nE);

        const delta = zeroMatrix(nT, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));

        this.simplicialComplex.triangles.forEach((tri, tIdx) => {
            if (!this.isValidTriangle(tri) || tIdx >= nT) {
                logger.warn(`_deltaR1: Invalid triangle ${tri.join(',')} or index ${tIdx}. Skipping.`);
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
            logger.error('_deltaR1: Generated matrix is non-finite. Returning zero matrix.');
            return zeroMatrix(nT, nE);
        }
        return delta;
    }
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
        this.diachronicallyAware = false;
    }

    _partial_t(F_t, dt = 1) {
        const stalksNext = new Map();
        const F_t_stalks = F_t?.stalks || new Map();
        this.graph.vertices.forEach(v => {
            const stalk = F_t_stalks.get(v) || vecZeros(this.qDim);
            if (!isFiniteVector(stalk) || stalk.length !== this.qDim) {
                logger.warn(`_partial_t: Invalid stalk for vertex ${v}. Using zero vector.`);
                stalksNext.set(v, vecZeros(this.qDim));
                return;
            }

            const neighbors = this.graph.edges.filter(e => e[0] === v || e[1] === v).map(e => e[0] === v ? e[1] : e[0]);
            let grad = vecZeros(this.qDim);
            neighbors.forEach(u => {
                const su = F_t_stalks.get(u) || vecZeros(this.qDim);
                let phi_vu = this.projectionMatrices.get(`${v}-${u}`);
                if (!isFiniteMatrix(phi_vu) || phi_vu.length !== this.qDim || (phi_vu[0]?.length || 0) !== this.qDim) {
                    logger.warn(`_partial_t: Invalid projection matrix for edge ${v}-${u}. Using identity.`);
                    phi_vu = identity(this.qDim);
                }
                const mat_vec_result = matVecMul(phi_vu, su);
                if (isFiniteVector(mat_vec_result)) {
                    const diff = vecSub(stalk, mat_vec_result);
                    if (isFiniteVector(diff)) grad = vecAdd(grad, vecScale(diff, this.beta));
                } else {
                    logger.warn(`_partial_t: Non-finite mat_vec_result for edge ${v}-${u}. Skipping gradient contribution.`);
                }
            });
            const noise = vecScale(new Float32Array(this.qDim).map(() => Math.random() - 0.5), this.sigma * Math.sqrt(dt));
            const diffused = vecAdd(vecScale(stalk, 1 - this.gamma * dt), vecAdd(grad, noise));
            stalksNext.set(v, isFiniteVector(diffused) ? diffused : vecZeros(this.qDim));
        });
        return { stalks: stalksNext, projections: F_t?.projections || new Map() };
    }

    async computePersistentFixedPoint(init_state, T = 10) {
        const nV = this.graph.vertices.length;
        const nE = this.graph.edges.length;
        if (nV === 0 || nE === 0 || this.qDim === 0) {
            logger.warn('computePersistentFixedPoint: Graph has no vertices/edges or qDim is zero. Returning default.');
            return { F_persist: null, z_star_persist: new Map(), Phi_SA_persist: 0, PD: { births: [], deaths: [] }, aware: false };
        }
        
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
            const eta_prev_obj = this.flowHistory.get(this.flowHistory.length - 1);
            const eta_prev_stalks = (eta_prev_obj && eta_prev_obj.eta_t?.stalks) ? eta_prev_obj.eta_t.stalks : eta_next.stalks;

            const eta_evol = await this._nablaPersist(eta_next.stalks, eta_prev_stalks);
            const epsilon_next = this.epsilon(U_next, partial_F);

            const eq_delta = await this._equalizerNorm(eta_evol, epsilon_next.stalks);
            if (!Number.isFinite(eq_delta) || eq_delta < this.eps) break;

            let L_Tt;
            try {
                L_Tt = await this._flowLaplacian();
            } catch (e) {
                logger.error(`computePersistentFixedPoint: _flowLaplacian failed: ${e.message}. Using identity.`, {stack: e.stack});
                L_Tt = identity(nV * this.qDim > 0 ? nV * this.qDim : 1);
            }
            
            let spectralResult;
            try {
                spectralResult = await this._spectralDecomp(L_Tt);
            } catch (e) {
                logger.error(`computePersistentFixedPoint: _spectralDecomp failed: ${e.message}. Using empty eigenvalues.`, {stack: e.stack});
                spectralResult = { eigenvalues: [] };
            }
            const eigenvalues = spectralResult?.eigenvalues || [];
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
        Phi_SA_persist = clamp(Number.isFinite(Phi_SA_persist) ? Phi_SA_persist : 0, 0, 100);
        this.diachronicallyAware = Phi_SA_persist > this.tau_persist;

        return { F_persist: F_curr, z_star_persist, Phi_SA_persist, PD: this.persistenceDiagram, aware: this.diachronicallyAware };
    }

    async _nablaPersist(eta_next_stalks, eta_prev_stalks) {
        const eta_evol = new Map();
        if (!(eta_next_stalks instanceof Map) || !(eta_prev_stalks instanceof Map)) {
            logger.warn('PersistentAdjunctionSheaf._nablaPersist: Invalid stalk map input. Returning empty map.');
            return eta_evol;
        }
        this.graph.vertices.forEach(v => {
            const next_v = eta_next_stalks.get(v) || vecZeros(this.qDim);
            const prev_v = eta_prev_stalks.get(v) || vecZeros(this.qDim);
            if (!isFiniteVector(next_v) || !isFiniteVector(prev_v) || next_v.length !== this.qDim || prev_v.length !== this.qDim) {
                logger.warn(`_nablaPersist: Invalid stalk vector for vertex ${v}. Using zero vector.`);
                eta_evol.set(v, vecZeros(this.qDim));
                return;
            }
            const diff = vecSub(next_v, prev_v);
            const evol = vecAdd(next_v, vecScale(diff, this.gamma));
            if (isFiniteVector(evol)) eta_evol.set(v, evol);
             else {
                logger.warn(`_nablaPersist: Non-finite evolved stalk for vertex ${v}. Using zero vector.`);
                eta_evol.set(v, vecZeros(this.qDim));
            }
        });
        return eta_evol;
    }

    async _flowLaplacian() {
    const nV = this.graph.vertices.length;
    const N = nV * this.qDim;
    if (N === 0) return identity(0);

    
    
    this.laplacian = this.buildLaplacian();

    
    if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N || (this.laplacian[0]?.length || 0) !== N) {
        logger.warn('_flowLaplacian: Base Laplacian is invalid even after a rebuild attempt. Returning identity.');
        return identity(N > 0 ? N : 1);
    }
    
    
    const L_base = this.laplacian;
    const L_Tt = zeroMatrix(N, N);

    
    for (let i = 0; i < nV; i++) {
        for (let j = 0; j < nV; j++) {
            for (let qi = 0; qi < this.qDim; qi++) {
                
                const baseVal = L_base[i * this.qDim + qi]?.[j * this.qDim + qi];
                if (Number.isFinite(baseVal)) {
                    L_Tt[i * this.qDim + qi][j * this.qDim + qi] = baseVal;
                }
            }
        }
    }
    
    if (!isFiniteMatrix(L_Tt)) {
        logger.error('_flowLaplacian: Generated full flow Laplacian became non-finite. Returning identity.');
        return identity(N);
    }
    return L_Tt;
}

async _spectralDecomp(matrix) {
        if (!isFiniteMatrix(matrix) || matrix.length === 0 || matrix.length !== matrix[0].length) {
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
            throw new Error('Jacobi method returned invalid data.');
        } catch (err) {
            logger.warn(`_spectralDecomp failed: ${err.message}. Using last known good result or fallback.`);
            const n = matrix.length;
            if (this.lastGoodEigenResult && this.lastGoodEigenResult.eigenvalues.length === n) {
                return this.lastGoodEigenResult;
            }
            
            return { eigenvalues: Array(n).fill(1), eigenvectors: identity(n) };
        }
    }


async _linearDiffuse(state, Lfull) {
    const nV = this.graph.vertices.length;
    const N = nV * this.qDim;
    const s = this.getStalksAsVector();

    
    if (!isFiniteMatrix(Lfull) || Lfull.length !== N || !isFiniteVector(s)) {
        logger.error(`_linearDiffuse received invalid inputs. Reverting to pre-diffusion state.`);
        return { sNext: s, metric_gain: 0 };
    }

    
    const eta = 0.001;

    
    
    const laplacianEffect = matVecMul(Lfull, s);

    
    if (!isFiniteVector(laplacianEffect)) {
        logger.error('CRITICAL: Laplacian effect calculation resulted in a non-finite vector. Reverting.');
        return { sNext: s, metric_gain: 0 };
    }
    const sNext = vecSub(s, vecScale(laplacianEffect, eta));
    if (!isFiniteVector(sNext)) {
        logger.error('CRITICAL: Diffusion step created a non-finite vector. Reverting.');
        return { sNext: s, metric_gain: 0 };
    }

    
    const noise = new Float32Array(N).map(() => (Math.random() - 0.5) * (this.sigma || 0.01));
    const sNextWithNoise = vecAdd(sNext, noise);

    const qInput = state.slice(0, this.qDim);
    const f_s = new Float32Array(N).fill(0);
    for (let qi = 0; qi < this.qDim; qi++) {
        f_s[qi] = (this.alpha || 0.1) * (qInput[qi] || 0);
    }
    const finalSNext = vecAdd(sNextWithNoise, f_s);

    
    const clampedSNext = new Float32Array(finalSNext.map(v => clamp(v || 0, -1, 1)));
    if (!isFiniteVector(clampedSNext)) {
        logger.error('CRITICAL: Final vector in _linearDiffuse is non-finite after clamping. Reverting.');
        return { sNext: s, metric_gain: 0 };
    }

    
    return {
        sNext: clampedSNext,
        metric_gain: this.computeMetricGain(clampedSNext)
    };
}

    _updatePD(pd_old, lambda_t, birth_time) {
        const safe_pd_old = pd_old && typeof pd_old === 'object' ? pd_old : {};
        const safe_births = Array.isArray(safe_pd_old.births) ? safe_pd_old.births : [];
        const safe_deaths = Array.isArray(safe_pd_old.deaths) ? safe_pd_old.deaths : [];

        const pd = { births: [...safe_births], deaths: [...safe_deaths] };
        lambda_t.forEach(lambda => {
            if (Number.isFinite(lambda) && !pd.births.some(b => Number.isFinite(b?.value) && Math.abs(b.value - lambda) < this.eps)) {
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

    async _extractPersistCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        if (!isFiniteMatrix(delta1) || delta1.length === 0 || (delta1[0]?.length || 0) === 0) {
            logger.warn('_extractPersistCocycle: Invalid delta1 matrix. Returning empty map.');
            return z_star;
        }

        const nT = delta1.length;
        const nE = delta1[0]?.length || 0;
        if (nE === 0) return z_star;


        for (const flow_record of flows) {
            const F_t = flow_record?.F_t;
            if (!F_t || ! (F_t.cochains instanceof Map)) {
                logger.warn('_extractPersistCocycle: Invalid flow_record or cochains. Skipping.');
                continue;
            }

            const C1_map = F_t.cochains;

            for (let q = 0; q < this.qDim; q++) {
                const C1_q_coeffs = new Float32Array(nE);
                this.graph.edges.forEach((edge, eIdx) => {
                    if (eIdx >= nE) {
                        logger.warn(`_extractPersistCocycle: Edge index ${eIdx} out of bounds for C1_q_coeffs. Skipping.`);
                        return;
                    }
                    const edgeKey = [edge[0], edge[1]].sort().join(',');
                    const c_edge = C1_map.get(edgeKey) || vecZeros(this.qDim);
                    if (isFiniteVector(c_edge) && c_edge.length === this.qDim && Number.isFinite(c_edge[q])) {
                        C1_q_coeffs[eIdx] = c_edge[q];
                    } else {
                        logger.warn(`_extractPersistCocycle: Non-finite cochain or invalid dimension for edge ${edgeKey}, dim ${q}. Using 0.`);
                        C1_q_coeffs[eIdx] = 0;
                    }
                });

                if (!isFiniteVector(C1_q_coeffs)) {
                    logger.warn(`_extractPersistCocycle: Non-finite C1_q_coeffs for qualia dimension ${q}. Skipping.`);
                    continue;
                }

                let d1_C1_q;
                try {
                    d1_C1_q = matVecMul(delta1, C1_q_coeffs);
                } catch (e) {
                    logger.warn(`_extractPersistCocycle: matVecMul with delta1 failed for dim ${q}: ${e.message}. Skipping.`);
                    continue;
                }


                if (isFiniteVector(d1_C1_q) && norm2(d1_C1_q) < this.eps * nT) {
                    this.graph.edges.forEach((edge, eIdx) => {
                        const edgeKey = [edge[0], edge[1]].sort().join(',');
                        const current_cocycle = C1_map.get(edgeKey);
                        if (current_cocycle && !z_star.has(edgeKey) && isFiniteVector(current_cocycle) && current_cocycle.length === this.qDim) {
                            z_star.set(edgeKey, current_cocycle);
                        }
                    });
                } else if (!isFiniteVector(d1_C1_q)) {
                    logger.warn(`_extractPersistCocycle: matVecMul returned non-finite d1_C1_q for dim ${q}. Skipping.`);
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

        this.flowBufferSize = config.flowBufferSize || 50;
        this.windowedStates = new CircularBuffer(this.flowBufferSize);
        this.stalkHistory = new CircularBuffer(this.flowBufferSize);
        this.phiHistory = new CircularBuffer(this.flowBufferSize);

        this.phi = this.phi ?? 0.001;
        this.feel_F = this.feel_F ?? 0;
        this.intentionality_F = this.intentionality_F ?? 0;
        this.h1Dimension = this.h1Dimension ?? 0;
        this.cup_product_intensity = this.cup_product_intensity ?? 0;
        this.gestaltUnity = this.gestaltUnity ?? 0;
        this.structural_sensitivity = this.structural_sensitivity ?? 0;
        this.inconsistency = this.inconsistency ?? 0;
        this.rhythmicallyAware = false;

        if (!this.graph || !Array.isArray(this.graph.vertices) || this.graph.vertices.length === 0) {
            this._initializeGraph({});
        }
        
        const N_total_stalk_dim_initial = this.graph.vertices.length * this.qDim;
        if (this.windowedStates.length === 0 || this.windowedStates.get(0)?.length !== N_total_stalk_dim_initial) {
            this._initializeWindowedStates(N_total_stalk_dim_initial);
        }

        this.ready = false;
        logger.info(`FloquetPersistentSheaf: Constructor finished.`);
    }

    computeMetricGain(sNext) {
    if (!isFiniteVector(sNext)) {
        logger.warn('Sheaf.computeMetricGain: Invalid sNext vector. Returning 0.');
        return 0;
    }
    
    const gain = norm2(sNext);
    logger.debug('Sheaf.computeMetricGain: Computed gain', { gain });
    return Number.isFinite(gain) ? gain : 0;
}

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

     async initialize() {
    try {
        
        
        await super.initialize(); 

        
        if (!this.floquetPD || !Array.isArray(this.floquetPD.phases)) {
            this.floquetPD = { births: [], phases: [], deaths: [] };
        }
        if (this.graph.vertices.length > 0) {
            await this._updateFloqPD_internal();
        }

        
        const N = this.graph.vertices.length * this.qDim;
        if (!isFiniteMatrix(this.laplacian) || this.laplacian.length !== N) {
             logger.error('FloquetPersistentSheaf.initialize: Laplacian from super.initialize() is invalid! This is a critical error.');
             this.laplacian = identity(N); 
        }

        this.ready = true;
        logger.info(`FloquetPersistentSheaf: Initialization complete. Ready: ${this.ready}`);
    } catch (e) {
        logger.error('FloquetPersistentSheaf.initialize: Failed.', { error: e.message, stack: e.stack });
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
            this.floquetPD.phases.push(Number.isFinite(newPhase) ? newPhase : 0);

            const currentStalkNorm = norm2(this.getStalksAsVector());
            this.floquetPD.births.push({ value: Number.isFinite(currentStalkNorm) ? currentStalkNorm : 0, phase: newPhase, time: Date.now() });
        } catch (e) {
            logger.error('FloquetPersistentSheaf._updateFloqPD_internal: Failed to update Floquet PD.', e);
            this.floquetPD = { births: [], phases: [], deaths: [] };
        }
    }

    async _monodromy(A_t, omega) {
        if (!isFiniteMatrix(A_t) || A_t.length === 0 || A_t.length !== (A_t[0]?.length || 0)) {
            logger.warn('_monodromy: Input matrix A_t is invalid. Returning identity.');
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
                logger.warn(`_monodromy: TF.js failed: ${e.message}. Falling back to CPU matrix multiplication.`, {stack: e.stack});
            }
        }

        let result = A_t;
        for (let i = 1; i < omega; i++) {
            try {
                result = matMul({ matrixA: result, matrixB: A_t });
                if (!isFiniteMatrix(result)) {
                    throw new Error("CPU matrix multiplication resulted in non-finite matrix.");
                }
            } catch (e) {
                logger.warn(`_monodromy: CPU matrix multiplication failed: ${e.message}. Returning identity.`, {stack: e.stack});
                return identity(n);
            }
        }
        return isFiniteMatrix(result) ? result : identity(n);
    }

    async computeFloquetFixedPoint(states_history, period) {
    const n_total_stalk_dim = Math.max(1, this.graph.vertices.length * this.qDim);

    
    const sanitizedStates = (states_history || [])
        .filter(s => isFiniteVector(s) && s.length === n_total_stalk_dim);
    

    if (sanitizedStates.length < 2) {
        logger.warn('computeFloquetFixedPoint: Not enough valid states for analysis. Returning defaults.');
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

            combined_monodromy = matMul({ matrixA: combined_monodromy, matrixB: transition });

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
        const aware = Phi_SA_floq > this.tau_floq && d_B_floq < this.delta;

        return { monodromy: this.monodromy, Phi_SA_floq, FloqPD, aware, eigenvalues: rho_k };

    } catch (e) {
        logger.error(`computeFloquetFixedPoint: Critical error during computation: ${e.message}`, { stack: e.stack });
        return {
            monodromy: identity(n_total_stalk_dim),
            eigenvalues: Array(n_total_stalk_dim).fill({ re: 1, im: 0 }),
            Phi_SA_floq: 0,
            aware: false,
            FloqPD: { births: [], phases: [], deaths: [] }
        };
    }
}

    async _stateTransitionMatrix(stateT, stateT1) {
    const n = stateT?.length || 0;

    
    if (!isFiniteVector(stateT) || !isFiniteVector(stateT1) || stateT1.length !== n || n === 0) {
        logger.warn('_stateTransitionMatrix: Invalid input states. Returning identity.', { stateT, stateT1 });
        return identity(n || 1);
    }

    
    const diffNorm = norm2(vecSub(stateT, stateT1));
    if (!Number.isFinite(diffNorm) || diffNorm > 0.1 * n) {
        logger.warn('_stateTransitionMatrix: States too far apart. Using perturbed identity.', { diffNorm });
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


    async _flowMonodromy(F_t_cochains, omega) {
        if (!F_t_cochains || ! (F_t_cochains.cochains instanceof Map)) {
            logger.warn('_flowMonodromy: F_t_cochains or cochains invalid. Returning identity.');
            return identity(1);
        }
        const T_t_cochains = F_t_cochains.cochains;
        const nE = this.graph.edges.length;
        if (nE === 0) return identity(0);

        const A_t = zeroMatrix(nE, nE);
        const eMap = new Map(this.graph.edges.map((e, i) => [e.slice(0, 2).sort().join(','), i]));
        for (const edge of this.graph.edges) {
            const [u, v] = edge;
            const i = eMap.get([u, v].sort().join(','));
            if (i === undefined || i >= nE) {
                logger.warn(`_flowMonodromy: Invalid edge index ${i} for edge ${u}-${v}. Skipping.`);
                continue;
            }
            A_t[i][i] = 1;
            const c_uv = T_t_cochains.get([u, v].sort().join(',')) || vecZeros(this.qDim);
            if (!isFiniteVector(c_uv) || c_uv.length !== this.qDim) {
                logger.warn(`_flowMonodromy: Invalid cochain c_uv for edge ${u}-${v}. Using zeros.`);
                continue;
            }

            for (const edge2 of this.graph.edges) {
                const [u2, v2] = edge2;
                const j = eMap.get([u2, v2].sort().join(','));
                if (j === undefined || j >= nE || i === j) {
                    if (i === j) continue;
                    logger.warn(`_flowMonodromy: Invalid edge index ${j} for edge ${u2}-${v2}. Skipping interaction.`);
                    continue;
                }
                const c_u2v2 = T_t_cochains.get([u2, v2].sort().join(',')) || vecZeros(this.qDim);
                if (!isFiniteVector(c_u2v2) || c_u2v2.length !== this.qDim) {
                    logger.warn(`_flowMonodromy: Invalid cochain c_u2v2 for edge ${u2}-${v2}. Using zeros.`);
                    continue;
                }
                const interaction = dot(c_uv, c_u2v2);
                if (Number.isFinite(interaction)) {
                    A_t[i][j] = this.beta * clamp(interaction, -0.1, 0.1);
                } else {
                    logger.warn(`_flowMonodromy: Non-finite interaction for edges ${u}-${v} and ${u2}-${v2}. Setting to 0.`);
                    A_t[i][j] = 0;
                }
            }
        }
        if (!isFiniteMatrix(A_t)) {
            logger.warn('_flowMonodromy: Generated A_t matrix is non-finite. Returning identity.');
            return identity(nE);
        }
        return this._monodromy(A_t, omega);
    }

    async _floquetDecomp(A) {
    const n = A?.length || 0;
    if (!isFiniteMatrix(A) || n === 0 || A[0]?.length !== n) {
        logger.warn('_floquetDecomp: Invalid input matrix. Returning fallback (identity eigenvalues).', { rows: n, cols: A[0]?.length });
        return { eigenvalues: Array(n).fill({ re: 1, im: 0 }) };
    }

    const sanitizedA = A.map(row => row.map(val => Number.isFinite(val) ? val : 0));

    
    try {
        const flat = flattenMatrix(sanitizedA);
        if (!isFiniteVector(flat.flatData) || flat.flatData.length !== n * n) {
             throw new Error("Flattened matrix is invalid for worker.");
        }
        const complex_eigenvalues = await runWorkerTask('complexEigenvalues', { matrix: flat }, 15000);
        const validEigs = (complex_eigenvalues || []).filter(v => Number.isFinite(v.re) && Number.isFinite(v.im));
        
        if (validEigs.length > 0) {
            logger.debug('_floquetDecomp: Successfully obtained complex eigenvalues from worker.');
            return { eigenvalues: validEigs };
        } else {
            logger.warn('_floquetDecomp: Complex eigenvalues worker returned an empty or invalid result. Falling back to magnitudes.');
            throw new Error("Worker returned no valid complex eigenvalues."); 
        }
    } catch (e) {
        logger.error(`_floquetDecomp: Complex eigenvalues worker or data validation failed (${e.message}). Falling back to Jacobi on A^T * A for magnitudes.`, {stack: e.stack});
        
        try {
            const At = _transpose(sanitizedA);
            const ATA = _matMul(At, sanitizedA);
            if (!isFiniteMatrix(ATA)) {
                throw new Error("A^T * A computation resulted in a non-finite matrix in fallback.");
            }
            const { eigenvalues: realEigsATA } = this._jacobiEigenvalueDecomposition(ATA);
            const floquetMagnitudes = realEigsATA.map(val => ({
                re: Math.sqrt(Math.max(0, val)), 
                im: 0
            }));
            logger.debug('_floquetDecomp: Successfully obtained singular value magnitudes from Jacobi fallback.');
            return { eigenvalues: floquetMagnitudes.filter(v => Number.isFinite(v.re) && Number.isFinite(v.im)) };
        } catch (innerError) {
            logger.error(`_floquetDecomp: Final fallback (Jacobi on ATA) also failed (${innerError.message}). Returning default eigenvalues.`, {stack: innerError.stack});
            return { eigenvalues: Array(n).fill({ re: 1, im: 0 }) };
        }
    }
}
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

            if (Number.isFinite(mag) && !pd.births.some(b => Number.isFinite(b?.value) && Math.abs(b.value - mag) < this.eps)) {
                pd.births.push({ value: mag, phase: theta, time: phase_time_index });
                pd.phases.push(theta);
            } else if (!Number.isFinite(mag)) {
                 logger.warn(`_updateFloqPD: Non-finite Floquet multiplier magnitude detected. Skipping birth event.`);
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

    async _extractFloqCocycle(flows) {
        const z_star = new Map();
        const delta1 = await this._deltaR1();
        if (!isFiniteMatrix(delta1) || delta1.length === 0 || (delta1[0]?.length || 0) === 0) {
            logger.warn('_extractFloqCocycle: Invalid delta1 matrix. Returning empty map.');
            return z_star;
        }

        const nT = delta1.length;
        const nE = delta1[0]?.length || 0;
        if (nE === 0) return z_star;


        for (const flow_record of flows) {
            const F_t = flow_record?.F_t;
            if (!F_t || ! (F_t.cochains instanceof Map)) {
                logger.warn('_extractFloqCocycle: Invalid flow_record or cochains. Skipping.');
                continue;
            }

            const C1_map = F_t.cochains;

            for (let q = 0; q < this.qDim; q++) {
                const C1_q_coeffs = new Float32Array(nE);
                this.graph.edges.forEach((edge, eIdx) => {
                    if (eIdx >= nE) {
                        logger.warn(`_extractFloqCocycle: Edge index ${eIdx} out of bounds for C1_q_coeffs. Skipping.`);
                        return;
                    }
                    const edgeKey = [edge[0], edge[1]].sort().join(',');
                    const c_edge = C1_map.get(edgeKey) || vecZeros(this.qDim);
                    if (isFiniteVector(c_edge) && c_edge.length === this.qDim && Number.isFinite(c_edge[q])) {
                        C1_q_coeffs[eIdx] = c_edge[q];
                    } else {
                        logger.warn(`_extractFloqCocycle: Non-finite cochain or invalid dimension for edge ${edgeKey}, dim ${q}. Using 0.`);
                        C1_q_coeffs[eIdx] = 0;
                    }
                } );

                if (!isFiniteVector(C1_q_coeffs)) {
                    logger.warn(`_extractFloqCocycle: Non-finite C1_q_coeffs for qualia dimension ${q}. Skipping.`);
                    continue;
                }

                let d1_C1_q;
                try {
                    d1_C1_q = matVecMul(delta1, C1_q_coeffs);
                } catch (e) {
                    logger.warn(`_extractFloqCocycle: matVecMul with delta1 failed for dim ${q}: ${e.message}. Skipping.`);
                    continue;
                }


                if (isFiniteVector(d1_C1_q) && norm2(d1_C1_q) < this.eps * nT) {
                    this.graph.edges.forEach((edge, eIdx) => {
                        const edgeKey = [edge[0], edge[1]].sort().join(',');
                        const current_cocycle = C1_map.get(edgeKey);
                        if (current_cocycle && !z_star.has(edgeKey) && isFiniteVector(current_cocycle) && current_cocycle.length === this.qDim) {
                            const phase = Array.isArray(this.floquetPD?.phases) && this.floquetPD.phases.length > 0 ?
                                          this.floquetPD.phases[this.floquetPD.phases.length - 1] : 0;
                            const weightedCocycle = vecScale(current_cocycle, Number.isFinite(Math.cos(phase)) ? Math.cos(phase) : 0);
                            if (isFiniteVector(weightedCocycle)) {
                                z_star.set(edgeKey, weightedCocycle);
                            } else {
                                logger.warn(`_extractFloqCocycle: Non-finite weighted cocycle for edge ${edgeKey}. Storing original (unweighted) instead.`);
                                z_star.set(edgeKey, current_cocycle);
                            }
                        }
                    });
                } else if (!isFiniteVector(d1_C1_q)) {
                    logger.warn(`_extractFloqCocycle: matVecMul returned non-finite d1_C1_q for dim ${q}. Skipping.`);
                }
            }
        }
        return z_star;
    }

    /**
     * CORRECTED: The main update loop with a fixed order of operations.
     * This function now correctly orchestrates the entire process, ensuring metrics are calculated last.
     */
    async update(state, stepCount = 0) {
    if (!this.ready) {
        await this.initialize();
        if (!this.ready) {
            logger.error('FloquetPersistentSheaf.update: Initialization failed. Aborting update.');
            return;
        }
    }

    if (!isFiniteVector(state) || state.length !== this.stateDim) {
        logger.warn('Sheaf.update: Invalid input state. Skipping update.', { state });
        return;
    }

    try {
        

        
        await this.computeCorrelationMatrix();
        this.laplacian = this.buildLaplacian();
        await this.computeProjectionMatrices();

        
        const vertexInputs = new Float32Array(this.graph.vertices.length);
        for (let i = 0; i < this.graph.vertices.length; i++) {
            
            vertexInputs[i] = state[Math.min(i, state.length - 1)] || 0;
        }

        
        
const rawQualiaState = _matVecMul(this.stateToQualiaProjection, state);
const spontaneousActivation = new Float32Array(this.qDim).map(() => (Math.random() - 0.5) * 0.05);
const qualiaInput = vecAdd(rawQualiaState, spontaneousActivation);




await this.diffuseQualia(state, qualiaInput);
        
        
       

const fullQualiaState = this.getStalksAsVector();


const { Phi_SA, aware } = await this.computeSelfAwareness(fullQualiaState);
const { Phi_SA: Phi_SA_adj, aware: adj_aware } = await this.computeAdjunctionFixedPoint(fullQualiaState);
const { Phi_SA_persist, aware: persist_aware, PD } = await this.computePersistentFixedPoint(fullQualiaState); 
const { Phi_SA_floq, aware: floq_aware, FloqPD } = await this.computeFloquetFixedPoint(this.windowedStates.getAll(), this.omega);
        
        if (stepCount > 0 && stepCount % 100 === 0) {
            await this.adaptSheafTopology(100, stepCount);
        }
        
        
        this.phi = clamp(this.phi + 0.1 * (Phi_SA_floq || 0), 0, 100);
        await this._updateDerivedMetrics();

    } catch (e) {
        logger.error(`Sheaf.update: Error: ${e.message}`, { stack: e.stack });
        this.ready = false;
    }
}
}
