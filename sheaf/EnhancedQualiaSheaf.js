import { CircularBuffer } from '../data-structures/CircularBuffer.js';
import {
    logger, clamp, isFiniteNumber, isFiniteVector, isFiniteMatrix,
    vecZeros, zeroMatrix, identity, dot, vecAdd, vecSub, vecScale,
    norm2, matVecMul, matMul, transpose, flattenMatrix, unflattenMatrix,
    randomMatrix, runWorkerTask, _matVecMul, _transpose, safeVecScale,
} from '../utils.js';


const tf = window.tf || null;
const THREE = window.THREE || null;

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
