// --- START OF FILE utils.js ---
// --- On-page Logger ---
const logElement = document.getElementById('log');
const MAX_LOG_ENTRIES = 200;

function appendLog(message, type = 'info') {
    if (!logElement) {
        console.warn('Log element not found, logging to console:', message);
        return;
    }
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logElement.appendChild(entry);
    if (logElement.children.length > MAX_LOG_ENTRIES) logElement.removeChild(logElement.children[0]);
    logElement.scrollTop = logElement.scrollHeight;
}

export const logger = {
    info: (msg, data) => appendLog(`INFO: ${msg}` + (data ? ' ' + JSON.stringify(data) : '')),
    warn: (msg, data) => appendLog(`WARN: ${msg}` + (data ? ' ' + JSON.stringify(data) : ''), 'warn'),
    error: (msg, data) => { appendLog(`ERROR: ${msg}` + (data ? ' ' + JSON.stringify(data) : ''), 'error'); console.error(msg, data); }
};

// --- Loading Overlay Manager ---
const loadingOverlays = {
    metrics: document.getElementById('metricsLoadingOverlay'),
    game: document.getElementById('gameLoadingOverlay'),
    mainBrain: document.getElementById('mainBrainLoadingOverlay'),
    opponentBrain: document.getElementById('opponentBrainLoadingOverlay'),
};
const loadingTexts = {
    metrics: document.getElementById('metricsLoadingText'),
    game: document.getElementById('gameLoadingText'),
    mainBrain: document.getElementById('mainBrainLoadingText'),
    opponentBrain: document.getElementById('opponentBrainLoadingText'),
};

export function showLoading(panel, message = 'Processing...') {
    if (loadingOverlays[panel]) {
        loadingOverlays[panel].classList.add('active');
        if (loadingTexts[panel]) loadingTexts[panel].textContent = message;
    }
}

export function hideLoading(panel) {
    if (loadingOverlays[panel]) {
        loadingOverlays[panel].classList.remove('active');
    }
}

// --- Web Worker for Heavy Computations ---
const workerLogicString = `
// Worker scope utility functions (must be self-contained)

function clamp(v, min, max) {
    const safeV = Number.isFinite(v) ? v : 0;
    const safeMin = Number.isFinite(min) ? min : -Infinity;
    const safeMax = Number.isFinite(max) ? max : Infinity;
    return Math.max(safeMin, Math.min(safeMax, safeV));
}

function dot(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let s = 0.0;
    for (let i = 0, n = a.length; i < n; ++i) {
        const val_a = a[i];
        const val_b = b[i];
        s += (Number.isFinite(val_a) ? val_a : 0) * (Number.isFinite(val_b) ? val_b : 0);
    }
    return s;
}

function norm2(v) {
    if (!v || v.length === 0) return 0;
    let s = 0;
    for (let i = 0; i < v.length; ++i) {
        const val = v[i];
        s += (Number.isFinite(val) ? val : 0) * (Number.isFinite(val) ? val : 0);
    }
    return Math.sqrt(s + 1e-12);
}

function isFiniteVector(v) {
    if (!v || !(Array.isArray(v) || v instanceof Float32Array)) return false;
    for (let i = 0; i < v.length; i++) {
        if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) return false;
    }
    return true;
}

function isFiniteMatrix(m) {
    if (!Array.isArray(m) || m.length === 0) return true;
    const firstRowLength = m[0] && m[0].length !== undefined ? m[0].length : 0;
    if (m.length > 0 && firstRowLength === 0) return true;
    return m.every(row => (Array.isArray(row) || row instanceof Float32Array) && row.length === firstRowLength && isFiniteVector(row));
}

function matVecMul(m, v) {
    const r = m && m.length !== undefined ? m.length : 0;
    if (r === 0) return new Float32Array(0);
    const c = m[0] && m[0].length !== undefined ? m[0].length : 0;
    
    if (!isFiniteMatrix(m)) return new Float32Array(r).fill(0);
    if (!isFiniteVector(v) || v.length !== c) return new Float32Array(r).fill(0);

    const out = new Float32Array(r);
    for (let i = 0; i < r; ++i) {
        let s = 0.0;
        const row = m[i];
        for (let j = 0; j < c; ++j) {
            s += (row[j] || 0) * (v[j] || 0);
        }
        out[i] = s;
    }
    return out;
}

function unflattenMatrix(data) {
    if (!data || !data.flatData || !isFiniteVector(data.flatData) || !Number.isFinite(data.rows) || !Number.isFinite(data.cols) || data.flatData.length !== data.rows * data.cols) {
        return [];
    }
    const { flatData, rows, cols } = data;
    const matrix = [];
    for (let i = 0; i < rows; i++) {
        const row = new Float32Array(cols);
        for(let j = 0; j < cols; j++) {
            row[j] = flatData[i * cols + j];
        }
        matrix.push(row);
    }
    return matrix;
}

function flattenMatrix(matrix) {
    if (!isFiniteMatrix(matrix)) {
        return { flatData: new Float32Array(0), rows: 0, cols: 0 };
    }
    const rows = matrix.length;
    const cols = (matrix[0] && matrix[0].length !== undefined) ? matrix[0].length : 0;
    if (rows === 0 || cols === 0) return { flatData: new Float32Array(0), rows, cols };

    const flatData = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        for(let j = 0; j < cols; j++) {
            flatData[i * cols + j] = matrix[i][j] || 0;
        }
    }
    return { flatData, rows, cols };
}

function transpose(matrix) {
    if (!isFiniteMatrix(matrix)) return [];
    const numRows = matrix.length;
    const numCols = matrix[0] && matrix[0].length !== undefined ? matrix[0].length : 0;
    if (numRows === 0 || numCols === 0) return [];

    const result = Array(numCols).fill(0).map(() => new Float32Array(numRows));
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            result[j][i] = matrix[i][j] || 0;
        }
    }
    return result;
}

// --- NEWLY IMPLEMENTED WORKER FUNCTIONS ---

function matMul(data) {
    const A = unflattenMatrix(data.matrixA);
    let B = unflattenMatrix(data.matrixB);

    if (data.transposeB) {
        B = transpose(B);
    }

    const A_rows = A.length;
    const A_cols = A[0] ? A[0].length : 0;
    const B_rows = B.length;
    const B_cols = B[0] ? B[0].length : 0;
    
    if (A_cols !== B_rows) {
        console.warn('Worker matMul: Incompatible matrix dimensions for multiplication.');
        return [];
    }

    const C = Array(A_rows).fill(0).map(() => new Float32Array(B_cols).fill(0));
    for (let i = 0; i < A_rows; i++) {
        for (let j = 0; j < B_cols; j++) {
            let sum = 0;
            for (let k = 0; k < A_cols; k++) {
                sum += (A[i][k] || 0) * (B[k][j] || 0);
            }
            C[i][j] = sum;
        }
    }
    return C;
}

function persistentHomology(data) {
    const filtration = data.filtration;
    const n = filtration.length;
    let edges = [];
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const correlation = filtration[i][j] || 0;
            edges.push({ u: i, v: j, weight: 1 - Math.abs(correlation) });
        }
    }
    edges.sort((a, b) => a.weight - b.weight);

    const parent = Array(n).fill(0).map((_, i) => i);
    const find = (i) => (parent[i] === i) ? i : (parent[i] = find(parent[i]));
    const union = (i, j) => {
        const rootI = find(i);
        const rootJ = find(j);
        if (rootI !== rootJ) {
            parent[rootI] = rootJ;
            return true;
        }
        return false;
    };

    let dimH1 = 0;
    for (const edge of edges) {
        if (!union(edge.u, edge.v)) {
            // This edge creates a cycle (a 1D hole is born)
            if (edge.weight < 0.5) { // Heuristic: only count strong, persistent cycles
                 dimH1++;
            }
        }
    }
    // Heuristic reduction for filled cycles
    dimH1 = Math.max(0, dimH1 - n / 3);
    return { dimH1: clamp(dimH1, 0, n / 2) };
}

function ksgMutualInformation(data) {
    const states = data.states;
    const k = data.k || 3;
    const n = states.length;
    if (n < 2 * k) return 0;
    const d = states[0].length;
    const d1 = Math.floor(d / 2);
    
    // Digamma function approximation
    const digamma = (x) => Math.log(x) - 1.0 / (2.0 * x);

    let mi = 0;
    for (let i = 0; i < n; i++) {
        const p_i = states[i];
        let distances = [];
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const p_j = states[j];
            let max_dist = 0;
            for (let dim = 0; dim < d; dim++) {
                max_dist = Math.max(max_dist, Math.abs(p_i[dim] - p_j[dim]));
            }
            distances.push(max_dist);
        }
        
        distances.sort((a, b) => a - b);
        const kth_dist = distances[k - 1];

        if(kth_dist <= 1e-9) continue;

        let nx = 0, ny = 0;
        for (let j = 0; j < n; j++) {
             if (i === j) continue;
             const p_j = states[j];
             let dist_x = 0;
             for (let dim = 0; dim < d1; dim++) dist_x = Math.max(dist_x, Math.abs(p_i[dim] - p_j[dim]));
             if(dist_x < kth_dist) nx++;

             let dist_y = 0;
             for (let dim = d1; dim < d; dim++) dist_y = Math.max(dist_y, Math.abs(p_i[dim] - p_j[dim]));
             if(dist_y < kth_dist) ny++;
        }
        mi += digamma(k) - (digamma(nx + 1) + digamma(ny + 1)) / 2;
    }
    
    return Math.max(0, (mi / n) + digamma(n));
}

// --- END OF NEWLY IMPLEMENTED FUNCTIONS ---

// ... (rest of existing worker functions like solveLinearSystemCG, covarianceMatrix, etc.)

function solveLinearSystemCG(A_flat_data, b, opts={tol:1e-6, maxIter:200}) {
    const A = unflattenMatrix(A_flat_data);
    const n = b.length;
    if (!isFiniteMatrix(A) || A.length !== n || (A.length > 0 && A[0].length !== n) || !isFiniteVector(b)) {
        return new Float32Array(b.map(x => Number.isFinite(x) ? clamp(x, -1, 1) : 0));
    }
    let x = new Float32Array(n).fill(0);
    let r = new Float32Array(b.map(v => Number.isFinite(v) ? v : 0));
    let p = new Float32Array(r);
    let rsold = dot(r, r);
    if (!Number.isFinite(rsold) || rsold < 1e-20) {
        return new Float32Array(x.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
    }
    const Ap = new Float32Array(n);
    const maxIter = Math.min(opts.maxIter || 200, n * 5);
    for (let it = 0; it < maxIter; ++it) {
        Ap.set(matVecMul(A, p));
        const denom = dot(p, Ap);
        if (!Number.isFinite(denom) || denom <= 1e-20) break;
        const alpha = rsold / denom;
        if (!Number.isFinite(alpha)) break;
        for (let i = 0; i < n; i++) x[i] += alpha * p[i];
        for (let i = 0; i < n; i++) r[i] -= alpha * Ap[i];
        const rsnew = dot(r, r);
        if (!Number.isFinite(rsnew)) break;
        if (Math.sqrt(rsnew) < (opts.tol || 1e-6)) break;
        const beta = rsnew / (rsold + 1e-20);
        if (!Number.isFinite(beta)) break;
        for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
    return new Float32Array(x.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
}

function covarianceMatrix(states_array, eps = 1e-3) {
    const states = states_array.filter(s => isFiniteVector(s));
    if (!Array.isArray(states) || states.length < 2) return [[eps]];
    const n = states.length;
    const d = states[0] ? states[0].length : 0;
    if (d === 0) return [[eps]];
    const mean = new Float32Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += states[i][j] || 0;
    for (let j = 0; j < d; j++) mean[j] /= n;
    const cov = Array(d).fill(0).map(() => new Float32Array(d).fill(0));
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < d; i++) {
            const di = (states[k][i] || 0) - mean[i];
            for (let j = i; j < d; j++) {
                const dj = (states[k][j] || 0) - mean[j];
                cov[i][j] += di * dj / Math.max(1, n - 1);
            }
        }
    }
    for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) cov[j][i] = cov[i][j];
    for (let i = 0; i < d; i++) cov[i][i] += eps;
    return cov;
}

function matrixSpectralNormApprox(M_flat_data, maxIter = 10) {
    const M = unflattenMatrix(M_flat_data);
    if (!isFiniteMatrix(M)) return 0;
    let n = M.length;
    if (n === 0) return 0;
    let v = new Float32Array(n).fill(1 / Math.sqrt(n));
    for (let i = 0; i < maxIter; i++) {
        const Av = matVecMul(M, v);
        const norm = norm2(Av);
        if (norm < 1e-10) return 0;
        for(let k=0; k<n; k++) v[k] = (Av[k] || 0) / norm;
    }
    return norm2(matVecMul(M, v));
}

function smithNormalFormGF2(matrixData) {
    if (!matrixData || !matrixData.flatData || !Array.isArray(matrixData.flatData) || !Number.isFinite(matrixData.rows) || !Number.isFinite(matrixData.cols)) {
        console.warn('Worker smithNormalFormGF2: Invalid matrixData. Returning zero diagonal.', { matrixData });
        return { diagonal: [], rank: 0 };
    }

    const M = unflattenMatrix(matrixData);
    if (!isFiniteMatrix(M) || M.length === 0) {
        console.warn('Worker smithNormalFormGF2: Input matrix is non-finite or empty. Returning zero diagonal.');
        return { diagonal: [], rank: 0 };
    }

    let rank = 0;
    const rows = M.length;
    const cols = M[0].length;
    const workingM = M.map(row => new Float32Array(row).map(x => Number.isFinite(x) ? Math.abs(x) % 2 : 0)); // Ensure GF(2)

    for (let col = 0; col < cols && rank < rows; col++) {
        let pivotRow = rank;
        for (let row = rank + 1; row < rows; row++) {
            if (workingM[row][col] > workingM[pivotRow][col]) {
                pivotRow = row;
            }
        }
        if (workingM[pivotRow][col] !== 1) continue;

        [workingM[rank], workingM[pivotRow]] = [workingM[pivotRow], workingM[rank]];

        for (let row = 0; row < rows; row++) {
            if (row !== rank && workingM[row][col] === 1) {
                for (let j = col; j < cols; j++) {
                    workingM[row][j] = (workingM[row][j] + workingM[rank][j]) % 2;
                }
            }
        }
        rank++;
    }

    // Construct diagonal array
    const diagonal = new Float32Array(Math.min(rows, cols)).fill(0);
    for (let i = 0; i < rank; i++) {
        diagonal[i] = 1; // GF(2): 1s for non-zero entries
    }

    return { diagonal, rank };
}

self.onmessage = function(e) {
    const { type, id, data } = e.data;
    let result;
    try {
        switch (type) {
            case 'transpose':
                result = flattenMatrix(transpose(unflattenMatrix(data.matrix)));
                break;
            case 'matVecMul':
                result = matVecMul(unflattenMatrix(data.matrix), data.vector);
                break;
            case 'solveLinearSystemCG':
                result = solveLinearSystemCG(data.A, data.b, data.opts);
                break;
            case 'covarianceMatrix':
                result = covarianceMatrix(data.states, data.eps);
                break;
            case 'matrixSpectralNormApprox':
                result = matrixSpectralNormApprox(data.matrix);
                break;
            case 'smithNormalForm':
                result = smithNormalFormGF2(data.matrix);
                break;
            // --- ADDED CASES FOR NEW TASKS ---
            case 'matMul':
                const matMulResult = matMul(data);
                result = isFiniteMatrix(matMulResult) ? matMulResult : [];
                break;
            case 'persistentHomology':
                result = persistentHomology(data);
                if (!Number.isFinite(result.dimH1)) result = { dimH1: 0 };
                break;
            case 'ksgMutualInformation':
                result = ksgMutualInformation(data);
                if (!Number.isFinite(result)) result = 0;
                break;
            default:
                result = { error: 'Unknown message type' };
                break;
        }
        self.postMessage({ type: type + 'Result', id, result });
    } catch (error) {
        console.error('Worker error for type ' + type + ':', error);
        self.postMessage({ type: type + 'Error', id, error: error.message || String(error) });
    }
};
`;

const worker = new Worker(URL.createObjectURL(new Blob([workerLogicString], { type: 'application/javascript' })));

const workerCallbacks = new Map();
let nextWorkerTaskId = 0;

export function runWorkerTask(type, data, timeout = 20000) {
    return new Promise((resolve, reject) => {
        const id = nextWorkerTaskId++;
        const timer = setTimeout(() => {
            workerCallbacks.delete(id);
            const errorMessage = `Worker task '${type}' (id: ${id}) timed out after ${timeout}ms.`;
            logger.error(errorMessage);
            reject(new Error(errorMessage));
        }, timeout);
        workerCallbacks.set(id, {
            resolve: (result) => {
                clearTimeout(timer);
                // --- ADDED SAFE DEFAULTS FOR NEW TASKS ---
                if (result === null || result === undefined) {
                    logger.error(`Worker task ${type} (id: ${id}) returned null/undefined result.`);
                    if (type === 'matVecMul' || type === 'solveLinearSystemCG') {
                         result = vecZeros((data.b || data.vector || []).length);
                    } else if (type === 'covarianceMatrix' || type === 'matMul') {
                         result = [];
                    } else if (type === 'matrixSpectralNormApprox' || type === 'ksgMutualInformation') {
                         result = 0;
                    } else if (type === 'persistentHomology') {
                        result = { dimH1: 0 };
                    } else if (type === 'smithNormalForm') {
                         result = { rank: 0 };
                    } else if (type === 'transpose') {
                         result = { flatData: new Float32Array(0), rows: 0, cols: 0 };
                    }
                }
                resolve(result);
            },
            reject: (error) => {
                clearTimeout(timer);
                logger.error(`Worker task ${type} (id: ${id}) error: ${error.message || error.toString()}. Rejecting.`);
                reject(error);
            },
        });
        worker.postMessage({ type, id, data });
    });
}


worker.onmessage = function(e) {
    const { id, result, error, type: messageType } = e.data;
    const callback = workerCallbacks.get(id);
    if (callback) {
        if (error) {
            logger.error(`Worker message error for type ${messageType.replace('Result', '')} (id: ${id}): ${error}`);
            callback.reject(new Error(error));
        } else {
            callback.resolve(result);
        }
        workerCallbacks.delete(id);
    }
};

worker.onerror = function(error) {
    logger.error('CRITICAL ERROR: Worker global error:', error.message || error.toString());
    workerCallbacks.forEach(cb => cb.reject(new Error('Worker crashed')));
    workerCallbacks.clear();
};


// --- CORE UTILITY FUNCTIONS (for main thread) ---
// ... (rest of the file is identical to the original and does not need to be changed)
export function clamp(v, min, max) {
    const safeV = Number.isFinite(v) ? v : 0;
    const safeMin = Number.isFinite(min) ? min : -Infinity;
    const safeMax = Number.isFinite(max) ? max : Infinity;
    return Math.max(safeMin, Math.min(safeMax, safeV));
}
export function dot(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let s = 0.0;
    for (let i = 0, n = a.length; i < n; ++i) {
        const ai = a[i], bi = b[i];
        if (!Number.isFinite(ai) || !Number.isFinite(bi)) return 0;
        s += ai * bi;
    }
    return s;
}
export function norm2(v) {
    if (!Array.isArray(v) && !(v instanceof Float32Array)) return 0;
    let s = 0;
    for (let i = 0; i < v.length; ++i) {
        const val = v[i];
        s += (Number.isFinite(val) ? val : 0) * (Number.isFinite(val) ? val : 0);
    }
    return Math.sqrt(s + 1e-12);
}
export function vecAdd(a, b) {
    const n = Math.max((a && a.length) || 0, (b && b.length) || 0);
    const out = new Float32Array(n);
    for (let i = 0; i < n; ++i) out[i] = (Number.isFinite(a[i]) ? a[i] : 0) + (Number.isFinite(b[i]) ? b[i] : 0);
    return out;
}
export function vecSub(a, b) {
    const n = Math.max((a && a.length) || 0, (b && b.length) || 0);
    const out = new Float32Array(n);
    for (let i = 0; i < n; ++i) out[i] = (Number.isFinite(a[i]) ? a[i] : 0) - (Number.isFinite(b[i]) ? b[i] : 0);
    return out;
}
export function vecScale(v, s) {
    const n = (v && v.length) || 0;
    const out = new Float32Array(n);
    for (let i = 0; i < n; ++i) out[i] = (Number.isFinite(v[i]) ? v[i] : 0) * (Number.isFinite(s) ? s : 0);
    return out;
}
export function tanhVec(v) {
    if (!isFiniteVector(v)) { logger.warn('tanhVec: Input vector is not finite. Returning zeros.'); return vecZeros((v && v.length) || 0); }
    return new Float32Array(v.map(x => Math.tanh(x)));
}
export function sigmoidVec(v) {
    if (!isFiniteVector(v)) { logger.warn('sigmoidVec: Input vector is not finite. Returning zeros.'); return vecZeros((v && v.length) || 0); }
    return new Float32Array(v.map(x => 1 / (1 + Math.exp(-x))));
}
export function vecMul(a, b) {
    const n = Math.max((a && a.length) || 0, (b && b.length) || 0);
    const out = new Float32Array(n);
    for (let i = 0; i < n; ++i) out[i] = (Number.isFinite(a[i]) ? a[i] : 0) * (Number.isFinite(b[i]) ? b[i] : 0);
    return out;
}
export function randomMatrix(r, c, scale) {
    return Array(r).fill().map(() => new Float32Array(c).fill().map(() => clamp((Math.random() - 0.5) * scale, -1, 1)));
}
export function vecZeros(n) { return new Float32Array(n).fill(0); }
export function zeroMatrix(r, c) { return Array(r).fill().map(() => new Float32Array(c).fill(0)); }
export function identity(n) {
    const M = zeroMatrix(n, n);
    for (let i = 0; i < n; i++) M[i][i] = 1;
    return M;
}
export function isFiniteVector(v) {
    if (!v || !(Array.isArray(v) || v instanceof Float32Array)) return false;
    if (v.length > 0 && typeof v[0] === 'object' && v[0] !== null) return false;
    for (let i = 0; i < v.length; i++) {
        if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) return false;
    }
    return true;
}
export function isFiniteMatrix(m) {
    if (!Array.isArray(m)) return false;
    if (m.length === 0) return true;
    const firstRowLength = m[0] && m[0].length !== undefined ? m[0].length : 0;
    return m.every(row => (Array.isArray(row) || row instanceof Float32Array) && row.length === firstRowLength && isFiniteVector(row));
}
export function flattenMatrix(matrix) {
    if (!isFiniteMatrix(matrix)) { logger.warn('flattenMatrix: Input matrix is not finite. Returning empty.'); return { flatData: new Float32Array(0), rows: 0, cols: 0 }; }
    const rows = matrix.length;
    const cols = (matrix[0] && matrix[0].length !== undefined) ? matrix[0].length : 0;
    if (rows === 0 || cols === 0) return { flatData: new Float32Array(0), rows, cols };
    const flatData = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) for(let j = 0; j < cols; j++) flatData[i * cols + j] = matrix[i][j];
    return { flatData, rows, cols };
}
export function unflattenMatrix(data) {
    if (!data || !data.flatData || !Number.isFinite(data.rows) || !Number.isFinite(data.cols) || data.flatData.length !== data.rows * data.cols) { return []; }
    const { flatData, rows, cols } = data;
    const matrix = [];
    for (let i = 0; i < rows; i++) {
        const row = new Float32Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = flatData[i * cols + j];
        }
        matrix.push(row);
    }
    return matrix;
}
export function logDeterminantFromDiagonal(M) {
    if (!Array.isArray(M) || M.length === 0) return Math.log(1e-12);
    let s = 0;
    for (let i=0;i<M.length;i++) {
        const val = (M[i] && M[i][i] !== undefined) ? M[i][i] : 0;
        if (!Number.isFinite(val)) { logger.warn('logDeterminantFromDiagonal: Non-finite diagonal element. Skipping.'); continue; }
        s += Math.log(Math.max(Math.abs(val), 1e-12));
    }
    return s;
}
export function softmax(logits) {
    if (!logits || !(logits instanceof Float32Array) || !isFiniteVector(logits) || logits.length === 0) {
        const fallbackLength = (logits && logits.length > 0) ? logits.length : 4;
        logger.warn(`Softmax input logits are invalid or empty. Returning uniform probabilities for length ${fallbackLength}.`);
        return vecZeros(fallbackLength).fill(1 / fallbackLength);
    }
    const maxLogit = Math.max(...logits);
    if (!Number.isFinite(maxLogit)) {
        logger.warn('Softmax maxLogit is non-finite. Returning uniform probabilities.');
        return vecZeros(logits.length).fill(1 / logits.length);
    }
    const exp_logits = new Float32Array(logits.length);
    for(let i = 0; i < logits.length; i++) {
        const val = Math.exp(logits[i] - maxLogit);
        exp_logits[i] = Number.isFinite(val) ? val : 0;
    }
    let sum_exp_logits = 0;
    for(let i = 0; i < exp_logits.length; i++) sum_exp_logits += exp_logits[i];
    const safe_sum_exp_logits = (Number.isFinite(sum_exp_logits) && sum_exp_logits > 1e-9) ? sum_exp_logits : 1e-9;
    const resultProbs = new Float32Array(logits.length);
    for(let i = 0; i < logits.length; i++) {
        const val = exp_logits[i] / safe_sum_exp_logits;
        resultProbs[i] = Number.isFinite(val) ? val : 0;
    }
    if (!isFiniteVector(resultProbs)) {
        logger.warn('Softmax output probabilities are non-finite after calculation. Returning uniform probabilities.');
        return vecZeros(logits.length).fill(1 / logits.length);
    }
    return resultProbs;
}
