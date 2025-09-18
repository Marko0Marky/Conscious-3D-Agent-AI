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

function topologicalScore(data) {
    const states = data.states;
    const filtration = data.filtration;
    if (!Array.isArray(states) || states.length < 2 || !isFiniteMatrix(filtration)) {
        return { score: 0 };
    }
    const { dimH1 } = persistentHomology({ filtration });
    const n = states.length;
    const d = states[0].length;
    let score = dimH1 * 0.5;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let dist = 0;
            for (let k = 0; k < d; k++) {
                dist += (states[i][k] - states[j][k]) ** 2;
            }
            score += clamp(1 - Math.sqrt(dist) / d, 0, 1) * 0.01;
        }
    }
    return { score: clamp(score / n, 0, 1) };
}

function matVecMul(matrixData, vector) {
    const A = Array.isArray(matrixData) ? matrixData : unflattenMatrix(matrixData || { flatData: [], rows: 0, cols: 0 });
    const v = vector && (Array.isArray(vector) || vector instanceof Float32Array) ? vector : [];

    if (!isFiniteMatrix(A) || A.length === 0 || !A[0] || A[0].length === 0 || A[0].length !== v.length) {
        console.warn('Worker matVecMul: Invalid matrix or vector (non-finite, empty, or dimension mismatch). Returning zero vector.');
        const resultLength = (A && A.length > 0) ? A.length : (vector ? vector.length : 1);
        return Array(resultLength).fill(0);
    }
    
    const n = A.length;
    const m = A[0].length;
    const result = new Float32Array(n);
    
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < m; j++) {
            sum += (A[i][j] || 0) * (v[j] || 0);
        }
        result[i] = sum;
    }
    
    return Array.from(result);
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

function matMul(data) {
    const A = unflattenMatrix(data.matrixA);
    let B = unflattenMatrix(data.matrixB);

    if (!isFiniteMatrix(A) || !isFiniteMatrix(B)) {
        return [];
    }

    const A_rows = A.length;
    const A_cols = A[0] ? A[0].length : 0;
    const B_rows = B.length;
    const B_cols = B[0] ? B[0].length : 0;
    
    if (A_rows === 0 || A_cols === 0 || B_rows === 0 || B_cols === 0 || A_cols !== B_rows) {
        return [];
    }

    if (data.transposeB) {
        B = transpose(B);
        if (!isFiniteMatrix(B)) return [];
    }

    const C = Array(A_rows).fill(0).map(() => new Float32Array(B_cols).fill(0));
    for (let i = 0; i < A_rows; i++) {
        for (let j = 0; j < B_cols; j++) {
            let sum = 0;
            for (let k = 0; k < A_cols; k++) {
                const a_ik = A[i][k] || 0;
                const b_kj = B[k][j] || 0;
                sum += a_ik * b_kj;
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
            if (edge.weight < 0.5) {
                dimH1++;
            }
        }
    }
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

        if (kth_dist <= 1e-9) continue;

        let nx = 0, ny = 0;
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const p_j = states[j];
            let dist_x = 0;
            for (let dim = 0; dim < d1; dim++) dist_x = Math.max(dist_x, Math.abs(p_i[dim] - p_j[dim]));
            if (dist_x < kth_dist) nx++;

            let dist_y = 0;
            for (let dim = d1; dim < d; dim++) dist_y = Math.max(dist_y, Math.abs(p_i[dim] - p_j[dim]));
            if (dist_y < kth_dist) ny++;
        }
        mi += digamma(k) - (digamma(nx + 1) + digamma(ny + 1)) / 2;
    }
    
    return Math.max(0, (mi / n) + digamma(n));
}

function eigenvalues(data) {
    const M = unflattenMatrix(data.matrix);
    if (!isFiniteMatrix(M) || M.length === 0 || M[0].length === 0 || M.length !== M[0].length) {
        return [];
    }
    const n = M.length;
    const maxIter = Math.min(n * 20, 1000);
    const tol = 1e-6;

    let v = new Float32Array(n).fill(1 / Math.sqrt(n));
    let lambda = 0;
    let eigenvalues = [];

    for (let k = 0; k < Math.min(n, 5); k++) {
        for (let iter = 0; iter < maxIter; iter++) {
            let v_new = matVecMul(M, v);
            const norm = norm2(v_new);
            if (norm < 1e-10) break;
            v_new = v_new.map(x => x / norm);
            const lambda_new = dot(v_new, matVecMul(M, v_new));
            if (Math.abs(lambda_new - lambda) < tol) {
                eigenvalues.push(Number.isFinite(lambda_new) ? lambda_new : 0);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        M[i][j] -= lambda_new * v[i] * v[j];
                    }
                }
                break;
            }
            lambda = lambda_new;
            v = v_new;
        }
        v = new Float32Array(n).fill(1 / Math.sqrt(n));
        lambda = 0;
    }
    return eigenvalues.filter(Number.isFinite);
}

function zeroMatrix(r, c) {
    return Array(r).fill(0).map(() => new Float32Array(c).fill(0));
}

function identity(n) {
    const M = zeroMatrix(n, n);
    for (let i = 0; i < n; i++) M[i][i] = 1;
    return M;
}


function complexEigenvalues(data) {
    const M = unflattenMatrix(data.matrix);
    if (!isFiniteMatrix(M) || M.length === 0 || M.length !== M[0]?.length) {
        return [{ re: 1, im: 0 }];
    }
    const n = M.length;
    if (n === 1) return [{ re: M[0][0], im: 0 }];

    let A = M.map(row => Array.from(row));
    const maxIter = 30;
    const tol = 1e-7;

    for (let k = 0; k < maxIter; k++) {
        // Simple QR decomposition (Gram-Schmidt)
        let Q = zeroMatrix(n, n).map(row => Array.from(row));
        let R = zeroMatrix(n, n).map(row => Array.from(row));
        for (let j = 0; j < n; j++) {
            let v = A.map(row => row[j]);
            for (let i = 0; i < j; i++) {
                const qi = Q.map(row => row[i]);
                const r_ij = dot(qi, v);
                for (let k = 0; k < n; k++) {
                    v[k] -= r_ij * qi[k];
                }
                R[i][j] = r_ij;
            }
            const norm_v = norm2(v);
            R[j][j] = norm_v;
            if (norm_v > tol) {
                for (let i = 0; i < n; i++) {
                    Q[i][j] = v[i] / norm_v;
                }
            }
        }
        // A_next = R * Q
        const nextA = zeroMatrix(n, n).map(row => Array.from(row));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let l = 0; l < n; l++) {
                    sum += R[i][l] * Q[l][j];
                }
                nextA[i][j] = sum;
            }
        }
        A = nextA;
    }

    const eigenvalues = [];
    for (let i = 0; i < n; ) {
        if (i + 1 < n && Math.abs(A[i+1][i]) > tol) {
            const a = A[i][i], b = A[i][i+1], c = A[i+1][i], d = A[i+1][i+1];
            const trace = a + d;
            const det = a * d - b * c;
            const discriminant = trace * trace - 4 * det;
            if (discriminant < 0) {
                eigenvalues.push({ re: trace / 2, im: Math.sqrt(-discriminant) / 2 });
                eigenvalues.push({ re: trace / 2, im: -Math.sqrt(-discriminant) / 2 });
            } else {
                eigenvalues.push({ re: (trace + Math.sqrt(discriminant)) / 2, im: 0 });
                eigenvalues.push({ re: (trace - Math.sqrt(discriminant)) / 2, im: 0 });
            }
            i += 2;
        } else {
            eigenvalues.push({ re: A[i][i], im: 0 });
            i++;
        }
    }
    return eigenvalues.filter(v => Number.isFinite(v.re) && Number.isFinite(v.im));
}


function solveLinearSystemCG(data) {
    const { A: A_flat, b, opts } = data || {};
    if (!A_flat || !b) return [];

    const A = unflattenMatrix(A_flat);
    const n = b.length;
    if (!isFiniteMatrix(A) || A.length !== n || A[0]?.length !== n || !isFiniteVector(b)) {
        return new Float32Array(n).fill(0);
    }
    
    let x = new Float32Array(n).fill(0);
    let r = new Float32Array(b);
    let p = new Float32Array(r);
    let rsold = dot(r, r);
    if (rsold < 1e-20) return x;
    
    const maxIter = (opts && opts.maxIter) || 200;
    const tol = (opts && opts.tol) || 1e-6;

    for (let it = 0; it < maxIter; ++it) {
        const Ap = matVecMul(A, p);
        const alpha = rsold / (dot(p, Ap) + 1e-20);
        
        for (let i = 0; i < n; i++) x[i] += alpha * p[i];
        for (let i = 0; i < n; i++) r[i] -= alpha * Ap[i];
        
        const rsnew = dot(r, r);
        if (Math.sqrt(rsnew) < tol) break;
        
        const beta = rsnew / (rsold + 1e-20);
        for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
    return Array.from(x);
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

function matrixSpectralNormApprox(data) {
    const M = unflattenMatrix(data.matrix);
    if (!isFiniteMatrix(M) || M.length === 0) return 0;
    const n = M.length;
    let v = new Float32Array(n).fill(1 / Math.sqrt(n));
    for (let i = 0; i < 10; i++) { // Power iteration
        const Av = matVecMul(M, v);
        const norm = norm2(Av);
        if (norm < 1e-10) return 0;
        v = Av.map(x => x / norm);
    }
    return norm2(matVecMul(M, v));
}

function smithNormalFormGF2(matrixData) {
    if (!matrixData || !matrixData.flatData || !Number.isFinite(matrixData.rows) || !Number.isFinite(matrixData.cols) || matrixData.flatData.length !== matrixData.rows * matrixData.cols) {
        return { diagonal: [], rank: 0 };
    }
    const M = unflattenMatrix(matrixData);
    if (!isFiniteMatrix(M) || M.length === 0) {
        return { diagonal: [], rank: 0 };
    }
    let rank = 0;
    const rows = M.length;
    const cols = M[0].length;
    const workingM = M.map(row => new Float32Array(row).map(x => Math.abs(x) % 2));

    for (let col = 0; col < cols && rank < rows; col++) {
        let pivotRow = -1;
        for (let row = rank; row < rows; row++) {
            if (workingM[row][col] === 1) {
                pivotRow = row;
                break;
            }
        }

        if (pivotRow !== -1) {
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
    }

    const diagonal = new Float32Array(Math.min(rows, cols)).fill(0);
    for (let i = 0; i < rank; i++) {
        diagonal[i] = 1;
    }
    return { diagonal, rank };
}

self.onmessage = function(e) {
    const { type, id, data } = e.data || {};
    if (!type || !id) {
        return;
    }
    let result;
    try {
        const guardedData = data || {};
        switch (type) {
            case 'transpose':
                result = flattenMatrix(transpose(unflattenMatrix(guardedData.matrix)));
                break;
            case 'matVecMul':
                result = matVecMul(guardedData.matrix, guardedData.vector);
                break;
            case 'solveLinearSystemCG':
                result = solveLinearSystemCG(guardedData);
                break;
            case 'covarianceMatrix':
                result = covarianceMatrix(guardedData.states, guardedData.eps);
                break;
            case 'matrixSpectralNormApprox':
                result = matrixSpectralNormApprox(guardedData);
                break;
            case 'smithNormalFormGF2': // Match the explicit name
                result = smithNormalFormGF2(guardedData.matrix);
                break;
            case 'matMul':
                result = matMul(guardedData);
                break;
            case 'persistentHomology':
                result = persistentHomology(guardedData);
                break;
            case 'topologicalScore':
                result = topologicalScore(guardedData);
                break;
            case 'ksgMutualInformation':
                result = ksgMutualInformation(guardedData);
                break;
            case 'eigenvalues':
                result = eigenvalues(guardedData);
                break;
            case 'complexEigenvalues':
                result = complexEigenvalues(guardedData);
                break;
            default:
                result = null;
                break;
        }
        self.postMessage({ type: type + 'Result', id, result });
    } catch (error) {
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
            logger.warn(errorMessage); // Use warn instead of error for timeouts
            resolve(null); // Resolve with null on timeout instead of rejecting
        }, timeout);
        workerCallbacks.set(id, {
            resolve: (result) => {
                clearTimeout(timer);
                resolve(result);
            },
            reject: (error) => {
                clearTimeout(timer);
                logger.error(`Worker task ${type} (id: ${id}) error: ${error.message || error.toString()}.`);
                resolve(null); // Resolve with null on error instead of rejecting
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
export function clamp(v, min, max) {
    const safeV = Number.isFinite(v) ? v : 0;
    const safeMin = Number.isFinite(min) ? min : -Infinity;
    const safeMax = Number.isFinite(max) ? max : Infinity;
    return Math.max(safeMin, Math.min(safeMax, safeV));
}

export function matVecMul(m, v) {
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

export function identityMatrix(n) {
    const matrix = Array(n).fill().map(() => new Float32Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        matrix[i][i] = 1;
    }
    return matrix;
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
    if (!isFiniteMatrix(matrix)) { 
        logger.warn('flattenMatrix: Input matrix is not finite. Returning empty.'); 
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

export function matMul(data) {
    const A = unflattenMatrix(data.matrixA);
    let B = unflattenMatrix(data.matrixB);

    if (!isFiniteMatrix(A) || !isFiniteMatrix(B)) {
        return [];
    }

    const A_rows = A.length;
    const A_cols = A[0] ? A[0].length : 0;
    const B_rows = B.length;
    const B_cols = B[0] ? B[0].length : 0;
    
    if (A_rows === 0 || A_cols === 0 || B_rows === 0 || B_cols === 0 || A_cols !== B_rows) {
        return [];
    }

    if (data.transposeB) {
        B = transpose(B);
        if (!isFiniteMatrix(B)) return [];
    }

    const C = Array(A_rows).fill(0).map(() => new Float32Array(B_cols).fill(0));
    for (let i = 0; i < A_rows; i++) {
        for (let j = 0; j < B_cols; j++) {
            let sum = 0;
            for (let k = 0; k < A_cols; k++) {
                const a_ik = Number.isFinite(A[i][k]) ? A[i][k] : 0;
                const b_kj = Number.isFinite(B[k][j]) ? B[k][j] : 0;
                sum += a_ik * b_kj;
            }
            C[i][j] = Number.isFinite(sum) ? sum : 0;
        }
    }
    return C;
}

export function covarianceMatrix(states_array, eps = 1e-3) {
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
// --- END OF FILE utils.js ---
