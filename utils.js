// --- UTILS.JS (STABILITY-ENHANCED & COMPLETE - V9) ---
// Fixes TypeScript errors in workerLogicString and runWorkerTask, removes type annotations,
// ensures proper template literal closure, and optimizes complexEigenvalues for 56x56 matrices.
export function safeVecScale(vector, scalar) {
    if (!isFiniteVector(vector) || !isFiniteNumber(scalar) || !Number.isFinite(scalar)) {
        
        return vecZeros(vector?.length || 1);
    }
    const result = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
        result[i] = vector[i] * scalar;
    }
    return result;
}


export function _matMul(matrixA, matrixB) {
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

export function _transpose(matrix) {
    if (!isFiniteMatrix(matrix) || matrix.length === 0) return [];
    const result = zeroMatrix(matrix[0].length, matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

export function _matVecMul(matrix, vector) {
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
const logElement = document.getElementById('log');
const MAX_LOG_ENTRIES = 200;

function appendLog(message, type = 'info') {
    if (!logElement) {
        const logMethod = console[type] || console.log;
        logMethod(`[${new Date().toLocaleTimeString()}] ${message}`);
        return;
    }
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logElement.appendChild(entry);
    if (logElement.children.length > MAX_LOG_ENTRIES) {
        logElement.removeChild(logElement.firstChild);
    }
    logElement.scrollTop = logElement.scrollHeight;
}

export const logger = {
    info: (msg, data) => appendLog(`INFO: ${msg}` + (data ? ' ' + JSON.stringify(data) : '')),
    warn: (msg, data) => appendLog(`WARN: ${msg}` + (data ? ' ' + JSON.stringify(data) : ''), 'warn'),
    error: (msg, data) => {
        appendLog(`ERROR: ${msg}` + (data ? ' ' + JSON.stringify(data) : ''), 'error');
        console.error(msg, data);
    },
    debug: (msg, data) => {
        // console.log(`DEBUG: ${msg}`, data);
    }
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

// --- Core Utility Functions ---
export function clamp(value, min, max) {
    if (!Number.isFinite(value)) return min;
    return Math.max(min, Math.min(value, max));
}

export function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

export function isFiniteVector(vec) {
    if (!vec) return false;
    if (typeof vec === 'object' && !Array.isArray(vec) && !(vec instanceof Float32Array)) {
        vec = Object.keys(vec).sort((a, b) => Number(a) - Number(b)).map(k => vec[k]);
    }
    if (Array.isArray(vec) || vec instanceof Float32Array) {
        for (let i = 0; i < vec.length; i++) {
            if (!isFiniteNumber(vec[i])) return false;
        }
        return vec.length > 0;
    }
    return false;
}

export function isFiniteMatrix(matrix) {
    if (!Array.isArray(matrix)) return false;
    if (matrix.length === 0) return true;
    const firstRow = matrix[0];
    if (!isFiniteVector(firstRow)) return false;
    const firstRowLength = firstRow.length;
    for (let i = 0; i < matrix.length; i++) {
        if (!isFiniteVector(matrix[i]) || matrix[i].length !== firstRowLength) return false;
    }
    return true;
}

export function vecZeros(dim) {
    if (!Number.isInteger(dim) || dim < 0) {
        logger.warn(`vecZeros: Invalid dimension (dim=${dim}). Returning 1-element vector.`);
        dim = 1;
    }
    return new Float32Array(dim).fill(0);
}

export function zeroMatrix(rows, cols) {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) return [];
    return Array.from({ length: rows }, () => new Float32Array(cols).fill(0));
}

export function identity(dim) {
    if (!Number.isInteger(dim) || dim < 0) return [];
    const mat = zeroMatrix(dim, dim);
    for (let i = 0; i < dim; i++) mat[i][i] = 1;
    return mat;
}

export function vectorAsRow(vec) {
    if (!isFiniteVector(vec) || vec.length === 0) {
        logger.error('vectorAsRow: Invalid or empty input vector. Returning empty matrix.', { vec });
        return [];
    }
    return [new Float32Array(vec)];
}

export function vectorAsCol(vec) {
    if (!isFiniteVector(vec) || vec.length === 0) {
        logger.error('vectorAsCol: Invalid or empty input vector. Returning empty matrix.', { vec });
        return [];
    }
    return Array.from(vec).map(x => new Float32Array([x]));
}

export function dot(vec1, vec2) {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) {
        logger.warn('dot: Invalid vectors or dimension mismatch.', { len1: vec1?.length, len2: vec2?.length });
        return 0;
    }
    if (!isFiniteVector(vec1) || !isFiniteVector(vec2)) {
        logger.warn('dot: Vectors contain non-finite numbers.');
        return 0;
    }
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
        sum += vec1[i] * vec2[i];
    }
    return isFiniteNumber(sum) ? sum : 0;
}

export const vecAdd = (a, b) => {
    if (typeof a === 'number' || typeof b === 'number') {
        logger.warn('vecAdd: Invalid input, one argument is a scalar. Returning zero vector.');
        return vecZeros(a?.length || b?.length || 1);
    }
    if (!isFiniteVector(a) || !isFiniteVector(b) || a.length !== b.length) {
        logger.warn('vecAdd: Invalid vectors or dimension mismatch. Returning zero vector.', { lenA: a?.length, lenB: b?.length });
        return vecZeros(a?.length || b?.length || 1);
    }
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) result[i] = clamp(a[i] + b[i], -1e6, 1e6);
    return result;
};

export const vecSub = (a, b) => {
    if (!isFiniteVector(a) || !isFiniteVector(b) || a.length !== b.length) {
        logger.warn('vecSub: Invalid vectors or dimension mismatch. Returning zero vector.', { lenA: a?.length, lenB: b?.length });
        return vecZeros(a?.length || b?.length || 1);
    }
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) result[i] = clamp(a[i] - b[i], -1e6, 1e6);
    return result;
};

export const vecScale = (vec, scalar) => {
    if (!isFiniteVector(vec) || !isFiniteNumber(scalar)) {
        logger.warn('vecScale: Invalid vector or scalar. Returning zero vector.');
        return vecZeros(vec?.length || 1);
    }
    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) result[i] = clamp(vec[i] * scalar, -1e6, 1e6);
    return result;
};

export function vecMul(vec1, vec2) {
    if (!isFiniteVector(vec1) || !isFiniteVector(vec2) || vec1.length !== vec2.length) {
        const fallbackDim = vec1?.length || vec2?.length || 1;
        logger.warn(`vecMul: Invalid vectors or dimension mismatch. Returning zero vector of dim ${fallbackDim}.`);
        return vecZeros(fallbackDim);
    }
    const result = new Float32Array(vec1.length);
    for (let i = 0; i < vec1.length; i++) {
        result[i] = clamp(vec1[i] * vec2[i], -1e6, 1e6);
    }
    return result;
}

export function norm2(vec) {
    if (!isFiniteVector(vec)) return 0;
    return Math.sqrt(dot(vec, vec) + 1e-12);
}

export function matVecMul(matrix, vector) {
    if (matrix && matrix.matrix_type === 'object') {
        logger.error('matVecMul FATAL: Received invalid matrix object. Tracing origin...');
        console.trace('Stack trace for invalid matrix object:');
        const defaultLength = vector?.length || matrix?.rows || 7;
        return vecZeros(defaultLength);
    }
    if (!isFiniteMatrix(matrix) || !isFiniteVector(vector) || matrix[0]?.length !== vector.length) {
        logger.warn('matVecMul: Invalid input or dimension mismatch.', {
            matrixRows: matrix?.length,
            matrixCols: matrix?.[0]?.length,
            vectorLength: vector?.length
        });
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

export function matMul(params) {
    const { matrixA, matrixB, transposeB = false } = params || {};
    let A = matrixA;
    let B = matrixB;
    if (!isFiniteMatrix(A) || !isFiniteMatrix(B)) return [];
    if (transposeB) B = transpose(B);
    if (!isFiniteMatrix(B) || A[0]?.length !== B.length) return [];
    const C = zeroMatrix(A.length, B[0].length);
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < A[0].length; k++) sum += (A[i][k] || 0) * (B[k][j] || 0);
            C[i][j] = clamp(sum, -1e6, 1e6);
        }
    }
    return C;
}

export function transpose(matrix) {
    if (!isFiniteMatrix(matrix) || matrix.length === 0) return [];
    const result = zeroMatrix(matrix[0].length, matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

export function flattenMatrix(matrix) {
    if (!matrix || !Array.isArray(matrix) || !isFiniteMatrix(matrix)) {
        logger.warn('flattenMatrix: Invalid input. Returning empty structure.', { matrix_type: typeof matrix });
        return { flatData: new Float32Array(0), rows: 0, cols: 0 };
    }
    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    if (rows === 0 || cols === 0) {
        return { flatData: new Float32Array(0), rows: 0, cols: 0 };
    }
    const flat = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        if (matrix[i] && matrix[i].length === cols) {
            flat.set(matrix[i], i * cols);
        }
    }
    return { flatData: flat, rows, cols };
}

export function unflattenMatrix(data) {
    if (!data || !data.flatData || !isFiniteVector(data.flatData) || data.flatData.length !== data.rows * data.cols) return [];
    const matrix = [];
    for (let i = 0; i < data.rows; i++) {
        matrix.push(data.flatData.slice(i * data.cols, (i + 1) * data.cols));
    }
    return matrix;
}

export function randomMatrix(rows, cols, scale = 0.1) {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
        logger.error(`randomMatrix: Invalid dimensions ${rows}x${cols}. Returning 1x1 identity equivalent.`);
        return { flatData: new Float32Array([1]), rows: 1, cols: 1 };
    }
    const flatData = new Float32Array(rows * cols);
    for (let i = 0; i < flatData.length; i++) {
        flatData[i] = (Math.random() * 2 - 1) * scale;
    }
    return { flatData, rows, cols };
}

export function covarianceMatrix(states_array, eps = 1e-6) {
    const states = states_array.filter(s => isFiniteVector(s));
    if (!Array.isArray(states) || states.length < 2) {
        logger.warn('covarianceMatrix: Insufficient valid states. Returning identity matrix.', { statesLength: states.length });
        return identity(states_array?.[0]?.length || 1);
    }
    const n = states.length;
    const d = states[0]?.length || 0;
    if (d === 0) {
        logger.warn('covarianceMatrix: Zero-dimensional states. Returning 1x1 identity.');
        return identity(1);
    }
    const mean = vecZeros(d);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += states[i][j] / n;
        }
    }
    const cov = zeroMatrix(d, d);
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < d; i++) {
            const di = states[k][i] - mean[i];
            for (let j = i; j < d; j++) {
                const dj = states[k][j] - mean[j];
                cov[i][j] += (di * dj) / Math.max(1, n - 1);
            }
        }
    }
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < i; j++) {
            cov[j][i] = cov[i][j];
        }
        cov[i][i] = clamp(cov[i][i] + eps, 1e-9, 1e9);
    }
    return isFiniteMatrix(cov) ? cov : identity(d);
}

export function logDeterminantFromDiagonal(matrix) {
    if (!isFiniteMatrix(matrix) || matrix.length === 0 || matrix.length !== matrix[0]?.length) {
        logger.warn('logDeterminantFromDiagonal: Invalid or non-square matrix. Returning 0.', { matrix });
        return 0;
    }
    let logDet = 0;
    for (let i = 0; i < matrix.length; i++) {
        const diagVal = matrix[i][i];
        if (!Number.isFinite(diagVal) || diagVal <= 0) {
            logger.warn(`logDeterminantFromDiagonal: Invalid diagonal element at index ${i}: ${diagVal}. Returning 0.`);
            return 0;
        }
        logDet += Math.log(diagVal);
    }
    return Number.isFinite(logDet) ? logDet : 0;
}

export function softmax(logits) {
    if (!isFiniteVector(logits) || logits.length === 0) {
        const fallbackLength = logits?.length || 4;
        logger.warn(`softmax: Invalid or empty logits. Returning uniform probabilities for length ${fallbackLength}.`);
        return vecZeros(fallbackLength).fill(1 / fallbackLength);
    }
    const maxLogit = Math.max(...logits);
    if (!Number.isFinite(maxLogit)) {
        logger.warn('softmax: Non-finite max logit. Returning uniform probabilities.');
        return vecZeros(logits.length).fill(1 / logits.length);
    }
    const exp_logits = new Float32Array(logits.length);
    let sum_exp_logits = 0;
    for (let i = 0; i < logits.length; i++) {
        const val = Math.exp(logits[i] - maxLogit);
        exp_logits[i] = Number.isFinite(val) ? val : 0;
        sum_exp_logits += exp_logits[i];
    }
    const safe_sum_exp_logits = (Number.isFinite(sum_exp_logits) && sum_exp_logits > 1e-9) ? sum_exp_logits : 1e-9;
    const resultProbs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        resultProbs[i] = Number.isFinite(exp_logits[i] / safe_sum_exp_logits) ? exp_logits[i] / safe_sum_exp_logits : 0;
    }
    if (!isFiniteVector(resultProbs)) {
        logger.warn('softmax: Non-finite output probabilities. Returning uniform probabilities.');
        return vecZeros(logits.length).fill(1 / logits.length);
    }
    return resultProbs;
}

export function sigmoidVec(vec) {
    if (!isFiniteVector(vec)) {
        logger.warn('sigmoidVec: Invalid vector. Returning zeros.', { vec });
        return vecZeros(vec?.length || 1);
    }
    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        result[i] = sigmoid(vec[i]);
    }
    return result;
}

export function sigmoid(x) {
    if (!Number.isFinite(x)) {
        logger.debug('sigmoid: Non-finite input. Returning 0.5.', { x });
        return 0.5;
    }
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    } else {
        const z = Math.exp(x);
        return z / (1 + z);
    }
}

export function tanhVec(vec) {
    if (!isFiniteVector(vec)) {
        logger.warn('tanhVec: Invalid vector. Returning zeros.', { vec });
        return vecZeros(vec?.length || 1);
    }
    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        result[i] = Math.tanh(vec[i]);
        if (!Number.isFinite(result[i])) {
            logger.debug(`tanhVec: Non-finite result at index ${i}. Setting to 0.`);
            result[i] = 0;
        }
    }
    return result;
}

export function ksgMutualInformation(data) {
    const { x, y, k = 3 } = data || {};
    if (!isFiniteVector(x) || !isFiniteVector(y) || x.length !== y.length || x.length < k + 1 || !Number.isInteger(k) || k < 1) {
        logger.warn('ksgMutualInformation: Invalid input or insufficient data points.', { xLength: x?.length, yLength: y?.length, k });
        return 0;
    }
    const n = x.length;
    const distances = [];
    for (let i = 0; i < n; i++) {
        const dist = [];
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                const dx = x[i] - x[j];
                const dy = y[i] - y[j];
                dist.push({ index: j, maxNorm: Math.max(Math.abs(dx), Math.abs(dy)) });
            }
        }
        dist.sort((a, b) => a.maxNorm - b.maxNorm);
        distances.push(dist);
    }
    let nx = new Array(n).fill(0);
    let ny = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        const kthDist = distances[i][k - 1].maxNorm + 1e-10;
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                if (Math.abs(x[i] - x[j]) < kthDist) nx[i]++;
                if (Math.abs(y[i] - y[j]) < kthDist) ny[i]++;
            }
        }
    }
    let mi = 0;
    for (let i = 0; i < n; i++) {
        mi += Math.log(n) - Math.log(nx[i] + 1) - Math.log(ny[i] + 1) + Math.log(k);
    }
    mi = mi / n + Math.log(n);
    return Number.isFinite(mi) ? clamp(mi, 0, 1e6) : 0;
}

export function matrixSpectralNormApprox(data) {
    const M = unflattenMatrix(data.matrix);
    if (!isFiniteMatrix(M) || M.length === 0 || M[0].length === 0) {
        logger.warn('matrixSpectralNormApprox: Invalid or empty matrix. Returning 0.', { matrix: M });
        return 0;
    }
    const rows = M.length;
    const cols = M[0].length;
    let v = new Float32Array(cols).map(() => Math.random());
    v = vecScale(v, 1 / norm2(v));
    const maxIter = 50;
    const tol = 1e-6;
    let sigma = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        let u = matVecMul(M, v);
        const uNorm = norm2(u);
        if (uNorm < tol) {
            logger.debug('matrixSpectralNormApprox: Intermediate vector norm too small.', { iter, uNorm });
            break;
        }
        u = vecScale(u, 1 / uNorm);
        v = matVecMul(transpose(M), u);
        const vNorm = norm2(v);
        if (vNorm < tol) {
            logger.debug('matrixSpectralNormApprox: Intermediate vector norm too small.', { iter, vNorm });
            break;
        }
        v = vecScale(v, 1 / vNorm);
        const newSigma = norm2(matVecMul(M, v));
        if (Math.abs(newSigma - sigma) < tol) {
            sigma = newSigma;
            break;
        }
        sigma = newSigma;
    }
    return Number.isFinite(sigma) ? clamp(sigma, 0, 1e6) : 0;
}

export function solveLinearSystemCG(data) {
    const A = unflattenMatrix(data.A);
    const b = data.b;
    if (!isFiniteMatrix(A) || !isFiniteVector(b) || A.length !== b.length || A.length === 0 || A.length !== A[0].length) {
        logger.warn('solveLinearSystemCG: Invalid input or dimension mismatch. Returning zero vector.', { ARows: A?.length, ACols: A?.[0]?.length, bLength: b?.length });
        return vecZeros(b?.length || 1);
    }
    const n = b.length;
    let x = vecZeros(n);
    let r = vecSub(b, matVecMul(A, x));
    let p = new Float32Array(r);
    let rsold = dot(r, r);
    if (rsold < 1e-10) {
        logger.debug('solveLinearSystemCG: Initial residual too small. Returning zero vector.');
        return x;
    }
    const maxIter = Math.min(n, 1000);
    const tol = 1e-6;
    for (let i = 0; i < maxIter; i++) {
        const Ap = matVecMul(A, p);
        const alpha = rsold / dot(p, Ap);
        if (!isFiniteNumber(alpha)) {
            logger.warn('solveLinearSystemCG: Non-finite alpha. Terminating early.', { iter: i, alpha });
            break;
        }
        x = vecAdd(x, vecScale(p, alpha));
        r = vecSub(r, vecScale(Ap, alpha));
        const rsnew = dot(r, r);
        if (rsnew < tol * tol) {
            logger.debug('solveLinearSystemCG: Converged.', { iter: i, residual: Math.sqrt(rsnew) });
            break;
        }
        p = vecAdd(r, vecScale(p, rsnew / rsold));
        rsold = rsnew;
    }
    return isFiniteVector(x) ? x : vecZeros(n);
}

export function safeEigenDecomposition(matrix, options) {
    options = options || { maxIterations: 100 };
    const n = matrix.length;
    if (!isFiniteMatrix(matrix) || matrix.length === 0 || matrix[0].length !== n) {
        logger.warn('safeEigenDecomposition: Invalid or non-square matrix. Returning identity.', {
            rows: matrix?.length,
            cols: matrix?.[0]?.length
        });
        return {
            lambda: { x: Array(n).fill(1), y: Array(n).fill(0) },
            E: { x: identity(n), y: zeroMatrix(n, n) }
        };
    }
    if (typeof Numeric !== 'undefined' && typeof Numeric.eig === 'function') {
        try {
            const start = performance.now();
            const result = Numeric.eig(matrix, options.maxIterations);
            const durationMs = performance.now() - start;
            logger.info('safeEigenDecomposition: numeric.eig execution time.', { durationMs, matrixSize: `${n}x${n}` });
            if (!result || !result.lambda || !result.E || !isFiniteVector(result.lambda.x) || !isFiniteMatrix(result.E.x)) {
                logger.warn('safeEigenDecomposition: Numeric.eig returned invalid data. Using identity.', { result });
                return {
                    lambda: { x: Array(n).fill(1), y: Array(n).fill(0) },
                    E: { x: identity(n), y: zeroMatrix(n, n) }
                };
            }
            return result;
        } catch (err) {
            logger.warn('safeEigenDecomposition: Numeric.eig failed.', { error: err.message });
        }
    }
    logger.info('safeEigenDecomposition: Using power iteration fallback.', { matrixSize: `${n}x${n}` });
    let v = vecZeros(n).map(() => Math.random());
    v = vecScale(v, 1 / norm2(v));
    let eigenvalue = 0;
    for (let i = 0; i < Math.min(options.maxIterations, 50); i++) {
        const vNext = matVecMul(matrix, v);
        eigenvalue = dot(vNext, v);
        v = vecScale(vNext, 1 / norm2(vNext));
        if (!isFiniteNumber(eigenvalue) || !isFiniteVector(v)) {
            logger.warn('safeEigenDecomposition: Power iteration produced non-finite results. Using identity.');
            return {
                lambda: { x: Array(n).fill(1), y: Array(n).fill(0) },
                E: { x: identity(n), y: zeroMatrix(n, n) }
            };
        }
    }
    return {
        lambda: { x: Array(n).fill(eigenvalue), y: Array(n).fill(0) },
        E: { x: Array(n).fill(v), y: zeroMatrix(n, n) }
    };
}

// --- Worker Logic ---
const workerLogicString = `
// --- START OF WORKER LOGIC ---
const clamp = function(v, min, max) {
    return Math.max(min, Math.min(max, Number.isFinite(v) ? v : min));
};
const isFiniteNumber = function(v) {
    return typeof v === 'number' && Number.isFinite(v);
};
const isFiniteVector = function(v) {
    if (!v || !(v instanceof Array || v instanceof Float32Array)) return false;
    for (let i = 0; i < v.length; i++) if (!isFiniteNumber(v[i])) return false;
    return true;
};
const isFiniteMatrix = function(m) {
    if (!Array.isArray(m)) return false;
    if (m.length === 0) return true;
    const len = m[0]?.length;
    if (typeof len !== 'number') return false;
    for (let i = 0; i < m.length; i++) if (!isFiniteVector(m[i]) || m[i].length !== len) return false;
    return true;
};
const vecZeros = function(d) {
    return new Float32Array(d);
};
const zeroMatrix = function(r, c) {
    return Array.from({ length: r }, () => new Float32Array(c));
};
const identity = function(d) {
    const m = zeroMatrix(d, d);
    for (let i = 0; i < d; i++) m[i][i] = 1;
    return m;
};
const dot = function(v1, v2) {
    let s = 0;
    for (let i = 0; i < v1.length; i++) s += v1[i] * v2[i];
    return s;
};
const norm2 = function(v) {
    return Math.sqrt(dot(v, v) + 1e-12);
};
const vecAdd = function(a, b) {
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) result[i] = clamp(a[i] + b[i], -1e6, 1e6);
    return result;
};
const vecSub = function(a, b) {
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) result[i] = clamp(a[i] - b[i], -1e6, 1e6);
    return result;
};
const vecScale = function(vec, scalar) {
    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) result[i] = clamp(vec[i] * scalar, -1e6, 1e6);
    return result;
};
const unflattenMatrix = function(d) {
    if (!d || !d.flatData || d.flatData.length !== d.rows * d.cols) {
        console.error('[Worker] unflattenMatrix: Invalid data structure.', { flatDataLength: d?.flatData?.length, rows: d?.rows, cols: d?.cols });
        return [];
    }
    const m = [];
    for (let i = 0; i < d.rows; i++) m.push(d.flatData.slice(i * d.cols, (i + 1) * d.cols));
    return m;
};
const flattenMatrix = function(matrix) {
    if (!isFiniteMatrix(matrix)) return { flatData: new Float32Array(), rows: 0, cols: 0 };
    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    const flatData = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) flatData.set(matrix[i], i * cols);
    return { flatData, rows, cols };
};
const transpose = function(matrix) {
    if (!isFiniteMatrix(matrix) || matrix.length === 0) return [];
    const result = zeroMatrix(matrix[0].length, matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
};
const matVecMul = function(data) {
    const matrix = (data.matrix && data.matrix.flatData) ? unflattenMatrix(data.matrix) : data.matrix;
    const vector = data.vector;
    if (!isFiniteMatrix(matrix) || !isFiniteVector(vector) || matrix[0]?.length !== vector.length) {
        console.error('[Worker] matVecMul: Invalid input or dimension mismatch.', { matrixRows: matrix?.length, matrixCols: matrix?.[0]?.length, vectorLength: vector?.length });
        return vecZeros(matrix?.length || 1);
    }
    const result = new Float32Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        result[i] = clamp(dot(matrix[i], vector), -1e6, 1e6);
    }
    return result;
};
const matMul = function(data) {
    const A = unflattenMatrix(data.matrixA);
    const B = unflattenMatrix(data.matrixB);
    if (!isFiniteMatrix(A) || !isFiniteMatrix(B)) return flattenMatrix([]);
    const B_final = data.transposeB ? transpose(B) : B;
    if (!isFiniteMatrix(B_final) || A[0]?.length !== B_final.length) return flattenMatrix([]);
    const C = zeroMatrix(A.length, B_final[0].length);
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B_final[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < A[0].length; k++) {
                sum += (A[i][k] || 0) * (B_final[k][j] || 0);
            }
            C[i][j] = clamp(sum, -1e6, 1e6);
        }
    }
    return flattenMatrix(C);
};
const covarianceMatrix = function(data) {
    // FIX: Extract the states array and eps from the input data object.
    const states_array = data.states;
    const eps = data.eps || 1e-6;

    // The rest of the function logic remains the same.
    if (!Array.isArray(states_array)) {
        console.error('[Worker] covarianceMatrix: Input states_array is not an array.');
        return identity(1);
    }
    
    const states = states_array.filter(s => isFiniteVector(s));
    if (states.length < 2) {
        console.warn('[Worker] covarianceMatrix: Insufficient valid states. Returning identity matrix.', { statesLength: states.length });
        return identity(states_array?.[0]?.length || 1);
    }
    const n = states.length;
    const d = states[0]?.length || 0;
    if (d === 0) {
        console.warn('[Worker] covarianceMatrix: Zero-dimensional states. Returning 1x1 identity.');
        return identity(1);
    }
    const mean = vecZeros(d);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += states[i][j] / n;
        }
    }
    const cov = zeroMatrix(d, d);
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < d; i++) {
            const di = states[k][i] - mean[i];
            for (let j = i; j < d; j++) {
                const dj = states[k][j] - mean[j];
                cov[i][j] += (di * dj) / Math.max(1, n - 1);
            }
        }
    }
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < i; j++) {
            cov[j][i] = cov[i][j];
        }
        cov[i][i] = clamp(cov[i][i] + eps, 1e-9, 1e9);
    }
    return isFiniteMatrix(cov) ? cov : identity(d);
};

const workerLogger = {
    info: function(msg, data) { console.log(\`[\${new Date().toLocaleTimeString()}] Worker INFO: \${msg}\` + (data ? ' ' + JSON.stringify(data) : '')); },
    warn: function(msg, data) { console.warn(\`[\${new Date().toLocaleTimeString()}] Worker WARN: \${msg}\` + (data ? ' ' + JSON.stringify(data) : '')); },
    error: function(msg, data) { console.error(\`[\${new Date().toLocaleTimeString()}] Worker ERROR: \${msg}\` + (data ? ' ' + JSON.stringify(data) : '')); },
    debug: function(msg, data) { console.log(\`[\${new Date().toLocaleTimeString()}] Worker DEBUG: \${msg}\`, data); }
};

workerLogger.info('Worker: Starting initialization.');

// Dynamically import Numeric.js into the worker's scope for complex eigenvalue decomposition.
// This assumes Numeric.js is available at this path relative to the worker script.
try {
    importScripts('https://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.min.js');
    workerLogger.info('Worker: Successfully imported numeric.min.js.');
} catch (e) {
    workerLogger.error('Worker: Failed to import numeric.min.js.', { error: e.message });
    // Numeric will remain undefined, forcing fallback paths
}
let eigenCache = null;
let lastLaplacianHash = null;

function hashMatrix(matrix) {
    let hash = 0;
    const rows = matrix.length;
    if (rows === 0) return 0;
    const cols = matrix[0].length;
    for (let i = 0; i < rows; i++) {
        hash = (hash * 31 + matrix[i][i]) | 0;
        hash = (hash * 31 + matrix[i][0]) | 0;
        hash = (hash * 31 + matrix[i][cols - 1]) | 0;
    }
    return hash;
}

function jacobiEigenvalues(matrix, maxIterations = 100) {
    const n = matrix.length;
    let A = matrix.map(row => [...row]);
    const tol = 1e-9;
    for (let iter = 0; iter < maxIterations; iter++) {
        let maxVal = 0;
        let p = 0, q = 1;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(A[i][j]) > maxVal) {
                    maxVal = Math.abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (maxVal < tol) {
            workerLogger.info('Jacobi method converged.', { iterations: iter });
            break;
        }
        let app = A[p][p];
        let aqq = A[q][q];
        let apq = A[p][q];
        let tau = (aqq - app) / (2 * apq);
        let t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        let c = 1 / Math.sqrt(1 + t * t);
        let s = c * t;
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = A[q][p] = 0.0;
        for (let i = 0; i < n; i++) {
            if (i !== p && i !== q) {
                let aip = A[i][p];
                let aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip - s * aiq;
                A[i][q] = A[q][i] = s * aip + c * aiq;
            }
        }
    }
    const eigenvalues = new Array(n);
    for (let i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues.sort((a, b) => b - a);
}

function complexEigenvalues(data) {
    workerLogger.info('complexEigenvalues: Task started.', { dataKeys: Object.keys(data), matrixPresent: !!data.matrix });
    if (!data || !data.matrix || !data.matrix.flatData || !Number.isInteger(data.matrix.rows) || !Number.isInteger(data.matrix.cols)) {
        workerLogger.error('complexEigenvalues: Invalid or missing matrix data.', { data: data });
        return Array(data?.matrix?.rows || 56).fill({ re: 1, im: 0 });
    }
    const matrix = unflattenMatrix(data.matrix);
    const matrixSize = matrix.length;
    if (!isFiniteMatrix(matrix) || matrix.length === 0 || matrix[0].length !== matrixSize) {
        workerLogger.error('complexEigenvalues: Invalid or non-square matrix.', { rows: matrix?.length, cols: matrix?.[0]?.length });
        return Array(matrixSize || 56).fill({ re: 1, im: 0 });
    }
    const matrixHash = hashMatrix(matrix);
    if (eigenCache && lastLaplacianHash === matrixHash) {
        workerLogger.info('complexEigenvalues: Using cached eigenvalue result.');
        return eigenCache;
    }
    // Sanitize matrix and add diagonal regularization for stability
    const sanitizedMatrix = matrix.map((row, i) => row.map((v, j) => isFiniteNumber(v) ? v : (i === j ? 1e-6 : 0)));
    const start = performance.now();
    let result;
    try {
        if (typeof Numeric !== 'undefined' && typeof Numeric.eig === 'function') {
            workerLogger.info('complexEigenvalues: Attempting Numeric.eig for full spectrum.');
            const eigResult = Numeric.eig(sanitizedMatrix, { maxiter: 200, tol: 1e-10 });
            if (eigResult && eigResult.lambda && isFiniteVector(eigResult.lambda.x) && isFiniteVector(eigResult.lambda.y)) {
                result = eigResult.lambda.x.map((re, i) => ({
                    re: clamp(isFiniteNumber(re) ? re : 1, -1e6, 1e6),
                    im: clamp(isFiniteNumber(eigResult.lambda.y[i]) ? eigResult.lambda.y[i] : 0, -1e6, 1e6)
                }));
                workerLogger.info('complexEigenvalues: Successfully used Numeric.eig.', { durationMs: performance.now() - start });
            } else {
                throw new Error('Invalid Numeric.eig result');
            }
        } else {
            throw new Error('Numeric.eig unavailable');
        }
    } catch (err) {
        workerLogger.warn('complexEigenvalues: Numeric.eig failed. Using enhanced power iteration.', { error: err.message });
        // Enhanced power iteration for dominant eigenvalue, suitable for 56x56 Laplacian
        let v = vecZeros(matrixSize).map(() => Math.random());
        v = vecScale(v, 1 / norm2(v));
        let eigenvalue = 0;
        const maxIter = 100;
        const tol = 1e-8;
        for (let i = 0; i < maxIter; i++) {
            const vNext = matVecMul({ matrix: sanitizedMatrix, vector: v });
            const vNextNorm = norm2(vNext);
            if (vNextNorm < tol) {
                workerLogger.warn('complexEigenvalues: Power iteration norm too small.', { iter: i });
                break;
            }
            v = vecScale(vNext, 1 / vNextNorm);
            eigenvalue = dot(vNext, v);
            if (!isFiniteNumber(eigenvalue)) {
                workerLogger.warn('complexEigenvalues: Non-finite eigenvalue. Breaking.', { iter: i });
                break;
            }
            if (i > 0 && Math.abs(eigenvalue - dot(matVecMul({ matrix: sanitizedMatrix, vector: v }), v)) < tol) {
                break;
            }
        }
        result = Array(matrixSize).fill({ re: isFiniteNumber(eigenvalue) ? clamp(eigenvalue, -1e6, 1e6) : 1, im: 0 });
        workerLogger.info('complexEigenvalues: Power iteration completed.', { durationMs: performance.now() - start });
    }
    const durationMs = performance.now() - start;
    if (!Array.isArray(result) || result.length !== matrixSize || !result.every(e => isFiniteNumber(e.re) && isFiniteNumber(e.im))) {
        workerLogger.error('complexEigenvalues: Invalid result. Returning default.', { resultLength: result?.length });
        result = Array(matrixSize).fill({ re: 1, im: 0 });
    }
    eigenCache = result;
    lastLaplacianHash = matrixHash;
    workerLogger.info('complexEigenvalues: Completed.', { durationMs, matrixSize: matrixSize + 'x' + matrixSize, eigenvalueSample: result.slice(0, 5) });
    return result;
}

function ksgMutualInformation(data) {
    const k = data && data.k !== undefined ? data.k : 3;
    const states = data && data.states ? data.states : [];
    if (!Array.isArray(states) || states.length < 2) {
        workerLogger.warn('ksgMutualInformation: Invalid or insufficient states.', { statesLength: states.length });
        return 0;
    }
    const n_total_dim = states[0].length;
    const n_half = Math.floor(n_total_dim / 2);
    if (n_half < 1) return 0;
    const x = states.map(function(s) { return s.slice(n_half); });
    const y = states.map(function(s) { return s.slice(0, n_half); });
    const n = x.length;
    if (n < k + 1) return 0;
    const distances = [];
    for (let i = 0; i < n; i++) {
        const dist = [];
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                const dx = norm2(vecSub(x[i], x[j]));
                const dy = norm2(vecSub(y[i], y[j]));
                dist.push({ index: j, maxNorm: Math.max(dx, dy) });
            }
        }
        dist.sort(function(a, b) { return a.maxNorm - b.maxNorm; });
        distances.push(dist);
    }
    let nx = new Array(n).fill(0);
    let ny = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        const kthDist = distances[i][k - 1].maxNorm + 1e-10;
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                if (norm2(vecSub(x[i], x[j])) < kthDist) nx[i]++;
                if (norm2(vecSub(y[i], y[j])) < kthDist) ny[i]++;
            }
        }
    }
    let mi = 0;
    for (let i = 0; i < n; i++) {
        mi += Math.log(n) - (nx[i] > 0 ? Math.log(nx[i]) : 0) - (ny[i] > 0 ? Math.log(ny[i]) : 0) + Math.log(k);
    }
    mi = mi / n;
    return Number.isFinite(mi) ? clamp(mi, 0, 1e6) : 0;
}

function topologicalScore(data) {
    const states = data && data.states ? data.states : [];
    const filtration = data && data.filtration ? data.filtration : [];
    if (!Array.isArray(states) || states.length === 0 || !isFiniteMatrix(filtration) || filtration.length === 0) {
        workerLogger.warn('topologicalScore: Invalid states or filtration.', { statesLength: states.length, filtrationRows: filtration.length });
        return { score: 0 };
    }
    const nV = filtration.length;
    let sumAbsCorrelation = 0;
    for (let i = 0; i < nV; i++) {
        for (let j = 0; j < nV; j++) {
            sumAbsCorrelation += Math.abs(filtration[i][j]);
        }
    }
    const avgCorrelation = (nV * nV > 0) ? sumAbsCorrelation / (nV * nV) : 0;
    let sumNormsSq = 0;
    for (const s of states) {
        sumNormsSq += norm2(s) * norm2(s);
    }
    const avgNormsSq = (states.length > 0) ? sumNormsSq / states.length : 0;
    const varianceNorms = (avgNormsSq - (sumNormsSq * sumNormsSq / (states.length * states.length))) / states.length;
    const complexityScore = clamp(avgCorrelation * 0.5 + Math.sqrt(Math.abs(varianceNorms)) * 0.1, 0, 1);
    return { score: complexityScore };
}

function matrixSpectralNormApprox(data) {
    const M = unflattenMatrix(data && data.matrix ? data.matrix : { flatData: new Float32Array(), rows: 0, cols: 0 });
    if (!isFiniteMatrix(M) || M.length === 0 || M[0].length === 0) {
        workerLogger.warn('matrixSpectralNormApprox: Invalid or empty matrix. Returning 0.', { matrix: M });
        return 0;
    }
    const rows = M.length;
    const cols = M[0].length;
    let v = new Float32Array(cols).map(function() { return Math.random(); });
    v = vecScale(v, 1 / norm2(v));
    const maxIter = 50;
    const tol = 1e-6;
    let sigma = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        let u = matVecMul({ matrix: M, vector: v });
        const uNorm = norm2(u);
        if (uNorm < tol) {
            workerLogger.debug('matrixSpectralNormApprox: Intermediate vector norm too small.', { iter: iter, uNorm: uNorm });
            break;
        }
        u = vecScale(u, 1 / uNorm);
        v = matVecMul({ matrix: transpose(M), vector: u });
        const vNorm = norm2(v);
        if (vNorm < tol) {
            workerLogger.debug('matrixSpectralNormApprox: Intermediate vector norm too small.', { iter: iter, vNorm: vNorm });
            break;
        }
        v = vecScale(v, 1 / vNorm);
        const newSigma = norm2(matVecMul({ matrix: M, vector: v }));
        if (Math.abs(newSigma - sigma) < tol) {
            sigma = newSigma;
            break;
        }
        sigma = newSigma;
    }
    return Number.isFinite(sigma) ? clamp(sigma, 0, 1e6) : 0;
}

function solveLinearSystemCG(data) {
    if (!data || !data.A || !data.b) {
        workerLogger.error('solveLinearSystemCG: Missing input data.', { dataKeys: Object.keys(data || {}) });
        return vecZeros(data?.b?.length || 1);
    }
    let A = data.A.flatData ? unflattenMatrix(data.A) : data.A;
    let b = data.b;
    if (typeof b === 'object' && !Array.isArray(b) && !(b instanceof Float32Array)) {
        b = Object.keys(b).sort((a, b) => Number(a) - Number(b)).map(k => b[k]);
    }
    if (!isFiniteMatrix(A) || !isFiniteVector(b) || A.length !== b.length || A.length === 0 || A.length !== A[0].length) {
        workerLogger.warn('solveLinearSystemCG: Invalid input or dimension mismatch. Returning zero vector.', { ARows: A?.length, ACols: A?.[0]?.length, bLength: b?.length });
        return vecZeros(b?.length || 1);
    }
    const n = b.length;
    let x = vecZeros(n);
    let r = vecSub(b, matVecMul({ matrix: A, vector: x }));
    let p = new Float32Array(r);
    let rsold = dot(r, r);
    if (rsold < 1e-10) {
        workerLogger.debug('solveLinearSystemCG: Initial residual too small. Returning zero vector.');
        return x;
    }
    const maxIter = Math.min(n, 1000);
    const tol = 1e-6;
    for (let i = 0; i < maxIter; i++) {
        const Ap = matVecMul({ matrix: A, vector: p });
        const alpha = rsold / dot(p, Ap);
        if (!isFiniteNumber(alpha)) {
            workerLogger.warn('solveLinearSystemCG: Non-finite alpha. Terminating early.', { iter: i, alpha: alpha });
            break;
        }
        x = vecAdd(x, vecScale(p, alpha));
        r = vecSub(r, vecScale(Ap, alpha));
        const rsnew = dot(r, r);
        if (rsnew < tol * tol) {
            workerLogger.debug('solveLinearSystemCG: Converged.', { iter: i, residual: Math.sqrt(rsnew) });
            break;
        }
        p = vecAdd(r, vecScale(p, rsnew / rsold));
        rsold = rsnew;
    }
    return isFiniteVector(x) ? x : vecZeros(n);
}

self.addEventListener('message', function(e) {
    const type = e.data && e.data.type;
    const id = e.data && e.data.id;
    const data = e.data && e.data.data;
    if (!type || id === undefined) {
        workerLogger.error('Worker: Invalid message format.', { data: e.data });
        self.postMessage({ type: 'error', id: id, error: 'Invalid message format' });
        return;
    }
    try {
        const handlers = {
            'matVecMul': matVecMul,
            'matMul': matMul,
            'covarianceMatrix': covarianceMatrix,
            'complexEigenvalues': complexEigenvalues,
            'ksg_mi': ksgMutualInformation,
            'solveLinearSystemCG': solveLinearSystemCG,
            'matrixSpectralNormApprox': matrixSpectralNormApprox,
            'topologicalScore': topologicalScore
        };
        if (handlers[type]) {
            const result = handlers[type](data);
            self.postMessage({ type: type + 'Result', id: id, result: result });
        } else {
            throw new Error('Unknown worker task type: ' + type);
        }
    } catch (error) {
        workerLogger.error('Worker: Error in task "' + type + '": ' + error.message, { stack: error.stack, data: e.data });
        self.postMessage({ type: type + 'Error', id: id, error: error.message || String(error) });
    }
}, false);

workerLogger.info('Worker: Initialization complete.');
// --- END OF WORKER LOGIC ---
`;


// --- Worker Management ---
const workerBlob = new Blob([workerLogicString], { type: 'application/javascript' });
const worker = new Worker(URL.createObjectURL(workerBlob));
const workerCallbacks = new Map();
let nextWorkerTaskId = 0;

export async function runWorkerTask(type, data, timeout, maxAttempts) {
    timeout = timeout || 5000;
    maxAttempts = maxAttempts || 3;

    if (typeof type !== 'string') {
        logger.error('runWorkerTask: Invalid task type (not a string). Resolving with null.');
        return null;
    }

    function attemptTask(attempt) {
        return new Promise(function(resolve, reject) {
            const id = nextWorkerTaskId++;
            const timer = setTimeout(function() {
                workerCallbacks.delete(id);
                logger.warn(`Worker task "${type}" (id: ${id}, attempt: ${attempt}) timed out.`, {
                    matrixSample: type === 'complexEigenvalues' ? (data?.matrix?.flatData?.slice(0, 5)) : undefined
                });
                reject(new Error('Timeout'));
            }, timeout);

            workerCallbacks.set(id, {
                resolve: function(workerResult) {
                    clearTimeout(timer);
                    workerCallbacks.delete(id);
                    let isValid = false;
                    let result = workerResult;
                    if (workerResult !== null && workerResult !== undefined) {
                        if (type === 'matVecMul') {
                            isValid = isFiniteVector(workerResult) && workerResult.length === data.expectedDim;
                        } else if (type === 'matMul') {
                            isValid = workerResult.flatData && isFiniteVector(workerResult.flatData);
                        } else if (type === 'complexEigenvalues') {
                            isValid = Array.isArray(workerResult) && workerResult.every(function(e) { return isFiniteNumber(e.re) && isFiniteNumber(e.im); });
                        } else if (type === 'ksg_mi') {
                            isValid = isFiniteNumber(workerResult);
                        } else if (type === 'topologicalScore') {
                            isValid = workerResult && isFiniteNumber(workerResult.score);
                        } else {
                            isValid = true;
                        }
                    }
                    if (!isValid) {
                        logger.warn(`Worker task "${type}" (id: ${id}, attempt: ${attempt}) returned invalid data.`, {
                            resultReceived: workerResult,
                            matrixSample: type === 'complexEigenvalues' ? (data?.matrix?.flatData?.slice(0, 5)) : undefined
                        });
                        reject(new Error('Invalid result'));
                    } else {
                        resolve(result);
                    }
                },
                reject: function(error) {
                    clearTimeout(timer);
                    workerCallbacks.delete(id);
                    logger.error(`Worker task "${type}" (id: ${id}, attempt: ${attempt}) failed.`, {
                        error: error.message,
                        matrixSample: type === 'complexEigenvalues' ? (data?.matrix?.flatData?.slice(0, 5)) : undefined
                    });
                    reject(error);
                }
            });

            if (type === 'complexEigenvalues' && (!data || !data.matrix || !data.matrix.flatData || !Number.isInteger(data.matrix.rows) || !Number.isInteger(data.matrix.cols))) {
                logger.error(`runWorkerTask: Invalid matrix data for "${type}".`, { data: data });
                reject(new Error('Invalid matrix data'));
                return;
            }
            worker.postMessage({ type: type, id: id, data: data });
        });
    }

    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            const result = await attemptTask(attempt);
            return result;
        } catch (error) {
            lastError = error;
            if (attempt < maxAttempts) {
                logger.warn(`Attempt ${attempt} failed for task "${type}". Retrying...`, {
                    error: error.message,
                    matrixSample: type === 'complexEigenvalues' ? (data?.matrix?.flatData?.slice(0, 5)) : undefined
                });
            }
        }
    }

    logger.error(`All ${maxAttempts} attempts failed for task "${type}".`, {
        error: lastError?.message || 'Unknown error',
        matrixSample: type === 'complexEigenvalues' ? (data?.matrix?.flatData?.slice(0, 5)) : undefined
    });

    let fallbackResult = null;
    if (type === 'matVecMul') {
        fallbackResult = vecZeros(data?.expectedDim || 1);
    } else if (type === 'matMul') {
        fallbackResult = { flatData: new Float32Array(), rows: 0, cols: 0 };
    } else if (type === 'ksg_mi') {
        fallbackResult = 0;
    } else if (type === 'topologicalScore') {
        fallbackResult = { score: 0 };
    } else if (type === 'solveLinearSystemCG') {
    fallbackResult = vecZeros(data?.b?.length || 1);
    logger.warn(`runWorkerTask: Using zero vector fallback for solveLinearSystemCG.`);
}
    return fallbackResult;
}

worker.onmessage = function(e) {
    logger.debug('Main thread: Worker message received.', { data: e.data });
    const id = e.data && e.data.id;
    const result = e.data && e.data.result;
    const error = e.data && e.data.error;
    const type = e.data && e.data.type;
    const callback = workerCallbacks.get(id);
    if (callback) {
        if (error) {
            logger.error('Worker task "' + (type || 'unknown') + '" (id: ' + id + ') returned error: ' + error, {
                rawData: e.data
            });
            callback.reject(new Error(error));
        } else {
            callback.resolve(result);
        }
    }
};

worker.onerror = function(error) {
    logger.error('CRITICAL WORKER ERROR:', { message: error.message, file: error.filename, line: error.lineno });
    workerCallbacks.forEach(function(cb) { cb.reject(new Error('Worker crashed')); });
    workerCallbacks.clear();
    logger.info('Attempting to restart worker after crash.');
    const newWorker = new Worker(URL.createObjectURL(workerBlob));
    Object.assign(worker, newWorker);
};
