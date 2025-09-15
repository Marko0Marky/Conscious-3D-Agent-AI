// --- START OF FILE worker-logic.js ---
// Self-contained worker utilities (no imports needed)

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
        if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) {
            return false;
        }
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
    
    if (!isFiniteMatrix(m)) {
        console.warn('Worker matVecMul: Input matrix is non-finite. Returning zeros.');
        return new Float32Array(r).fill(0);
    }
    if (!isFiniteVector(v) || v.length !== c) {
        console.warn('Worker matVecMul: Input vector is non-finite or length mismatch. Returning zeros.', {v_length: v ? v.length : 'null', c});
        return new Float32Array(r).fill(0);
    }

    const out = new Float32Array(r);
    for (let i = 0; i < r; ++i) {
        let s = 0.0;
        const row = m[i];
        for (let j = 0; j < c; ++j) {
            const val_m = row[j];
            const val_v = v[j];
            const product = (Number.isFinite(val_m) ? val_m : 0) * (Number.isFinite(val_v) ? val_v : 0);
            s += Number.isFinite(product) ? product : 0;
        }
        out[i] = Number.isFinite(s) ? s : 0;
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
    if (rows === 0 || cols === 0) return { flatData: new Float32Array(0), rows: rows, cols: cols };

    const flatData = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        for(let j = 0; j < cols; j++) {
            const val = matrix[i][j];
            flatData[i * cols + j] = Number.isFinite(val) ? val : 0;
        }
    }
    return { flatData, rows, cols };
}

function transpose(matrix) {
    if (!isFiniteMatrix(matrix)) {
        return [];
    }
    const numRows = matrix.length;
    const numCols = matrix[0] && matrix[0].length !== undefined ? matrix[0].length : 0;
    if (numRows === 0 || numCols === 0) return [];

    const result = Array(numCols).fill(0).map(() => new Float32Array(numRows));
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const val = matrix[i][j];
            result[j][i] = Number.isFinite(val) ? val : 0;
        }
    }
    return result;
}

function solveLinearSystemCG(A_flat_data, b, opts={tol:1e-6, maxIter:200}) {
    const A = unflattenMatrix(A_flat_data);
    const n = b.length;

    if (!isFiniteMatrix(A) || A.length !== n || (A.length > 0 && A[0].length !== n) || !isFiniteVector(b)) {
        console.warn('Worker solveLinearSystemCG: Invalid input matrix A or vector b. Returning sanitized b.');
        return new Float32Array(b.map(x => Number.isFinite(x) ? clamp(x, -1, 1) : 0));
    }

    let x = new Float32Array(n).fill(0);
    let r = new Float32Array(b.map(v => Number.isFinite(v) ? v : 0));
    let p = new Float32Array(r);
    let rsold = dot(r, r);

    if (!Number.isFinite(rsold) || rsold < 1e-20) {
        console.warn('Worker solveLinearSystemCG: Initial rsold is non-finite or too small. Returning sanitized x.');
        return new Float32Array(x.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
    }

    const Ap = new Float32Array(n);
    const maxIter = Math.min(opts.maxIter || 200, n * 5);

    for (let it = 0; it < maxIter; ++it) {
        Ap.set(matVecMul(A, p));

        const denom = dot(p, Ap);
        if (!Number.isFinite(denom) || denom <= 1e-20) {
            console.warn('Worker solveLinearSystemCG: Denominator is non-finite or too small. Breaking CG loop.');
            break;
        }

        const alpha = rsold / denom;
        if (!Number.isFinite(alpha)) {
            console.warn('Worker solveLinearSystemCG: Alpha is non-finite. Breaking CG loop.');
            break;
        }

        for (let i = 0; i < n; i++) x[i] = (Number.isFinite(x[i]) ? x[i] : 0) + (Number.isFinite(alpha) ? alpha : 0) * (Number.isFinite(p[i]) ? p[i] : 0);
        for (let i = 0; i < n; i++) r[i] = (Number.isFinite(r[i]) ? r[i] : 0) - (Number.isFinite(alpha) ? alpha : 0) * (Number.isFinite(Ap[i]) ? Ap[i] : 0);

        const rsnew = dot(r, r);
        if (!Number.isFinite(rsnew)) {
            console.warn('Worker solveLinearSystemCG: Rsnew is non-finite. Breaking CG loop.');
            break;
        }
        if (Math.sqrt(rsnew) < (opts.tol || 1e-6)) break;

        const beta = rsnew / (rsold + 1e-20);
        if (!Number.isFinite(beta)) {
            console.warn('Worker solveLinearSystemCG: Beta is non-finite. Breaking CG loop.');
            break;
        }

        for (let i = 0; i < n; i++) p[i] = (Number.isFinite(r[i]) ? r[i] : 0) + (Number.isFinite(beta) ? beta : 0) * (Number.isFinite(p[i]) ? p[i] : 0);
        rsold = rsnew;
    }
    return new Float32Array(x.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));
}

function covarianceMatrix(states_array, eps = 1e-3) {
    const states = states_array.filter(s => isFiniteVector(s));
    if (!Array.isArray(states) || states.length < 2) {
        console.warn('Worker covarianceMatrix: Not enough valid states. Returning minimal matrix.');
        return [[eps]];
    }
    
    const n = states.length;
    const d = states[0] && states[0].length !== undefined ? states[0].length : 0;
    if (d === 0) {
        console.warn('Worker covarianceMatrix: State dimension is zero. Returning minimal matrix.');
        return [[eps]];
    }

    const mean = new Float32Array(d).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            const val = states[i][j];
            mean[j] += (Number.isFinite(val) ? val : 0);
        }
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const cov = Array(d).fill(0).map(() => new Float32Array(d).fill(0));
    for (let k = 0; k < n; k++) {
        for (let i = 0; i < d; i++) {
            const di = (Number.isFinite(states[k][i]) ? states[k][i] : 0) - (Number.isFinite(mean[i]) ? mean[i] : 0);
            for (let j = i; j < d; j++) {
                const dj = (Number.isFinite(states[k][j]) ? states[k][j] : 0) - (Number.isFinite(mean[j]) ? mean[j] : 0);
                const product = di * dj;
                cov[i][j] += (Number.isFinite(product) ? product : 0) / Math.max(1, n - 1);
            }
        }
    }
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < i; j++) cov[j][i] = cov[i][j];
    }
    for (let i = 0; i < d; i++) cov[i][i] = (Number.isFinite(cov[i][i]) ? cov[i][i] : 0) + eps;
    
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
            if (!Number.isFinite(cov[i][j])) {
                console.warn('Worker covarianceMatrix: Non-finite element in covariance matrix. Setting to 0.');
                cov[i][j] = 0;
            }
        }
    }

    return cov;
}

function matrixSpectralNormApprox(M_flat_data, maxIter = 10) {
    const M = unflattenMatrix(M_flat_data);
    if (!isFiniteMatrix(M)) {
        console.warn('Worker matrixSpectralNormApprox: Input matrix is non-finite. Returning 0.');
        return 0;
    }
    let n = M.length;
    if (n === 0) return 0;
    
    let v = new Float32Array(n).fill(1 / Math.sqrt(n));

    for (let i = 0; i < maxIter; i++) {
        const Av = matVecMul(M, v);
        if (!isFiniteVector(Av)) {
            console.warn('Worker matrixSpectralNormApprox: Intermediate Av is non-finite. Returning 0.');
            return 0;
        }
        const norm = norm2(Av);
        if (norm < 1e-10) return 0;
        for(let k=0; k<n; k++) v[k] = (Av[k] || 0) / norm;
    }
    const finalAv = matVecMul(M, v);
    if (!isFiniteVector(finalAv)) {
        console.warn('Worker matrixSpectralNormApprox: Final Av is non-finite. Returning 0.');
        return 0;
    }
    return norm2(finalAv);
}

function matrixRank(M_flat_data) {
    const M = unflattenMatrix(M_flat_data);
    if (!isFiniteMatrix(M)) {
        console.warn('Worker matrixRank: Input matrix is non-finite. Returning 0.');
        return 0;
    }
    const m = M.length;
    if (m === 0) return 0;
    const n = M[0] && M[0].length !== undefined ? M[0].length : 0;
    if (n === 0) return 0;

    const A = M.map(row => new Float32Array(row));
    let rank = 0;
    const tol = 1e-10;

    for (let col = 0; col < n && rank < m; col++) {
        let pivotRow = rank;
        for (let i = rank + 1; i < m; i++) {
            if (Math.abs(A[i][col]) > Math.abs(A[pivotRow][col])) pivotRow = i;
        }

        if (Math.abs(A[pivotRow][col]) < tol) continue;

        [A[rank], A[pivotRow]] = [A[pivotRow], A[rank]];

        const pivot = A[rank][col];
        if (!Number.isFinite(pivot) || Math.abs(pivot) < tol) {
            console.warn('Worker matrixRank: Pivot is non-finite or too small. Continuing.');
             continue;
        }
        for (let j = col; j < n; j++) A[rank][j] = (A[rank][j] || 0) / pivot;

        for (let i = 0; i < m; i++) {
            if (i === rank) continue;
            const factor = A[i][col];
            if (!Number.isFinite(factor)) {
                console.warn('Worker matrixRank: Factor is non-finite. Continuing.');
                continue;
            }
            for (let j = col; j < n; j++) A[i][j] = (A[i][j] || 0) - (factor * (A[rank][j] || 0));
        }
        rank++;
    }
    return rank;
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

    console.log(`Worker smithNormalFormGF2: Computed rank=${rank}, rows=${rows}, cols=${cols}, diagonal_length=${diagonal.length}`);
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
            case 'matrixRank':
                result = matrixRank(data.matrix);
                break;
            case 'smithNormalForm':
                result = smithNormalFormGF2(data.matrix);
                break;
            default:
                result = { error: 'Unknown message type' };
                break;
        }
        self.postMessage({ type: type + 'Result', id, result });
    } catch (error) {
        console.error(`Worker error for type ${type}:`, error);
        self.postMessage({ type: type + 'Error', id, error: error.message || String(error) });
    }
    
};
// --- END OF FILE worker-logic.js ---
