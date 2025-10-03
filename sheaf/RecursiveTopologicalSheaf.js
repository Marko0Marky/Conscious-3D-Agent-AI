import { EnhancedQualiaSheaf } from './EnhancedQualiaSheaf.js';
import { CircularBuffer } from '../data-structures/CircularBuffer.js';
import { logger, vecZeros, identity, vecAdd, vecScale, matVecMul, vecSub, norm2, clamp, isFiniteVector, isFiniteMatrix, _matVecMul, _transpose, zeroMatrix, _matMul, dot, isFiniteNumber, runWorkerTask } from '../utils.js';

/**
 * Theorem 14: RecursiveTopologicalSheaf – Fixed-Point Cohomology Extension.
 * Induces Banach contractions on cochains for reflexive fixed points.
 */
export class RecursiveTopologicalSheaf extends EnhancedQualiaSheaf {
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
