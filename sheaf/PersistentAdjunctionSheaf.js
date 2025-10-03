// src/sheaf/PersistentAdjunctionSheaf.ts

import { AdjunctionReflexiveSheaf } from './AdjunctionReflexiveSheaf.js';
import { CircularBuffer } from '../data-structures/CircularBuffer.js';
import { logger, vecZeros, identity, vecAdd, vecScale, matVecMul, vecSub, norm2, clamp, isFiniteVector, isFiniteMatrix, _matVecMul, _transpose, zeroMatrix, _matMul, isFiniteNumber } from '../utils.js';
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

/**
 * Theorem 16: PersistentAdjunctionSheaf – Flow Persistence Extension.
 * Bottleneck PD over filtrations for diachronic invariants.
 */
export class PersistentAdjunctionSheaf extends AdjunctionReflexiveSheaf {
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