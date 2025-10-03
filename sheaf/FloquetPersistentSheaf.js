// src/sheaf/FloquetPersistentSheaf.js
import { CircularBuffer } from '../data-structures/CircularBuffer.js';
import { PersistentAdjunctionSheaf } from './PersistentAdjunctionSheaf.js';
import { logger, clamp, identity, matMul, vecZeros, isFiniteVector, isFiniteMatrix, norm2, flattenMatrix, _matVecMul, _transpose, vecAdd, zeroMatrix, vecSub, _matMul, isFiniteNumber, runWorkerTask} from '../utils.js';



const tf = window.tf || null;

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