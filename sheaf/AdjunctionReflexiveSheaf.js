// src/sheaf/AdjunctionReflexiveSheaf.ts

import { RecursiveTopologicalSheaf } from './RecursiveTopologicalSheaf.js';
import { logger, vecZeros, identity, vecAdd, vecScale, matVecMul, vecSub, norm2, isFiniteVector, isFiniteMatrix, _matVecMul, _transpose, zeroMatrix, clamp, dot, _matMul, isFiniteNumber, runWorkerTask } from '../utils.js';


/**
 * Theorem 15: AdjunctionReflexiveSheaf – Categorical Monad Extension.
 * Monadic T = UF for hierarchical reflexivity, equalizer fixed sheaves.
 */
export class AdjunctionReflexiveSheaf extends RecursiveTopologicalSheaf {
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