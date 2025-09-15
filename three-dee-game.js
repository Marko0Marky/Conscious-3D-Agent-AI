
import { clamp, logger } from './utils.js';
import { LevelGenerator } from './level-generator.js';
import * as THREE from 'three';
import { BufferGeometryUtils } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export class ThreeDeeGame {
    static WORLD_SIZE = 100;
    static AGENT_SPEED = 0.8;
    static AGENT_TURN_SPEED = 0.05;
    static MAX_RAY_DISTANCE = 30;

    constructor(canvas) {
        this.canvas = canvas;
        this.width = Math.max(1, canvas.clientWidth);
        this.height = Math.max(1, canvas.clientHeight);

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a15);
        
        this.camera = new THREE.PerspectiveCamera(75, this.width / this.height, 0.1, 1000);
        this.camera.position.set(0, 80, 60);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;

        const ambientLight = new THREE.AmbientLight(0x404060, 1.5);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(20, 50, 20);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        this.raycaster = new THREE.Raycaster();
        
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        this.player = null;
        this.ai = null;
        this.playerTarget = null;
        this.aiTarget = null;
        this.ground = null;
        this.obstacles = [];
        this.mergedWallsMesh = null;
        this.collidables = [];

        this.levelGenerator = new LevelGenerator(ThreeDeeGame.WORLD_SIZE, THREE); 
        this.reset();
        logger.info('ThreeDeeGame initialized.');
    }
    
    async resumeAudioContext() {
        if (this.audioContext.state === 'suspended') {
            return this.audioContext.resume().catch(e => logger.error("Failed to resume AudioContext:", e));
        }
        return Promise.resolve();
    }

    normalizeAngle(angle) {
        const twoPi = 2 * Math.PI;
        angle = angle % twoPi;
        if (angle < 0) angle += twoPi;
        return Number.isFinite(angle) ? angle : 0;
    }

    reset() {
        const objectsToDisposeAndRemove = [];
        if (this.player) objectsToDisposeAndRemove.push(this.player);
        if (this.ai) objectsToDisposeAndRemove.push(this.ai);
        if (this.playerTarget) objectsToDisposeAndRemove.push(this.playerTarget);
        if (this.aiTarget) objectsToDisposeAndRemove.push(this.aiTarget);
        this.obstacles.forEach(o => objectsToDisposeAndRemove.push(o));
        if (this.ground) objectsToDisposeAndRemove.push(this.ground);
        if (this.mergedWallsMesh) objectsToDisposeAndRemove.push(this.mergedWallsMesh);

        objectsToDisposeAndRemove.forEach(obj => {
            if (obj.parent) {
                obj.parent.remove(obj);
            }
            if (obj.geometry) {
                obj.geometry.dispose();
            }
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(m => m.dispose());
                } else {
                    obj.material.dispose();
                }
            }
        });

        this.obstacles = [];
        this.collidables = [];
        this.score = { ai: 0, player: 0 };
        
        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2;
        
        const groundGeom = new THREE.PlaneGeometry(ThreeDeeGame.WORLD_SIZE, ThreeDeeGame.WORLD_SIZE);
        const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a4a });
        this.ground = new THREE.Mesh(groundGeom, groundMat);
        this.ground.rotation.x = -Math.PI / 2;
        this.ground.receiveShadow = true;
        this.ground.name = 'ground';
        this.scene.add(this.ground);
        this.collidables.push(this.ground);

        const levelData = this.levelGenerator.generateLevel(8);
        
        const wallGeometries = [];
        levelData.walls.forEach(wall => {
            wall.updateMatrixWorld(true);
            const geom = wall.geometry.clone();
            geom.applyMatrix4(wall.matrixWorld);
            wallGeometries.push(geom);
        });

        if (wallGeometries.length > 0) {
            const mergedWallGeometry = BufferGeometryUtils.mergeBufferGeometries(wallGeometries);
            const wallMat = new THREE.MeshStandardMaterial({ color: 0x4a4a6a });
            this.mergedWallsMesh = new THREE.Mesh(mergedWallGeometry, wallMat);
            this.mergedWallsMesh.name = 'mergedWalls';
            this.mergedWallsMesh.castShadow = true;
            this.mergedWallsMesh.receiveShadow = true;
            this.scene.add(this.mergedWallsMesh);
            this.collidables.push(this.mergedWallsMesh);
        }

        levelData.obstacles.forEach(obstacle => {
            this.scene.add(obstacle);
            this.obstacles.push(obstacle);
            this.collidables.push(obstacle);
        });

        const playerGeom = new THREE.BoxGeometry(4, 4, 4);
        const playerMat = new THREE.MeshStandardMaterial({ color: 0xff9900 });
        this.player = new THREE.Mesh(playerGeom, playerMat);
        this.player.castShadow = true;
        this.player.name = 'player';
        this.player.position.copy(levelData.spawnPoints.player);
        this.player.position.y = 2;
        this.player.rotation.y = this.normalizeAngle(Math.random() * 2 * Math.PI); // Random initial rotation
        this.scene.add(this.player);
        this.collidables.push(this.player);

        const aiGeom = new THREE.BoxGeometry(4, 4, 4);
        const aiMat = new THREE.MeshStandardMaterial({ color: 0x44aaff });
        this.ai = new THREE.Mesh(aiGeom, aiMat);
        this.ai.castShadow = true;
        this.ai.name = 'ai';
        this.ai.position.copy(levelData.spawnPoints.ai);
        this.ai.position.y = 2;
        this.ai.rotation.y = this.normalizeAngle(Math.random() * 2 * Math.PI); // Random initial rotation
        this.scene.add(this.ai);
        this.collidables.push(this.ai);
        
        const pTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const pTargetMat = new THREE.MeshStandardMaterial({ color: 0xff9900, emissive: 0xaa6600 });
        this.playerTarget = new THREE.Mesh(pTargetGeom, pTargetMat);
        this.playerTarget.name = 'playerTarget';
        this.playerTarget.position.copy(levelData.targetPoints.player);
        this.playerTarget.position.y = 3;
        this.scene.add(this.playerTarget);
        this.collidables.push(this.playerTarget);

        const aiTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const aiTargetMat = new THREE.MeshStandardMaterial({ color: 0x44aaff, emissive: 0x2288dd });
        this.aiTarget = new THREE.Mesh(aiTargetGeom, aiTargetMat);
        this.aiTarget.name = 'aiTarget';
        this.aiTarget.position.copy(levelData.targetPoints.ai);
        this.aiTarget.position.y = 3;
        this.scene.add(this.aiTarget);
        this.collidables.push(this.aiTarget);
        
        this.collidables = this.collidables.filter(obj => obj && obj.isObject3D);

        logger.info(`ThreeDeeGame reset completed. Total collidables: ${this.collidables.length}`);
    }

    respawnTarget(target) {
        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2;
        const spawnRadius = worldHalf * 0.8;
        let newPosFound = false;
        let attempts = 0;
        const maxAttempts = 50;
        const minTargetDistance = 10;

        while (!newPosFound && attempts < maxAttempts) {
            const tx = clamp((Math.random() - 0.5) * spawnRadius, -worldHalf + 5, worldHalf - 5);
            const tz = clamp((Math.random() - 0.5) * spawnRadius, -worldHalf + 5, worldHalf - 5);
            const potentialPos = new THREE.Vector3(tx, 3, tz);

            let tooClose = false;
            if (potentialPos.distanceTo(this.player.position) < minTargetDistance ||
                potentialPos.distanceTo(this.ai.position) < minTargetDistance) {
                tooClose = true;
            }
            if (target === this.playerTarget && potentialPos.distanceTo(this.aiTarget.position) < minTargetDistance) {
                tooClose = true;
            }
            if (target === this.aiTarget && potentialPos.distanceTo(this.playerTarget.position) < minTargetDistance) {
                tooClose = true;
            }
            for (const obstacle of this.obstacles) {
                const obstacleBB = new THREE.Box3().setFromObject(obstacle);
                const targetBB = new THREE.Box3().setFromCenterAndSize(potentialPos, new THREE.Vector3(6, 6, 6));
                if (obstacleBB.intersectsBox(targetBB)) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                target.position.copy(potentialPos);
                newPosFound = true;
            }
            attempts++;
        }

        if (!newPosFound) {
            logger.warn(`Could not find a clear spot for ${target.name} respawn after ${maxAttempts} attempts. Placing at (0,3,0).`);
            target.position.set(0, 3, 0);
        }
    }

    update() {
        let aReward = 0;
        let pReward = 0;

        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2 - 2;
        this.player.position.x = clamp(this.player.position.x, -worldHalf, worldHalf);
        this.player.position.z = clamp(this.player.position.z, -worldHalf, worldHalf);
        this.ai.position.x = clamp(this.ai.position.x, -worldHalf, worldHalf);
        this.ai.position.z = clamp(this.ai.position.z, -worldHalf, worldHalf);

        const playerBB = new THREE.Box3().setFromObject(this.player);
        const aiBB = new THREE.Box3().setFromObject(this.ai);
        
        for (const obstacle of this.obstacles) {
            const obstacleBB = new THREE.Box3().setFromObject(obstacle);
            if (playerBB.intersectsBox(obstacleBB)) pReward -= 0.05;
            if (aiBB.intersectsBox(obstacleBB)) aReward -= 0.05;
        }

        const pDist = this.player.position.distanceTo(this.playerTarget.position);
        const aDist = this.ai.position.distanceTo(this.aiTarget.position);

        pReward -= pDist * 0.001;
        aReward -= aDist * 0.001;

        if (pDist < 5) {
            pReward += 1.0;
            this.score.player++;
            this.respawnTarget(this.playerTarget);
            this.playSound('win');
        }
        if (aDist < 5) {
            aReward += 1.0;
            this.score.ai++;
            this.respawnTarget(this.aiTarget);
            this.playSound('win');
        }

        if (!Number.isFinite(aReward) || !Number.isFinite(pReward)) {
            logger.warn(`Invalid rewards: aReward=${aReward}, pReward=${pReward}, resetting to 0`);
            aReward = 0;
            pReward = 0;
        }

        return { aReward, pReward, isDone: false };
    }
    
    getRaycastDetections(agentMesh) {
        const results = { left: 0, center: 0, right: 0 };
        const origin = agentMesh.position.clone().add(new THREE.Vector3(0, 1, 0));
        
        if (!Number.isFinite(origin.x) || !Number.isFinite(origin.y) || !Number.isFinite(origin.z)) {
            logger.error(`Raycast origin for ${agentMesh.name || 'agent'} is non-finite:`, origin);
            return { left: 0, center: 0, right: 0 };
        }

        const angles = { left: Math.PI / 4, center: 0, right: -Math.PI / 4 };
        const agentDirection = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.normalizeAngle(agentMesh.rotation.y));
        
        if (!Number.isFinite(agentDirection.x) || !Number.isFinite(agentDirection.y) || !Number.isFinite(agentDirection.z)) {
            logger.error(`Raycast base direction for ${agentMesh.name || 'agent'} is non-finite:`, agentDirection);
            return { left: 0, center: 0, right: 0 };
        }
        
        const rayCollidables = this.collidables.filter(c => c && c.isObject3D && c !== agentMesh);
        if (rayCollidables.length === 0) {
            logger.warn(`No valid collidables for raycasting for ${agentMesh.name || 'agent'}.`);
            return { left: 0, center: 0, right: 0 };
        }

        for (const key in angles) {
            const direction = agentDirection.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), angles[key]);
            
            if (!Number.isFinite(direction.x) || !Number.isFinite(direction.y) || !Number.isFinite(direction.z)) {
                logger.warn(`Ray direction for ${key} ray of ${agentMesh.name || 'agent'} is non-finite, skipping this ray.`, direction);
                continue;
            }

            this.raycaster.set(origin, direction);
            const intersects = this.raycaster.intersectObjects(rayCollidables);

            if (intersects.length > 0 && intersects[0].distance < ThreeDeeGame.MAX_RAY_DISTANCE) {
                results[key] = 1.0 - clamp(intersects[0].distance / ThreeDeeGame.MAX_RAY_DISTANCE, 0, 1);
            }
        }
        return results;
    }

    setAIAction(action) {
        this.applyActionToObject(action, this.ai);
    }

    setPlayerAction(action) {
        this.applyActionToObject(action, this.player);
    }
    
    applyActionToObject(action, object) {
        const prevPosition = object.position.clone();
        let forward = 0, left = 0, right = 0;

        // Handle both string and array actions
        if (typeof action === 'string') {
            switch (action) {
                case 'FORWARD': forward = 1; break;
                case 'LEFT': left = 1; break;
                case 'RIGHT': right = 1; break;
                case 'IDLE': break;
                default: logger.warn(`Unknown string action for ${object.name}: ${action}`);
            }
        } else if (Array.isArray(action) && action.length >= 3) {
            [forward, left, right] = action;
            if (!Number.isFinite(forward) || !Number.isFinite(left) || !Number.isFinite(right)) {
                logger.warn(`Invalid array action for ${object.name}:`, action);
                return;
            }
        } else {
            logger.warn(`Invalid action format for ${object.name}:`, action);
            return;
        }

        if (forward === 1) {
            object.position.x += Math.sin(this.normalizeAngle(object.rotation.y)) * ThreeDeeGame.AGENT_SPEED;
            object.position.z += Math.cos(this.normalizeAngle(object.rotation.y)) * ThreeDeeGame.AGENT_SPEED;
        }
        if (left === 1) {
            object.rotation.y += ThreeDeeGame.AGENT_TURN_SPEED;
        }
        if (right === 1) {
            object.rotation.y -= ThreeDeeGame.AGENT_TURN_SPEED;
        }

        object.rotation.y = this.normalizeAngle(object.rotation.y);

        if (object && object.isObject3D) {
            const agentBB = new THREE.Box3().setFromObject(object);
            for (const obstacle of this.obstacles) {
                if (obstacle && obstacle.isObject3D) {
                    if (agentBB.intersectsBox(new THREE.Box3().setFromObject(obstacle))) {
                        object.position.copy(prevPosition);
                        break;
                    }
                }
            }
        }
    }

    // ThreeDeeGame.js
async update(agentAction, playerAction) {
    this.applyAction(this.ai, agentAction.action);
    this.applyAction(this.player, playerAction);
    const state = this.getState();
    await this.qualiaSheaf.diffuseQualia(state);
    if (this.qualiaSheaf.ready) {
        await this.qualiaSheaf.computeStructuralSensitivity();
        this.qualiaSheaf.visualizeActivity(this.scene, this.camera, this.renderer);
    }
    if (this.checkCollision(this.ai, this.aiTarget)) {
        this.score.ai += 1;
        this.playSound('win');
        this.resetTargets();
    }
    if (this.checkCollision(this.player, this.playerTarget)) {
        this.score.player += 1;
        this.playSound('win');
        this.resetTargets();
    }
    logger.info(`Game update: AI score=${this.score.ai}, Player score=${this.score.player}, phi=${this.qualiaSheaf.phi.toFixed(3)}`);
    this.render();
}

    getState() {
        const state = {
            player: { 
                x: Number.isFinite(this.player.position.x) ? this.player.position.x : 0, 
                z: Number.isFinite(this.player.position.z) ? this.player.position.z : 0, 
                rot: this.normalizeAngle(this.player.rotation.y) 
            },
            ai: { 
                x: Number.isFinite(this.ai.position.x) ? this.ai.position.x : 0, 
                z: Number.isFinite(this.ai.position.z) ? this.ai.position.z : 0, 
                rot: this.normalizeAngle(this.ai.rotation.y) 
            },
            playerTarget: { 
                x: Number.isFinite(this.playerTarget.position.x) ? this.playerTarget.position.x : 0, 
                z: Number.isFinite(this.playerTarget.position.z) ? this.playerTarget.position.z : 0, 
                rot: 0 
            },
            aiTarget: { 
                x: Number.isFinite(this.aiTarget.position.x) ? this.aiTarget.position.x : 0, 
                z: Number.isFinite(this.aiTarget.position.z) ? this.aiTarget.position.z : 0, 
                rot: 0 
            }
        };
        return state;
    }
    
    resize(width, height) {
        this.width = Math.max(1, width);
        this.height = Math.max(1, height);
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    playSound(type = 'collision') {
        try {
            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume().then(() => this._playActualSound(type));
            } else if (this.audioContext.state === 'running') {
                this._playActualSound(type);
            }
        } catch (e) {
            logger.warn('Audio playback failed or context error:', e.message);
        }
    }

    _playActualSound(type) {
        if (this.audioContext.state !== 'running') return;
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        if (type === 'win') {
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(587.33, this.audioContext.currentTime);
            gainNode.gain.setValueAtTime(0.3, this.audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 0.3);
            oscillator.start(this.audioContext.currentTime);
            oscillator.stop(this.audioContext.currentTime + 0.3);
        }
    }
}
