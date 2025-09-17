// --- START OF FILE three-dee-game.js ---


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
        // FIX: Ensure input angle is finite before calculation
        if (!Number.isFinite(angle)) {
            logger.error(`normalizeAngle: Input angle ${angle} is non-finite. Returning 0.`);
            return 0;
        }
        const normalized = angle % twoPi;
        // FIX: Ensure normalized angle is always finite and within [0, 2pi)
        if (Number.isFinite(normalized)) {
            return normalized < 0 ? normalized + twoPi : normalized;
        }
        logger.error(`normalizeAngle: Input angle ${angle} resulted in non-finite normalized angle. Returning 0.`);
        return 0; // Fallback to 0 if angle is non-finite
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
        // FIX: Ensure positions are finite before assignment, and clamp
        if (Number.isFinite(levelData.spawnPoints.player.x) && Number.isFinite(levelData.spawnPoints.player.z)) {
            this.player.position.copy(levelData.spawnPoints.player);
            this.player.position.y = 2;
        } else {
            logger.error('Reset: Player spawn point is non-finite. Defaulting to (0,2,0).');
            this.player.position.set(0,2,0);
        }
        this.player.rotation.y = this.normalizeAngle(Math.random() * 2 * Math.PI); // Random initial rotation
        this.scene.add(this.player);
        this.collidables.push(this.player);

        const aiGeom = new THREE.BoxGeometry(4, 4, 4);
        const aiMat = new THREE.MeshStandardMaterial({ color: 0x44aaff });
        this.ai = new THREE.Mesh(aiGeom, aiMat);
        this.ai.castShadow = true;
        this.ai.name = 'ai';
        // FIX: Ensure positions are finite before assignment, and clamp
        if (Number.isFinite(levelData.spawnPoints.ai.x) && Number.isFinite(levelData.spawnPoints.ai.z)) {
            this.ai.position.copy(levelData.spawnPoints.ai);
            this.ai.position.y = 2;
        } else {
            logger.error('Reset: AI spawn point is non-finite. Defaulting to (0,2,0).');
            this.ai.position.set(0,2,0);
        }
        this.ai.rotation.y = this.normalizeAngle(Math.random() * 2 * Math.PI); // Random initial rotation
        this.scene.add(this.ai);
        this.collidables.push(this.ai);
        
        const pTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const pTargetMat = new THREE.MeshStandardMaterial({ color: 0xff9900, emissive: 0xaa6600 });
        this.playerTarget = new THREE.Mesh(pTargetGeom, pTargetMat);
        this.playerTarget.name = 'playerTarget';
        // FIX: Ensure positions are finite before assignment, and clamp
        if (Number.isFinite(levelData.targetPoints.player.x) && Number.isFinite(levelData.targetPoints.player.z)) {
            this.playerTarget.position.copy(levelData.targetPoints.player);
            this.playerTarget.position.y = 3;
        } else {
            logger.error('Reset: Player target point is non-finite. Defaulting to (0,3,0).');
            this.playerTarget.position.set(0,3,0);
        }
        this.scene.add(this.playerTarget);
        this.collidables.push(this.playerTarget);

        const aiTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const aiTargetMat = new THREE.MeshStandardMaterial({ color: 0x44aaff, emissive: 0x2288dd });
        this.aiTarget = new THREE.Mesh(aiTargetGeom, aiTargetMat);
        this.aiTarget.name = 'aiTarget';
        // FIX: Ensure positions are finite before assignment, and clamp
        if (Number.isFinite(levelData.targetPoints.ai.x) && Number.isFinite(levelData.targetPoints.ai.z)) {
            this.aiTarget.position.copy(levelData.targetPoints.ai);
            this.aiTarget.position.y = 3;
        } else {
            logger.error('Reset: AI target point is non-finite. Defaulting to (0,3,0).');
            this.aiTarget.position.set(0,3,0);
        }
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
            const tx_raw = (Math.random() - 0.5) * spawnRadius;
            const tz_raw = (Math.random() - 0.5) * spawnRadius;
            const tx = clamp(tx_raw, -worldHalf + 5, worldHalf - 5);
            const tz = clamp(tz_raw, -worldHalf + 5, worldHalf - 5);

            // FIX: Validate generated coordinates immediately
            if (!Number.isFinite(tx) || !Number.isFinite(tz)) {
                 logger.warn(`respawnTarget: Generated non-finite coordinates for ${target.name}. Retrying.`);
                 attempts++;
                 continue;
            }
            
            const potentialPos = new THREE.Vector3(tx, 3, tz);

            let tooClose = false;
            // FIX: Ensure potential position is finite before doing distance checks
            if (!Number.isFinite(potentialPos.x) || !Number.isFinite(potentialPos.z)) {
                 logger.warn(`respawnTarget: Potential position for ${target.name} is non-finite. Skipping collision checks.`);
                 tooClose = true; // Treat as too close if non-finite
            } else {
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
            }


            if (!tooClose) {
                target.position.copy(potentialPos);
                newPosFound = true;
            }
            attempts++;
        }

        if (!newPosFound) {
            logger.warn(`Could not find a clear spot for ${target.name} respawn after ${maxAttempts} attempts. Placing at (0,3,0).`);
            // FIX: Ensure fallback position is finite
            target.position.set(0, 3, 0);
        }
    }

    update() {
        let aReward = 0;
        let pReward = 0;

        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2 - 2;

        // FIX: Add explicit checks for agent/target positions before clamping and distance calculations
        const sanitizePosition = (pos, name) => {
            if (!Number.isFinite(pos.x) || !Number.isFinite(pos.y) || !Number.isFinite(pos.z)) {
                logger.error(`update: Non-finite position for ${name} detected. Resetting to (0,2,0).`, {pos});
                pos.set(0, 2, 0); // Reset to a safe, finite default
            }
            return pos;
        };
        sanitizePosition(this.player.position, 'player');
        sanitizePosition(this.ai.position, 'ai');
        sanitizePosition(this.playerTarget.position, 'playerTarget');
        sanitizePosition(this.aiTarget.position, 'aiTarget');


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

        // FIX: Ensure pDist and aDist are finite before use
        let pDist = this.player.position.distanceTo(this.playerTarget.position);
        let aDist = this.ai.position.distanceTo(this.aiTarget.position);

        if (!Number.isFinite(pDist)) {
            logger.warn(`update: player distance is non-finite. Setting to a large value.`);
            pDist = ThreeDeeGame.WORLD_SIZE * 2;
        }
        if (!Number.isFinite(aDist)) {
            logger.warn(`update: AI distance is non-finite. Setting to a large value.`);
            aDist = ThreeDeeGame.WORLD_SIZE * 2;
        }

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

        // Final sanity check for rewards
        if (!Number.isFinite(aReward) || !Number.isFinite(pReward)) {
            logger.warn(`Invalid rewards AFTER all calculations: aReward=${aReward}, pReward=${pReward}. Resetting to 0.`);
            aReward = 0;
            pReward = 0;
        }

        return { aReward, pReward, isDone: false };
    }
    
    getRaycastDetections(agentMesh) {
        const results = { left: 0, center: 0, right: 0 };
        const origin = agentMesh.position.clone().add(new THREE.Vector3(0, 1, 0));
        
        // FIX: Validate origin before raycasting
        if (!Number.isFinite(origin.x) || !Number.isFinite(origin.y) || !Number.isFinite(origin.z)) {
            logger.error(`Raycast origin for ${agentMesh.name || 'agent'} is non-finite:`, origin);
            return { left: 0, center: 0, right: 0 };
        }

        const angles = { left: Math.PI / 4, center: 0, right: -Math.PI / 4 };
        const agentDirection = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.normalizeAngle(agentMesh.rotation.y));
        
        // FIX: Validate base direction
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
            
            // FIX: Validate per-ray direction
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

        let currentRotY = this.normalizeAngle(object.rotation.y); // Use normalized angle
        // FIX: Ensure rotation is finite before use
        if (!Number.isFinite(currentRotY)) {
             logger.error(`applyActionToObject: Non-finite rotation for ${object.name}. Resetting to 0.`);
             currentRotY = 0;
             object.rotation.y = 0;
        }

        if (forward === 1) {
            const dx = Math.sin(currentRotY) * ThreeDeeGame.AGENT_SPEED;
            const dz = Math.cos(currentRotY) * ThreeDeeGame.AGENT_SPEED;
            
            // FIX: Validate dx, dz before adding to position
            if (Number.isFinite(dx) && Number.isFinite(dz)) {
                object.position.x += dx;
                object.position.z += dz;
            } else {
                logger.warn(`applyActionToObject: Non-finite movement vector for ${object.name}. Skipping position update.`);
            }
        }
        if (left === 1) {
            object.rotation.y += ThreeDeeGame.AGENT_TURN_SPEED;
        }
        if (right === 1) {
            object.rotation.y -= ThreeDeeGame.AGENT_TURN_SPEED;
        }

        object.rotation.y = this.normalizeAngle(object.rotation.y);

        if (object && object.isObject3D) {
            // FIX: Validate object's position BEFORE creating bounding box and collision checks
            if (!Number.isFinite(object.position.x) || !Number.isFinite(object.position.y) || !Number.isFinite(object.position.z)) {
                logger.error(`applyActionToObject: Non-finite position for ${object.name} after movement. Resetting to prevPosition.`);
                object.position.copy(prevPosition); // Revert to known good position
                return; // Skip collision checks if position is bad
            }

            const agentBB = new THREE.Box3().setFromObject(object);
            for (const obstacle of this.obstacles) {
                if (obstacle && obstacle.isObject3D) {
                    const obstacleBB = new THREE.Box3().setFromObject(obstacle);
                    if (agentBB.intersectsBox(obstacleBB)) {
                        object.position.copy(prevPosition);
                        break;
                    }
                }
            }
        }
    }

    getState() {
        // FIX: Sanitize all state values before returning
        const sanitizeCoord = (val) => Number.isFinite(val) ? val : 0;
        const sanitizeRot = (val) => this.normalizeAngle(val);

        const state = {
            player: { 
                x: sanitizeCoord(this.player.position.x), 
                z: sanitizeCoord(this.player.position.z), 
                rot: sanitizeRot(this.player.rotation.y) 
            },
            ai: { 
                x: sanitizeCoord(this.ai.position.x), 
                z: sanitizeCoord(this.ai.position.z), 
                rot: sanitizeRot(this.ai.rotation.y) 
            },
            playerTarget: { 
                x: sanitizeCoord(this.playerTarget.position.x), 
                z: sanitizeCoord(this.playerTarget.position.z), 
                rot: 0 // Target has no rotation
            },
            aiTarget: { 
                x: sanitizeCoord(this.aiTarget.position.x), 
                z: sanitizeCoord(this.aiTarget.position.z), 
                rot: 0 // Target has no rotation
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
