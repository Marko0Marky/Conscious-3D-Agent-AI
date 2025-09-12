// --- START OF FILE three-dee-game.js ---
import { clamp, logger } from './utils.js';

// Assume THREE is globally available via CDN script tag in index.html

/**
 * Represents the 3D game environment, handling physics, scoring, and rendering with Three.js.
 */
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
        this.walls = [];
        this.collidables = [];
        
        this.reset();
    }
    
    async resumeAudioContext() {
        if (this.audioContext.state === 'suspended') {
            return this.audioContext.resume().catch(e => logger.error("Failed to resume AudioContext in 3DGame:", e));
        }
        return Promise.resolve();
    }

    reset() {
        const objectsToDisposeAndRemove = [];
        if (this.player) objectsToDisposeAndRemove.push(this.player);
        if (this.ai) objectsToDisposeAndRemove.push(this.ai);
        if (this.playerTarget) objectsToDisposeAndRemove.push(this.playerTarget);
        if (this.aiTarget) objectsToDisposeAndRemove.push(this.aiTarget);
        this.walls.forEach(w => objectsToDisposeAndRemove.push(w));
        this.obstacles.forEach(o => objectsToDisposeAndRemove.push(o));
        if (this.ground) objectsToDisposeAndRemove.push(this.ground);

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
        this.walls = [];
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


        const wallMat = new THREE.MeshStandardMaterial({ color: 0x4a4a6a });
        const wallGeomH = new THREE.BoxGeometry(ThreeDeeGame.WORLD_SIZE + 2, 10, 2);
        const wallGeomV = new THREE.BoxGeometry(2, 10, ThreeDeeGame.WORLD_SIZE + 2);
        
        const createAndAddWall = (geom, mat, x, z, rotY = 0, name) => {
            const wall = new THREE.Mesh(geom, mat);
            wall.position.set(x, 5, z);
            wall.rotation.y = rotY;
            wall.castShadow = true;
            wall.receiveShadow = true;
            wall.name = name;
            wall.position.x = clamp(wall.position.x, -worldHalf - 1, worldHalf + 1);
            wall.position.z = clamp(wall.position.z, -worldHalf - 1, worldHalf + 1);

            this.scene.add(wall);
            this.walls.push(wall);
            this.collidables.push(wall);
        };

        createAndAddWall(wallGeomH, wallMat, 0, -worldHalf, 0, 'wall1');
        createAndAddWall(wallGeomH, wallMat, 0, worldHalf, 0, 'wall2');
        createAndAddWall(wallGeomV, wallMat, -worldHalf, 0, 0, 'wall3');
        createAndAddWall(wallGeomV, wallMat, worldHalf, 0, 0, 'wall4');


        this.addObstacles(8);
        this.obstacles.forEach(obstacle => {
            if (obstacle && obstacle.isObject3D) {
                this.collidables.push(obstacle);
            } else {
                logger.warn("Invalid obstacle found during reset; not adding to collidables.");
            }
        });

        const playerGeom = new THREE.BoxGeometry(4, 4, 4);
        const playerMat = new THREE.MeshStandardMaterial({ color: 0xff9900 });
        this.player = new THREE.Mesh(playerGeom, playerMat);
        this.player.castShadow = true;
        this.player.name = 'player';
        this.player.position.set(clamp(-worldHalf / 2, -worldHalf + 2, worldHalf - 2), 2, 0);
        this.scene.add(this.player);
        this.collidables.push(this.player);

        const aiGeom = new THREE.BoxGeometry(4, 4, 4);
        const aiMat = new THREE.MeshStandardMaterial({ color: 0x44aaff });
        this.ai = new THREE.Mesh(aiGeom, aiMat);
        this.ai.castShadow = true;
        this.ai.name = 'ai';
        this.ai.position.set(clamp(worldHalf / 2, -worldHalf + 2, worldHalf - 2), 2, 0);
        this.scene.add(this.ai);
        this.collidables.push(this.ai);
        
        const pTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const pTargetMat = new THREE.MeshStandardMaterial({ color: 0xff9900, emissive: 0xaa6600 });
        this.playerTarget = new THREE.Mesh(pTargetGeom, pTargetMat);
        this.playerTarget.name = 'playerTarget';
        this.scene.add(this.playerTarget);
        this.respawnTarget(this.playerTarget);
        this.collidables.push(this.playerTarget);

        const aiTargetGeom = new THREE.SphereGeometry(3, 16, 16);
        const aiTargetMat = new THREE.MeshStandardMaterial({ color: 0x44aaff, emissive: 0x2288dd });
        this.aiTarget = new THREE.Mesh(aiTargetGeom, aiTargetMat);
        this.aiTarget.name = 'aiTarget';
        this.scene.add(this.aiTarget);
        this.respawnTarget(this.aiTarget);
        this.collidables.push(this.aiTarget);
        
        this.collidables = this.collidables.filter(obj => obj && obj.isObject3D);

        logger.info(`ThreeDeeGame reset completed. Total collidables: ${this.collidables.length}`);
    }

    addObstacles(count) {
        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2;
        const spawnArea = worldHalf * 0.8;
        const obstacleMat = new THREE.MeshStandardMaterial({ color: 0x886688 });

        for(let i = 0; i < count; i++) {
            const sx = clamp(4 + Math.random() * 12, 1, 20);
            const sz = clamp(4 + Math.random() * 12, 1, 20);
            
            const obstacleGeom = new THREE.BoxGeometry(sx, 10, sz);
            const obstacle = new THREE.Mesh(obstacleGeom, obstacleMat);
            
            const ox = clamp((Math.random() - 0.5) * spawnArea, -worldHalf + sx / 2, worldHalf - sx / 2);
            const oz = clamp((Math.random() - 0.5) * spawnArea, -worldHalf + sz / 2, worldHalf - sz / 2);

            obstacle.position.set(ox, 5, oz);
            obstacle.name = `obstacle-${i}`;
            
            if (!Number.isFinite(obstacle.position.x) || !Number.isFinite(obstacle.position.z)) {
                logger.error(`Non-finite obstacle position detected during creation for obstacle ${i}! Skipping this obstacle.`);
                continue;
            }

            obstacle.castShadow = true;
            this.obstacles.push(obstacle);
            this.scene.add(obstacle);
        }
    }


    respawnTarget(target) {
        const worldHalf = ThreeDeeGame.WORLD_SIZE / 2;
        const spawnRadius = worldHalf * 0.8;
        const tx = clamp((Math.random()-0.5) * spawnRadius, -worldHalf + 5, worldHalf - 5);
        const tz = clamp((Math.random()-0.5) * spawnRadius, -worldHalf + 5, worldHalf - 5);

        target.position.set(tx, 3, tz);
        
        if (!Number.isFinite(target.position.x) || !Number.isFinite(target.position.z)) {
            logger.error(`Non-finite target position detected during respawn! Resetting to default.`);
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
        
        for(const obstacle of this.obstacles) {
            const obstacleBB = new THREE.Box3().setFromObject(obstacle);
            if(playerBB.intersectsBox(obstacleBB)) pReward -= 0.5;
            if(aiBB.intersectsBox(obstacleBB)) aReward -= 0.5;
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
        const agentDirection = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), agentMesh.rotation.y);
        
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

            if(intersects.length > 0 && intersects[0].distance < ThreeDeeGame.MAX_RAY_DISTANCE) {
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

        if (action[0] === 1) {
            object.position.x += Math.sin(object.rotation.y) * ThreeDeeGame.AGENT_SPEED;
            object.position.z += Math.cos(object.rotation.y) * ThreeDeeGame.AGENT_SPEED;
        }
        if (action[1] === 1) {
            object.rotation.y += ThreeDeeGame.AGENT_TURN_SPEED;
        }
        if (action[2] === 1) {
            object.rotation.y -= ThreeDeeGame.AGENT_TURN_SPEED;
        }

        if (object && object.isObject3D) {
            const agentBB = new THREE.Box3().setFromObject(object);
            for(const obstacle of this.obstacles) {
                if (obstacle && obstacle.isObject3D) {
                    if (agentBB.intersectsBox(new THREE.Box3().setFromObject(obstacle))) {
                        object.position.copy(prevPosition);
                        break;
                    }
                }
            }
        }
    }

    getState() {
        return {
            player: { x: this.player.position.x, z: this.player.position.z, rotY: this.player.rotation.y },
            ai: { x: this.ai.position.x, z: this.ai.position.z, rotY: this.ai.rotation.y },
            playerTarget: { x: this.playerTarget.position.x, z: this.playerTarget.position.z },
            aiTarget: { x: this.aiTarget.position.x, z: this.aiTarget.position.z },
        };
    }
    
    resize(width, height){
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
            logger.warn('Audio playback failed or context error', e.message);
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
// --- END OF FILE three-dee-game.js ---