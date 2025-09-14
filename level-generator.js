// --- START OF FILE level-generator.js ---
import { clamp, logger } from './utils.js';

// Assume THREE is globally available via CDN script tag in index.html

/**
 * Manages procedural generation of the 3D game environment, including walls,
 * obstacles, and spawn/target locations.
 */
export class LevelGenerator {
    /**
     * @param {number} worldSize - The overall size of the square world.
     * @param {THREE} THREE_LIB - The Three.js library instance.
     */
    constructor(worldSize, THREE_LIB) {
        this.worldSize = worldSize;
        this.THREE = THREE_LIB;
        this.gridSize = 10; // Number of cells along one dimension (e.g., 10x10 grid)
        this.cellSize = worldSize / this.gridSize; // Size of each grid cell
        this.wallHeight = 10;
        this.obstacleHeight = 5;
        this.obstacleMinSize = 4;
        this.obstacleMaxSize = 12;
        this.minAgentsTargetsDistance = 20; // Minimum distance between initial agent/target spawns
    }

    /**
     * Generates a new level layout with walls, obstacles, and defines spawn/target positions.
     * @param {number} numObstacles - Number of dynamic obstacles to place.
     * @returns {{
     *   walls: THREE.Mesh[],
     *   obstacles: THREE.Mesh[],
     *   spawnPoints: { player: THREE.Vector3, ai: THREE.Vector3 },
     *   targetPoints: { player: THREE.Vector3, ai: THREE.Vector3 }
     * }} Generated level data.
     */
    generateLevel(numObstacles = 8) {
        logger.info(`Generating new level: worldSize=${this.worldSize}, gridSize=${this.gridSize}, numObstacles=${numObstacles}`);
        const walls = [];
        const obstacles = [];
        const worldHalf = this.worldSize / 2;

        // 1. Generate boundary walls (these are static for all levels)
        const wallMat = new this.THREE.MeshStandardMaterial({ color: 0x4a4a6a });
        const wallGeomH = new this.THREE.BoxGeometry(this.worldSize + 2, this.wallHeight, 2); // Horizontal (along X)
        const wallGeomV = new this.THREE.BoxGeometry(2, this.wallHeight, this.worldSize + 2); // Vertical (along Z)

        const createWall = (geom, mat, x, z, rotY = 0, name) => {
            const wall = new this.THREE.Mesh(geom, mat);
            wall.position.set(x, this.wallHeight / 2, z);
            wall.rotation.y = rotY;
            wall.castShadow = true;
            wall.receiveShadow = true;
            wall.name = name;
            return wall;
        };

        walls.push(createWall(wallGeomH, wallMat, 0, -worldHalf, 0, 'wall-north'));
        walls.push(createWall(wallGeomH, wallMat, 0, worldHalf, 0, 'wall-south'));
        walls.push(createWall(wallGeomV, wallMat, -worldHalf, 0, 0, 'wall-west'));
        walls.push(createWall(wallGeomV, wallMat, worldHalf, 0, 0, 'wall-east'));

        // 2. Generate dynamic obstacles
        const obstacleMat = new this.THREE.MeshStandardMaterial({ color: 0x886688 });
        const spawnPadding = this.cellSize * 1.5; // Keep a clear area around the edges for initial spawns

        for (let i = 0; i < numObstacles; i++) {
            const ox = clamp((Math.random() - 0.5) * (this.worldSize - 2 * spawnPadding), -worldHalf + spawnPadding, worldHalf - spawnPadding);
            const oz = clamp((Math.random() - 0.5) * (this.worldSize - 2 * spawnPadding), -worldHalf + spawnPadding, worldHalf - spawnPadding);
            
            // Randomly choose between a box or cylinder for variety
            const geomType = Math.random() < 0.5 ? 'box' : 'cylinder';
            let obstacleGeom;
            const sizeX = clamp(this.obstacleMinSize + Math.random() * (this.obstacleMaxSize - this.obstacleMinSize), 1, this.worldSize / 4);
            const sizeZ = clamp(this.obstacleMinSize + Math.random() * (this.obstacleMaxSize - this.obstacleMinSize), 1, this.worldSize / 4);
            const radius = Math.max(sizeX, sizeZ) / 2; // For cylinder, use largest dim as radius base

            if (geomType === 'box') {
                obstacleGeom = new this.THREE.BoxGeometry(sizeX, this.obstacleHeight, sizeZ);
            } else {
                obstacleGeom = new this.THREE.CylinderGeometry(radius, radius, this.obstacleHeight, 16);
            }
            
            const obstacle = new this.THREE.Mesh(obstacleGeom, obstacleMat);
            obstacle.position.set(ox, this.obstacleHeight / 2, oz);
            obstacle.name = `obstacle-${i}`;
            obstacle.castShadow = true;
            obstacle.receiveShadow = true; // Obstacles can cast shadows on each other/ground

            if (Number.isFinite(obstacle.position.x) && Number.isFinite(obstacle.position.z)) {
                obstacles.push(obstacle);
            } else {
                logger.error(`Non-finite obstacle position detected during creation for obstacle ${i}! Skipping.`);
            }
        }

        // 3. Determine spawn and target points
        const spawnPoints = {
            player: new this.THREE.Vector3(),
            ai: new this.THREE.Vector3()
        };
        const targetPoints = {
            player: new this.THREE.Vector3(),
            ai: new this.THREE.Vector3()
        };

        // Place agents and targets ensuring a minimum distance
        this._placePoints(spawnPoints.player, spawnPoints.ai, worldHalf, this.minAgentsTargetsDistance);
        this._placePoints(targetPoints.player, targetPoints.ai, worldHalf, this.minAgentsTargetsDistance);
        
        // Ensure targets are not placed on top of agents
        while (spawnPoints.player.distanceTo(targetPoints.player) < 10) { // Small buffer
             this._placePoints(targetPoints.player, targetPoints.ai, worldHalf, this.minAgentsTargetsDistance);
        }
        while (spawnPoints.ai.distanceTo(targetPoints.ai) < 10) {
             this._placePoints(targetPoints.player, targetPoints.ai, worldHalf, this.minAgentsTargetsDistance);
        }

        logger.info('Level generation complete.');
        return { walls, obstacles, spawnPoints, targetPoints };
    }

    /**
     * Helper to place two points randomly while ensuring a minimum distance between them.
     * @param {THREE.Vector3} p1 - First point to place.
     * @param {THREE.Vector3} p2 - Second point to place.
     * @param {number} bounds - Half the world size for placement.
     * @param {number} minDist - Minimum required distance between p1 and p2.
     * @private
     */
    _placePoints(p1, p2, bounds, minDist) {
        const spawnRadius = bounds * 0.8;
        let placed = false;
        let attempts = 0;
        const maxAttempts = 100;

        while (!placed && attempts < maxAttempts) {
            const x1 = (Math.random() - 0.5) * spawnRadius * 2;
            const z1 = (Math.random() - 0.5) * spawnRadius * 2;
            const x2 = (Math.random() - 0.5) * spawnRadius * 2;
            const z2 = (Math.random() - 0.5) * spawnRadius * 2;

            p1.set(clamp(x1, -bounds + 5, bounds - 5), 0, clamp(z1, -bounds + 5, bounds - 5));
            p2.set(clamp(x2, -bounds + 5, bounds - 5), 0, clamp(z2, -bounds + 5, bounds - 5));

            if (p1.distanceTo(p2) >= minDist) {
                placed = true;
            }
            attempts++;
        }

        if (!placed) {
            logger.warn(`Could not place points with minimum distance after ${maxAttempts} attempts. Placing them close.`);
            p1.set(0, 0, -bounds / 2);
            p2.set(0, 0, bounds / 2);
        }
        // Ensure Y-coordinate is appropriate for agent/target height
        p1.y = 2; 
        p2.y = 2;
    }
}
// --- END OF FILE level-generator.js ---
